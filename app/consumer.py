import asyncio
import base64
import json
import logging
import ssl
from datetime import datetime, timezone

from pydantic import ValidationError

from app.events import PhotoEvent
from app.pipeline import Pipeline, RetryableError
from app.state import ContractError, State

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.propagate = False


def log(action, **fields):
    logger.info(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                component='kafka', action=action, **fields)))


class KafkaWorker:
    def __init__(self, settings, advisor):
        self.settings = settings
        self.state = State(settings.state_path)
        self.pipeline = Pipeline(settings, advisor, self.state)
        self.stopping = asyncio.Event()
        self.task = None
        self.consumer = self.producer = None

    async def start(self):
        try:
            await self._start()
        except BaseException:
            await self.stop()
            raise

    async def _start(self):
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

        config = self.settings
        connection = dict(bootstrap_servers=config.kafka_bootstrap_servers,
                          security_protocol=config.kafka_security_protocol)
        if 'SSL' in config.kafka_security_protocol:
            connection['ssl_context'] = ssl.create_default_context(cafile=config.kafka_ssl_cafile)
        if config.kafka_security_protocol.startswith('SASL'):
            connection.update(sasl_mechanism=config.kafka_sasl_mechanism,
                              sasl_plain_username=config.kafka_sasl_username,
                              sasl_plain_password=config.kafka_sasl_password.get_secret_value())
        self.consumer = AIOKafkaConsumer(
            config.kafka_input_topic, **connection, group_id=config.kafka_group_id,
            enable_auto_commit=False, auto_offset_reset='earliest',
            max_poll_records=1, max_poll_interval_ms=3_600_000,
            fetch_max_bytes=1_048_576, max_partition_fetch_bytes=1_048_576)
        self.producer = AIOKafkaProducer(**connection, enable_idempotence=True,
                                        max_request_size=2_097_152)
        await self.producer.start()
        await self.consumer.start()
        self.task = asyncio.create_task(self.run(), name='photo-advisor-kafka')

    @property
    def ready(self):
        return self.task is not None and not self.task.done() and not self.stopping.is_set()

    async def flush(self):
        while row := self.state.pending():
            topic = (self.settings.kafka_result_topic if row['kind'] == 'result'
                     else self.settings.kafka_dlq_topic)
            await asyncio.wait_for(self.producer.send_and_wait(
                topic, row['payload'].encode(), key=row['listing_id'].encode()), timeout=30)
            self.state.sent(row['seq'])

    async def handle(self, record):
        record_id = f'{record.topic}:{record.partition}:{record.offset}'
        event = None
        try:
            if record.value is None or len(record.value) > 262144:
                raise ContractError('invalid_event_size')
            event = PhotoEvent.model_validate_json(record.value)
            if record.key != event.listing_id.encode():
                raise ContractError('message_key_must_be_listing_id')
            self.pipeline.validate(event)
            if not self.state.needed(event):
                self.state.save(event, None)
                log('duplicate_or_stale', event_id=event.event_id, listing_id=event.listing_id,
                    correlation_id=event.correlation_id)
                return
        except (ValidationError, ContractError) as exc:
            self.reject(record, record_id, 'invalid_contract', event)
            log('rejected', record_id=record_id, error=type(exc).__name__)
            return

        attempt_id = f'event:{event.event_id}'
        while not self.stopping.is_set():
            attempt = self.state.attempt(attempt_id)
            try:
                if attempt > self.settings.kafka_max_attempts:
                    raise RetryableError('attempts_exhausted')
                result = await asyncio.to_thread(self.pipeline.analyze, event)
                if any(p.get('error') for p in result['photos']):
                    self.reject(record, record_id, 'photo_analysis_failed', event, result)
                else:
                    self.state.save(event, result)
                log('processed', event_id=event.event_id, correlation_id=event.correlation_id,
                    listing_id=event.listing_id, revision=event.listing_revision, status=result['status'])
                return
            except RetryableError:
                log('retry', event_id=event.event_id, correlation_id=event.correlation_id, attempt=attempt)
                if attempt >= self.settings.kafka_max_attempts:
                    result = self.pipeline.envelope(event, 'failed')
                    result['error'] = 'attempts_exhausted'
                    self.reject(record, record_id, 'attempts_exhausted', event, result)
                    return
                try:
                    await asyncio.wait_for(self.stopping.wait(),
                                           timeout=self.settings.kafka_retry_backoff * 2**(attempt-1))
                except asyncio.TimeoutError:
                    pass
        raise InterruptedError('shutdown before processing completed')

    def reject(self, record, record_id, error, event=None, result=None):
        payload = dict(schema_version=1, dlq_id=record_id, error=error,
                       source=dict(topic=record.topic, partition=record.partition, offset=record.offset),
                       event_id=event.event_id if event else None,
                       correlation_id=event.correlation_id if event else None,
                       payload_base64=base64.b64encode((record.value or b'')[:262144]).decode(),
                       payload_truncated=bool(record.value and len(record.value) > 262144))
        self.state.rejected(record_id, payload, event, result)

    async def run(self):
        from aiokafka import TopicPartition

        try:
            await self.flush()
            while not self.stopping.is_set():
                batches = await self.consumer.getmany(timeout_ms=500, max_records=1)
                for records in batches.values():
                    for record in records:
                        await self.handle(record)
                        await self.flush()
                        await self.consumer.commit({
                            TopicPartition(record.topic, record.partition): record.offset + 1})
        except InterruptedError:
            if not self.stopping.is_set():
                raise
        except Exception as exc:
            log('worker_stopped', error=type(exc).__name__)
            raise

    async def stop(self):
        self.stopping.set()
        try:
            if self.task is not None:
                await self.task
        finally:
            try:
                if self.consumer is not None:
                    await self.consumer.stop()
            finally:
                try:
                    if self.producer is not None:
                        await asyncio.wait_for(self.producer.stop(), timeout=30)
                finally:
                    self.state.close()
