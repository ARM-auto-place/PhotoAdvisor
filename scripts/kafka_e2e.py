"""Disposable real Kafka + MinIO + containerized Advisor integration check.

Uses only uniquely named test containers; does not read .env or touch project topics.
Requires existing Docker images and prepared local model weights.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import uuid


def docker(*args):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return result.stdout.strip()


async def inside(phase):
    import boto3
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError
    from botocore.config import Config

    s3 = boto3.client('s3', endpoint_url='http://minio:9000',
                      aws_access_key_id=os.environ['TEST_ACCESS_KEY'],
                      aws_secret_access_key=os.environ['TEST_SECRET_KEY'],
                      region_name='us-east-1', config=Config(connect_timeout=2, read_timeout=3))
    bucket, key = 'photo-advisor-e2e', 'listings/e2e/front.jpg'
    if phase == 'setup':
        for attempt in range(60):
            admin = AIOKafkaAdminClient(bootstrap_servers='kafka:9092')
            try:
                await admin.start()
                try:
                    await admin.create_topics([NewTopic(name, 1, 1) for name in ('pa-input', 'pa-result', 'pa-dlq')])
                except TopicAlreadyExistsError:
                    pass
                s3.create_bucket(Bucket=bucket)
                s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={'Status': 'Enabled'})
                first = s3.put_object(Bucket=bucket, Key=key, Body=Path('/fixtures/front_left.jpg').read_bytes())
                second = s3.put_object(Bucket=bucket, Key=key, Body=b'corrupt replacement')
                fixture = dict(original=first['VersionId'], corrupt=second['VersionId'])
                s3.put_object(Bucket=bucket, Key='test-fixture.json', Body=json.dumps(fixture).encode())
                return
            except Exception:
                if attempt == 59:
                    raise
                await asyncio.sleep(1)
            finally:
                await admin.close()
    fixture = json.loads(s3.get_object(Bucket=bucket, Key='test-fixture.json')['Body'].read())
    producer = AIOKafkaProducer(bootstrap_servers='kafka:9092', enable_idempotence=True)
    consumer = AIOKafkaConsumer('pa-result', 'pa-dlq', bootstrap_servers='kafka:9092',
        group_id=f'test-{phase}', auto_offset_reset='earliest', enable_auto_commit=False)
    await producer.start()
    await consumer.start()
    def event(revision, kind='photoset.finalized', manual=None, version=None):
        photo = dict(photo_id='p1', bucket=bucket, object_key=key,
                     version_id=version or fixture['original'])
        if manual:
            photo['manual_category'] = manual
        value = dict(event_id=f'e2e-{revision}', correlation_id='e2e-request', listing_id='e2e',
                     listing_revision=revision, event_type=kind)
        if kind == 'image.upserted':
            value.update(image_id='p1', photo=photo)
        elif kind == 'image.deleted':
            value['image_id'] = 'p1'
        else:
            value['photos'] = [] if revision == 7 else [photo]
        return value
    async def send(value):
        await producer.send_and_wait('pa-input', json.dumps(value).encode(), key=b'e2e')
    try:
        if phase == 'before':
            await send(event(1, 'image.upserted'))
            await send(event(2))
            await send(event(2))
            await send(event(3, manual='front'))
            await send(event(4, manual='front'))
            await send(event(5, version=fixture['corrupt']))
            await send(event(6, 'image.deleted'))
            await send(event(7))
            await send(event(4, manual='front'))
            await producer.send_and_wait('pa-input', b'broken-json', key=b'e2e')
            expected, expected_dlq = set(range(1, 8)), 2
        else:
            await send(event(7))
            await send(event(9, manual='rear'))
            expected, expected_dlq = {*range(1, 8), 9}, 2
        results, dlq = [], []
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            batches = await consumer.getmany(timeout_ms=1000)
            for tp, messages in batches.items():
                for message in messages:
                    value = json.loads(message.value)
                    (results if tp.topic == 'pa-result' else dlq).append(value)
            if {r['listing_revision'] for r in results} >= expected and len(dlq) >= expected_dlq:
                break
        by_revision = {r['listing_revision']: r for r in results}
        assert set(by_revision) == expected, f'unexpected/missing revisions: {set(by_revision)}'
        assert len(results) == len(expected), 'duplicate result from replay'
        assert by_revision[1]['status'] == 'processing' and by_revision[1]['missing'] is None
        assert by_revision[2]['status'] == 'needs_review' and by_revision[2]['recognized'] == 0
        assert by_revision[3]['recognized'] == 1 and 'front' not in by_revision[3]['missing']
        assert by_revision[4]['recognized'] == 1, 'versioned original lost after replacement'
        assert by_revision[5]['status'] == 'failed' and by_revision[5]['recognized'] == 0
        assert by_revision[6]['status'] == 'processing'
        assert len(by_revision[7]['missing']) == 12
        if phase == 'after':
            assert by_revision[9]['recognized'] == 1 and 'rear' not in by_revision[9]['missing']
            assert 'front' in by_revision[9]['missing']
        assert len(dlq) == expected_dlq
        Path(f'/reports/kafka-e2e-{phase}.json').write_text(
            json.dumps(dict(results=results, dlq_errors=[d['error'] for d in dlq]), indent=2), encoding='utf-8')
        print(f'{phase}: {len(results)} real Kafka results, {len(dlq)} DLQ records; assertions passed')
    finally:
        await consumer.stop()
        await producer.stop()
        s3.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inside', choices=['setup', 'before', 'after'])
    parser.add_argument('--image', default='photo-advisor:quality-20261102')
    parser.add_argument('--backend', choices=['clip', 'trained'], default='clip')
    parser.add_argument('--model-dir', default='models/view-classifier-v2')
    args = parser.parse_args()
    if args.inside:
        asyncio.run(inside(args.inside))
        return
    root = Path(__file__).resolve().parents[1]
    run_id = f'photo-advisor-test-{uuid.uuid4().hex[:8]}'
    network, volume = run_id, f'{run_id}-state'
    names = {kind: f'{run_id}-{kind}' for kind in ('kafka', 'minio', 'advisor')}
    secret = secrets.token_urlsafe(24)
    containers = []
    network_created = volume_created = False
    def run(kind, image, options, command=()):
        name = names[kind]
        docker('run', '-d', '--name', name, '--network', network, '--network-alias', kind,
               *options, image, *command)
        containers.append(name)
    def driver(phase):
        print(docker('run', '--rm', '--network', network,
            '-e', 'TEST_ACCESS_KEY=advisor-test', '-e', f'TEST_SECRET_KEY={secret}',
            '--mount', f'type=bind,src={root / "scripts"},dst=/checks,readonly',
            '--mount', f'type=bind,src={root / "datasets" / "smoke"},dst=/fixtures,readonly',
            '--mount', f'type=bind,src={root / "reports"},dst=/reports',
            args.image, 'python', '/checks/kafka_e2e.py', '--inside', phase))
    try:
        docker('network', 'create', network)
        network_created = True
        docker('volume', 'create', volume)
        volume_created = True
        env = {
            'KAFKA_NODE_ID': '1', 'KAFKA_PROCESS_ROLES': 'broker,controller',
            'KAFKA_LISTENERS': 'PLAINTEXT://:9092,CONTROLLER://:9093',
            'KAFKA_ADVERTISED_LISTENERS': 'PLAINTEXT://kafka:9092',
            'KAFKA_CONTROLLER_LISTENER_NAMES': 'CONTROLLER',
            'KAFKA_CONTROLLER_QUORUM_VOTERS': '1@kafka:9093',
            'KAFKA_LISTENER_SECURITY_PROTOCOL_MAP': 'CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT',
            'KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR': '1',
            'KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR': '1',
            'KAFKA_TRANSACTION_STATE_LOG_MIN_ISR': '1',
            'KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS': '0',
            'KAFKA_HEAP_OPTS': '-Xms256m -Xmx256m',
        }
        run('kafka', 'apache/kafka:4.1.1', [item for k, v in env.items() for item in ('-e', f'{k}={v}')])
        run('minio', 'alfa-knowledge-minio:2025-10-15',
            ['-e', 'MINIO_ROOT_USER=advisor-test', '-e', f'MINIO_ROOT_PASSWORD={secret}'], ['server', '/data'])
        driver('setup')
        env = dict(SERVICE_TOKEN=secret, S3_ENDPOINT_URL='http://minio:9000',
            S3_ACCESS_KEY_ID='advisor-test', S3_SECRET_ACCESS_KEY=secret, S3_BUCKET='photo-advisor-e2e',
            KAFKA_ENABLED='true', KAFKA_BOOTSTRAP_SERVERS='kafka:9092', KAFKA_GROUP_ID='advisor-e2e',
            KAFKA_INPUT_TOPIC='pa-input', KAFKA_RESULT_TOPIC='pa-result', KAFKA_DLQ_TOPIC='pa-dlq')
        env.update(CV_BACKEND=args.backend, TRAINED_MODEL_DIR=args.model_dir)
        options = [item for k, v in env.items() for item in ('-e', f'{k}={v}')]
        options += ['--read-only', '--tmpfs', '/tmp:size=64m', '--memory', '2g', '--cpus', '2',
                    '--mount', f'type=bind,src={root / "models"},dst=/service/models,readonly',
                    '--mount', f'type=volume,src={volume},dst=/service/data']
        run('advisor', args.image, options)
        driver('before')
        docker('restart', '--time', '60', names['advisor'])
        driver('after')
        print('Real Kafka + MinIO + Advisor E2E passed, including persistent restart.')
    except BaseException:
        for name in containers:
            print(f'--- {name} ---\n{docker("logs", "--tail", "25", name)}')
        raise
    finally:
        for name in reversed(containers):
            docker('rm', '-f', name)
        if volume_created:
            docker('volume', 'rm', volume)
        if network_created:
            docker('network', 'rm', network)


if __name__ == '__main__':
    main()
