import asyncio
import json
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from app.consumer import KafkaWorker
from app.events import PhotoEvent, REQUIRED
from app.pipeline import Pipeline
from app.settings import Settings
from app.state import State
from app.storage import S3Storage, StorageError


def config(tmp_path):
    return Settings(_env_file=None, s3_bucket='test-bucket', kafka_enabled=True,
                    kafka_bootstrap_servers='unused:9092', kafka_group_id='test-group',
                    kafka_input_topic='test-input', kafka_result_topic='test-output',
                    kafka_dlq_topic='test-dlq', kafka_retry_backoff=0,
                    state_path=tmp_path / 'state.sqlite3')


def photo(**changes):
    return dict(photo_id='p1', bucket='test-bucket', object_key='listings/a/p1.jpg',
                etag='"original"', **changes)


def event(revision=1, kind='photoset.finalized', **changes):
    values = dict(event_id=f'e{revision}', correlation_id='correlation', listing_id='a',
                  listing_revision=revision, event_type=kind)
    if kind == 'photoset.finalized':
        values['photos'] = [photo()]
    elif kind == 'image.upserted':
        values.update(image_id='p1', photo=photo())
    else:
        values['image_id'] = 'p1'
    values.update(changes)
    return PhotoEvent.model_validate(values)


def record(value, offset=0):
    return SimpleNamespace(topic='test-input', partition=0, offset=offset, key=b'a',
                           value=value.model_dump_json().encode() if isinstance(value, PhotoEvent) else value)


def advisor():
    obj = Mock()
    obj.lock = Lock()
    obj.evaluate.return_value = dict(photo_results=[dict(
        photo_id='p1', category='front', recognition='known', accepted=True,
        confidence=None, similarity=.4, quality='good', error=None)], model={'version': 'test-double'})
    return obj


def latest(worker):
    return json.loads(worker.state.db.execute('SELECT result FROM listings WHERE id="a"').fetchone()[0])


def test_duplicate_and_restart_preserve_outbox(tmp_path):
    async def scenario():
        cv = advisor()
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        await worker.handle(record(event(), 1))
        assert cv.evaluate.call_count == 1
        assert worker.state.db.execute('SELECT count(*) FROM outbox').fetchone()[0] == 1
        worker.state.close()
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        worker.producer = SimpleNamespace(send_and_wait=AsyncMock())
        await worker.flush()
        assert cv.evaluate.call_count == 1
        assert not worker.state.pending()
        assert worker.producer.send_and_wait.call_args.kwargs['key'] == b'a'
        worker.state.close()
    asyncio.run(scenario())


def test_processing_finalized_delete_and_out_of_order(tmp_path):
    async def scenario():
        cv = advisor()
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event(kind='image.upserted')))
        assert latest(worker)['status'] == 'processing'
        assert latest(worker)['missing'] is None
        cv.evaluate.assert_not_called()
        await worker.handle(record(event(3)))
        assert latest(worker)['recognized'] == 1
        assert latest(worker)['status'] == 'completed'
        assert len(latest(worker)['missing']) == 11
        await worker.handle(record(event(2, 'image.deleted')))
        assert latest(worker)['listing_revision'] == 3
        await worker.handle(record(event(4, 'image.deleted')))
        assert latest(worker)['missing'] is None
        cv.evaluate.return_value['photo_results'] = []
        await worker.handle(record(event(5, photos=[])))
        assert latest(worker)['recognized'] == 0
        assert latest(worker)['missing'] == list(REQUIRED)
        worker.state.close()
    asyncio.run(scenario())


def test_replacement_and_manual_category(tmp_path):
    async def scenario():
        cv = advisor()
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        replacement = photo(manual_category='fender_front_left', version_id='version-2')
        await worker.handle(record(event(2, photos=[replacement])))
        supplied = cv.evaluate.call_args.args[0].photos[0]
        assert supplied.version_id == 'version-2'
        result = latest(worker)
        assert 'front' in result['missing']
        assert 'fender_front_left' not in result['missing']
        assert result['photos'][0]['source'] == 'manual'
        assert result['photos'][0]['confidence'] is None
        assert result['photos'][0]['candidate'] == 'front'
        assert result['photos'][0]['category'] == 'fender_front_left'
        feedback = worker.state.db.execute('SELECT payload FROM review_feedback').fetchall()
        assert len(feedback) == 1
        correction = json.loads(feedback[0][0])
        assert correction['model_candidate'] == 'front'
        assert correction['corrected_category'] == 'fender_front_left'
        assert correction['training_permission'] is False
        assert correction['review_status'] == 'pending'
        await worker.handle(record(event(2, photos=[replacement])))
        assert worker.state.db.execute('SELECT COUNT(*) FROM review_feedback').fetchone()[0] == 1
        from scripts.export_feedback import export_feedback
        assert export_feedback(config(tmp_path).state_path, tmp_path/'feedback.json') == 1
        with pytest.raises(ValueError, match='immutable'):
            export_feedback(config(tmp_path).state_path, tmp_path/'feedback.json')
        worker.state.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('error', ['s3_unavailable', 'inference_failed'])
def test_retries_exhausted_atomic_failed_and_dlq(tmp_path, error):
    async def scenario():
        cv = advisor()
        cv.evaluate.return_value['photo_results'][0]['error'] = error
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        assert cv.evaluate.call_count == 3
        assert latest(worker)['status'] == 'failed'
        assert latest(worker)['missing'] is None
        assert [r[0] for r in worker.state.db.execute('SELECT kind FROM outbox ORDER BY seq')] == ['result', 'dlq']
        await worker.handle(record(event()))
        assert cv.evaluate.call_count == 3
        worker.state.close()
    asyncio.run(scenario())


def test_retry_budget_survives_restart(tmp_path):
    async def scenario():
        cv = advisor()
        worker = KafkaWorker(config(tmp_path), cv)
        for _ in range(3):
            worker.state.attempt('event:e1')
        worker.state.close()
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        cv.evaluate.assert_not_called()
        assert latest(worker)['status'] == 'failed'
        worker.state.close()
    asyncio.run(scenario())


def test_transient_error_recovers(tmp_path):
    async def scenario():
        cv = advisor()
        good = cv.evaluate.return_value
        bad = dict(good, photo_results=[dict(good['photo_results'][0], error='s3_unavailable')])
        cv.evaluate.side_effect = [bad, good]
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        assert cv.evaluate.call_count == 2
        assert latest(worker)['status'] == 'completed'
        assert worker.state.db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 0
        worker.state.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('recognition,error', [('uncertain', None), ('unknown', None),
                                               ('error', 'unsupported_format')])
def test_uncertain_or_corrupt_never_covers_category(tmp_path, recognition, error):
    async def scenario():
        cv = advisor()
        cv.evaluate.return_value['photo_results'][0].update(
            recognition=recognition, accepted=False, error=error)
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(event()))
        result = latest(worker)
        assert result['status'] == ('failed' if error else 'needs_review')
        assert result['recognized'] == 0
        assert 'front' in result['missing']
        assert result['uncertain'] == ['p1']
        worker.state.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('value', [b'not json', b'null', b'{}', None])
def test_invalid_event_goes_to_dlq(tmp_path, value):
    async def scenario():
        worker = KafkaWorker(config(tmp_path), advisor())
        await worker.handle(record(value))
        assert worker.state.pending()['kind'] == 'dlq'
        assert worker.state.db.execute('SELECT count(*) FROM listings').fetchone()[0] == 0
        worker.state.close()
    asyncio.run(scenario())


def test_conflicting_id_and_revision_do_not_overwrite(tmp_path):
    async def scenario():
        worker = KafkaWorker(config(tmp_path), advisor())
        await worker.handle(record(event()))
        await worker.handle(record(event(2, event_id='e1'), 1))
        await worker.handle(record(event(event_id='another'), 2))
        assert latest(worker)['listing_revision'] == 1
        assert worker.state.db.execute('SELECT count(*) FROM outbox WHERE kind="dlq"').fetchone()[0] == 2
        worker.state.close()
    asyncio.run(scenario())


def test_publish_failure_keeps_outbox_for_restart(tmp_path):
    async def scenario():
        worker = KafkaWorker(config(tmp_path), advisor())
        await worker.handle(record(event()))
        worker.producer = SimpleNamespace(send_and_wait=AsyncMock(side_effect=OSError()))
        with pytest.raises(OSError):
            await worker.flush()
        assert worker.state.pending()
        worker.state.close()
    asyncio.run(scenario())


def test_run_commits_only_after_publication(tmp_path):
    async def scenario():
        from aiokafka import TopicPartition
        worker = KafkaWorker(config(tmp_path), advisor())
        calls = []
        async def getmany(**kwargs):
            return {TopicPartition('test-input', 0): [record(event(), 7)]}
        async def publish(*args, **kwargs):
            calls.append('publish')
        async def commit(offsets):
            assert offsets == {TopicPartition('test-input', 0): 8}
            calls.append('commit')
            worker.stopping.set()
        worker.consumer = SimpleNamespace(getmany=getmany, commit=commit)
        worker.producer = SimpleNamespace(send_and_wait=publish)
        await worker.run()
        assert calls == ['publish', 'commit']
        worker.state.close()
    asyncio.run(scenario())


def test_transaction_rolls_back_if_outbox_write_fails(tmp_path, monkeypatch):
    state = State(tmp_path / 'state.sqlite3')
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(state, 'enqueue', fail)
    with pytest.raises(OSError):
        state.save(event(), {'result_id': 'result'})
    assert state.db.execute('SELECT count(*) FROM events').fetchone()[0] == 0
    assert state.db.execute('SELECT count(*) FROM listings').fetchone()[0] == 0
    state.close()


def test_s3_reads_exact_version_with_etag(tmp_path):
    client = Mock()
    client.get_object.return_value = {'Body': Mock(read=Mock(side_effect=[b'image', b'']))}
    storage = S3Storage(config(tmp_path), client)
    obj = event(photos=[photo(version_id='v2')]).photos[0]
    assert storage.fetch('a', obj) == b'image'
    client.get_object.assert_called_once_with(Bucket='test-bucket', Key='listings/a/p1.jpg',
                                              VersionId='v2', IfMatch='"original"')
    client.get_object.return_value['Body'].close.assert_called_once()


def test_s3_replacement_is_permanent_error(tmp_path):
    client = Mock()
    client.get_object.side_effect = ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'GetObject')
    with pytest.raises(StorageError, match='s3_object_unavailable'):
        S3Storage(config(tmp_path), client).fetch('a', event().photos[0])


def test_manifest_requires_object_identity():
    p = photo()
    del p['etag']
    with pytest.raises(ValidationError):
        event(photos=[p])


def test_all_twelve_categories_can_be_covered(tmp_path):
    cv = advisor()
    manifest = [dict(photo(), photo_id=str(i), manual_category=c) for i, c in enumerate(REQUIRED)]
    cv.evaluate.return_value['photo_results'] = [dict(
        photo_id=str(i), category='unknown', recognition='unknown', accepted=False,
        confidence=None, quality='good', error=None) for i in range(12)]
    result = Pipeline(config(tmp_path), cv, None).analyze(event(photos=manifest))
    assert result['recognized'] == 12
    assert result['missing'] == []
    assert result['status'] == 'completed'


@pytest.mark.parametrize('quality', ['poor', 'uncertain', 'unavailable'])
def test_manual_correction_cannot_bypass_quality(tmp_path, quality):
    cv = advisor()
    cv.evaluate.return_value['photo_results'][0].update(quality=quality, accepted=False)
    result = Pipeline(config(tmp_path), cv, None).analyze(
        event(photos=[photo(manual_category='rear')]))
    assert result['recognized'] == 0
    assert result['status'] == 'needs_review'
    assert result['uncertain'] == ['p1']
    p = result['photos'][0]
    assert p['manual_category'] == 'rear'
    assert p['manual_applied'] is False
    assert p['manual_rejection_reason'] == 'quality_not_good'
    assert p['source'] == 'model'
    assert p['candidate'] == 'front'


def test_manual_correction_cannot_bypass_permanent_photo_error(tmp_path):
    cv = advisor()
    cv.evaluate.return_value['photo_results'][0].update(
        quality='unavailable', accepted=False, recognition='error', error='unsupported_format')
    result = Pipeline(config(tmp_path), cv, None).analyze(
        event(photos=[photo(manual_category='rear')]))
    assert result['recognized'] == 0
    assert result['status'] == 'failed'
    assert result['photos'][0]['manual_rejection_reason'] == 'photo_error'


@pytest.mark.parametrize('kind', ['exact', 'near'])
@pytest.mark.parametrize('correct_root', [True, False])
def test_manual_duplicate_group_has_one_category_and_survives_restart(tmp_path, kind, correct_root):
    async def scenario():
        import copy
        cv = advisor()
        base = cv.evaluate.return_value['photo_results'][0]
        cv.evaluate.return_value['photo_results'] = [
            dict(base, candidate='front'),
            dict(base, photo_id='p2', candidate='front', accepted=False,
                 duplicate_of='p1', duplicate_kind=kind),
            dict(base, photo_id='p3', candidate='front', accepted=False,
                 duplicate_of='p1', duplicate_kind='exact')]
        original = copy.deepcopy(cv.evaluate.return_value)
        references = [dict(photo(), manual_category='rear' if correct_root else None),
                      dict(photo(), photo_id='p2', manual_category='left_side'),
                      dict(photo(), photo_id='p3', manual_category='right_side')]
        expected_id, category = ('p1', 'rear') if correct_root else ('p2', 'left_side')
        worker = KafkaWorker(config(tmp_path), cv)
        evt = event(photos=references)
        await worker.handle(record(evt))
        result = latest(worker)
        assert result['recognized'] == 1
        assert result['missing'] == [c for c in REQUIRED if c != category]
        assert result['uncertain'] == []
        assert result['status'] == 'completed'
        assert cv.evaluate.return_value == original
        for p in result['photos']:
            assert p['accepted'] is (p['image_id'] == expected_id)
            assert p['manual_applied'] is (p['image_id'] == expected_id)
            assert p['duplicate_of'] == (None if p['image_id'] == expected_id else expected_id)
            if p['image_id'] != expected_id and p['manual_category'] is not None:
                assert p['manual_rejection_reason'] == 'duplicate_photo'
        feedback = [json.loads(r[0]) for r in worker.state.db.execute('SELECT payload FROM review_feedback')]
        assert len(feedback) == 1
        assert feedback[0]['image_id'] == expected_id
        assert feedback[0]['corrected_category'] == category
        worker.state.close()
        worker = KafkaWorker(config(tmp_path), cv)
        await worker.handle(record(evt, 1))
        assert latest(worker) == result
        assert cv.evaluate.call_count == 1
        assert worker.state.db.execute('SELECT count(*) FROM outbox').fetchone()[0] == 1
        worker.state.close()
    asyncio.run(scenario())


def test_publish_failure_prevents_offset_commit(tmp_path):
    async def scenario():
        from aiokafka import TopicPartition
        worker = KafkaWorker(config(tmp_path), advisor())
        worker.consumer = SimpleNamespace(
            getmany=AsyncMock(return_value={TopicPartition('test-input', 0): [record(event())]}),
            commit=AsyncMock())
        worker.producer = SimpleNamespace(send_and_wait=AsyncMock(side_effect=OSError()))
        with pytest.raises(OSError):
            await worker.run()
        worker.consumer.commit.assert_not_awaited()
        assert worker.state.pending()
        worker.state.close()
    asyncio.run(scenario())


def test_graceful_stop_waits_for_current_inference(tmp_path):
    async def scenario():
        worker = KafkaWorker(config(tmp_path), advisor())
        entered, finish = asyncio.Event(), asyncio.Event()
        async def current_work():
            entered.set()
            await finish.wait()
            worker.state.save(event(), worker.pipeline.envelope(event(), 'completed'))
        worker.consumer = SimpleNamespace(stop=AsyncMock())
        worker.producer = SimpleNamespace(stop=AsyncMock())
        worker.task = asyncio.create_task(current_work())
        await entered.wait()
        stopping = asyncio.create_task(worker.stop())
        await asyncio.sleep(0)
        worker.consumer.stop.assert_not_awaited()
        finish.set()
        await stopping
        worker.consumer.stop.assert_awaited_once()
        worker.producer.stop.assert_awaited_once()
        state = State(config(tmp_path).state_path)
        assert state.pending()
        state.close()
    asyncio.run(scenario())


def test_readiness_tracks_worker_and_shutdown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    cv = advisor()
    worker = SimpleNamespace(ready=True, task=object(), start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr('app.consumer.KafkaWorker', lambda *_: worker)
    settings = config(tmp_path).model_copy(update={'service_token': Settings(service_token='test').service_token})
    with TestClient(create_app(settings, cv)) as client:
        assert client.get('/health/ready').status_code == 200
        worker.ready = False
        assert client.get('/health/ready').status_code == 503
    worker.stop.assert_awaited_once()
    cv.close.assert_called_once()
