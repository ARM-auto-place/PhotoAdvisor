import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from botocore.exceptions import ReadTimeoutError
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter
from pydantic import ValidationError

from app.imaging import decode, duplicate_kind, fingerprint, quality
from app.main import create_app
from app.rules import summarize
from app.schemas import EvaluateParams, Photo, PhotoResult
from app.service import Advisor
from app.settings import Settings
from app.storage import S3Storage, StorageError

CHECKLIST = json.loads(Path('configs/checklist.json').read_text(encoding='utf-8'))


def picture(seed=1, size=(600, 400)):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    data = BytesIO()
    image.save(data, format='PNG')
    return data.getvalue()


def settings(**kw):
    return Settings(_env_file=None, service_token='test-secret', s3_bucket='test-private', **kw)


def test_blank_env_uses_iam_chain():
    config = settings(s3_endpoint_url='', s3_access_key_id='', s3_secret_access_key='')
    assert config.s3_endpoint_url is None
    assert config.s3_access_key_id is None


class Storage:
    def __init__(self, data):
        self.data = data
    def fetch(self, listing_id, photo):
        value = self.data[photo.object_key]
        if isinstance(value, Exception):
            raise value
        return value
    def validate_key(self, listing_id, key):
        if not key.startswith(f'listings/{listing_id}/'):
            raise StorageError('denied')
    def close(self):
        pass


class FakeVision:
    """Contract-only test double. Never used to claim recognition accuracy."""
    checklist = CHECKLIST
    calibration = {'validated': True}
    version = 'test-double'
    def __init__(self, labels):
        self.labels = iter(labels)
    def predict(self, images):
        results = []
        for image in images:
            label = next(self.labels)
            results.append(dict(category=label, candidate=label, recognition='known',
                                similarity=.3, margin=.1, quality_issues=[],
                                embedding=np.array([1., 0.], dtype=np.float32)))
        return results


def params(keys):
    return EvaluateParams(listing_id='a', photos=[dict(photo_id=str(i), object_key=k) for i, k in enumerate(keys)])


@pytest.mark.parametrize('label', list(CHECKLIST['classes']))
def test_category_coverage_contract(label):
    photo = PhotoResult(photo_id='1', category=label, candidate=label, accepted=True, quality='good', recognition='known')
    result = summarize('a', [photo], CHECKLIST, 'ru', 'test', True)
    present = result['required']['present'] + result['recommended']['present']
    assert len(present) <= 1
    assert present == ([label] if label in CHECKLIST['coverage'] else [])


def test_complete_required_and_conditional_damage():
    photos = [PhotoResult(photo_id=str(i), category=c['id'], accepted=True, recognition='known', quality='good')
              for i, c in enumerate(CHECKLIST['categories']) if c['priority']=='required']
    result = summarize('a', photos, CHECKLIST, 'ru', 'test', True)
    assert result['status'] == 'complete'
    assert next(s for s in result['suggestions'] if s['category']=='damage')['conditional']
    assert len(result['categories']) == 14


def test_uncertain_not_confirmed():
    p = PhotoResult(photo_id='1', category='left_side', candidate='left_side', recognition='uncertain')
    result = summarize('a', [p], CHECKLIST, 'ru', 'test', False)
    assert result['required']['uncertain'] == ['left_side']
    assert not result['required']['present']
    assert result['status']=='needs_review'


@pytest.mark.parametrize('key', ['https://host/a', '../a', 'listings/a/../b', '/a', 'a\\b', 'a/%2e/b', 'a//b', 'a/\x00b'])
def test_unsafe_keys(key):
    with pytest.raises(ValidationError):
        Photo(photo_id='a', object_key=key)


@pytest.mark.parametrize('data', [b'', b'not a picture', b'\xff\xd8broken'])
def test_broken_images(data):
    with pytest.raises(ValueError):
        decode(data, 1000000)


def test_unsupported_and_pixels():
    data = BytesIO()
    Image.new('RGB', (400, 300)).save(data, format='BMP')
    with pytest.raises(ValueError, match='unsupported'):
        decode(data.getvalue(), 1000000)
    with pytest.raises(ValueError, match='pixels'):
        decode(picture(), 10000)


def test_exif_orientation_and_rgb():
    data = BytesIO()
    im = Image.new('RGB', (300, 400), (240, 10, 20))
    exif = Image.Exif()
    exif[274] = 6
    im.save(data, format='JPEG', exif=exif)
    decoded, size = decode(data.getvalue(), 1000000)
    assert size == (400, 300)
    assert decoded.getpixel((10,10))[0] > 230


def test_blur_night_and_small():
    sharp, size = decode(picture(), 1000000)
    assert 'blurred' not in quality(sharp, size)
    blurred = sharp.filter(ImageFilter.GaussianBlur(15))
    assert 'blurred' in quality(blurred, size)
    assert 'too_small' in quality(sharp, (120,120))
    # A dark photograph with visible texture is not rejected merely for being dark.
    arr = np.asarray(sharp).copy() // 3
    assert 'underexposed' not in quality(Image.fromarray(arr), size)


def test_duplicates_and_useful_similar():
    data = picture()
    im, _ = decode(data, 1000000)
    fp = fingerprint(data, im)
    assert duplicate_kind(fp, fp, True, 1.)=='exact'
    another, _ = decode(picture(2), 1000000)
    assert duplicate_kind(fp, fingerprint(picture(2), another), True, .999) is None
    modified = (str('other hash'), fp[1], fp[2])
    assert duplicate_kind(fp, modified, True, .999)=='near'
    assert duplicate_kind(fp, modified, False, .999) is None


def test_reanalysis_deleted_replaced_and_partial_errors():
    storage = Storage({'listings/a/1':picture(), 'listings/a/2':picture(),
                       'listings/a/bad':b'broken', 'listings/a/3':picture(3)})
    advisor = Advisor(settings(), storage, FakeVision(['front','front','rear']))
    first = advisor.evaluate(params(['listings/a/1','listings/a/2','listings/a/bad']))
    assert first['required']['present']==['front']
    assert first['photo_results'][1]['duplicate_of']=='0'
    assert first['processing_status']=='partial'
    second = advisor.evaluate(params(['listings/a/3']))
    assert second['required']['present']==['rear']
    assert not second['photo_results'][0]['duplicate_of']
    empty = advisor.evaluate(params([]))
    assert empty['status']=='incomplete' and empty['processing_status']=='succeeded'
    advisor.close()


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_duplicate_promotion_keeps_one_representative_across_batches(batch_size):
    class InitiallyUncertain(FakeVision):
        count = 0

        def predict(self, images):
            predictions = super().predict(images)
            for prediction in predictions:
                if self.count < 2:
                    prediction.update(category='unknown', recognition='uncertain')
                self.count += 1
            return predictions

    keys = [f'listings/a/{i}' for i in range(4)]
    data = picture()
    advisor = Advisor(settings(batch_size=batch_size), Storage(dict.fromkeys(keys, data)),
                      InitiallyUncertain(['front'] * 4))
    try:
        result = advisor.evaluate(params(keys))
        photos = result['photo_results']
        assert [p['photo_id'] for p in photos if p['accepted']] == ['2']
        assert [p['duplicate_of'] for p in photos] == ['2', '2', None, '2']
        assert result['required']['present'] == ['front']
        assert result['required']['uncertain'] == []
        assert CHECKLIST['messages']['ru']['uncertain'] not in result['notices']
        assert result['status'] == 'incomplete'
    finally:
        advisor.close()


def test_poor_known_photo_requires_review_in_direct_api():
    photo = PhotoResult(photo_id='bad', category='front', candidate='front',
                        recognition='known', quality='poor', quality_issues=['blurred'])
    result = summarize('a', [photo], CHECKLIST, 'ru', 'test', True)
    assert result['status'] == 'needs_review'
    assert result['required']['present'] == []
    assert result['required']['uncertain'] == ['front']


def test_duplicate_promotion_preserves_exact_vs_near_evidence(monkeypatch):
    first, replacement = picture(), picture(2)
    monkeypatch.setattr('app.service.fingerprint', lambda data, _: (
        'first' if data == first else 'replacement', np.zeros(63, dtype=bool),
        np.zeros((32, 32), dtype=np.float32)))
    vision = FakeVision(['front'] * 4)
    original_predict = vision.predict
    count = iter(range(4))

    def predict(images):
        outputs = original_predict(images)
        for output in outputs:
            if next(count) < 2:
                output.update(category='unknown', recognition='uncertain')
        return outputs

    vision.predict = predict
    keys = [f'listings/a/{i}' for i in range(4)]
    advisor = Advisor(settings(batch_size=1), Storage(dict(zip(
        keys, [first, first, replacement, replacement]))), vision)
    try:
        photos = advisor.evaluate(params(keys))['photo_results']
        assert [p['duplicate_of'] for p in photos] == ['2', '2', None, '2']
        assert [p['duplicate_kind'] for p in photos] == ['near', 'near', None, 'exact']
        assert sum(p['accepted'] for p in photos) == 1
    finally:
        advisor.close()


def test_superseded_poor_duplicate_does_not_require_review():
    photos = [PhotoResult(photo_id='bad', category='front', candidate='front',
                          recognition='known', quality='poor', duplicate_of='good'),
              PhotoResult(photo_id='good', category='front', candidate='front',
                          recognition='known', quality='good', accepted=True)]
    result = summarize('a', photos, CHECKLIST, 'ru', 'test', True)
    assert result['status'] == 'incomplete'
    assert result['required']['present'] == ['front']
    assert not result['required']['uncertain']


def test_s3_bounds_closing_and_scope():
    client = Mock()
    body = BytesIO(b'x'*2048)
    client.get_object.return_value = {'Body':body, 'ContentLength':2048}
    storage = S3Storage(settings(max_image_bytes=1024), client)
    with pytest.raises(StorageError, match='too_large'):
        storage.fetch('a', Photo(photo_id='1',object_key='listings/a/1'))
    assert body.closed
    body = BytesIO(b'x'*2048)
    client.get_object.return_value = {'Body':body}
    with pytest.raises(StorageError, match='too_large'):
        storage.fetch('a', Photo(photo_id='1',object_key='listings/a/1'))
    assert body.closed
    with pytest.raises(StorageError, match='not_allowed'):
        storage.validate_key('a', 'listings/ab/1')


def test_s3_timeout():
    client = Mock()
    client.get_object.side_effect = ReadTimeoutError(endpoint_url='http://local')
    with pytest.raises(StorageError, match='s3_unavailable'):
        S3Storage(settings(),client).fetch('a',Photo(photo_id='1', object_key='listings/a/1'))


@pytest.fixture
def api():
    advisor = Advisor(settings(), Storage({}), FakeVision([]))
    with TestClient(create_app(settings(), advisor)) as client:
        client.headers['Authorization']='Bearer test-secret'
        yield client


def request(**kw):
    return {'jsonrpc':'2.0','id':1,'method':'photos.evaluate','params':{'listing_id':'a','photos':[]},**kw}


def test_rpc(api):
    assert api.get('/health/ready').status_code==200
    assert api.post('/rpc',json=request()).json()['result']['status']=='incomplete'
    assert api.post('/rpc',json=request(method='missing')).json()['error']['code']==-32601
    assert api.post('/rpc',content='{').json()['error']['code']==-32700
    assert api.post('/rpc',json=request(id=True)).json()['error']['code']==-32600
    assert api.post('/rpc',json=[]).json()['error']['code']==-32600
    assert api.post('/rpc',json=request(params={'listing_id':'a','photos':[], 'url':'http://evil'})).json()['error']['code']==-32602
    assert api.post('/rpc',json=request(),headers={'Authorization':'wrong'}).status_code==401
    notification = request()
    del notification['id']
    assert api.post('/rpc',json=notification).status_code==204
    assert len(api.post('/rpc',json=[notification,request()]).json())==1
    assert api.post('/rpc',content=b'x'*65537).status_code==413


@pytest.mark.parametrize('content', [
    b'{"jsonrpc":"2.0","id":1e999,"method":"photos.evaluate","params":{"listing_id":"a","photos":[]}}',
    b'[' * 2000 + b'0' + b']' * 2000,
])
def test_rpc_invalid_numbers_and_deep_json_do_not_crash(api, content):
    response = api.post('/rpc', content=content)
    assert response.status_code == 200
    assert response.json()['error']['code'] == -32700
    assert api.post('/rpc', json=request()).json()['result']['status'] == 'incomplete'


def test_rpc_busy():
    advisor = Advisor(settings(), Storage({}), FakeVision([]))
    with TestClient(create_app(settings(), advisor)) as client:
        advisor.lock.acquire()
        try:
            response = client.post('/rpc',json=request(),headers={'Authorization':'Bearer test-secret'})
            assert response.json()['error']['code']==-32003
        finally:
            advisor.lock.release()


def test_detection_visibility_rules():
    from app.inference import vehicle_visibility
    assert vehicle_visibility('interior_front', [], (600,400))==[]
    assert vehicle_visibility('engine_bay', [], (600,400))==[]
    assert vehicle_visibility('front', [], (600,400))==['car_not_detected']
    assert vehicle_visibility('front', [[20,20,300,300]]*2, (600,400))==['multiple_cars']
    assert 'possible_crop' in vehicle_visibility('front', [[0,20,590,390]], (600,400))
    assert 'car_too_small' in vehicle_visibility('rear', [[30,30,60,60]], (600,400))


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        EvaluateParams(listing_id='a',photos=[{'photo_id':'same','object_key':'listings/a/1'}]*2)


def test_rpc_photo_limit(api):
    photos = [{'photo_id':str(i),'object_key':f'listings/a/{i}'} for i in range(21)]
    assert api.post('/rpc',json=request(params={'listing_id':'a','photos':photos})).json()['error']['code']==-32602
