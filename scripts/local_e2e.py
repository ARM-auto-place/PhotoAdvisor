"""Real models + FastAPI + boto3 with stubbed S3 transport, no cloud or secrets.

This does not replace a real MinIO/S3 deployment integration test.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import secrets
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from app.inference import load_vision
from app.main import create_app
from app.service import Advisor
from app.settings import Settings
from app.storage import S3Storage

parser = argparse.ArgumentParser()
parser.add_argument('photo', type=Path)
parser.add_argument('--output', type=Path, default=Path('reports/local-e2e.json'))
parser.add_argument('--trained-model', type=Path,
                    help='Explicit local trained checkpoint; defaults to the CLIP backend')
args = parser.parse_args()
overrides = ({'cv_backend': 'trained', 'trained_model_dir': args.trained_model}
             if args.trained_model else {})
# Stubber consumes requests in order; one downloader keeps this transport test deterministic.
settings = Settings(_env_file=None, service_token=secrets.token_urlsafe(32),
                    s3_bucket='local-test-bucket', download_workers=1, batch_size=2, **overrides)
vision = load_vision(settings)
client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='local-test', aws_secret_access_key='local-test')
stub = Stubber(client)
photo = args.photo.read_bytes()


def object_response(key, data):
    stub.add_response('get_object', {'Body': StreamingBody(BytesIO(data), len(data)), 'ContentLength':len(data)},
                      {'Bucket':settings.s3_bucket, 'Key':key})


object_response('listings/test/1',photo)
object_response('listings/test/2',photo)
object_response('listings/test/3',photo)
stub.add_client_error('get_object', service_error_code='NoSuchKey', http_status_code=404,
                      expected_params={'Bucket':settings.s3_bucket,'Key':'listings/test/missing'})
object_response('listings/test/replaced',b'broken image')
advisor = Advisor(settings,S3Storage(settings,client),vision)
stub.activate()
try:
    with TestClient(create_app(settings,advisor)) as api:
        api.headers['Authorization'] = f'Bearer {settings.service_token.get_secret_value()}'
        def call(photos):
            started=time.perf_counter()
            response=api.post('/rpc',json={'jsonrpc':'2.0','id':'e2e','method':'photos.evaluate',
                              'params':{'listing_id':'test','photos':photos}})
            assert response.status_code==200
            return response.json()['result'], time.perf_counter()-started
        first, elapsed=call([{'photo_id':'1','object_key':'listings/test/1'},
                             {'photo_id':'2','object_key':'listings/test/2'},
                             {'photo_id':'3','object_key':'listings/test/3'},
                             {'photo_id':'missing','object_key':'listings/test/missing'}])
        assert first['processing_status']=='partial'
        assert first['photo_results'][1]['duplicate_of']=='1'
        assert first['photo_results'][2]['duplicate_of']=='1'
        assert first['photo_results'][3]['error']=='s3_object_unavailable'
        assert sum(p['accepted'] for p in first['photo_results']) <= 1
        second,_=call([{'photo_id':'new','object_key':'listings/test/replaced'}])
        assert second['status']=='failed'
        assert not second['required']['present']
        empty,_=call([])
        assert empty['status']=='incomplete'
        for malformed in (b'['*2000+b'0'+b']'*2000,
                          b'{"jsonrpc":"2.0","id":1e999,"method":"photos.evaluate"}'):
            invalid=api.post('/rpc',content=malformed)
            assert invalid.status_code==200
            assert invalid.json()['error']['code']==-32700
        assert api.get('/health/ready').status_code==200
        report={'kind':'real_models_fastapi_boto3_stubbed_transport','batch_seconds':elapsed,
                'model_version':vision.version,
                'backend':settings.cv_backend, 'duplicate_batch_boundary_checked':True,
                'malformed_json_recovery_checked':True,
                'first':first,'replacement':second,'empty_status':empty['status']}
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    stub.assert_no_pending_responses()
    print(f'Local E2E passed; first batch {elapsed:.3f}s; S3 transport was stubbed.')
finally:
    stub.deactivate()
