import hmac
import json
import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.schemas import EvaluateParams
from app.settings import Settings


def rpc_error(code, message, request_id=None):
    return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('nonfinite JSON number')
    return parsed


def create_app(settings=None, advisor=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        if not settings.service_token.get_secret_value() or not settings.s3_bucket:
            raise RuntimeError('SERVICE_TOKEN and S3_BUCKET must be configured')
        if advisor is None:
            from app.inference import load_vision
            from app.service import Advisor
            from app.storage import S3Storage
            cv_settings = settings.model_copy(update={'checklist_path': settings.kafka_checklist_path}) \
                if settings.kafka_enabled else settings
            vision = load_vision(cv_settings)
            app.state.advisor = Advisor(settings, S3Storage(settings), vision)
        else:
            app.state.advisor = advisor
        worker = None
        try:
            if settings.kafka_enabled:
                from app.consumer import KafkaWorker
                worker = KafkaWorker(settings, app.state.advisor)
                await worker.start()
            app.state.kafka_worker = worker
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                if worker is not None and worker.task is not None:
                    await worker.stop()
            finally:
                app.state.advisor.close()

    app = FastAPI(title='Photo Advisor', lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get('/health/live')
    def live():
        return {'status': 'alive'}

    @app.get('/health/ready')
    def ready():
        ok = getattr(app.state, 'ready', False)
        worker = getattr(app.state, 'kafka_worker', None)
        ok = ok and (worker is None or worker.ready)
        return JSONResponse({'status': 'ready' if ok else 'not_ready'}, status_code=200 if ok else 503)

    @app.post('/rpc')
    async def rpc(request: Request):
        expected = settings.service_token.get_secret_value()
        supplied = request.headers.get('authorization', '')
        if not expected or not hmac.compare_digest(supplied.encode(), f'Bearer {expected}'.encode()):
            return JSONResponse(rpc_error(-32001, 'Unauthorized'), status_code=401)
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > settings.max_rpc_bytes:
                return JSONResponse(rpc_error(-32600, 'Request too large'), status_code=413)
            raw.extend(chunk)
        try:
            payload = json.loads(raw, parse_float=finite_float,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, UnicodeError, RecursionError):
            return JSONResponse(rpc_error(-32700, 'Parse error'))

        async def dispatch(item):
            if not isinstance(item, dict):
                return rpc_error(-32600, 'Invalid Request')
            request_id = item.get('id')
            valid_id = request_id is None or (type(request_id) in {str, int, float})
            if (item.get('jsonrpc') != '2.0' or not isinstance(item.get('method'), str)
                    or not valid_id or set(item) - {'jsonrpc', 'id', 'method', 'params'}
                    or ('params' in item and not isinstance(item['params'], (dict, list)))):
                return rpc_error(-32600, 'Invalid Request')
            notification = 'id' not in item

            def failure(code, message):
                return None if notification else rpc_error(code, message, request_id)

            if item['method'] != 'photos.evaluate':
                return failure(-32601, 'Method not found')
            try:
                params = EvaluateParams.model_validate(item.get('params', {}))
                if len(params.photos) > settings.max_photos:
                    raise ValueError('too many photos')
                for photo in params.photos:
                    app.state.advisor.storage.validate_key(params.listing_id, photo.object_key)
            except (ValidationError, ValueError):
                return failure(-32602, 'Invalid params')
            except Exception:
                return failure(-32602, 'Object key outside allowed prefix')
            worker = app.state.advisor
            # Acquire and release in the worker thread, so disconnect cancellation cannot
            # release the lock while inference is still running.
            def evaluate():
                if not worker.lock.acquire(blocking=False):
                    return None
                try:
                    return worker.evaluate(params)
                finally:
                    worker.lock.release()
            try:
                result = await run_in_threadpool(evaluate)
            except Exception:
                logging.getLogger(__name__).error('evaluation_failed')
                return failure(-32603, 'Internal error')
            if result is None:
                return failure(-32003, 'Service busy; retry with backoff')
            return None if notification else {'jsonrpc': '2.0', 'id': request_id, 'result': result}

        if isinstance(payload, list):
            if not payload or len(payload) > 10:
                return JSONResponse(rpc_error(-32600, 'Invalid Request'))
            outputs = []
            for item in payload:
                result = await dispatch(item)
                if result is not None:
                    outputs.append(result)
            return JSONResponse(outputs) if outputs else Response(status_code=204)
        result = await dispatch(payload)
        return JSONResponse(result) if result is not None else Response(status_code=204)

    return app


app = create_app()
