"""Common HTTP protections for the two local annotation applications."""

import hmac

from fastapi import HTTPException
from fastapi.responses import Response
from starlette.middleware.trustedhost import TrustedHostMiddleware


SECURITY_HEADERS = {
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
    'Content-Security-Policy': (
        "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
    ),
}


def install_local_guard(app, token, header):
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def guard(request, call_next):
        if (request.headers.get('sec-fetch-site') == 'cross-site'
                or (request.method not in {'GET', 'HEAD'} and not hmac.compare_digest(
                    request.headers.get(header, '').encode(), token.encode()))):
            return Response(status_code=403)
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response


async def read_body(request, limit, message='Слишком большой файл или запрос.'):
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > limit:
            raise HTTPException(413, message)
        content.extend(chunk)
    return bytes(content)
