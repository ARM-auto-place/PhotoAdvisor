"""Loopback-only assisted pilot; model-assisted feedback is never a blind test set."""
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import secrets
import sqlite3
from threading import RLock
from urllib.parse import unquote
import zipfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.events import REQUIRED
from app.imaging import decode
from app.local_web import install_local_guard, read_body
from app.inference import load_vision
from app.rules import summarize
from app.schemas import EvaluateParams, PhotoResult
from app.service import Advisor


class NewSession(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1, max_length=100)
    training_permission: bool = Field(default=False, strict=True)


class Correction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = Field(min_length=1, max_length=40)
    revision: int = Field(ge=0, strict=True)


class Permission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    training_permission: bool = Field(strict=True)
    revision: int = Field(ge=0, strict=True)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class PilotWorkspace:
    def __init__(self, root, settings, max_bytes=512 * 1024 * 1024):
        self.root, self.settings = Path(root).resolve(), settings
        self.root.mkdir(parents=True, exist_ok=True)
        self.images = self.root/'images'
        self.images.mkdir(exist_ok=True)
        self.owner = FileLock(str(self.root/'pilot.lock'), thread_local=False)
        try:
            self.owner.acquire(timeout=0)
        except TimeoutError as exc:
            raise RuntimeError('pilot workspace already owned by another process') from exc
        self.lock, self.token, self.max_bytes = RLock(), secrets.token_urlsafe(32), max_bytes
        try:
            with self.connect() as db:
                db.execute('PRAGMA journal_mode=WAL')
                db.executescript('''
                    CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                        created TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                        training_permission INTEGER NOT NULL, report TEXT);
                    CREATE TABLE IF NOT EXISTS photos(id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        filename TEXT NOT NULL, sha256 TEXT NOT NULL, original_sha256 TEXT NOT NULL,
                        bytes INTEGER NOT NULL, decision TEXT, created TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS history(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                        label TEXT NOT NULL, session_revision INTEGER NOT NULL, created TEXT NOT NULL);
                ''')
        except BaseException:
            self.owner.release()
            raise

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root/'pilot.sqlite3', timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def close(self):
        self.owner.release()

    def session(self, db, sid, revision=None):
        row = db.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Проверка не найдена.')
        if revision is not None and revision != row['revision']:
            raise HTTPException(409, 'Данные изменились. Обновите результат и повторите действие.')
        return row

    def create(self, value):
        with self.lock, self.connect() as db:
            if db.execute('SELECT count(*) FROM sessions').fetchone()[0] >= 100:
                raise HTTPException(413, 'Сохранено 100 проверок. Удалите ненужные.')
            sid = secrets.token_hex(16)
            db.execute('INSERT INTO sessions(id,title,created,training_permission) VALUES(?,?,?,?)',
                       (sid, value.title.strip() or 'Автомобиль', timestamp(), value.training_permission))
        return sid

    def listings(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT s.id,s.title,s.created,s.training_permission,
                count(p.id) AS photo_count FROM sessions s LEFT JOIN photos p ON p.session_id=s.id
                GROUP BY s.id ORDER BY s.created DESC''')]

    def snapshot(self, sid):
        with self.lock, self.connect() as db:
            session = dict(self.session(db, sid))
            session['photos'] = [dict(r) for r in db.execute(
                'SELECT * FROM photos WHERE session_id=? ORDER BY created,rowid', (sid,))]
        session['report'] = json.loads(session['report']) if session['report'] else None
        session['training_permission'] = bool(session['training_permission'])
        return session

    def add(self, sid, data, filename):
        image, _ = decode(data, self.settings.max_pixels)
        try:
            output = BytesIO()
            image.save(output, format='JPEG', quality=95)  # No EXIF/GPS or user-controlled file paths.
            normalized = output.getvalue()
        finally:
            image.close()
        pid = secrets.token_hex(16)
        path = self.images/f'{pid}.jpg'
        with self.lock:
            try:
                with self.connect() as db:
                    self.session(db, sid)
                    if db.execute('SELECT count(*) FROM photos WHERE session_id=?', (sid,)).fetchone()[0] >= self.settings.max_photos:
                        raise HTTPException(413, f'Можно добавить до {self.settings.max_photos} фото.')
                    total = db.execute('SELECT coalesce(sum(bytes),0) FROM photos').fetchone()[0]
                    if total + len(normalized) > self.max_bytes:
                        raise HTTPException(413, 'Локальное хранилище заполнено. Удалите ненужные проверки.')
                    path.write_bytes(normalized)
                    db.execute('INSERT INTO photos VALUES(?,?,?,?,?,?,?,?)', (
                        pid, sid, filename[:200], hashlib.sha256(normalized).hexdigest(),
                        hashlib.sha256(data).hexdigest(), len(normalized), None, timestamp()))
                    db.execute('UPDATE sessions SET revision=revision+1,report=NULL WHERE id=?', (sid,))
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return pid

    def remove(self, sid, pid=None):
        with self.lock, self.connect() as db:
            self.session(db, sid)
            photos = db.execute('SELECT id FROM photos WHERE session_id=?' + (' AND id=?' if pid else ''),
                                (sid, pid) if pid else (sid,)).fetchall()
            if pid and not photos:
                raise HTTPException(404, 'Фото не найдено.')
            if pid:
                db.execute('DELETE FROM photos WHERE id=?', (pid,))
                db.execute('UPDATE sessions SET revision=revision+1,report=NULL WHERE id=?', (sid,))
            else:
                db.execute('DELETE FROM sessions WHERE id=?', (sid,))
        for photo in photos:
            (self.images/f'{photo["id"]}.jpg').unlink(missing_ok=True)

    def save_report(self, sid, revision, report):
        with self.lock, self.connect() as db:
            self.session(db, sid, revision)
            db.execute('UPDATE sessions SET report=? WHERE id=?',
                       (json.dumps(report, ensure_ascii=False, allow_nan=False), sid))

    def correct(self, sid, pid, value):
        if value.label not in (*REQUIRED, 'other'):
            raise HTTPException(422, 'Выберите один из 12 ракурсов или «Не подходит».')
        with self.lock, self.connect() as db:
            session = self.session(db, sid, value.revision)
            if not session['report']:
                raise HTTPException(409, 'Сначала запустите проверку фотографий.')
            photo = db.execute('SELECT * FROM photos WHERE id=? AND session_id=?', (pid, sid)).fetchone()
            if not photo:
                raise HTTPException(404, 'Фото не найдено.')
            prediction = next((p for p in json.loads(session['report'])['photo_results'] if p['photo_id'] == pid), None)
            if prediction is None or prediction.get('error'):
                raise HTTPException(409, 'Фото не обработано. Повторите проверку.')
            db.execute('UPDATE photos SET decision=? WHERE id=?', (value.label, pid))
            db.execute('INSERT INTO history(photo_id,label,session_revision,created) VALUES(?,?,?,?)',
                       (pid, value.label, value.revision, timestamp()))
            db.execute('UPDATE sessions SET revision=revision+1 WHERE id=?', (sid,))

    def permission(self, sid, value):
        with self.lock, self.connect() as db:
            self.session(db, sid, value.revision)
            db.execute('UPDATE sessions SET training_permission=?,revision=revision+1 WHERE id=?',
                       (value.training_permission, sid))

    def result(self, sid, checklist):
        state = self.snapshot(sid)
        if not state['report']:
            return state
        photos = {r['id']: r for r in state['photos']}
        effective, groups = [], {}
        originals = {p['photo_id']: p for p in state['report']['photo_results']}
        for original in state['report']['photo_results']:
            photo = PhotoResult.model_validate(original)
            decision = photos[photo.photo_id]['decision']
            if decision:
                photo.category, photo.candidate, photo.recognition = decision, decision, 'known'
                photo.confidence, photo.score, photo.score_kind = None, None, None
                photo.confidence_kind = 'manual_review'
                photo.accepted = (decision in REQUIRED and photo.quality == 'good'
                                  and not photo.error and not photo.duplicate_of)
            effective.append(photo)
            groups.setdefault(original.get('duplicate_of') or photo.photo_id, []).append(photo)
        for root, members in groups.items():
            confirmed = [p for p in members if photos[p.photo_id]['decision']
                         and p.quality == 'good' and not p.error]
            if not confirmed:
                continue
            chosen = next((p for p in confirmed if p.photo_id == root), confirmed[0])
            for photo in members:
                photo.accepted = photo is chosen and photo.category in REQUIRED
                photo.duplicate_of = None if photo is chosen else chosen.photo_id
                photo.duplicate_kind = (None if photo is chosen else 'exact'
                    if photos[photo.photo_id]['sha256'] == photos[chosen.photo_id]['sha256'] else 'near')
        enriched = [dict(p.model_dump(), model_candidate=originals[p.photo_id].get('candidate'),
                         source='manual' if photos[p.photo_id]['decision'] else 'model',
                         confirmed=bool(photos[p.photo_id]['decision'])) for p in effective]
        report = summarize(sid, effective, checklist, 'ru', state['report']['model']['version'],
                           state['report']['model']['calibrated'])
        report['photo_results'] = enriched
        state['report'] = report
        return state

    def export(self, sid):
        # Lock out deletions/consent changes for the entire bounded archive creation.
        with self.lock:
            state = self.snapshot(sid)
            if not state['training_permission']:
                raise HTTPException(403, 'Для выгрузки фото на обучение нужно разрешение владельца.')
            predictions = {p['photo_id']: p for p in (state['report'] or {}).get('photo_results', [])}
            selected = [p for p in state['photos'] if p['decision'] and p['id'] in predictions]
            if not selected:
                raise HTTPException(409, 'Сначала проверьте фото и подтвердите хотя бы один ракурс.')
            rows, archive = [], BytesIO()
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for photo in selected:
                    prediction = predictions[photo['id']]
                    relative = f'images/{photo["id"]}.jpg'
                    data = (self.images/f'{photo["id"]}.jpg').read_bytes()
                    if hashlib.sha256(data).hexdigest() != photo['sha256']:
                        raise HTTPException(409, 'Файл изменился. Выгрузка остановлена.')
                    bundle.writestr(relative, data)
                    rows.append(dict(path=relative, label=photo['decision'], sha256=photo['sha256'],
                                     original_sha256=photo['original_sha256'], vehicle_group=sid,
                                     source_kind='own', training_permission=True, reviewer_type='unverified',
                                     review_origin='interactive_pilot',
                                     reviewer_identity_verified=False, vehicle_identity_verified=False,
                                     review_status='pilot_feedback', model_assisted=True,
                                     eligible_for_independent_test=False, model=state['report']['model'],
                                     original_prediction=prediction, created=timestamp()))
                bundle.writestr('feedback.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
                bundle.writestr('README.txt', 'Model-assisted pilot feedback. Not a training manifest or blind test.\n'
                                'Review labels, permissions, quality and vehicle grouping before dataset import.\n')
            return archive.getvalue()


class PilotStorage:
    def __init__(self, workspace):
        self.workspace = workspace

    def fetch(self, listing_id, photo):
        with self.workspace.connect() as db:
            row = db.execute('SELECT id FROM photos WHERE session_id=? AND id=?',
                             (listing_id, photo.photo_id)).fetchone()
        if not row:
            from app.storage import StorageError
            raise StorageError('s3_object_unavailable')
        return (self.workspace.images/f'{row["id"]}.jpg').read_bytes()

    def close(self):
        pass


def create_pilot_app(workspace, vision=None):
    @asynccontextmanager
    async def lifespan(app):
        try:
            model = vision if vision is not None else load_vision(workspace.settings)
            app.state.advisor = Advisor(workspace.settings, PilotStorage(workspace), model)
            yield
        finally:
            if hasattr(app.state, 'advisor'):
                app.state.advisor.close()
            workspace.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    install_local_guard(app, workspace.token, 'x-pilot-token')

    async def value(request, schema):
        from pydantic import ValidationError
        try:
            return schema.model_validate_json(await read_body(request, 16384))
        except ValidationError as exc:
            raise HTTPException(422, 'Проверьте заполненные поля.') from exc

    @app.get('/')
    def home():
        return HTMLResponse(Path(__file__).with_name('pilot.html').read_text(encoding='utf-8')
                            .replace('__PILOT_TOKEN__', workspace.token))

    @app.get('/pilot-feedback.js')
    def feedback_script():
        return Response(Path(__file__).with_name('pilot_feedback.js').read_text(encoding='utf-8'),
                        media_type='application/javascript')

    @app.get('/api/sessions')
    def sessions():
        return workspace.listings()

    @app.post('/api/sessions')
    async def create(request: Request):
        sid = await run_in_threadpool(workspace.create, await value(request, NewSession))
        return workspace.snapshot(sid)

    @app.get('/api/sessions/{sid}')
    def result(sid):
        return workspace.result(sid, app.state.advisor.vision.checklist)

    @app.post('/api/sessions/{sid}/photos')
    async def upload(sid, request: Request):
        data = await read_body(request, workspace.settings.max_image_bytes)
        try:
            pid = await run_in_threadpool(workspace.add, sid, data, unquote(request.headers.get('x-filename', 'Фото')))
        except (ValueError, OSError) as exc:
            raise HTTPException(422, 'Не удалось открыть фото. Поддерживаются JPEG, PNG и WEBP без анимации.') from exc
        return {'photo_id': pid}

    @app.delete('/api/sessions/{sid}/photos/{pid}')
    def delete_photo(sid, pid):
        workspace.remove(sid, pid)
        return {'deleted': True}

    @app.delete('/api/sessions/{sid}')
    def delete_session(sid):
        workspace.remove(sid)
        return {'deleted': True}

    @app.post('/api/sessions/{sid}/analyze')
    def analyze(sid):
        advisor = app.state.advisor
        if not advisor.lock.acquire(blocking=False):
            raise HTTPException(409, 'Идёт другая проверка. Попробуйте через несколько секунд.')
        try:
            with workspace.lock:
                state = workspace.snapshot(sid)
                params = EvaluateParams(listing_id=sid, photos=[dict(photo_id=p['id'],
                    object_key=f'listings/{sid}/{p["id"]}.jpg') for p in state['photos']])
                # Upload/deletion cannot remove files while inference uses this snapshot.
                report = advisor.evaluate(params)
                workspace.save_report(sid, state['revision'], report)
                return workspace.result(sid, advisor.vision.checklist)
        finally:
            advisor.lock.release()

    @app.post('/api/sessions/{sid}/photos/{pid}/decision')
    async def decision(sid, pid, request: Request):
        await run_in_threadpool(workspace.correct, sid, pid, await value(request, Correction))
        return workspace.result(sid, app.state.advisor.vision.checklist)

    @app.post('/api/sessions/{sid}/permission')
    async def permission(sid, request: Request):
        await run_in_threadpool(workspace.permission, sid, await value(request, Permission))
        return workspace.result(sid, app.state.advisor.vision.checklist)

    @app.get('/media/{sid}/{pid}')
    def media(sid, pid):
        with workspace.lock, workspace.connect() as db:
            row = db.execute('SELECT id FROM photos WHERE session_id=? AND id=?', (sid, pid)).fetchone()
            if not row:
                raise HTTPException(404, 'Фото не найдено.')
            try:
                data = (workspace.images/f'{row["id"]}.jpg').read_bytes()
            except FileNotFoundError as exc:
                raise HTTPException(404, 'Фото не найдено.') from exc
        return Response(data, media_type='image/jpeg')

    @app.get('/api/sessions/{sid}/export')
    def export(sid):
        return Response(workspace.export(sid), media_type='application/zip',
                        headers={'Content-Disposition': f'attachment; filename="pilot-{sid}.zip"'})

    return app
