"""Local annotation workspace; never exposes production credentials or model suggestions."""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
from pathlib import Path
import secrets
import shutil
import sqlite3
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from filelock import FileLock

from app.events import REQUIRED
from app.data_guard import RESERVATIONS, is_reserved, load_reservations
from app.imaging import decode
from app.data_sources import author_group, allowed_license
from app.local_web import install_local_guard, read_body


def now():
    return datetime.now(timezone.utc).isoformat()


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: str = Field(pattern='^(reviewed|excluded)$')
    label: str | None = None
    reviewer: str = Field(min_length=1, max_length=80)
    reviewer_type: str = Field(default='human',pattern='^(human|assistant)$')
    vehicle_group: str = Field(min_length=1, max_length=128, pattern=r'^[\w-]+$')
    vehicle_verified: bool = False
    split: str = Field(pattern='^(train|calibration|test)$')
    version: int = Field(ge=0)
    notes: str = Field(default='', max_length=2000)


class ReviewWorkspace:
    def __init__(self, state, sources, previous=None, registry=RESERVATIONS):
        self.root = Path(state).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'uploads').mkdir(exist_ok=True)
        (self.root/'exports').mkdir(exist_ok=True)
        self.db_path = self.root/'review.sqlite3'
        self.registry_path = Path(registry)
        self.export_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.fixed, self.seen_hashes, self.seen_ids = {}, set(), set()
        if previous:
            from app.evaluation import load_manifest
            for row in load_manifest(previous, (*REQUIRED, 'other'), REQUIRED):
                self.fixed[row['vehicle_id']] = row['split']
                self.seen_hashes.add(row['sha256'])
                self.seen_ids.add((row.get('source_dataset'), row.get('source_id')))
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY, source TEXT NOT NULL,
                    source_id TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                    record TEXT NOT NULL, vehicle_group TEXT NOT NULL, locked_split TEXT,
                    seen_before INTEGER NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY REFERENCES samples(id),
                    decision TEXT NOT NULL, version INTEGER NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_history(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL, decision TEXT NOT NULL, version INTEGER NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY, split TEXT NOT NULL);
            ''')
            for group, split in self.fixed.items():
                old = db.execute('SELECT split FROM groups WHERE id=?', (group,)).fetchone()
                if old and old['split'] != split:
                    raise ValueError('existing review group conflicts with previous dataset')
                db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)', (group, split))
        for source in sources:
            self.import_source(Path(source))
        with self.connect() as db:
            for row in db.execute('SELECT * FROM samples').fetchall():
                seen = (row['vehicle_group'] in self.fixed or row['sha256'] in self.seen_hashes
                        or (row['source'], row['source_id']) in self.seen_ids)
                locked = self.fixed.get(row['vehicle_group']) or row['locked_split']
                db.execute('UPDATE samples SET seen_before=MAX(seen_before,?),locked_split=? WHERE id=?',
                           (int(seen), locked, row['id']))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, source, source_id, path, record):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = hashlib.sha256(f'{source}:{source_id}'.encode()).hexdigest()
        group = record.get('vehicle_group') or author_group(record)
        locked = self.fixed.get(group) or record.get('fixed_split')
        seen = group in self.fixed or digest in self.seen_hashes or (source, source_id) in self.seen_ids
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)',
                (key, source, source_id, str(path), digest, json.dumps(record), group, locked, int(seen), now()))
            found = db.execute('SELECT id FROM samples WHERE sha256=?', (digest,)).fetchone()
        return found['id']

    def import_source(self, source):
        source = source.resolve()
        for item in json.loads((source/'candidates.json').read_text(encoding='utf-8')):
            own = item.get('source_kind') == 'own' and item.get('training_permission') is True
            if not (own or allowed_license(item.get('license', ''))) or not item.get('author') or not item.get('source_url'):
                continue
            path = (source/item['path']).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError('candidate image outside source or missing')
            source_id = str(item.get('image_id', item.get('page_id')))
            self.add(source.name, source_id, path, item)

    def next(self, split=None, status='pending', after=None, fresh=False, cohort=None, natural=False, fenders_only=False):
        with self.connect() as db:
            rows = db.execute('SELECT s.*,r.decision,r.version FROM samples s LEFT JOIN reviews r USING(id) '
                              'ORDER BY s.created,s.rowid').fetchall()
        priority = {'under_trunk': 0, 'trunk': 1, 'interior': 2}
        rows = sorted(rows, key=lambda r: priority.get(json.loads(r['record']).get('label_hint'), 3))
        passed = after is None
        for row in rows:
            if not passed:
                passed = row['id'] == after
                continue
            decision = json.loads(row['decision']) if row['decision'] else None
            actual_status = decision['status'] if decision else 'pending'
            if fenders_only and (not decision or not (decision.get('label') or '').startswith('fender_')):
                continue
            actual_split = decision['split'] if decision else (row['locked_split'] or 'train')
            if actual_status != status or (split and actual_split != split) or (fresh and row['seen_before']):
                continue
            record = json.loads(row['record'])
            if cohort and record.get('evaluation_cohort') != cohort:
                continue
            if natural and record.get('derived'):
                continue
            # No candidate/model score/source view hint in the annotation response.
            return dict(id=row['id'], image=f"/media/{row['id']}", version=row['version'] or 0,
                source_url=record['source_url'], author=record['author'], license=record['license'],
                vehicle_group=decision['vehicle_group'] if decision else row['vehicle_group'],
                split=actual_split or 'train', locked_split=row['locked_split'], seen_before=bool(row['seen_before']),
                derived=bool(record.get('derived')), decision=decision)
        return None

    def save(self, key, decision):
        if decision.status == 'reviewed' and decision.label not in (*REQUIRED, 'other'):
            raise ValueError('choose a valid category')
        if not decision.reviewer.strip():
            raise ValueError('reviewer name required')
        if decision.vehicle_verified and decision.vehicle_group.startswith(('source-group-', 'hitl-')):
            raise ValueError('Для проверенной машины задайте свой ID, а не группу автора или имя файла.')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT s.*,r.version FROM samples s LEFT JOIN reviews r USING(id) WHERE s.id=?', (key,)).fetchone()
            if not row:
                raise KeyError(key)
            if (row['version'] or 0) != decision.version:
                raise RuntimeError('annotation_changed')
            if row['locked_split'] and row['locked_split'] != decision.split:
                raise ValueError('previously assigned photo cannot change split')
            group = db.execute('SELECT split FROM groups WHERE id=?', (decision.vehicle_group,)).fetchone()
            if group and group['split'] != decision.split:
                raise ValueError('all photos of the same vehicle must stay in one split')
            db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)', (decision.vehicle_group, decision.split))
            payload = dict(decision.model_dump(), reviewed_at=now())
            serialized = json.dumps(payload, ensure_ascii=False)
            version = decision.version+1
            db.execute('INSERT INTO reviews VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE '
                       'SET decision=excluded.decision,version=excluded.version,updated=excluded.updated',
                       (key, serialized, version, now()))
            db.execute('INSERT INTO review_history(sample_id,decision,version,updated) VALUES (?,?,?,?)',
                       (key, serialized, version, now()))
        return version

    def stats(self, cohort=None, fresh=False, natural=False, split=None, fenders_only=False):
        with self.connect() as db:
            rows = db.execute('SELECT s.record,s.seen_before,s.locked_split,r.decision FROM samples s LEFT JOIN reviews r USING(id)').fetchall()
            rows = [r for r in rows if (not cohort or json.loads(r['record']).get('evaluation_cohort') == cohort)
                    and (not fresh or not r['seen_before'])
                    and (not natural or not json.loads(r['record']).get('derived'))
                    and (not fenders_only or (r['decision'] and (json.loads(r['decision']).get('label') or '').startswith('fender_')))
                    and (not split or (json.loads(r['decision'])['split'] if r['decision'] else (r['locked_split'] or 'train')) == split)]
            total = len(rows)
            decisions = [json.loads(r['decision']) for r in rows if r['decision']]
        return dict(total=total, pending=total-len(decisions),
            reviewed=sum(r['status']=='reviewed' for r in decisions),
            excluded=sum(r['status']=='excluded' for r in decisions))

    def export(self, holdout=False):
        with self.export_lock, self.connect() as db:
            rows = db.execute('SELECT s.*,r.decision FROM samples s JOIN reviews r USING(id) ORDER BY s.id').fetchall()
            selected = []
            reservation = load_reservations(self.registry_path)
            for row in rows:
                review, record = json.loads(row['decision']), json.loads(row['record'])
                if review['status'] != 'reviewed':
                    continue
                if holdout and (review.get('reviewer_type') != 'human' or review['split'] != 'test' or not review['vehicle_verified']
                        or row['seen_before'] or record.get('derived')):
                    continue
                if record.get('source_kind') == 'own' and record.get('training_permission') is not True:
                    continue
                if not holdout and is_reserved(dict(vehicle_id=review['vehicle_group'],
                        origin_vehicle_group=row['vehicle_group'], sha256=row['sha256']), reservation):
                    continue
                if not holdout and review['split'] == 'test' and not row['seen_before']:
                    continue
                selected.append((row, review, record))
            if not selected:
                raise ValueError('no eligible reviewed images; fresh test needs new original photos and verified vehicle IDs')
            for row, _, _ in selected:
                if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('source photo changed after import; reimport into a new workspace')
            if holdout:
                self.registry_path.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(str(self.registry_path)+'.lock', timeout=5):
                    existing = load_reservations(self.registry_path)
                    protected = {g for row, review, _ in selected for g in (row['vehicle_group'],review['vehicle_group'])}
                    reservation = dict(schema_version=1,
                        groups=sorted(set(existing.get('groups',[])) | protected),
                        sha256=sorted(set(existing.get('sha256',[])) | {row['sha256'] for row,_,_ in selected}),
                        reason='Reserved independent photos; never use in model development.')
                    temporary = self.registry_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(reservation,indent=2),encoding='utf-8')
                    temporary.replace(self.registry_path)
            output = self.root/'exports'/f"{'holdout' if holdout else 'reviewed'}-{secrets.token_hex(8)}"
            (output/'images').mkdir(parents=True)
            candidates, reviews, manifest = [], [], []
            for row, review, record in selected:
                relative = f"images/{row['id']}.jpg"
                shutil.copyfile(row['path'], output/relative)
                if hashlib.sha256((output/relative).read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('source photo changed while exporting')
                item = dict(record, path=relative, image_id=row['id'], vehicle_group=review['vehicle_group'],
                            fixed_split=review['split'], origin_vehicle_group=row['vehicle_group'],
                            origin_source_dataset=row['source'], origin_source_id=row['source_id'])
                candidates.append(item)
                reviews.append(dict(image_id=row['id'], status='reviewed', label=review['label'],
                    reviewer=review['reviewer'], reviewer_type=review['reviewer_type'],
                    label_origin='human_review' if review['reviewer_type']=='human' else 'assistant_visual_review',
                    vehicle_group=review['vehicle_group'], vehicle_identity_verified=review['vehicle_verified'],
                    fixed_split=review['split'], note=review['notes']))
                manifest.append(dict(path=relative, label=review['label'], split=review['split'],
                    vehicle_id=review['vehicle_group'], listing_id=review['vehicle_group'],
                    covers=[] if review['label']=='other' else [review['label']], review_status='reviewed',
                    reviewer=review['reviewer'], reviewer_type=review['reviewer_type'],
                    label_origin='human_review' if review['reviewer_type']=='human' else 'assistant_visual_review',
                    vehicle_identity_verified=review['vehicle_verified'], sha256=row['sha256'],
                    origin_vehicle_group=row['vehicle_group'],
                    origin_source_dataset=row['source'], origin_source_id=row['source_id'],
                    derived=bool(record.get('derived')), source_url=record['source_url'], author=record['author'],
                    license=record['license'], license_url=record.get('license_url', ''),
                    source_kind=record.get('source_kind'), training_permission=record.get('training_permission', False)))
            (output/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
            (output/'reviewed-labels.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding='utf-8')
            data = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in manifest)
            (output/'manifest.jsonl').write_text(data, encoding='utf-8')
            human = all(review.get('reviewer_type')=='human' for _,review,_ in selected)
            frozen = dict(kind='fresh_human_holdout' if holdout else ('human_reviewed_export' if human else 'mixed_reviewed_export'),
                created_at=now(), count=len(manifest), manifest_sha256=hashlib.sha256((output/'manifest.jsonl').read_bytes()).hexdigest(),
                blind=True, production_approved=False)
            (output/'snapshot.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
            return dict(directory=str(output), **frozen)


def create_review_app(workspace):
    app = FastAPI(docs_url=None, redoc_url=None)
    install_local_guard(app, workspace.token, 'x-review-token')

    @app.get('/')
    def home():
        html = Path(__file__).with_name('review.html').read_text(encoding='utf-8')
        return HTMLResponse(html.replace('__REVIEW_TOKEN__', workspace.token))

    @app.get('/api/stats')
    def stats(cohort: str | None = None, fresh: bool = False, natural: bool = False, split: str | None = None,
              fenders_only: bool = False):
        return workspace.stats(cohort, fresh, natural, split, fenders_only)

    @app.get('/api/next')
    def next_image(split: str | None = None, status: str = 'pending', after: str | None = None,
                   fresh: bool = False, cohort: str | None = None, natural: bool = False, fenders_only: bool = False):
        return workspace.next(split, status, after, fresh, cohort, natural, fenders_only)

    @app.get('/media/{key}')
    def media(key):
        with workspace.connect() as db:
            row = db.execute('SELECT path,sha256 FROM samples WHERE id=?', (key,)).fetchone()
        if not row:
            raise HTTPException(404)
        data = Path(row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise HTTPException(409, 'Исходный файл изменился после импорта.')
        try:
            image, _ = decode(data, 24_000_000)
        except ValueError:
            raise HTTPException(422, 'Повреждённое изображение.') from None
        try:
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=92)
            return Response(buffer.getvalue(), media_type='image/jpeg')
        finally:
            image.close()

    @app.post('/api/reviews/{key}')
    def review(key, decision: Decision):
        try:
            return {'version': workspace.save(key, decision)}
        except KeyError:
            raise HTTPException(404) from None
        except RuntimeError:
            raise HTTPException(409, 'Фото уже изменено. Обновите страницу.') from None
        except ValueError as exc:
            messages = {
                'choose a valid category': 'Выберите ракурс.',
                'reviewer name required': 'Укажите, кто проверяет фотографию.',
                'previously assigned photo cannot change split': 'Это фото уже закреплено за выборкой. Перенос в другую запрещён.',
                'all photos of the same vehicle must stay in one split': 'Фотографии одной машины должны оставаться в одной выборке.',
            }
            raise HTTPException(422, messages.get(str(exc), str(exc))) from None

    @app.post('/api/export')
    def export(holdout: bool = False):
        try:
            return workspace.export(holdout)
        except ValueError as exc:
            message = (('Для новой проверки нужны вручную размеченные оригиналы новых машин с подтверждёнными ID.'
                        if holdout else 'Нет проверенных фото для обучения. Независимая проверка экспортируется отдельно.')
                       if str(exc).startswith('no eligible')
                       else 'Исходный файл изменился. Экспорт остановлен, чтобы не сохранить неверную разметку.')
            raise HTTPException(409, message) from None

    @app.post('/api/upload')
    async def upload(request: Request, vehicle_id: str, author: str, training_permission: bool = False):
        if not training_permission or not author.strip():
            raise HTTPException(422, 'Нужны имя автора и разрешение на обучение.')
        # Reuse the exact vehicle identity constraints of the review endpoint.
        try:
            Decision(status='excluded', reviewer=author, vehicle_group=vehicle_id, split='train', version=0)
        except ValueError:
            raise HTTPException(422, 'Укажите ID машины без пробелов и имя автора.') from None
        data = await read_body(request, 12*1024*1024, 'Файл слишком большой.')
        try:
            image, _ = decode(bytes(data), 24_000_000)
        except ValueError:
            raise HTTPException(422, 'Нужна фотография JPEG, PNG или WebP.') from None
        try:
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
        finally:
            image.close()
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        path = workspace.root/'uploads'/f'{digest}.jpg'
        if not path.exists():
            path.write_bytes(buffer.getvalue())
        item = dict(source_kind='own', training_permission=True, author=author.strip(),
            source_url='own-photo:'+digest, license='Own data', license_url='',
            vehicle_group='own-'+vehicle_id, derived=False, permission_recorded_at=now())
        return {'id': workspace.add('own', digest, path, item)}

    return app
