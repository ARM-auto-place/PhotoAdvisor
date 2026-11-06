import hashlib
import json
import sqlite3
from pathlib import Path

from filelock import FileLock, Timeout


class ContractError(ValueError):
    pass


class State:
    """One process owns this database; Kafka and outbox run on its event loop."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(Path(path).resolve()) + '.lock')
        try:
            self._lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError('state_database_already_owned') from exc
        self.db = None
        try:
            self.db = sqlite3.connect(path)
            self._initialize()
        except BaseException:
            if self.db is not None:
                self.db.close()
            self._lock.release()
            raise

    def _initialize(self):
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS listings (
                id TEXT PRIMARY KEY, revision INTEGER NOT NULL, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL, listing_id TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_feedback (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        ''')

    @staticmethod
    def digest(event):
        return hashlib.sha256(event.model_dump_json().encode()).hexdigest()

    def needed(self, event):
        row = self.db.execute('SELECT digest FROM events WHERE id=?', (event.event_id,)).fetchone()
        if row:
            if row['digest'] != self.digest(event):
                raise ContractError('event_id_reused')
            return False
        row = self.db.execute('SELECT revision FROM listings WHERE id=?', (event.listing_id,)).fetchone()
        if row and row['revision'] == event.listing_revision:
            raise ContractError('listing_revision_reused')
        return not row or row['revision'] < event.listing_revision

    def _save(self, event, result):
        needed = self.needed(event)
        self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?)',
                        (event.event_id, self.digest(event)))
        if needed and result is not None:
            self.db.execute('INSERT INTO listings VALUES (?,?,?) ON CONFLICT(id) DO UPDATE '
                            'SET revision=excluded.revision,result=excluded.result',
                            (event.listing_id, event.listing_revision, json.dumps(result)))
            self.enqueue('result', result['result_id'], event.listing_id, result)
            if event.event_type == 'photoset.finalized':
                references = {p.photo_id: p for p in event.photos}
                for photo in result.get('photos', []):
                    if photo.get('source') != 'manual':
                        continue
                    reference = references[photo['image_id']]
                    feedback_id = hashlib.sha256(f'{event.event_id}:{reference.photo_id}'.encode()).hexdigest()
                    feedback = dict(schema_version=1, event_id=event.event_id,
                        listing_id=event.listing_id, listing_revision=event.listing_revision,
                        image_id=reference.photo_id, bucket=reference.bucket, object_key=reference.object_key,
                        version_id=reference.version_id, etag=reference.etag,
                        model_candidate=photo.get('candidate'), corrected_category=reference.manual_category,
                        model=result.get('model'), review_status='pending', training_permission=False,
                        note='User correction is a review lead, not independently verified training data.')
                    self.db.execute('INSERT OR IGNORE INTO review_feedback VALUES (?,?)',
                                    (feedback_id, json.dumps(feedback, ensure_ascii=False)))
        self.db.execute('DELETE FROM attempts WHERE id=?', (f'event:{event.event_id}',))

    def save(self, event, result):
        with self.db:
            self._save(event, result)

    def enqueue(self, kind, message_id, listing_id, payload):
        self.db.execute('INSERT OR IGNORE INTO outbox(id,kind,listing_id,payload) VALUES (?,?,?,?)',
                        (f'{kind}:{message_id}', kind, listing_id, json.dumps(payload, allow_nan=False)))

    def attempt(self, record_id):
        with self.db:
            self.db.execute('INSERT INTO attempts VALUES (?,1) ON CONFLICT(id) '
                            'DO UPDATE SET count=count+1', (record_id,))
        return self.db.execute('SELECT count FROM attempts WHERE id=?', (record_id,)).fetchone()[0]

    def rejected(self, record_id, payload, event=None, result=None):
        with self.db:
            if event is not None and result is not None:
                self._save(event, result)
            self.enqueue('dlq', record_id, event.listing_id if event else record_id, payload)

    def pending(self):
        return self.db.execute('SELECT * FROM outbox ORDER BY seq LIMIT 1').fetchone()

    def sent(self, seq):
        with self.db:
            self.db.execute('DELETE FROM outbox WHERE seq=?', (seq,))

    def close(self):
        try:
            self.db.close()
        finally:
            self._lock.release()
