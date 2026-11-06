"""Export pending correction metadata read-only; never download photos or grant consent."""
import argparse
import json
from pathlib import Path
import sqlite3


def export_feedback(state, output):
    output = Path(output)
    if output.exists():
        raise ValueError('choose a new output; feedback snapshots are immutable')
    with sqlite3.connect(Path(state).resolve().as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute('SELECT id,payload FROM review_feedback ORDER BY id').fetchall()
    data = [dict(id=key, **json.loads(payload)) for key, payload in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    return len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = export_feedback(args.state, args.output)
    print(f'Exported {count} pending correction records; no photos downloaded or admitted to training.')
