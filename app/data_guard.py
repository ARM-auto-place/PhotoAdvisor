"""Keep reserved blind photos completely outside training and development evaluation."""
import json
from pathlib import Path


RESERVATIONS = Path(__file__).resolve().parents[1]/'datasets/holdout-reservations.json'


def load_reservations(path=None):
    path = Path(path) if path else RESERVATIONS
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'groups': [], 'sha256': []}


def is_reserved(row, reservation):
    groups = set(reservation.get('groups', []))
    return bool({row.get('vehicle_id'), row.get('vehicle_group'), row.get('origin_vehicle_group')} & groups
                or row.get('sha256') in set(reservation.get('sha256', [])))
