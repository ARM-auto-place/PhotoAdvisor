import pytest

from app.data_guard import is_reserved
from scripts.train_views import check_rows


def test_reservation_covers_renamed_files_and_source_groups():
    registry = {'groups':['reserved-car'], 'sha256':['reserved-image']}
    assert is_reserved({'vehicle_id':'renamed', 'sha256':'reserved-image'}, registry)
    assert is_reserved({'vehicle_id':'renamed', 'origin_vehicle_group':'reserved-car'}, registry)
    assert not is_reserved({'vehicle_id':'other', 'sha256':'other-image'}, registry)


def test_reserved_photos_cannot_enter_training_even_as_test(monkeypatch):
    monkeypatch.setattr('scripts.train_views.load_reservations', lambda: {'groups':['reserved-car'], 'sha256':[]})
    for split in ('train','calibration','test'):
        with pytest.raises(ValueError, match='reserved blind holdout'):
            check_rows([dict(vehicle_id='reserved-car', split=split)], True)
