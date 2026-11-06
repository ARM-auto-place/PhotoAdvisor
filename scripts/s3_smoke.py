"""Opt-in real S3/MinIO integration check using one existing private object."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.imaging import decode
from app.schemas import Photo
from app.settings import Settings
from app.storage import S3Storage

parser = argparse.ArgumentParser()
parser.add_argument('listing_id')
parser.add_argument('object_key')
args = parser.parse_args()
settings = Settings()
storage = S3Storage(settings)
try:
    data = storage.fetch(args.listing_id, Photo(photo_id='smoke', object_key=args.object_key))
    image, size = decode(data, settings.max_pixels)
    image.close()
    print(f'Private object read and decoded successfully: {size[0]}x{size[1]}')
finally:
    storage.close()
