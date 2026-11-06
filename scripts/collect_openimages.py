"""Collect a small CC-BY Open Images subset for manual view annotation."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from io import BytesIO
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.data_sources import author_group


def session():
    client = requests.Session()
    client.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1,
        status_forcelist=[500, 502, 503, 504], allowed_methods=['GET'])))
    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('datasets/openimages-candidates'))
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--metadata-dir', type=Path)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--exclude-manifest', type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 2000 or args.offset < 0:
        parser.error('limit must be 1..2000 and offset nonnegative')
    root = args.output
    (root / 'images').mkdir(parents=True, exist_ok=True)
    metadata_root = args.metadata_dir or root
    metadata = {r['ImageID']: r for r in csv.DictReader((metadata_root/'image-info.csv').open(encoding='utf-8'))}
    boxes = defaultdict(list)
    for row in csv.DictReader((metadata_root/'boxes.csv').open(encoding='utf-8')):
        if row['LabelName'] == '/m/0k4j':
            boxes[row['ImageID']].append(row)
    ids = []
    for image_id, items in boxes.items():
        if len(items) != 1:
            continue
        box = items[0]
        area = (float(box['XMax'])-float(box['XMin']))*(float(box['YMax'])-float(box['YMin']))
        info = metadata[image_id]
        if (area >= .25 and all(box[c] == '0' for c in ('IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsOccluded'))
                and info['License'].rstrip('/') == 'https://creativecommons.org/licenses/by/2.0'
                and info['Author']):
            ids.append(image_id)
    random.Random(42).shuffle(ids)
    excluded_groups, excluded_ids = set(), set()
    if args.exclude_manifest:
        used = [json.loads(line) for line in args.exclude_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        excluded_groups = {r['vehicle_id'] for r in used}
        excluded_ids = {r.get('source_id') for r in used}
    def unused(image_id):
        info = metadata[image_id]
        return (image_id not in excluded_ids and author_group(dict(author=info['Author'],
                author_url=info['AuthorProfileURL'])) not in excluded_groups)
    selected = [i for i in ids[args.offset:] if unused(i)][:args.limit]
    if args.details:
        details = set()
        for row in csv.DictReader((metadata_root/'human-labels.csv').open(encoding='utf-8')):
            if row['Confidence'] == '1' and row['LabelName'] in {'/m/02mk9', '/m/0268t21', '/m/06mr9c'}:
                details.add(row['ImageID'])
        selected += [i for i in sorted(details) if i not in selected and unused(i) and metadata[i]['Author']
                     and metadata[i]['License'].rstrip('/') == 'https://creativecommons.org/licenses/by/2.0']
    def download(image_id):
        path = root / 'images' / f'{image_id}.jpg'
        url = f'https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg'
        if not path.exists():
            with session() as client:
                response = client.get(url, timeout=25)
                response.raise_for_status()
                if len(response.content) > 12*1024*1024:
                    raise ValueError('file limit')
                with Image.open(BytesIO(response.content)) as source:
                    if source.width*source.height > 24_000_000:
                        raise ValueError('pixel limit')
                    image = source.convert('RGB')
                    image.thumbnail((800, 800))
                    image.save(path, quality=93)
                    image.close()
        info = metadata[image_id]
        return dict(image_id=image_id, path=f'images/{image_id}.jpg', title=info['Title'],
            source_url=info['OriginalLandingURL'], download_url=url,
            author=info['Author'], author_url=info['AuthorProfileURL'], license='CC BY 2.0',
            license_url=info['License'], review_status='pending', box=boxes[image_id][0],
            derivative='resized JPEG', dataset='Open Images validation (candidate source, not our split)')
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(image_id, pool.submit(download, image_id)) for image_id in selected]
        for image_id, future in futures:
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(dict(image_id=image_id, error=type(exc).__name__))
            (root/'candidates.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            if len(records) % 20 == 0:
                print(f'{len(records)} candidates downloaded', flush=True)
    (root/'download-errors.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    print(f'Collected {len(records)} CC-BY candidates; {len(failures)} unavailable. Labels pending visual review.')


if __name__ == '__main__':
    main()
