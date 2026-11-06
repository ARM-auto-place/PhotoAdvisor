"""Collect openly licensed Commons candidates, never automatic ground-truth labels."""
import argparse
import hashlib
from html import unescape
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_sources import allowed_license as allowed_license

QUERIES = {
    'front': 'automobile front view',
    'rear': 'car rear view',
    'left_side': 'car left side view',
    'right_side': 'car right side view',
    'engine_bay': 'car engine bay',
    'trunk': 'car open trunk',
    'under_trunk': 'spare wheel trunk',
    'interior': 'car interior seats',
    'fenders': 'automobile fender close up',
    'negative': 'car dashboard display',
}


def text(value):
    return unescape(re.sub('<[^>]+>', ' ', value)).strip()


def metadata_field(metadata, key):
    return text(str(metadata.get(key, {}).get('value', '')))


def get(session, url, **kwargs):
    for attempt in range(3):
        response = session.get(url, timeout=30, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = int(response.headers.get('Retry-After', 60))
        if wait > 60 or attempt == 2:
            raise RuntimeError(f'source rate limited; retry after {wait} seconds')
        print(f'Source rate limit: waiting {wait} seconds', flush=True)
        time.sleep(max(wait, 1))
    raise RuntimeError('source unavailable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('datasets/commons-candidates'))
    parser.add_argument('--limit', type=int, default=16, choices=range(1, 51))
    parser.add_argument('--group', choices=list(QUERIES), action='append')
    parser.add_argument('--titles', type=Path, help='JSON list of exact Commons File: titles')
    parser.add_argument('--query', help='Custom candidate search; never a ground-truth label')
    parser.add_argument('--category', help='Commons category name without Category:')
    parser.add_argument('--filter', help='Optional title/description regex before image downloads')
    parser.add_argument('--page', type=int, default=0, choices=range(21), help='Additional page of category/search results')
    args = parser.parse_args()
    root = args.output
    (root / 'images').mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'PhotoAdvisorResearch/0.1 (user-directed local CV dataset preparation)'
    index_path = root / 'candidates.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else []
    seen = {r['page_id'] for r in index}
    failures = []
    try:
        special = args.titles or args.query or args.category
        for group in (['targeted'] if special else (args.group or QUERIES)):
            params = dict(action='query', prop='imageinfo', iiprop='url|extmetadata',
                          iiurlwidth=640, format='json')
            if args.titles:
                params['titles'] = '|'.join(json.loads(args.titles.read_text(encoding='utf-8')))
            elif args.category:
                params.update(generator='categorymembers', gcmtitle='Category:'+args.category,
                              gcmtype='file', gcmlimit=args.limit)
            else:
                params.update(generator='search', gsrsearch=f'{args.query or QUERIES[group]} filetype:bitmap',
                              gsrnamespace=6, gsrlimit=args.limit, gsroffset=args.page*args.limit)
            key = hashlib.sha256(json.dumps([params, args.page], sort_keys=True).encode()).hexdigest()[:12]
            metadata_path = root / f'metadata-{group}-{key}.json'
            if metadata_path.exists():
                pages = json.loads(metadata_path.read_text(encoding='utf-8'))
            else:
                for page_number in range(args.page+1 if args.category else 1):
                    response = get(session, 'https://commons.wikimedia.org/w/api.php', params=params)
                    payload = response.json()
                    if page_number == args.page or not args.category:
                        break
                    if not payload.get('continue'):
                        payload = {}
                        break
                    params.update(payload['continue'])
                pages = payload.get('query', {}).get('pages', {})
                metadata_path.write_text(json.dumps(pages, ensure_ascii=False), encoding='utf-8')
            for page in sorted(pages.values(), key=lambda p: p.get('index', 0)):
                if page['pageid'] in seen or not page.get('imageinfo'):
                    continue
                info = page['imageinfo'][0]
                meta = info.get('extmetadata', {})
                license_name = metadata_field(meta, 'LicenseShortName')
                if not allowed_license(license_name):
                    continue
                if args.filter and not re.search(args.filter,
                        page['title']+' '+metadata_field(meta, 'ImageDescription'), re.IGNORECASE):
                    continue
                image_url = info.get('thumburl', info['url'])
                if urlparse(image_url).hostname not in {'upload.wikimedia.org', 'thumb.wikimedia.org'}:
                    continue
                try:
                    image_response = get(session, image_url)
                    if len(image_response.content) > 12 * 1024 * 1024:
                        raise ValueError('image too large')
                    with Image.open(BytesIO(image_response.content)) as source:
                        if source.width * source.height > 24_000_000:
                            raise ValueError('pixel limit')
                        image = source.convert('RGB')
                        image.thumbnail((640, 640))
                        relative = f'images/{page["pageid"]}.jpg'
                        image.save(root / relative, quality=93)
                        image.close()
                    index.append(dict(page_id=page['pageid'], path=relative, search_group=group,
                        title=page['title'], description=metadata_field(meta, 'ImageDescription'),
                        source_url=info['descriptionurl'], download_url=image_url,
                        author=metadata_field(meta, 'Artist'), license=license_name,
                        license_url=metadata_field(meta, 'LicenseUrl'),
                        review_status='pending', derivative='resized JPEG, no semantic changes'))
                    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
                    seen.add(page['pageid'])
                except (requests.RequestException, OSError, ValueError) as exc:
                    failures.append(dict(page_id=page['pageid'], error=str(exc)))
                finally:
                    time.sleep(1)
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{group}: {len(index)} licensed candidates total', flush=True)
    finally:
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
        (root / 'download-errors.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
        session.close()
    print(f'Collected {len(index)} candidates. All require visual review before training.')


if __name__ == '__main__':
    main()
