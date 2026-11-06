"""Real-model regression on existing photos and explicitly derived degradations.

This is an engineering check, never an independent accuracy benchmark.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import psutil

from app.inference import Vision
from app.schemas import EvaluateParams
from app.service import Advisor
from app.settings import Settings


def encoded(image):
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/robustness.json'))
    args = parser.parse_args()
    settings = Settings(_env_file=None, checklist_path=Path('configs/checklist-kafka.json'))
    vision = Vision(settings)
    data, cases = {}, []
    for name in ('front_left', 'rear_left', 'dog', 'bus'):
        original = Path(f'datasets/smoke/{name}.jpg').read_bytes()
        data[name] = original
        cases.append((name, 'real_original', name))
        with Image.open(BytesIO(original)) as source:
            base = ImageOps.exif_transpose(source).convert('RGB')
        base.thumbnail((1024, 1024))
        variants = {
            'dark': ImageEnhance.Brightness(base).enhance(.03),
            'bright': ImageEnhance.Brightness(base).enhance(5),
            'blur': base.filter(ImageFilter.GaussianBlur(20)),
            'mirror': ImageOps.mirror(base),
            'small': base.resize((96, 64)),
        }
        for transform, image in variants.items():
            key = f'{name}-{transform}'
            data[key] = encoded(image)
            cases.append((key, transform, name))
            image.close()
        base.close()
    data['corrupt'] = b'invalid image bytes'
    cases.append(('corrupt', 'corrupt', None))

    class LocalStorage:
        def fetch(self, listing_id, photo):
            return data[photo.object_key]
        def close(self):
            pass

    worker = Advisor(settings, LocalStorage(), vision)
    reports = []
    started = time.perf_counter()
    try:
        for key, transform, source in cases:
            result = worker.evaluate(EvaluateParams(listing_id='robustness',
                photos=[dict(photo_id=key, object_key=key)]))
            photo = result['photo_results'][0]
            if transform in {'dark', 'blur', 'small', 'corrupt'}:
                assert not photo['accepted'], f'unsafe degradation accepted: {key}'
            reports.append(dict(source=source, transform=transform, result=photo))
        duplicate = worker.evaluate(EvaluateParams(listing_id='robustness', photos=[
            dict(photo_id='first', object_key='front_left'),
            dict(photo_id='second', object_key='front_left')]))
        assert duplicate['photo_results'][1]['duplicate_of'] == 'first'
        ordered = [dict(photo_id=name, object_key=name) for name in ('front_left', 'rear_left', 'dog', 'bus')]
        a = worker.evaluate(EvaluateParams(listing_id='robustness', photos=ordered))
        b = worker.evaluate(EvaluateParams(listing_id='robustness', photos=list(reversed(ordered))))
        assert a['required'] == b['required'], 'coverage depends on upload order'
        report = dict(kind='real_model_derived_robustness_not_independent_accuracy',
                      independent_originals=4, cases=len(reports), predictions=reports,
                      duplicate_check='passed', order_check='passed', model=a['model'],
                      seconds=time.perf_counter()-started,
                      rss_bytes=psutil.Process().memory_info().rss)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Passed {len(reports)} real-model cases + duplicate/order checks; wrote {args.output}')
    finally:
        worker.close()


if __name__ == '__main__':
    main()
