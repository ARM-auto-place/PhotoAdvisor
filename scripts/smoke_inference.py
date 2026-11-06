"""Actual model execution on local images. This is a smoke check, not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil

from app.imaging import decode, quality
from app.inference import load_vision
from app.settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='+', type=Path)
parser.add_argument('--output', type=Path, default=Path('reports/smoke.json'))
args = parser.parse_args()
settings = Settings()
started = time.perf_counter()
vision = load_vision(settings)
load_seconds = time.perf_counter() - started
results = []
for path in args.images:
    image, size = decode(path.read_bytes(), settings.max_pixels)
    start = time.perf_counter()
    prediction = vision.predict([image])[0]
    elapsed = time.perf_counter() - start
    prediction.pop('embedding')
    results.append(dict(path=str(path), seconds=elapsed, prediction=prediction,
                        heuristic_quality_issues=quality(image, size)))
    image.close()
report = dict(kind='real_inference_smoke_not_accuracy', model_version=vision.version,
              load_seconds=load_seconds, device=settings.device,
              rss_bytes=psutil.Process().memory_info().rss, results=results)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2))
