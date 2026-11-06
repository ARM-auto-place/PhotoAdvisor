"""Measure a local frozen classifier in one fresh CPU process; never score accuracy."""
import argparse
import gc
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil

from app.evaluation import load_manifest
from app.imaging import decode
from app.settings import Settings
from app.trained import LABELS, TrainedVision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--loader',choices=('mapped','eager'),default='mapped')
    parser.add_argument('--limit',type=int,default=12)
    parser.add_argument('--repeat',type=int,default=3)
    args = parser.parse_args()
    if args.output.exists() or args.limit<1 or args.repeat<1:
        parser.error('choose new output and positive sample sizes')
    rows = [r for r in load_manifest(args.manifest,LABELS,LABELS[:-1]) if r['split']=='calibration']
    if not rows:
        parser.error('calibration images required for a fixed runtime workload')
    rows = [rows[i] for i in np.linspace(0,len(rows)-1,min(args.limit,len(rows)),dtype=int)]
    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]
    def sample():
        while not stop.wait(.01):
            peak[0] = max(peak[0],process.memory_info().rss)
    sampler = threading.Thread(target=sample,daemon=True)
    images = []
    sampler.start()
    try:
        started = time.perf_counter()
        vision = TrainedVision(Settings(_env_file=None,cv_backend='trained',trained_model_dir=args.model,
            cpu_threads=2,device='cpu'),memory_efficient=args.loader=='mapped')
        load_seconds = time.perf_counter()-started
        for row in rows:
            image,_ = decode((args.manifest.parent/row['path']).read_bytes(),24_000_000)
            images.append(image)
        vision.predict(images[:1])
        gc.collect()
        warm_rss = process.memory_info().rss
        seconds = []
        cpu_started = time.process_time()
        for _ in range(args.repeat):
            for image in images:
                started = time.perf_counter()
                vision.predict([image])
                seconds.append(time.perf_counter()-started)
        cpu_seconds = time.process_time()-cpu_started
        peak[0] = max(peak[0],process.memory_info().rss)
        report = dict(kind='isolated_cpu_runtime_not_accuracy',model=str(args.model),model_version=vision.version,
            loader=args.loader,threads=2,samples=len(seconds),paths=[r['path'] for r in rows],
            load_seconds=load_seconds,warm_rss_bytes=warm_rss,sampled_peak_rss_bytes=peak[0],
            median_seconds=float(np.median(seconds)),p95_seconds=float(np.percentile(seconds,95)),
            total_cpu_seconds=cpu_seconds,production_approved=False,
            note='Fresh process, CPU, one image per inference after warmup. Same calibration workload. '
                 'RSS includes Python/libraries and decoded photos. Peak sampled every 10ms after imports; '
                 'not a full service load test and not a peak-memory guarantee.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k not in ('paths','note')}))
    finally:
        stop.set()
        sampler.join()
        for image in images:
            image.close()


if __name__=='__main__':
    main()
