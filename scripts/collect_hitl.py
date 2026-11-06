"""Download the public CC0 photo/mask mirror at an immutable revision."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = 'mitbersh/car-parts-segmentation-raw'
REVISION = '002a354280dd63e4940db9f3cec0ebcd7921c27e'
LICENSE_SOURCE = 'https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('datasets/hitl-source'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(repo=REPO, revision=REVISION, image_license='CC0 1.0', license_source=LICENSE_SOURCE,
        warning='Additional side/view tags are automatically generated and must not be treated as gold labels.')
    source = args.output/'source.json'
    if source.exists() and json.loads(source.read_text(encoding='utf-8')) != metadata:
        raise ValueError('output contains a different source; choose a new directory')
    for filename in ('README.md', 'meta.json', 'data/train-00000-of-00002.parquet',
                     'data/train-00001-of-00002.parquet'):
        hf_hub_download(REPO, filename, repo_type='dataset', revision=REVISION,
                        token=False, local_dir=args.output)
    source.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Downloaded pinned public source to {args.output}; run import_hitl.py for candidate proposals')


if __name__ == '__main__':
    main()
