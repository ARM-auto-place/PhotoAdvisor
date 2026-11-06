"""Download the pinned Apache-2.0 DINOv2 source, without credentials or remote code."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


MODEL_ID = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
WEIGHT_SHA256 = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/dinov2-small-apache'))
    args = parser.parse_args()
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION or getattr(info.card_data, 'license', None) != 'apache-2.0':
        raise ValueError('unexpected source revision or license')
    upstream = next(s.lfs.sha256 for s in info.siblings if s.rfilename == 'model.safetensors')
    if upstream != WEIGHT_SHA256:
        raise ValueError('unexpected publisher weight hash')
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json', 'README.md'):
        path = Path(hf_hub_download(MODEL_ID, name, revision=REVISION, local_dir=args.output, token=False))
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if hashes['model.safetensors'] != upstream:
        raise ValueError('downloaded weight checksum mismatch')
    if not (args.output/'README.md').read_text(encoding='utf-8').startswith('---\nlicense: apache-2.0'):
        raise ValueError('model card license mismatch')
    manifest = dict(model_id=MODEL_ID, revision=REVISION, license='Apache-2.0',
                    source_url=f'https://huggingface.co/{MODEL_ID}/tree/{REVISION}',
                    sha256=hashes, upstream_weights_sha256=upstream)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Verified pinned Apache-2.0 source: '+str(args.output))


if __name__ == '__main__':
    main()
