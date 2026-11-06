"""Run once with network access. Serving itself never downloads weights."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import snapshot_download
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

from app.inference import MODEL_ID, MODEL_REVISION


def main():
    root = Path('models')
    root.mkdir(exist_ok=True)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=root / 'clip',
                      allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin'])
    # Torchvision pins the URL with a content hash and verifies it on download.
    torch.hub.set_dir(str(root / 'torch-cache'))
    state = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(progress=True, check_hash=True)
    torch.save(state, root / 'ssdlite.pt')
    files = [p for p in (root / 'clip').iterdir() if p.is_file() and p.name != 'manifest.json']
    checksums = {}
    for file in files + [root / 'ssdlite.pt']:
        with file.open('rb') as stream:
            checksums[str(file.relative_to(root)).replace('\\', '/')] = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                'detector': 'torchvision/ssdlite320_mobilenet_v3_large/COCO_V1',
                'sha256': checksums}
    (root / 'clip' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Local model artifact prepared. No accuracy claim; run dataset evaluation next.')


if __name__ == '__main__':
    main()
