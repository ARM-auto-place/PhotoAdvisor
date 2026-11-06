"""Run the local photo-upload pilot. Does not connect to production S3/Kafka."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from app.pilot import PilotWorkspace, create_pilot_app
from app.settings import Settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=Path('data/pilot'))
    parser.add_argument('--model', type=Path, default=Path('models/view-classifier-triple-v19-apache-balanced'))
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    settings = Settings(_env_file=None, cv_backend='trained', trained_model_dir=args.model)
    workspace = PilotWorkspace(args.state, settings)
    print(f'Photo pilot: http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run(create_pilot_app(workspace), host='127.0.0.1', port=args.port, access_log=False)
