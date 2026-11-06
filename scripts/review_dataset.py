"""Start the local blind annotation workspace on loopback only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from app.review import ReviewWorkspace, create_review_app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, action='append', default=[])
    parser.add_argument('--state', type=Path, default=Path('data/review-workspace'))
    parser.add_argument('--previous', type=Path, default=Path('datasets/view-training-v10/manifest.jsonl'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    workspace = ReviewWorkspace(args.state, args.source, args.previous)
    print(f'Annotation workspace: http://127.0.0.1:{args.port}')
    uvicorn.run(create_review_app(workspace), host='127.0.0.1', port=args.port, access_log=False)
