"""Download one completed run without relying on CLI directory glob handling."""
import argparse
import json
from pathlib import Path, PurePosixPath

import modal
from modal.volume import FileEntryType

parser = argparse.ArgumentParser()
parser.add_argument('run_name')
args = parser.parse_args()
if Path(args.run_name).name != args.run_name or '..' in args.run_name:
    raise ValueError('Expected a run directory name')
remote_root = PurePosixPath('results') / args.run_name
destination = Path(__file__).parent / 'modal_results' / args.run_name
volume = modal.Volume.from_name('knightec-vla-stage1')
downloaded = []
for entry in volume.iterdir(str(remote_root), recursive=True):
    if entry.type != FileEntryType.FILE:
        continue
    relative = PurePosixPath(entry.path.lstrip('/')).relative_to(remote_root)
    if '..' in relative.parts:
        raise ValueError('Unexpected remote path')
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as stream:
        for data in volume.read_file(entry.path):
            stream.write(data)
    if target.stat().st_size != entry.size:
        raise RuntimeError(f'Incomplete download: {relative}')
    downloaded.append({'path': str(relative), 'bytes': entry.size})
print(json.dumps({'destination': str(destination), 'files': len(downloaded)}, indent=2))
(destination / 'download_manifest.json').write_text(json.dumps(downloaded, indent=2))
