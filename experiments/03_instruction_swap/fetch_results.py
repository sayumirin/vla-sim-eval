"""Download one completed run without relying on CLI directory glob handling."""
import argparse
import concurrent.futures
import json
from pathlib import Path, PurePosixPath

import modal
from modal.volume import FileEntryType

parser = argparse.ArgumentParser()
parser.add_argument('run_name')
parser.add_argument('--diagnostic', action='store_true')
parser.add_argument('--control-states', action='store_true')
args = parser.parse_args()
if Path(args.run_name).name != args.run_name or '..' in args.run_name:
    raise ValueError('Expected a run directory name')
assert not (args.diagnostic and args.control_states)
namespace = 'rq2/control_states' if args.control_states else ('rq2/diagnostics' if args.diagnostic else 'results_rq2')
remote_root = PurePosixPath(namespace) / args.run_name
folder = 'control_states' if args.control_states else ('diagnostics' if args.diagnostic else 'modal_results')
destination = Path(__file__).parent / folder / args.run_name
volume = modal.Volume.from_name('knightec-vla-stage1')
def download(entry):
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
    return {'path': str(relative), 'bytes': entry.size}


entries = [e for e in volume.iterdir(str(remote_root), recursive=True) if e.type == FileEntryType.FILE]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    downloaded = list(executor.map(download, entries))
print(json.dumps({'destination': str(destination), 'files': len(downloaded)}, indent=2))
(destination / 'download_manifest.json').write_text(json.dumps(downloaded, indent=2))


