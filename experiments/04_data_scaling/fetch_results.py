"""Download run metadata and videos; checkpoints stay in the private Modal volume."""
import argparse
import concurrent.futures
import json
from pathlib import Path,PurePosixPath
import modal
from modal.volume import FileEntryType

parser=argparse.ArgumentParser()
parser.add_argument('run_name',nargs='?')
parser.add_argument('--all-completed',action='store_true')
parser.add_argument('--audit',action='store_true')
parser.add_argument('--attempt',action='store_true')
args=parser.parse_args()
assert not (args.audit and args.attempt)
volume=modal.Volume.from_name('knightec-vla-stage3b')
local_root=Path(__file__).parent/('attempts' if args.attempt else ('audit_results' if args.audit else 'modal_results'))
remote_root=PurePosixPath('audit_results' if args.audit else 'results')
if args.all_completed:
    names=[p.stem for p in sorted(local_root.glob('formal_A10_*.json'))
           if json.loads(p.read_text())['status']=='completed' or
              json.loads(p.read_text()).get('reviewed_execution_status')=='completed_with_replay_differences']
else:
    assert args.run_name
    names=[args.run_name]


def download(entry,root,destination):
    relative=PurePosixPath(entry.path.lstrip('/')).relative_to(root)
    assert '..' not in relative.parts
    target=destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or target.stat().st_size!=entry.size:
        temporary=target.with_name(target.name+'.partial')
        with temporary.open('wb') as stream:
            for chunk in volume.read_file(entry.path):stream.write(chunk)
        assert temporary.stat().st_size==entry.size
        temporary.replace(target)
    assert target.stat().st_size==entry.size
    return {'path':str(relative),'bytes':entry.size}


for name in names:
    assert Path(name).name==name and '..' not in name
    root=remote_root/name
    destination=local_root/name
    entries=[e for e in volume.iterdir(str(root),recursive=True)
             if e.type==FileEntryType.FILE and ('/checkpoints/' not in e.path
                 or ('/pretrained_model/' in e.path and (e.path.endswith('.json')
                     or ('processor' in PurePosixPath(e.path).name and e.size < 1000000))))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files=list(pool.map(lambda e:download(e,root,destination),entries))
    (destination/'download_manifest.json').write_text(json.dumps(files,indent=2))
    print(json.dumps({'destination':str(destination),'files':len(files)}),flush=True)
