"""Verify that stopping the old queue did not produce duplicate camera episodes."""
import json
from pathlib import Path
import modal
from modal.volume import FileEntryType

root = Path(__file__).parent
volume = modal.Volume.from_name('knightec-vla-stage1')
correct = 'camera_100_seed0_20261029T010844Z'
entries = [e for e in volume.iterdir('results_plus', recursive=True) if e.type == FileEntryType.FILE]
runs = sorted({e.path.split('/')[1] for e in entries if '/camera_100_' in e.path})
records = []
for run in runs:
    prefix = 'results_plus/' + run + '/'
    files = [e.path for e in entries if e.path.startswith(prefix)]
    progress_path = prefix + 'progress.jsonl'
    episodes = len(b''.join(volume.read_file(progress_path)).decode().splitlines()) if progress_path in files else 0
    videos = sum(p.endswith('.mp4') for p in files)
    if run != correct:
        assert (root / 'parallel_coordination.json').exists()
        assert episodes == videos == 0, (run, episodes, videos)
        assert prefix + 'eval_info.json' not in files
    records.append({'run': run, 'recorded_episodes': episodes, 'videos': videos,
                    'role': 'formal_camera_run' if run == correct else 'cancelled_before_any_episode',
                    'files': len(files)})
(root / 'audit/parallel_runs.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
