"""Check canonical video mappings against the recorded action-step counts."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import cv2

root = Path(__file__).parent
results = root / 'results/rq2'
episode_file = results / 'episodes.jsonl'
rows = [json.loads(line) for line in episode_file.read_text().splitlines()]
assert len(rows) == 500 and len({(r['condition'],r['task_id'],r['episode']) for r in rows}) == 500
runs = {c: json.loads((results / f'{c}.json').read_text()) for c in {r['condition'] for r in rows}}


def check(row):
    folder = root / 'modal_results' / Path(runs[row['condition']]['output']).name
    video = folder / 'videos' / f'libero_goal_{row["task_id"]}' / f'eval_episode_{row["episode"]}.mp4'
    capture = cv2.VideoCapture(str(video))
    try:
        assert capture.isOpened(), str(video)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frames == row['steps'], (str(video), frames, row['steps'])
    finally:
        capture.release()
    return {'condition': row['condition'], 'frames': frames}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    checks = list(pool.map(check, rows))
audit = {'unique_condition_task_episode_keys': len(rows), 'videos_checked': len(checks),
         'all_video_frame_counts_match_recorded_steps': True,
         'episodes_jsonl_sha256': hashlib.sha256(episode_file.read_bytes()).hexdigest(),
         'per_condition': {c: {'videos': sum(r['condition']==c for r in checks),
             'total_action_steps': sum(r['frames'] for r in checks if r['condition']==c)} for c in sorted(runs)}}
(results / 'artifact_audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
