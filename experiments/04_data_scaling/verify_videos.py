"""Offline video integrity checks and a representative contact sheet (OpenCV)."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--partial', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root/'results'
out.mkdir(exist_ok=True)
checks, previews = [], []
first_frames = {}
for branch in ['modal_results','audit_results']:
    folders = sorted(p for p in (root/branch).glob('formal_A10_*') if p.is_dir())
    if not args.partial:
        assert len(folders) == 15, (branch,len(folders))
    for folder in folders:
        episode_path = folder/'eval/episodes.jsonl'
        if args.partial and not episode_path.exists():
            continue
        rows = {e['episode']:e for e in map(json.loads, episode_path.read_text().splitlines())}
        for index in [0,1]:
            video = folder/f'eval/videos/aloha_0/eval_episode_{index}.mp4'
            if args.partial and not video.exists():
                continue
            cap = cv2.VideoCapture(str(video))
            assert cap.isOpened(), video
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frames == rows[index]['steps'], (str(video), frames, rows[index]['steps'])
            ok, first = cap.read()
            assert ok and first.std() > 5, video
            first_frames.setdefault((branch,index),[]).append(first)
            if branch == 'modal_results' and index == 0 and rows[index]['train_seed']==0 and rows[index]['n_demonstrations'] in [5,35]:
                panels = []
                for frame_index in [0,frames//2,frames-1]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                    ok, frame = cap.read()
                    assert ok
                    frame = cv2.resize(frame,(480,360))
                    panel = np.full((405,480,3),255,dtype=np.uint8)
                    panel[45:]=frame
                    label = f"N={rows[index]['n_demonstrations']}, seed=0, frame={frame_index}"
                    cv2.putText(panel,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(35,35,35),1,cv2.LINE_AA)
                    panels.append(panel)
                previews.append((rows[index]['n_demonstrations'],np.concatenate(panels,axis=1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES,frames-1)
            ok, final = cap.read()
            assert ok and final.std()>5
            cap.release()
            checks.append({'path':str(video.relative_to(root)), 'frames':frames,
                'success':rows[index]['success'], 'sha256':hashlib.sha256(video.read_bytes()).hexdigest()})
if previews:
    sheet = np.concatenate([x[1] for x in sorted(previews)],axis=0)
    cv2.imwrite(str(out/'example_rollouts.jpg'),sheet)
differences = {}
for (branch,index), images in first_frames.items():
    ref = images[0].astype(float)
    differences[f'{branch}/episode_{index}'] = {
        'videos':len(images),
        'max_mean_absolute_pixel_difference':max(float(np.abs(im.astype(float)-ref).mean()) for im in images),
        'note':'Lossy encoded first-frame comparison is descriptive; exact pairing uses seeded initial qpos.'}
audit = {'partial':args.partial,'videos_checked':len(checks),'all_frame_counts_match':True,
    'first_frames':differences,'videos':checks}
(out/('video_audit_partial.json' if args.partial else 'video_audit.json')).write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='videos'},indent=2))
