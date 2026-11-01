"""CPU-only image build: pinned data, lossless frame cache, subset-only statistics."""
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames_pyav
import torch
from torchvision.models import resnet18, ResNet18_Weights

root = Path('/opt/rq1')
plan = json.loads((root/'plan.json').read_text())
dataset_root = root/'dataset'
snapshot_download(plan['dataset_repo'], repo_type='dataset', revision=plan['dataset_revision'], local_dir=dataset_root)
meta = LeRobotDatasetMetadata(plan['dataset_repo'], root=dataset_root, revision=plan['dataset_revision'])
camera = 'observation.images.top'
episodes = [meta.episodes[i] for i in range(50)]
assert all(e['length'] == 400 for e in episodes)
mapping = {}
for ep in episodes:
    path = str(meta.get_video_file_path(ep['episode_index'], camera))
    start = ep[f'videos/{camera}/from_timestamp']
    for frame in range(ep['length']):
        timestamp = start+frame/meta.fps
        key = (path, round(timestamp*meta.fps))
        assert key not in mapping
        mapping[key] = (ep['dataset_from_index']+frame, timestamp)
frames_dir = root/'frames'
frames_dir.mkdir(exist_ok=True)
count = 0
for video in sorted({k[0] for k in mapping}):
    with av.open(dataset_root/video) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for frame in container.decode(stream):
            timestamp = float(frame.pts*stream.time_base)
            key = (video, round(timestamp*meta.fps))
            if key not in mapping:
                continue
            index, expected = mapping[key]
            assert abs(timestamp-expected) < 1e-4
            array = frame.to_ndarray(format='rgb24')
            Image.fromarray(array).save(frames_dir/f'{index:06d}.png', compress_level=1)
            count += 1
            if count % 2000 == 0:
                print(f'Lossless cache: {count}/20000 frames', flush=True)
assert count == len(mapping) == 20000
checks = []
for ep_id, frame_id in [(0,0),(0,399),(1,1),(14,137),(28,200),(43,199),(49,399)]:
    ep = episodes[ep_id]
    timestamp = ep[f'videos/{camera}/from_timestamp']+frame_id/meta.fps
    original = decode_video_frames_pyav(dataset_root/meta.get_video_file_path(ep_id,camera),[timestamp],1e-4,return_uint8=True)[0]
    cached = torch.from_numpy(np.array(Image.open(frames_dir/f'{ep["dataset_from_index"]+frame_id:06d}.png'))).permute(2,0,1)
    assert torch.equal(original,cached)
    checks.append({'episode':ep_id,'frame':frame_id,'exact_rgb_equal':True})
table = pq.read_table(sorted((dataset_root/'data').rglob('*.parquet')))
ep_indices = np.asarray(table['episode_index'].to_numpy())
statistics = {}
for size, selected in plan['subsets'].items():
    mask = np.isin(ep_indices,selected)
    assert int(mask.sum()) == int(size)*400
    statistics[size] = {}
    for feature in ['observation.state','action']:
        values = np.asarray(table[feature].to_pylist(),dtype=np.float64)[mask]
        statistics[size][feature] = {key: value.tolist() for key,value in {
            'mean':values.mean(0), 'std':values.std(0), 'min':values.min(0),
            'max':values.max(0), 'count':np.array([len(values)])}.items()}
(root/'subset_stats.json').write_text(json.dumps(statistics,indent=2))
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
weight_file = next(Path('/opt/rq1/torch_cache/hub/checkpoints').glob('resnet18-*.pth'))
audit = {'repo':plan['dataset_repo'],'revision':plan['dataset_revision'],'episodes':50,'frames':count,
         'length_per_episode':400,'fps':meta.fps,'cache':'lossless PNG decoded with the official PyAV RGB conversion',
         'cache_equivalence_checks':checks,'normalization':'state/action statistics computed only from selected training episodes; fixed ImageNet image statistics',
         'resnet_weights_sha256':hashlib.sha256(weight_file.read_bytes()).hexdigest(),
         'frame_cache_bytes':sum(p.stat().st_size for p in frames_dir.iterdir())}
(root/'data_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
