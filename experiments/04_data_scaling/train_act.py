"""Official ACT training with equivalent PNG reads and subset-only normalization."""
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import gym_aloha  # Register ALOHA in spawned evaluation workers as well as the parent.
from gym_aloha.env import AlohaEnv
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets import factory as dataset_factory
from lerobot.scripts import lerobot_train

# AsyncVectorEnv closes workers when an attribute query raises. ACT does not use
# language, but the official evaluator probes this metadata field.
AlohaEnv.task_description = property(lambda self: self.task.replace('_', ' '))

ROOT = Path('/opt/rq1')
OUTPUT = Path(os.environ['RQ1_OUTPUT'])
plan = json.loads((ROOT/'plan.json').read_text())
subset_stats = json.loads((ROOT/'subset_stats.json').read_text())
original_dataset = dataset_factory.make_dataset
original_update = lerobot_train.update_policy
original_eval = lerobot_train.eval_policy_all
step = int(os.environ.get('RQ1_START_STEP','0'))
started = time.monotonic()


def cached_video_frames(self, queries, ep_idx):
    assert self.episodes is None or ep_idx in self.episodes
    episode = self._meta.episodes[ep_idx]
    result = {}
    for key, timestamps in queries.items():
        assert key == 'observation.images.top'
        frames = []
        for timestamp in timestamps:
            frame = round(timestamp*self._meta.fps)
            assert abs(timestamp-frame/self._meta.fps) <= self._tolerance_s
            assert 0 <= frame < episode['length']
            index = episode['dataset_from_index']+frame
            with Image.open(ROOT/'frames'/f'{index:06d}.png') as image:
                tensor = torch.from_numpy(np.array(image)).permute(2,0,1).contiguous()
            frames.append(tensor if self._return_uint8 else tensor.float()/255)
        result[key] = torch.stack(frames).squeeze(0)
    return result


DatasetReader._query_videos = cached_video_frames


def dataset_with_subset_stats(cfg):
    dataset = original_dataset(cfg)
    selected = [int(e) for e in cfg.dataset.episodes]
    size = str(len(selected))
    assert set(selected) == set(plan['subsets'][size])
    actual = {int(e) for e in dataset.hf_dataset.unique('episode_index')}
    assert actual == set(selected) and dataset.num_frames == len(selected)*400
    for feature, stats in subset_stats[size].items():
        dataset.meta.stats[feature] = {k:np.asarray(v) for k,v in stats.items()}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'subset_audit.json').write_text(json.dumps({'selected_episodes':selected,
        'frames':dataset.num_frames,'normalization_frames':len(selected)*400,
        'dataset_revision':plan['dataset_revision'],'image_cache':'exact decoded RGB; no image resizing or recompression loss'},indent=2))
    return dataset


def update_with_log(*args, **kwargs):
    global step
    metrics, extra = original_update(*args, **kwargs)
    assert np.isfinite(float(metrics.loss.val)), 'Non-finite training loss'
    step += 1
    if step == 1 or step % 50 == 0:
        record = {'step':step,'elapsed_seconds':time.monotonic()-started,
                  'loss':float(metrics.loss.val),'update_seconds':float(metrics.update_s.val)}
        with (OUTPUT/'training_progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('RQ1_TRAIN '+json.dumps(record),flush=True)
    return metrics,extra


def pilot_evaluation(*args, **kwargs):
    kwargs['start_seed'] = plan['pilot_eval_seeds'][0]
    kwargs['max_episodes_rendered'] = 2
    result = original_eval(*args, **kwargs)
    (OUTPUT/f'pilot_eval_step{step}.json').write_text(json.dumps(result,indent=2))
    print('RQ1_PILOT '+json.dumps({'step':step,'successes':result['overall']['n_success'],
        'episodes':result['overall']['n_episodes'],'elapsed_seconds':time.monotonic()-started}),flush=True)
    return result


dataset_factory.make_dataset = dataset_with_subset_stats
lerobot_train.update_policy = update_with_log
lerobot_train.eval_policy_all = pilot_evaluation
if __name__ == '__main__':
    lerobot_train.main()
