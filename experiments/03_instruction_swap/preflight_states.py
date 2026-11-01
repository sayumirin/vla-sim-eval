"""Verify all paired resets on this backend BEFORE loading the VLA model."""
import json
import os
from pathlib import Path
import numpy as np
import run_swap
from lerobot.envs import make_env, close_envs
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.random_utils import set_seed


def state_vector(self):
    return self._env.env.sim.get_state().flatten().tolist()


LiberoEnv.rq2_state_vector = state_vector
set_seed(0)
reference = Path(os.environ['RQ2_REFERENCE'])
output = run_swap.OUTPUT
(output / 'settled_states').mkdir(exist_ok=True)
offsets = {int(k):int(v) for k,v in json.loads(os.environ.get('RQ2_RESUME_OFFSETS', '{}')).items()}
ids = [t for t in range(10) if offsets.get(t,0)<10]
expected_pairs = sum(10-offsets.get(t,0) for t in ids)
envs = make_env(LiberoConfig(task='libero_goal', task_ids=ids, init_states=True, hard_reset=True),
                n_envs=1, use_async_envs=True)
count = 0
try:
    for task_id in ids:
        env = envs['libero_goal'][task_id]
        try:
            for episode in range(offsets.get(task_id,0),10):
                env.reset(seed=[episode], options={NEW_ROLLOUT_OPTION: True})
                audit = env.call('rq2_reset_audit')[0]
                stem = f'task{task_id}_episode{episode}'
                state = np.asarray(env.call('rq2_state_vector')[0], dtype=np.float64)
                np.save(output / 'settled_states' / f'{stem}.npy', state)
                baseline = json.loads((reference / 'episode_audits' / f'{stem}.json').read_text())
                for field in ['supplied_state_sha256', 'settled_state_sha256', 'visual_model_hashes']:
                    assert audit[field] == baseline[field], (task_id, episode, field)
                assert run_swap.array_hash(state) == baseline['settled_state_sha256']
                with np.load(reference / 'initial_images' / f'{stem}.npz') as expected, np.load(output / 'initial_images' / f'{stem}.npz') as actual:
                    metrics = {}
                    for key in expected.files:
                        diff = np.abs(expected[key].astype(float)-actual[key].astype(float))
                        metrics[key] = {'max_abs_difference': float(diff.max()),
                                        'fraction_different_channels': float((diff>0).mean())}
                        assert diff.max() <= run_swap.CONFIG['image_pair_tolerance']['max_abs_channel_difference']
                        assert (diff>0).mean() <= run_swap.CONFIG['image_pair_tolerance']['max_fraction_different_channels']
                with (output / 'checks.jsonl').open('a') as stream:
                    stream.write(json.dumps({'task_id': task_id, 'episode': episode,
                        'settled_state_sha256': audit['settled_state_sha256'], 'image_metrics': metrics})+'\n')
                # These are temporary probe images; formal initial images remain retained.
                (output / 'initial_images' / f'{stem}.npz').unlink()
                (output / 'initial_images' / f'{stem}.json').unlink()
                count += 1
            print(f'RQ2_PREFLIGHT verified={count}/{expected_pairs}', flush=True)
        finally:
            env.close()
finally:
    close_envs(envs)
(output / 'summary.json').write_text(json.dumps({'verified_pairs': count, 'expected_pairs': expected_pairs,
    'cloud': os.environ.get('MODAL_CLOUD_PROVIDER'), 'region': os.environ.get('MODAL_REGION')}, indent=2))
assert count == expected_pairs
