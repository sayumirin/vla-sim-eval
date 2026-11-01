"""Recover control state vectors only when their hashes equal the original C0 hashes."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import numpy as np
import run_swap
from lerobot.envs import make_env, close_envs
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.random_utils import set_seed
from libero.libero.envs.env_wrapper import ControlEnv


def state_vector(self):
    return self._env.env.sim.get_state().flatten().tolist()


def set_index(self, index):
    self.init_state_id = index


LiberoEnv.rq2_state_vector = state_vector
LiberoEnv.rq2_set_index = set_index
reference = Path(os.environ['RQ2_REFERENCE'])
cache = Path('/data/rq2/control_states') / reference.name
cache.mkdir(parents=True, exist_ok=True)
audits = {f.stem: json.loads(f.read_text()) for f in (reference / 'episode_audits').glob('*.json')}
assert len(audits) == 100
sources = {}
for source in Path('/data/results_rq2').glob('*/preflight/settled_states/*.npy'):
    if source.stem not in audits:
        continue
    state = np.load(source)
    if run_swap.array_hash(state) == audits[source.stem]['settled_state_sha256']:
        np.save(cache / source.name, state)
        sources[source.stem] = str(source)
missing = [key for key in audits if not (cache / f'{key}.npy').exists()]
print(f'Recovered {100-len(missing)}/100 exact C0 state vectors from prior reset probes.', flush=True)
set_seed(0)
ids = sorted({audits[key]['task_id'] for key in missing})
if ids:
    envs = make_env(LiberoConfig(task='libero_goal', task_ids=ids, init_states=True, hard_reset=True),
                    n_envs=1, use_async_envs=True)
    try:
        for task_id in ids:
            env = envs['libero_goal'][task_id]
            try:
                for key in sorted(k for k in missing if audits[k]['task_id'] == task_id):
                    episode = audits[key]['episode']
                    env.call('rq2_set_index', episode)
                    env.reset(seed=[episode], options={NEW_ROLLOUT_OPTION: True})
                    current = env.call('rq2_reset_audit')[0]
                    for field in ['supplied_state_sha256', 'settled_state_sha256', 'visual_model_hashes']:
                        assert current[field] == audits[key][field], (key, field)
                    state = np.asarray(env.call('rq2_state_vector')[0], dtype=np.float64)
                    assert run_swap.array_hash(state) == audits[key]['settled_state_sha256']
                    np.save(cache / f'{key}.npy', state)
                    sources[key] = 'independent reset probe; exact hash matched original C0 before copying'
                    print(f'Exact control snapshot verified: {key}', flush=True)
            finally:
                env.close()
    finally:
        close_envs(envs)
for key, audit in audits.items():
    assert run_swap.array_hash(np.load(cache / f'{key}.npy')) == audit['settled_state_sha256']
result = {'control_output': str(reference), 'verified_states': 100,
          'state_hashes': {k: a['settled_state_sha256'] for k,a in audits.items()},
          'sources': sources, 'cloud': os.environ.get('MODAL_CLOUD_PROVIDER'),
          'region': os.environ.get('MODAL_REGION'),
          'method': 'Recovered vectors are accepted only if their exact SHA256 equals the state hash recorded in the original C0 before its first action. No policy outcomes are recomputed or selected.'}
(cache / 'manifest.json').write_text(json.dumps(result, indent=2))
(run_swap.OUTPUT / 'state_cache.json').write_text(json.dumps(result, indent=2))
(run_swap.OUTPUT / 'state_restore_api.txt').write_text(inspect.getsource(ControlEnv.set_init_state)+'\n'+inspect.getsource(ControlEnv.regenerate_obs_from_state))
print(json.dumps({'cache': str(cache), 'verified_states': 100}), flush=True)
