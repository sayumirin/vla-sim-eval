"""Render a fixed reset without loading the VLA model, for pairing diagnostics."""
import json
import torch
import run_swap
from lerobot.envs import make_env, close_envs
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.random_utils import set_seed

set_seed(0)
torch.cuda.init()
envs = make_env(LiberoEnv(task='libero_goal', task_ids=[0], init_states=True, hard_reset=True),
                n_envs=1, use_async_envs=True)
env = envs['libero_goal'][0]
try:
    for seed in range(2):
        env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        print(json.dumps(env.call('rq2_reset_audit')[0]), flush=True)
finally:
    close_envs(envs)
