"""Compare the official and custom loops using the same real observation tape."""
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
import torch
import run_swap
from lerobot.envs.libero import LiberoEnv
from lerobot.scripts import lerobot_eval
from lerobot.utils.constants import ACTION

official_rollout = lerobot_eval.rollout
EPISODE = 5


def set_initial_index(self, index):
    self.init_state_id = index


LiberoEnv.rq2_set_initial_index = set_initial_index


class RecordingEnv:
    def __init__(self, env):
        self.env = env
        self.tape = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        self.env.call('rq2_set_initial_index', EPISODE)
        value = self.env.reset(*args, **kwargs)
        self.initial = deepcopy(value)
        self.metadata_values = {name: deepcopy(self.env.call(name)) for name in
            ['task_description', 'task', 'task_id', '_max_episode_steps', 'rq2_reset_audit']}
        reference = Path(os.environ['LOOP_REFERENCE']) / 'rng' / f'task2_episode{EPISODE}.pt'
        state = torch.load(reference, map_location='cpu', weights_only=True)
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state_all(state['cuda'])
        self.rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
        return value

    def step(self, action):
        value = self.env.step(action)
        self.tape.append(deepcopy(value))
        return value


class ReplayEnv:
    num_envs = 1

    def __init__(self, recording):
        self.recording = recording

    def reset(self, *args, **kwargs):
        self.index = 0
        return deepcopy(self.recording.initial)

    def call(self, name, *args, **kwargs):
        return deepcopy(self.recording.metadata_values[name])

    def step(self, action):
        value = deepcopy(self.recording.tape[self.index])
        self.index += 1
        return value


def compare(env, **kwargs):
    kwargs['seeds'] = [EPISODE]
    recording = RecordingEnv(env)
    reference = official_rollout(recording, **kwargs)
    torch.set_rng_state(recording.rng['cpu'])
    torch.cuda.set_rng_state_all(recording.rng['cuda'])
    replay_kwargs = dict(kwargs)
    replay_kwargs['render_callback'] = None
    candidate = run_swap.rollout(ReplayEnv(recording), **replay_kwargs)
    matches = {key: torch.equal(reference[key], candidate[key]) for key in [ACTION, 'reward', 'success', 'done']}
    result = {'task_id': 2, 'initial_state_index': EPISODE, 'environment_seed': EPISODE,
              'steps': reference[ACTION].shape[1], 'exact_tensor_equality': matches,
              'max_absolute_action_difference': float(torch.max(torch.abs(reference[ACTION]-candidate[ACTION]))),
              'method': 'Official loop records a real simulation trajectory; custom loop replays identical raw observations and transition results with the same trained policy, processors and restored initial CPU/CUDA RNG.',
              'all_checks_passed': all(matches.values())}
    (run_swap.OUTPUT / 'loop_equivalence.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(run_swap.OUTPUT / 'action_comparison.npz',
                        official=reference[ACTION].numpy(), custom=candidate[ACTION].numpy())
    assert result['all_checks_passed'], result
    print(json.dumps(result, indent=2), flush=True)
    return reference


if __name__ == '__main__':
    lerobot_eval.rollout = compare
    lerobot_eval.main()
