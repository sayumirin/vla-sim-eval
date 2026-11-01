"""Align numerical reset tails to verified C0 snapshots and resume only missing pairs."""
import json
import os
from pathlib import Path
import runpy
import numpy as np
import run_swap
from lerobot.envs.libero import LiberoEnv

overrides = json.loads(Path('/opt/rq2/validation_overrides.json').read_text())
assert overrides['scope'] == 'initial_state_alignment_and_validation'
assert set(overrides['image_pair_tolerance']) == {'max_abs_channel_difference', 'max_fraction_different_channels'}
run_swap.CONFIG['image_pair_tolerance'] = overrides['image_pair_tolerance']
offsets = {int(k): int(v) for k,v in json.loads(os.environ.get('RQ2_RESUME_OFFSETS', '{}')).items()}
cache_path = Path(os.environ['RQ2_CONTROL_STATES']) if os.environ.get('RQ2_CONTROL_STATES') else None
cache = json.loads((cache_path / 'manifest.json').read_text()) if cache_path else None
if cache is not None:
    assert cache['control_output'] == str(run_swap.REFERENCE) and cache['verified_states'] == 100

original_init = LiberoEnv.__init__
original_physics_reset = run_swap.original_reset
original_audit_reset = LiberoEnv.reset


def init_with_offset(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.init_state_id = offsets.get(self.task_id, self.init_state_id)


def align_initial_state(self, *args, **kwargs):
    observation, info = original_physics_reset(self, *args, **kwargs)
    self.rq2_alignment = {'state_alignment_applied': False, 'state_alignment_max_abs_difference': 0.0}
    if cache is not None:
        episode = self.init_state_id-self._reset_stride
        stem = f'task{self.task_id}_episode{episode}'
        target = np.load(cache_path / f'{stem}.npy')
        assert run_swap.array_hash(target) == cache['state_hashes'][stem]
        current = self._env.env.sim.get_state().flatten()
        assert target.shape == current.shape
        difference = float(np.max(np.abs(current-target)))
        assert np.isfinite(difference) and difference <= overrides['max_initial_state_alignment_difference'], (stem, difference)
        self.rq2_alignment = {'state_alignment_applied': difference != 0,
            'state_alignment_max_abs_difference': difference,
            'state_before_alignment_sha256': run_swap.array_hash(current),
            'state_snapshot_source': str(cache_path / f'{stem}.npy')}
        if difference != 0:
            # Correct only floating-point tails. Preserve the benchmark's cached
            # observation/controller timing; regenerating observations here would
            # perform an extra sensor update compared with C0.
            self._env.env.sim.set_state_from_flattened(target)
        assert run_swap.array_hash(self._env.env.sim.get_state().flatten()) == cache['state_hashes'][stem]
    return observation, info


def reset_with_alignment_metadata(self, *args, **kwargs):
    result = original_audit_reset(self, *args, **kwargs)
    self.rq2_reset_audit.update(self.rq2_alignment)
    stem = f'task{self.task_id}_episode{self.rq2_reset_audit["episode"]}'
    (run_swap.OUTPUT / 'initial_images' / f'{stem}.json').write_text(json.dumps(self.rq2_reset_audit, indent=2))
    return result


LiberoEnv.__init__ = init_with_offset
run_swap.original_reset = align_initial_state
LiberoEnv.reset = reset_with_alignment_metadata

original_eval_policy = run_swap.lerobot_eval.eval_policy


def eval_policy_remaining(env, *args, **kwargs):
    task_id = int(env.call('task_id')[0])
    offset = offsets.get(task_id, 0)
    if offset:
        kwargs['n_episodes'] = run_swap.CONFIG['episodes_per_task']-offset
        kwargs['start_seed'] = offset
    return original_eval_policy(env, *args, **kwargs)


run_swap.lerobot_eval.eval_policy = eval_policy_remaining
if __name__ == '__main__':
    if os.environ.get('RQ2_RUN_MODE') == 'preflight':
        runpy.run_path('/opt/rq2/preflight_states.py', run_name='__main__')
    else:
        run_swap.lerobot_eval.rollout = run_swap.rollout
        run_swap.lerobot_eval.main()
