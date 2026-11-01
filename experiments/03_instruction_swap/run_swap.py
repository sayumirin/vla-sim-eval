"""Explicit batch-one paired rollout; use pinned LeRobot setup, metrics and video writing.

The loop follows LeRobot's Apache-2.0 rollout at commit
72e1d9e5e76ee719bacfc78bc34d817bee5fb521, with task injection before preprocessing.
No simulation, target predicate, observation preprocessing or action logic changes.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.envs import preprocess_observation
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.processor import bind_relative_anchor
from lerobot.scripts import lerobot_eval
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS

CONFIG = json.loads(Path(os.environ['RQ2_CONFIG']).read_text())
OUTPUT = Path(os.environ['RQ2_OUTPUT'])
CONDITION = os.environ['RQ2_CONDITION']
REFERENCE = Path(os.environ['RQ2_REFERENCE']) if os.environ.get('RQ2_REFERENCE') else None
original_reset = LiberoEnv.reset


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def reset_with_audit(self, *args, **kwargs):
    """Observe the exact supplied and settled state without changing reset behavior."""
    episode = self.init_state_id
    supplied = self._init_states[episode % len(self._init_states)]
    observation, info = original_reset(self, *args, **kwargs)
    self.rq2_reset_audit = {
        'task_id': self.task_id, 'task_name': self.task,
        'episode': episode, 'initial_state_index': episode % len(self._init_states),
        'supplied_state_sha256': array_hash(supplied),
        'settled_state_sha256': array_hash(self._env.env.sim.get_state().flatten()),
        'initial_image_hashes': {k: array_hash(v) for k, v in observation['pixels'].items()},
    }
    model = self._env.env.sim.model
    self.rq2_reset_audit['visual_model_hashes'] = {
        key: array_hash(getattr(model, key)) for key in [
            'body_pos', 'body_quat', 'cam_pos', 'cam_quat', 'cam_fovy',
            'light_pos', 'light_dir', 'light_diffuse', 'light_ambient',
            'geom_rgba', 'mat_rgba', 'tex_rgb', 'tex_data'] if hasattr(model, key)}
    (OUTPUT / 'initial_images').mkdir(exist_ok=True)
    np.savez_compressed(OUTPUT / 'initial_images' / f'task{self.task_id}_episode{episode}.npz',
                        **observation['pixels'])
    (OUTPUT / 'initial_images' / f'task{self.task_id}_episode{episode}.json').write_text(json.dumps(self.rq2_reset_audit, indent=2))
    return observation, info


LiberoEnv.reset = reset_with_audit


def paired_rng(audit):
    """C0 records its untouched RNG stream; other conditions restore each paired start."""
    name = f'task{audit["task_id"]}_episode{audit["episode"]}.pt'
    if CONDITION != 'C0_original':
        state = torch.load(REFERENCE / 'rng' / name, map_location='cpu', weights_only=True)
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state_all(state['cuda'])
    state = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
    if CONDITION == 'C0_original':
        (OUTPUT / 'rng').mkdir(exist_ok=True)
        torch.save(state, OUTPUT / 'rng' / name)
    return {'cpu_rng_sha256': array_hash(state['cpu'].numpy()),
            'cuda_rng_sha256': [array_hash(s.numpy()) for s in state['cuda']]}


def successes_from_info(info, batch_size):
    if 'final_info' in info:
        final = info['final_info']
        if not isinstance(final, dict):
            return [bool(v.get('is_success', False)) if isinstance(v, dict) else False for v in final]
        value = final.get('is_success', [False] * batch_size)
    else:
        value = info.get('is_success', [False] * batch_size)
    return np.asarray(value, dtype=bool).reshape(batch_size).tolist()


def rollout(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor,
            seeds=None, return_observations=False, render_callback=None, recording_dir=None,
            env_features=None, recording_repo_id=None, recording_private=False,
            predicted_latents_callback=None):
    """Run one fixed-state episode and change only the task text sent to the policy."""
    assert env.num_envs == 1 and not return_observations and recording_dir is None
    assert predicted_latents_callback is None
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    observation, _ = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
    if render_callback is not None:
        render_callback(env)
    audit = dict(env.call('rq2_reset_audit')[0])
    task = CONFIG['tasks'][audit['task_id']]
    assert task['name'] == audit['task_name']
    original_instruction = list(env.call('task_description'))[0]
    assert original_instruction == task['instructions']['C0_original']
    instruction = task['instructions'][CONDITION]
    audit.update(paired_rng(audit))
    reference = None
    image_pair_metrics = None
    if REFERENCE is not None:
        reference = json.loads((REFERENCE / 'episode_audits' / f'task{audit["task_id"]}_episode{audit["episode"]}.json').read_text())
        for field in ['initial_state_index', 'supplied_state_sha256', 'settled_state_sha256',
                      'visual_model_hashes', 'cpu_rng_sha256', 'cuda_rng_sha256']:
            assert audit[field] == reference[field], (audit['task_id'], audit['episode'], field)
        name = f'task{audit["task_id"]}_episode{audit["episode"]}.npz'
        image_pair_metrics = {}
        with np.load(REFERENCE / 'initial_images' / name) as original, np.load(OUTPUT / 'initial_images' / name) as current:
            for key in original.files:
                diff = np.abs(original[key].astype(np.float32)-current[key].astype(np.float32))
                metrics = {'mean_abs_difference': float(diff.mean()), 'max_abs_difference': float(diff.max()),
                           'fraction_different_channels': float((diff>0).mean())}
                tolerance = CONFIG['image_pair_tolerance']
                assert metrics['max_abs_difference'] <= tolerance['max_abs_channel_difference'], (name, key, metrics)
                assert metrics['fraction_different_channels'] <= tolerance['max_fraction_different_channels'], (name, key, metrics)
                image_pair_metrics[key] = metrics
    bind_relative_anchor(policy, preprocessor)
    done = np.array([False])
    max_steps = env.call('_max_episode_steps')[0]
    actions, rewards, successes, dones = [], [], [], []
    first_prompt, first_tokens = None, None
    for step in range(max_steps):
        observation = preprocess_observation(observation)
        # Experimental intervention: only this natural-language field changes.
        observation['task'] = [instruction]
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)
        if step == 0:
            first_prompt = observation['task'][0]
            first_tokens = array_hash(observation[OBS_LANGUAGE_TOKENS].detach().cpu().numpy())
            prefix, separator, state_text = first_prompt.partition(', State:')
            assert separator and prefix == 'Task: ' + instruction.strip().replace('_', ' ').replace('\n', ' ')
            if reference is not None:
                assert first_tokens != reference['first_tokens_sha256']
                assert state_text == reference['first_formatted_prompt'].partition(', State:')[2]
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_numpy = action.to('cpu').numpy()
        assert action_numpy.ndim == 2
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)
        success = successes_from_info(info, env.num_envs)
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        actions.append(torch.from_numpy(action_numpy))
        rewards.append(torch.from_numpy(reward))
        successes.append(torch.tensor(success))
        dones.append(torch.from_numpy(done))
        if np.all(done):
            break
    record = {**audit, 'condition': CONDITION, 'run_seed': CONFIG['seed'],
              'environment_seed': seeds[0], 'original_instruction': original_instruction,
              'instruction': instruction, 'swap_target_task_id': task['swap_target_task_id'] if CONDITION == 'C2_swap' else None,
              'first_formatted_prompt': first_prompt, 'first_tokens_sha256': first_tokens,
              'success': bool(torch.stack(successes).any()), 'steps': len(actions),
              'paired_reference_checked': reference is not None,
              'image_pair_metrics': image_pair_metrics,
              'model_revision': CONFIG['model_revision'], 'lerobot_commit': CONFIG['lerobot_commit']}
    if CONDITION == 'C0_original':
        (OUTPUT / 'episode_audits').mkdir(exist_ok=True)
        (OUTPUT / 'episode_audits' / f'task{audit["task_id"]}_episode{audit["episode"]}.json').write_text(json.dumps(record, indent=2))
    with (OUTPUT / 'episodes.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'RQ2_PROGRESS condition={CONDITION} task={audit["task_id"]} episode={audit["episode"]} success={record["success"]} steps={len(actions)}', flush=True)
    if hasattr(policy, 'use_original_modules'):
        policy.use_original_modules()
    return {ACTION: torch.stack(actions, dim=1), 'reward': torch.stack(rewards, dim=1),
            'success': torch.stack(successes, dim=1), 'done': torch.stack(dones, dim=1)}


if __name__ == '__main__':
    lerobot_eval.rollout = rollout
    lerobot_eval.main()
