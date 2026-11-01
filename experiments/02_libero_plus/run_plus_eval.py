"""Keep visual-perturbation prompts identical to stage-one base task prompts."""
import json
import os
from pathlib import Path

from lerobot.envs.libero import LiberoEnv

config = json.loads(Path('/opt/knightec/plus_tasks.json').read_text())
mapping = {r['name']: r['prompt'] for c in config['categories'].values() for r in c['tasks']}
original_init = LiberoEnv.__init__
original_reset = LiberoEnv.reset
original_step = LiberoEnv.step


def reset_with_audit(self, *args, **kwargs):
    observation, info = original_reset(self, *args, **kwargs)
    self._audit_steps = 0
    if os.environ.get('KNIGHTEC_SMOKE') == '1':
        from PIL import Image
        import numpy as np
        output = Path(os.environ['KNIGHTEC_OUTPUT'])
        Image.fromarray(observation['pixels']['image']).save(output / f'initial_{self.task_id}.png')
        sim = self._env.env.sim
        audit = {'task': self.task, 'prompt': self.task_description,
                 'camera_positions': np.asarray(sim.model.cam_pos).tolist(),
                 'camera_quaternions': np.asarray(sim.model.cam_quat).tolist(),
                 'light_positions': np.asarray(sim.model.light_pos).tolist(),
                 'light_diffuse': np.asarray(sim.model.light_diffuse).tolist(),
                 'initial_qpos': np.asarray(sim.data.qpos).tolist()}
        (output / f'environment_{self.task_id}.json').write_text(json.dumps(audit, indent=2))
    return observation, info


def step_with_count(self, action):
    result = original_step(self, action)
    self._audit_steps += 1
    return result


def init_with_base_prompt(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self.is_libero_plus:
        self.task_description = mapping[self.task]


LiberoEnv.__init__ = init_with_base_prompt
LiberoEnv.reset = reset_with_audit
LiberoEnv.step = step_with_count

if __name__ == '__main__':
    from lerobot.scripts import lerobot_eval
    original_run_one = lerobot_eval.run_one

    def run_one_with_record(task_group, task_id, env, **kwargs):
        result = original_run_one(task_group, task_id, env, **kwargs)
        tg, tid, metrics = result
        record = {'suite': tg, 'task_id': tid, 'seed': 0,
                  'steps': list(env.call('_audit_steps')), 'metrics': metrics}
        with (Path(os.environ['KNIGHTEC_OUTPUT']) / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(f'KNIGHTEC_PROGRESS task={tid} success={metrics["successes"]}', flush=True)
        return result

    lerobot_eval.run_one = run_one_with_record
    lerobot_eval.main()
