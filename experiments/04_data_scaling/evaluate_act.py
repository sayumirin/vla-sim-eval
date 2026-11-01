"""Official batched ALOHA evaluation, with per-episode outcome logging."""
import json
import os
from pathlib import Path
import numpy as np
import gym_aloha
from gym_aloha.env import AlohaEnv
from lerobot.scripts import lerobot_eval

AlohaEnv.task_description = property(lambda self: self.task.replace('_', ' '))

output = Path(os.environ['RQ1_EVAL_OUTPUT'])
metadata = json.loads(os.environ['RQ1_EVAL_METADATA'])
original_rollout = lerobot_eval.rollout
original_all = lerobot_eval.eval_policy_all
original_reset = AlohaEnv.reset


def reset_with_pose(self,*args,**kwargs):
    value=original_reset(self,*args,**kwargs)
    self.rq1_initial_qpos=np.asarray(self._env.physics.data.qpos).tolist()
    return value


AlohaEnv.reset=reset_with_pose


def recorded_rollout(*args, **kwargs):
    result = original_rollout(*args, **kwargs)
    seeds = kwargs['seeds']
    env=kwargs.get('env',args[0] if args else None)
    poses=list(env.call('rq1_initial_qpos'))
    for i, seed in enumerate(seeds):
        steps = int(result['done'][i].to(int).argmax())+1
        record = {**metadata,'environment_seed':seed,
                  'episode':seed-metadata['eval_seed_start'],
                  'success':bool(result['success'][i,:steps].any()),'steps':steps,
                  'initial_qpos':poses[i]}
        with (output/'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
    print(f'RQ1_EVAL completed_through_seed={seeds[-1]}',flush=True)
    return result


def limited_videos(*args, **kwargs):
    kwargs['max_episodes_rendered'] = 2
    return original_all(*args, **kwargs)


lerobot_eval.rollout = recorded_rollout
lerobot_eval.eval_policy_all = limited_videos
if __name__ == '__main__':
    lerobot_eval.main()
