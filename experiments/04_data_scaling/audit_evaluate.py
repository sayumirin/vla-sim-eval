"""Passive reset-metadata fix; policy, simulator, seeds and official rollout unchanged.

SAME_STEP autoreset calls reset(seed=None) after a terminal transition. Preserve
the explicitly seeded episode's pose instead of overwriting it during autoreset.
"""
import numpy as np
import evaluate_act as original


def reset_preserving_seeded_pose(self, *args, **kwargs):
    value = original.original_reset(self, *args, **kwargs)
    seed = kwargs.get('seed', args[0] if args else None)
    if seed is not None:
        self.rq1_initial_qpos = np.asarray(self._env.physics.data.qpos).tolist()
    return value


original.AlohaEnv.reset = reset_preserving_seeded_pose
if __name__ == '__main__':
    original.lerobot_eval.main()
