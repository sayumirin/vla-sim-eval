"""Apply a declared validation-only tolerance; leave the executed rollout unchanged."""
import json
import os
from pathlib import Path
import runpy
import run_swap

overrides = json.loads(Path('/opt/rq2/validation_overrides.json').read_text())
assert overrides['scope'] == 'validation_only'
assert set(overrides['image_pair_tolerance']) == {'max_abs_channel_difference', 'max_fraction_different_channels'}
run_swap.CONFIG['image_pair_tolerance'] = overrides['image_pair_tolerance']
if os.environ.get('RQ2_RUN_MODE') == 'preflight':
    runpy.run_path('/opt/rq2/preflight_states.py', run_name='__main__')
else:
    run_swap.lerobot_eval.rollout = run_swap.rollout
    run_swap.lerobot_eval.main()
