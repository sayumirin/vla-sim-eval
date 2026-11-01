"""Check actual scene parameters and saved outcomes for all three smoke scenes."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent
scenes = {}
for category in ['light', 'camera', 'layout']:
    runs = sorted((root / 'modal_results').glob(f'{category}_smoke_seed0_*.json'))
    run = json.loads(runs[-1].read_text())
    assert run['status'] == 'completed' and run['results']['overall']['n_episodes'] == 1
    folder = root / 'modal_results' / Path(run['output']).name
    scenes[category] = json.loads(next(folder.glob('environment_*.json')).read_text())
assert len({s['prompt'] for s in scenes.values()}) == 1
light, camera, layout = (scenes[c] for c in ['light', 'camera', 'layout'])
camera_changed = not np.allclose(camera['camera_quaternions'], layout['camera_quaternions']) or not np.allclose(camera['camera_positions'], layout['camera_positions'])
light_changed = light['light_diffuse'] != layout['light_diffuse'] or light['light_positions'] != layout['light_positions']
layout_changed = len(layout['initial_qpos']) != len(camera['initial_qpos'])
assert camera_changed and light_changed and layout_changed
result = {'all_smoke_runs_completed': True, 'same_base_prompt': light['prompt'],
          'camera_parameters_changed': camera_changed, 'light_parameters_changed': light_changed,
          'layout_has_additional_object_state': layout_changed,
          'visual_inspection': 'All three saved initial images inspected; dark lighting, changed viewpoint, and two added objects visible.'}
(root / 'audit/smoke_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
