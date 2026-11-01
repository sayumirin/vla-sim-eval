"""Read public Hub metadata and freeze the dataset and nested episode subsets."""
import json
from pathlib import Path
import random
import urllib.request

root = Path(__file__).parent
repo = 'lerobot/aloha_sim_transfer_cube_scripted'
with urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo, timeout=60) as response:
    metadata = json.load(response)
revision = metadata['sha']
with urllib.request.urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/meta/info.json', timeout=60) as response:
    info = json.load(response)
assert info['total_episodes'] == 50 and info['total_tasks'] == 1
order = list(range(50))
random.Random(0).shuffle(order)
plan = {'dataset_repo': repo, 'dataset_revision': revision, 'dataset_info': info,
        'lerobot_commit': '72e1d9e5e76ee719bacfc78bc34d817bee5fb521',
        'subset_seed': 0, 'episode_permutation': order,
        'subsets': {str(n): order[:n] for n in [5,10,20,35,50]},
        'full_design': {'sizes': [5,10,20,35,50], 'training_seeds': [0,1,2]},
        'reduced_design': {'sizes': [10,20,50], 'training_seeds': [0,1]},
        'formal_eval_seeds': list(range(10000,10050)),
        'pilot_training_seed': 777, 'pilot_eval_seeds': list(range(20000,20020)),
        'budget_usd': 15, 'planned_reservation_budget_usd': 12,
        'protocol': 'Freeze one training-step budget after the 50-demo pilot; use the same batch size and final checkpoint for every formal run. Subset-only action/state statistics; fixed ImageNet visual normalization. Training and pilot evaluation are separate from formal evaluation.'}
(root/'configs').mkdir(parents=True, exist_ok=True)
content = json.dumps(plan, indent=2)
(root/'configs/rq1_plan.json').write_text(content)
(root/'configs/rq1_subsets.yaml').write_text(content)
print(json.dumps({'repo':repo,'revision':revision,'episodes':info['total_episodes'],
                  'frames':info['total_frames'],'features':info['features'], 'subsets':plan['subsets']},indent=2))
