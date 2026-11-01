"""Generate an auditable, seeded sample using names to map one-based IDs."""
import collections
import hashlib
import json
from pathlib import Path
import random
import runpy
import subprocess

ROOT = Path(__file__).parent
REPO = ROOT / 'LIBERO-plus'
BENCH = REPO / 'libero/libero/benchmark'
COMMIT = '4976dc30028e805ff8094b55501d532c48fec182'
CATEGORIES = {'light': 'Light Conditions', 'camera': 'Camera Viewpoints',
              'layout': 'Objects Layout'}


def select():
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    source = BENCH / 'task_classification.json'
    classified = json.loads(source.read_text())['libero_goal']
    registry = runpy.run_path(str(BENCH / 'libero_suite_task_map.py'))['libero_task_map']['libero_goal']
    index = {name: i for i, name in enumerate(registry)}
    assert len(index) == len(registry) == len(classified) == 2591
    assert all(index[r['name']] == r['id'] - 1 for r in classified)
    base_names = sorted({r['name'].split('_light_')[0] for r in classified if r['category'] == CATEGORIES['light']})
    assert len(base_names) == 10
    config = {'suite': 'libero_goal', 'seed': 0, 'episodes_per_variant': 1,
              'sampling': 'Uniform sample without replacement, 100 per dimension, independent Random(0) per category',
              'libero_plus_commit': COMMIT, 'classification_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'prompt_protocol': 'Original base-task filename converted to spaces, matching stage-one LeRobot prompts; no perturbation suffixes.',
              'categories': {}}
    for key, category in CATEGORIES.items():
        pool = [r for r in classified if r['category'] == category]
        sampled = sorted(random.Random(0).sample(pool, 100), key=lambda r: index[r['name']])
        records = []
        for row in sampled:
            bases = [b for b in base_names if row['name'].startswith(b + '_')]
            assert len(bases) == 1, row
            records.append({**row, 'task_id': index[row['name']], 'base_task': bases[0],
                            'prompt': bases[0].replace('_', ' ')})
        config['categories'][key] = {'category': category, 'available': len(pool),
            'task_ids': [r['task_id'] for r in records], 'tasks': records,
            'base_task_counts': dict(collections.Counter(r['base_task'] for r in records)),
            'difficulty_counts': dict(collections.Counter(r['difficulty_level'] for r in records))}
    out = ROOT / 'configs'
    out.mkdir(exist_ok=True)
    # JSON is valid YAML 1.2; use identical bytes for both portable formats.
    content = json.dumps(config, indent=2) + '\n'
    (out / 'plus_tasks.json').write_text(content)
    (out / 'plus_tasks.yaml').write_text(content)
    print(json.dumps({k: {f: v[f] for f in ['available', 'base_task_counts', 'difficulty_counts']} for k,v in config['categories'].items()}, indent=2))


if __name__ == '__main__':
    select()
