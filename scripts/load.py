"""Read the committed per-episode records of all four experiments."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / 'experiments'

PLUS_CATEGORIES = {'light': 'Lighting', 'camera': 'Camera viewpoint', 'layout': 'Object layout'}
CONDITIONS = {'C0_original': 'Original', 'C1_paraphrase': 'Paraphrased',
              'C2_swap': 'Swapped task', 'C3_empty': 'Empty text', 'C4_unrelated': 'Unrelated text'}
DEMONSTRATIONS = [5, 10, 20, 35, 50]
TRAINING_SEEDS = [0, 1, 2]


def jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def baseline():
    """Experiment 1: list of (task_id, episode, success)."""
    rows = jsonl(EXPERIMENTS / '01_baseline/results/episodes.jsonl')
    assert len(rows) == 100 and len({(r['task_id'], r['episode']) for r in rows}) == 100
    return rows


def plus():
    """Experiment 2: success outcomes grouped by perturbation category."""
    rows = jsonl(EXPERIMENTS / '02_libero_plus/results/episodes.jsonl')
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['category']].append(bool(r['success']))
    assert set(grouped) == set(PLUS_CATEGORIES) and all(len(v) == 100 for v in grouped.values())
    return grouped


def instruction_pairs():
    """Experiment 3: {condition: {(task_id, episode): success}}."""
    rows = jsonl(EXPERIMENTS / '03_instruction_swap/results/episodes.jsonl')
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r['condition']][r['task_id'], r['episode']] = bool(r['success'])
    keys = {(t, e) for t in range(10) for e in range(10)}
    assert set(pairs) == set(CONDITIONS) and all(set(v) == keys for v in pairs.values())
    return pairs


def scaling_cube():
    """Experiment 4: 0/1 array of shape (demonstration counts, training seeds, 50 test cases)."""
    rows = jsonl(EXPERIMENTS / '04_data_scaling/results/episodes.jsonl')
    cube = np.full((len(DEMONSTRATIONS), len(TRAINING_SEEDS), 50), np.nan)
    for r in rows:
        i = DEMONSTRATIONS.index(r['n_demonstrations'])
        j = TRAINING_SEEDS.index(r['train_seed'])
        cube[i, j, r['environment_seed'] - 10000] = r['success']
    assert len(rows) == 750 and not np.isnan(cube).any()
    return cube
