"""Statistics shared by the checks and figures. Same formulas as the per-experiment validators."""
import math

import numpy as np

Z95 = 1.959963984540054


def wilson(k, n):
    """Wilson 95% interval for k successes out of n, in percent."""
    p = k / n
    den = 1 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / den
    low = 0. if k == 0 else max(0., center - half)
    high = 1. if k == n else min(1., center + half)
    return 100 * low, 100 * high


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if not n:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.] * len(p_values)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p_values[index])
        adjusted[index] = min(1., running)
    return adjusted


def two_way_bootstrap(cube, replicates=10000, seed=20261030):
    """Resample training seeds and shared test cases together.

    cube has shape (conditions, training_seeds, test_cases) and holds 0/1 outcomes.
    Returns the 2.5% and 97.5% quantiles of the mean success rate per condition.
    """
    rng = np.random.default_rng(seed)
    _, n_seeds, n_cases = cube.shape
    boot = np.empty((replicates, cube.shape[0]))
    for b in range(replicates):
        seeds = rng.integers(0, n_seeds, n_seeds)
        cases = rng.integers(0, n_cases, n_cases)
        boot[b] = cube[:, seeds, :][:, :, cases].mean(axis=(1, 2))
    return np.quantile(boot, [.025, .975], axis=0)
