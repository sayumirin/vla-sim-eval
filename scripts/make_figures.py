"""Redraw every figure in figures/ from the committed per-episode records.

Needs numpy and matplotlib. Run from the repository root:  python scripts/make_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import load
from stats import two_way_bootstrap, wilson

OUT = load.ROOT / 'figures'

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
REFERENCE = '#b4b2a9'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlesize': 11.5,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK_2, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'grid.color': GRID, 'grid.linewidth': .8, 'legend.frameon': False,
})


def percent_axis(ax, top=108):
    ax.set_ylim(0, top)
    ax.set_yticks([0, 25, 50, 75, 100], labels=['0', '25', '50', '75', '100%'])
    ax.grid(axis='y', zorder=0)
    ax.spines['left'].set_visible(False)


def bars(ax, labels, counts, n, colors, label_size=9.5):
    """Bars with Wilson 95% error bars and the k/n count above each interval."""
    x = np.arange(len(labels))
    rates = [100 * k / n for k in counts]
    intervals = [wilson(k, n) for k in counts]
    ax.bar(x, rates, width=.44, color=colors, zorder=3)
    ax.errorbar(x, rates, yerr=[[r - lo for r, (lo, _) in zip(rates, intervals)],
                                [hi - r for r, (_, hi) in zip(rates, intervals)]],
                fmt='none', ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
    for i, (k, (_, hi)) in enumerate(zip(counts, intervals)):
        ax.text(i, hi + 2.5, f'{k}/{n}', ha='center', va='bottom', fontsize=label_size, color=INK)
    ax.set_xticks(x, labels=labels)
    return x


def instruction_panel(ax, compact=False):
    pairs = load.instruction_pairs()
    counts = [sum(pairs[c].values()) for c in load.CONDITIONS]
    labels = ['Original', 'Para-\nphrased', 'Swapped\ntask', 'Empty', 'Unrelated'] if compact \
        else list(load.CONDITIONS.values())
    bars(ax, labels, counts, 100, [MUTED, BLUE, BLUE, BLUE, BLUE])
    percent_axis(ax, 118)
    ax.set_ylabel('Original-task success')


def plus_panel(ax):
    grouped = load.plus()
    baseline = sum(r['success'] for r in load.baseline())
    labels = ['None', *[load.PLUS_CATEGORIES[c].replace(' ', '\n') for c in load.PLUS_CATEGORIES]]
    counts = [baseline, *[sum(grouped[c]) for c in load.PLUS_CATEGORIES]]
    bars(ax, labels, counts, 100, [MUTED, BLUE, BLUE, BLUE])
    percent_axis(ax, 118)
    ax.set_ylabel('Success rate')


def scaling_panel(ax, cube, annotate=True, top=108):
    means = cube.mean(axis=(1, 2)) * 100
    low, high = two_way_bootstrap(cube) * 100
    ax.fill_between(load.DEMONSTRATIONS, low, high, color=BLUE, alpha=.12, linewidth=0, zorder=2)
    ax.plot(load.DEMONSTRATIONS, means, color=BLUE, linewidth=2, solid_capstyle='round', zorder=3)
    ax.scatter(load.DEMONSTRATIONS, means, s=42, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
    if annotate:
        for n, m in zip(load.DEMONSTRATIONS, means):
            ax.annotate(f'{m:.0f}%', (n, m), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=9.5, color=INK)
    ax.set_xscale('log')
    ax.set_xticks(load.DEMONSTRATIONS, labels=[str(n) for n in load.DEMONSTRATIONS])
    ax.minorticks_off()
    ax.set_xlabel('Training demonstrations')
    percent_axis(ax, top)
    ax.set_ylabel('Success rate')


def overview():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={'width_ratios': [1.25, 1, 1]},
                             layout='constrained')
    instruction_panel(axes[0], compact=True)
    axes[0].set_title('Change the instruction: the original task fails')
    plus_panel(axes[1])
    axes[1].set_title('Camera and layout shifts: ~25 points lower')
    scaling_panel(axes[2], load.scaling_cube(), top=118)
    axes[2].set_title('ACT: 23% with 5 demos, ~88% with 35–50')
    fig.text(0.005, -0.02, 'Left, middle: π0.5 on LIBERO-Goal, 100 episodes per bar, 95% Wilson intervals; the left '
             'panel is paired (same initial states) and scored on the original task.\nRight: ACT on ALOHA '
             'transfer-cube, mean of 3 training seeds × 50 shared test cases; band = 95% two-way bootstrap.',
             fontsize=8.5, color=INK_2, ha='left', va='top')
    fig.savefig(OUT / 'overview.png', dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def instruction_figure():
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    instruction_panel(ax)
    ax.set_title('π0.5 on LIBERO-Goal: same scene, different instruction')
    fig.text(0.01, -0.03, '100 paired episodes per condition (10 tasks × 10 initial states), identical initial '
             'state and policy RNG across conditions.\nError bars: 95% Wilson intervals. Success is scored on the '
             'original task; completing the swapped-in task is not measured.\nSwapped, empty and unrelated each '
             'differ from Original at exact McNemar p < 1e-25 (Holm-adjusted over four comparisons).',
             fontsize=8.5, color=INK_2,
             ha='left', va='top')
    fig.savefig(OUT / 'instruction_swap.png', dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def plus_figure():
    fig, ax = plt.subplots(figsize=(7, 4.6), layout='constrained')
    plus_panel(ax)
    ax.set_title('π0.5 on LIBERO-Goal: visual perturbations (LIBERO-plus)')
    fig.text(0.01, -0.03, '"None" is the unperturbed baseline (10 tasks × 10 initial states). Each perturbation: '
             '100 sampled variants,\none episode each. Error bars: 95% Wilson intervals. Not a paired comparison.',
             fontsize=8.5, color=INK_2, ha='left', va='top')
    fig.savefig(OUT / 'visual_robustness.png', dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def scaling_figure():
    cube = load.scaling_cube()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
    scaling_panel(axes[0], cube)
    axes[0].set_title('Mean of 3 training seeds')
    ax = axes[1]
    for j, (seed, color) in enumerate(zip(load.TRAINING_SEEDS, [BLUE, ORANGE, AQUA])):
        x = np.array(load.DEMONSTRATIONS) * np.exp((j - 1) * .05)
        counts = cube[:, j].sum(axis=1).astype(int)
        rates = counts * 2
        intervals = np.array([wilson(k, 50) for k in counts])
        ax.errorbar(x, rates, yerr=[rates - intervals[:, 0], intervals[:, 1] - rates], fmt='none',
                    ecolor=color, elinewidth=1.2, capsize=2.5, zorder=3)
        ax.scatter(x, rates, s=42, color=color, edgecolor=SURFACE, linewidth=2, zorder=4,
                   label=f'Training seed {seed}')
    ax.set_xscale('log')
    ax.set_xticks(load.DEMONSTRATIONS, labels=[str(n) for n in load.DEMONSTRATIONS])
    ax.minorticks_off()
    ax.set_xlabel('Training demonstrations')
    percent_axis(ax)
    ax.set_title('Each policy, 50 test cases (95% Wilson)')
    ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('ACT on AlohaTransferCube-v0, 6,000 updates per model, nested demonstration subsets',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold')
    fig.text(0.01, -0.03, 'Band: 95% two-way bootstrap over training seeds and shared test cases (10,000 replicates). '
             'Same 50 test cases for every model.', fontsize=8.5, color=INK_2, ha='left', va='top')
    fig.savefig(OUT / 'data_scaling.png', dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    overview()
    instruction_figure()
    plus_figure()
    scaling_figure()
    print('Wrote', ', '.join(sorted(p.name for p in OUT.glob('*.png'))))
