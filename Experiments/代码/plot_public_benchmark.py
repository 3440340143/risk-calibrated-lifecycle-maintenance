#!/usr/bin/env python3
"""Plot Figure 6 directly from the public benchmark CSV records."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.results / 'cmapss_primary_summary.csv')
    frontier = pd.read_csv(args.results / 'cmapss_exact_frontiers.csv')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 9.5, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(3, 1, figsize=(6.8, 8.6), constrained_layout=True,
                                gridspec_kw={'height_ratios': [1, 1, 1.1]})
    chosen = ['Cycle_CP', 'Ridge_ERM_EqualBudget', 'Ridge_Constrained', 'Linex_Constrained']
    names = ['Cycle CP (Ridge)', 'Equal-budget ERM (Ridge)', 'Constrained sweep (Ridge)', 'Constrained sweep (Linex)']
    colors = ['#6C757D', '#BC6C25', '#1B365D', '#2A9D8F']
    x = np.arange(4)
    for index, (method, name, color) in enumerate(zip(chosen, names, colors)):
        values = table[table.method == method].set_index('dataset').loc[['FD001', 'FD002', 'FD003', 'FD004']]
        for ax, column in [(axes[0], 'j_life'), (axes[1], 'fail_pct')]:
            ax.bar(x + (index - 1.5) * .18, values[column], width=.17, color=color, label=name)
    axes[0].set_ylabel('Cost per 100 useful cycles')
    axes[0].set_title('(a) Primary holdout: lifecycle cost', loc='left', fontweight='bold')
    axes[0].legend(ncols=2, loc='upper center', bbox_to_anchor=(.5, 1.02), frameon=False)
    axes[0].set_ylim(0, table[table.method.isin(chosen)].j_life.max() * 1.35)
    axes[1].axhline(5.0, ls='--', lw=1, color='#B23A48')
    axes[1].set_title('(b) Primary holdout: observed breakdowns', loc='left', fontweight='bold')
    axes[1].set_ylabel('Breakdowns (%)')
    axes[1].text(3.45, 5.3, '5% target', ha='right', fontsize=8, color='#B23A48')
    axes[1].set_ylim(0, max(15, table[table.method.isin(chosen)].fail_pct.max() * 1.3))
    for ax in axes[:2]:
        ax.set_xticks(x, ['FD001\n40 engines', 'FD002\n104 engines', 'FD003\n40 engines', 'FD004\n102 engines'])
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    selected = table[(table.dataset == 'FD004') & table.method.str.startswith('Ridge')].set_index('method')
    curve = frontier[(frontier.dataset == 'FD004') & (frontier['head'] == 'Ridge') & (frontier.constraint == 'unconstrained')]
    max_display = max(selected.loc['Ridge_Constrained', 'margin'], selected.loc['Ridge_Floor', 'margin'],
                      selected.loc['Ridge_ERM_Tune', 'margin']) * 1.35
    curve = curve[curve.margin <= max_display]
    axes[2].step(curve.margin, curve.j_life, where='post', color='#1B365D', lw=1.2)
    for method, label, color, marker in [('Ridge_ERM_Tune', 'Unconstrained optimum', '#6C757D', 'o'),
             ('Ridge_Floor', 'Max(tuned margin, q)', '#BC6C25', 's'),
             ('Ridge_Constrained', 'Constrained optimum', '#2A9D8F', 'D')]:
        margin = selected.loc[method, 'margin']
        preceding = curve[curve.margin <= margin + 1e-12].iloc[-1]
        axes[2].scatter(margin, preceding.j_life, color=color, marker=marker, s=38, zorder=4, label=label)
    q = selected.loc['Ridge_UnitCP', 'margin']
    axes[2].axvline(q, color='#BC6C25', ls='--', lw=1)
    axes[2].set_xlim(0, max_display)
    axes[2].set_title('(c) FD004 Ridge tuning frontier: local view', loc='left', fontweight='bold')
    axes[2].set_xlabel('Safety margin')
    axes[2].set_ylabel('Tuning cost per 100 cycles')
    axes[2].grid(alpha=.18)
    axes[2].legend(loc='upper right', frameon=False)
    target = args.output / 'fig6_public_benchmark_and_exact_frontier'
    figure.savefig(target.with_suffix('.pdf'))
    figure.savefig(target.with_suffix('.png'), dpi=600)
    plt.close(figure)
    print(json.dumps({'figure': str(target.with_suffix('.pdf')), 'source': 'Public benchmark CSVs; no manually supplied result arrays.'}))


if __name__ == '__main__':
    main()
