"""Long-evaluation numbers and complete-trajectory raw/balanced gap curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reproduction.plot_training_correct_wrong_gap import configure, mean_at_positions

ARMS = ('baseline', 'mean_kl', 'smooth')
NAMES = {'baseline': 'A 原 OPSD', 'mean_kl': 'B 平均 KL', 'smooth': 'C 平均 KL＋平滑项'}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def curve_stats(series):
    if not series:
        return {'trajectories': 0, 'positions': []}
    mean, n = mean_at_positions(series, max(map(len, series)))
    windows = []
    for end in (100, 500, 1000, 2000, 4000, 8000):
        fixed = [x[1:end] for x in series if len(x) >= end and np.isfinite(x[1:end]).all()]
        if len(fixed) < 2:
            windows.append({'end_token': end, 'fixed_contributors': len(fixed), 'mean_curve_roughness': None,
                            'mean_trajectory_roughness': None})
            continue
        arr = np.asarray(fixed)
        curve = arr.mean(0)
        rough = float(np.abs(np.diff(curve)).mean())
        mean_gap = float(curve.mean())
        windows.append({'end_token': end, 'fixed_contributors': len(fixed),
                        'mean_curve_roughness': rough,
                        'relative_mean_curve_roughness': rough / mean_gap if mean_gap > 0 else None,
                        'mean_gap': mean_gap,
                        'mean_trajectory_roughness': float(np.abs(np.diff(arr, axis=1)).mean())})
    return {'trajectories': len(series), 'maximum_length': len(mean), 'positions': windows}


def run(directory):
    out = directory / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    coverage = json.loads((directory / 'coverage.json').read_text())
    rounds = coverage['completed_rounds']
    evaluations = {}
    for label in ('start_step35', *ARMS):
        raw = json.loads((directory / 'evaluation' / label / 'results.json').read_text())
        if not raw['summary']['complete']:
            raise ValueError(f'Incomplete long evaluation: {label}')
        evaluations[label] = raw['summary']
    dump(out / 'accuracy.json', evaluations)
    with (out / 'accuracy.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint', 'correct', 'total', 'accuracy', 'completed_incorrect', 'truncated', 'ungraded'])
        for label, s in evaluations.items():
            w.writerow([label, s['strict_correct'], s['planned_generations'], s['accuracy'],
                        s['completed_incorrect'], s['truncated'], s['ungraded_or_unknown_finish']])
    groups, all_metadata = {}, {}
    for arm in ARMS:
        records, values = [], []
        for r in range(rounds):
            folder = directory / 'diagnostics' / arm / f'round_{r}' / 'scores'
            rows = json.loads((folder / 'rows.json').read_text())
            with np.load(folder / 'flattened_gap.npz', allow_pickle=False) as z:
                gaps, offsets, grades = z['gaps'], z['offsets'], z['grades'].astype(str)
                if len(rows) != len(grades):
                    raise ValueError('Score metadata mismatch')
                for i, row in enumerate(rows):
                    if grades[i] not in ('correct', 'incorrect') or row['grade'] != grades[i]:
                        raise ValueError('Unclassified trajectory in graded comparison')
                    values.append(gaps[offsets[i]:offsets[i + 1]].copy())
                    records.append({**row, 'round_index': r})
        groups[arm] = {'all': values, **{
            g: [v for v, row in zip(values, records) if row['grade'] == g] for g in ('correct', 'incorrect')}}
        all_metadata[arm] = records
        for grade in ('correct', 'incorrect'):
            expected = coverage['per_arm'][arm]['counts'].get(grade, 0)
            if len(groups[arm][grade]) != expected:
                raise ValueError(f'Missing scored trajectories: {arm}/{grade}, expected {expected}')
    balanced_n = min(30, *(len(groups[a][g]) for a in ARMS for g in ('correct', 'incorrect')))
    generator = np.random.default_rng(20261113)
    balanced, selection = {}, {}
    for arm in ARMS:
        balanced[arm], selection[arm] = {}, {}
        for grade in ('correct', 'incorrect'):
            values = groups[arm][grade]
            indices = sorted(generator.choice(len(values), balanced_n, replace=False).tolist()) if balanced_n else []
            balanced[arm][grade] = [values[i] for i in indices]
            records = [r for r in all_metadata[arm] if r['grade'] == grade]
            selection[arm][grade] = [records[i] for i in indices]
        balanced[arm]['all'] = balanced[arm]['correct'] + balanced[arm]['incorrect']
    dump(out / 'balanced_selection.json', {'n_per_class': balanced_n, 'seed': 20261113,
                                           'selection': selection, 'gap_used_for_selection': False})
    summary = {'evaluation': evaluations, 'coverage': coverage, 'balanced_n_per_class': balanced_n,
               'raw': {a: {g: curve_stats(v) for g, v in groups[a].items()} for a in ARMS},
               'balanced': {a: {g: curve_stats(v) for g, v in balanced[a].items()} for a in ARMS},
               'notes': ['Endpoint frozen-model diagnostics; not historical training curves.',
                         'Diagnostic teacher sees full reference solution; student thinking disabled as in training.',
                         'AIME evaluation enables thinking and uses a 38912-token cap for all four checkpoints.',
                         'Correctness is automated final-answer grading, not verification of every reasoning step.',
                         'No zero padding or curve smoothing. Trajectory counts accompany every position.',
                         'Balanced selection equalizes initial counts, not tail counts or question/length composition.',
                         'Adaptive diagnostic outcomes are not a benchmark accuracy estimate.']}
    dump(out / 'summary.json', summary)
    configure()
    csv_rows = []
    for name, pool in [('raw', groups), ('balanced', balanced)]:
        for arm in ARMS:
            for grade, vals in pool[arm].items():
                means, counts = mean_at_positions(vals, max(map(len, vals), default=0))
                csv_rows.extend({'view': name, 'arm': arm, 'grade': grade, 'token_position': i + 1,
                                 'n': int(counts[i]), 'mean_gap': float(means[i]) if counts[i] else None}
                                for i in range(len(means)))
        fig, axes = plt.subplots(3, 3, figsize=(19, 12), squeeze=False)
        for i, arm in enumerate(ARMS):
            for grade, color, title in [('correct', '#16877A', '正确'), ('incorrect', '#C44E52', '错误')]:
                vals = pool[arm][grade]
                length = max(map(len, vals), default=0)
                mean, counts = mean_at_positions(vals, length)
                x = np.arange(1, length + 1)
                label = f'{title} {len(vals)} 条'
                axes[i, 0].plot(x, mean, color=color, lw=.7, label=label)
                axes[i, 1].plot(x[1:], mean[1:], color=color, lw=.7, label=label)
                axes[i, 2].plot(x, counts, color=color, lw=1., label=label)
            for j, ax in enumerate(axes[i]):
                ax.grid(alpha=.2); ax.legend(fontsize=9)
                ax.set_title(NAMES[arm] + ' · ' + ['全部位置', '位置 2 起放大', '参与平均的轨迹数'][j], loc='left')
                ax.set_xlabel('回答 token 位置')
                ax.set_ylabel('轨迹数' if j == 2 else '平均 gap / nat')
        for j in (0, 1):
            limit = max(ax.get_ylim()[1] for ax in axes[:, j])
            for ax in axes[:, j]:
                ax.set_ylim(0, limit)
        suffix = '全部完整判分轨迹' if name == 'raw' else f'各组等量抽取：每类 {balanced_n} 条'
        fig.suptitle('补采后的正确 / 错误平均师生 gap · ' + suffix, fontsize=18)
        fig.tight_layout(rect=(0, .035, 1, .96))
        fig.text(.06, .015, '原始 token 位置，短轨迹不补零，均值未做平滑；完整参考解教师，学生关闭 thinking。', fontsize=10)
        for ext in ('png', 'pdf'):
            fig.savefig(figures / f'correct_wrong_{name}.{ext}', dpi=160)
        plt.close(fig)
    # Unconditional curve over all completed, graded diagnostic trajectories.
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    for arm, color in zip(ARMS, ('#63758A', '#377EB8', '#D66A2D')):
        vals = groups[arm]['all']
        mean, counts = mean_at_positions(vals, max(map(len, vals), default=0))
        x = np.arange(1, len(mean) + 1)
        axes[0].plot(x, mean, color=color, lw=.8, label=NAMES[arm])
        axes[1].plot(x[1:], mean[1:], color=color, lw=.8, label=NAMES[arm])
        axes[2].plot(x, counts, color=color, lw=1., label=NAMES[arm])
    for ax, title in zip(axes, ('全部位置', '位置 2 起放大', '参与平均的完整轨迹数')):
        ax.set_title(title, loc='left'); ax.legend(); ax.grid(alpha=.2); ax.set_xlabel('回答 token 位置')
    fig.suptitle('A/B/C 终点：全部完整判分轨迹的平均 gap', fontsize=18)
    fig.tight_layout(rect=(0, .03, 1, .96))
    for ext in ('png', 'pdf'):
        fig.savefig(figures / f'all_arms_complete_gap.{ext}', dpi=160)
    plt.close(fig)
    with (out / 'position_means.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['view', 'arm', 'grade', 'token_position', 'n', 'mean_gap'])
        w.writeheader(); w.writerows(csv_rows)
    print('FOLLOWUP_ANALYSIS_COMPLETE', out / 'summary.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
