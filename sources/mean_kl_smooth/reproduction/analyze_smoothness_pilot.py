"""Compare the bounded pilot without smoothing or padding observed curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reproduction.plot_training_correct_wrong_gap import configure, mean_at_positions

ARMS = ('baseline', 'mean_kl', 'smooth')
NAMES = {'baseline': 'A 原 OPSD', 'mean_kl': 'B 额外平均 KL', 'smooth': 'C 平均 KL＋平滑项'}
COLORS = {'baseline': '#63758A', 'mean_kl': '#377EB8', 'smooth': '#D66A2D'}


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def roughness(values):
    values = np.asarray(values, dtype=float)
    pair = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    if not pair.any():
        return {'adjacent_abs_difference': None, 'relative_adjacent_difference': None}
    r = float(np.abs(np.diff(values))[pair].mean())
    mean = float(np.nanmean(values))
    return {'adjacent_abs_difference': r,
            'relative_adjacent_difference': r / mean if mean > 0 else None}


def run(directory):
    out = directory / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((directory / 'config.json').read_text())
    pools, metadata, report, flat_rows = {}, {}, {}, []
    for arm in ARMS:
        source = directory / 'figures' / arm / 'input'
        with np.load(source / 'flattened_gap.npz', allow_pickle=False) as z:
            gaps, offsets, grades = z['gaps'], z['offsets'], z['grades'].astype(str)
            series = [gaps[offsets[i]:offsets[i + 1]].copy() for i in range(len(grades))]
        metadata[arm] = json.loads((source / 'rows.json').read_text())
        pools[arm] = {'all': series,
                      **{label: [v for v, g in zip(series, grades) if g == label]
                         for label in ('correct', 'incorrect')}}
        records = []
        for category, group in pools[arm].items():
            maxlen = max(map(len, group), default=0)
            mean, counts = mean_at_positions(group, maxlen)
            for t in range(maxlen):
                flat_rows.append({'arm': arm, 'group': category, 'token_position': t + 1,
                                  'n': int(counts[t]), 'mean_gap': float(mean[t]) if counts[t] else None})
            for end in (100, 500, 800):
                # Match a fixed interval; do not silently use shorter trajectories.
                complete = [v[1:end] for v in group if len(v) >= end and np.isfinite(v[1:end]).all()]
                available = min(end, len(mean))
                positions = mean[1:available].copy()
                positions[counts[1:available] < 5] = np.nan
                individual = [roughness(v) for v in complete]
                relative = [v['relative_adjacent_difference'] for v in individual
                            if v['relative_adjacent_difference'] is not None]
                records.append({
                    'group': category, 'positions': [2, end], 'group_trajectories': len(group),
                    'full_window_trajectories': len(complete),
                    'observed_mean_curve_min_n5': roughness(positions),
                    'fixed_contributors_mean_curve': roughness(np.mean(complete, axis=0)) if complete else None,
                    'individual_mean_roughness': float(np.mean([v['adjacent_abs_difference'] for v in individual])) if individual else None,
                    'individual_mean_relative_roughness': float(np.mean(relative)) if relative else None,
                    'individual_relative_roughness_count': len(relative),
                    'mean_gap_fixed_window': float(np.mean(complete)) if complete else None,
                    'p95_gap_fixed_window': float(np.quantile(complete, .95)) if complete else None,
                })
        train = read_jsonl(directory / 'training' / arm / 'metrics/training_metrics.jsonl')
        auxiliary = read_jsonl(directory / 'training' / arm / 'smoothness_metrics.jsonl')
        if len(train) != cfg['pilot']['updates_per_arm'] or len(series) != cfg['pilot']['expected_trajectories_per_arm']:
            raise ValueError(f'Incomplete training collection for {arm}')
        report[arm] = {
            'curves': records, 'grade_counts': {x: int(np.sum(grades == x)) for x in sorted(set(grades))},
            'training_steps': [r['step'] for r in train],
            'loss_finite': all(np.isfinite(r['loss']) for r in train),
            'gradient_clip_threshold': cfg['training']['max_grad_norm'],
            'updates_above_gradient_clip_threshold': sum(r['grad_norm'] > cfg['training']['max_grad_norm'] for r in train),
            'generated_tokens': sum(r['rollout/tokens'] for r in train),
            'truncated_fraction': float(np.mean([r['truncated'] for r in metadata[arm]])),
            'auxiliary_records': len(auxiliary),
            'start_witness': json.loads((directory / 'training' / arm / 'branch_start.json').read_text()),
        }
    problems = [[r['problem'] for r in metadata[arm]] for arm in ARMS]
    if not all(x == problems[0] for x in problems):
        raise ValueError('Training problem order differs between arms')
    first_rollouts = []
    for arm in ARMS:
        rows = read_jsonl(directory / 'training' / arm / 'metrics/generation_trace.rank0.jsonl')
        first = [r for r in rows if r['optimizer_step'] == cfg['pilot']['start_step'] + 1]
        expected = cfg['training']['micro_batch'] * cfg['training']['gradient_accumulation']
        if len(first) != expected:
            raise ValueError(f'Incomplete first-update traces for {arm}: {len(first)} != {expected}')
        first_rollouts.append([r['completion_token_ids'] for r in first])
    evaluation = {}
    for label in ('start_step35', *ARMS):
        raw = json.loads((directory / 'evaluation' / label / 'results.json').read_text())
        evaluation[label] = raw.get('summary', raw.get('metrics', {}))
    summary = {
        'pilot': cfg['pilot'], 'arms': report, 'evaluation': evaluation,
        'problem_order_equal': True,
        'first_update_rollouts_equal': all(x == first_rollouts[0] for x in first_rollouts),
        'notes': ['No plotting smoothing; no zero-padding.',
                  'Correctness groups contain only completed, conservatively graded numeric answers.',
                  'All group contains all observed trajectories including truncated/ungraded.',
                  'Fixed-window individual metrics and contributor counts accompany mean curves.',
                  'Single-seed pilot with 30 questions x 2 generations; no significance claim.',
                  'Gap curves are training-forward observations, not endpoint holdout gap measurements.'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with (out / 'position_means.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        w.writeheader()
        w.writerows(flat_rows)
    configure()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1.4, .7]})
    for arm in ARMS:
        group = pools[arm]['all']
        mean, count = mean_at_positions(group, max(map(len, group)))
        x = np.arange(1, len(mean) + 1)
        axes[0].plot(x, mean, color=COLORS[arm], label=NAMES[arm], lw=.9)
        axes[1].plot(x[1:], mean[1:], color=COLORS[arm], label=NAMES[arm], lw=.9)
        axes[2].plot(x, count, color=COLORS[arm], label=NAMES[arm], lw=.9)
    for ax, title in zip(axes, ['全部位置', '位置 2 起放大', '每位置参与平均的轨迹数']):
        ax.set_title(title, loc='left'); ax.legend(); ax.grid(alpha=.2)
        ax.set_xlabel('回答 token 位置'); ax.set_ylabel('轨迹数' if ax is axes[2] else '平均 gap / nat')
    fig.suptitle('小实验 step 36–40：三组全部轨迹的平均师生 gap', fontsize=18)
    fig.tight_layout(rect=(0, .025, 1, .96))
    fig.text(.08, .01, '均值未经平滑；短轨迹结束后不补零。每组 160 条，训练后 5 个更新步。', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / 'figures' / f'all_arms_mean_gap.{suffix}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    for i, arm in enumerate(ARMS):
        for category, color, name in [('correct', '#16877A', '正确'), ('incorrect', '#C44E52', '错误')]:
            group = pools[arm][category]
            if not group:
                continue
            mean, count = mean_at_positions(group, max(map(len, group)))
            x = np.arange(1, len(mean) + 1)
            axes[i, 0].plot(x[1:], mean[1:], color=color, label=f'{name} {len(group)} 条', lw=.85)
            axes[i, 1].plot(x, count, color=color, label=name, lw=.85)
        for ax in axes[i]:
            ax.grid(alpha=.2); ax.legend(); ax.set_xlabel('回答 token 位置')
        axes[i, 0].set_title(NAMES[arm] + '：位置 2 起平均 gap', loc='left')
        axes[i, 1].set_title('实际参与平均的轨迹数', loc='left')
    limit = max(ax.get_ylim()[1] for ax in axes[:, 0])
    for ax in axes[:, 0]:
        ax.set_ylim(0, limit); ax.set_ylabel('平均 gap / nat')
    fig.suptitle('小实验：各组正确 / 错误轨迹对比（统一纵轴）', fontsize=18)
    fig.tight_layout(rect=(0, .025, 1, .96))
    fig.text(.08, .01, '截断、非数值和未判定轨迹不进入正误组；尾部可能只有少量轨迹。', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / 'figures' / f'correct_wrong_across_arms.{suffix}', dpi=160)
    plt.close(fig)
    print(json.dumps({'report': str(out / 'summary.json'), 'evaluation': evaluation}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    run(p.parse_args().run_dir.resolve())
