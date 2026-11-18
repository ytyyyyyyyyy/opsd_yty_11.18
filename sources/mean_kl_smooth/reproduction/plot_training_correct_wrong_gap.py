"""CPU plots of training-trajectory gap by final-answer correctness.

Input: flattened_gap.npz (gaps, offsets, steps, grades, snapshot_time) and
rows.json (one metadata object per trajectory). Only correct/incorrect labels
enter the curves; absent suffixes are excluded from means, never zero-filled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

BRANCHES = ("correct", "incorrect")
COLORS = {"correct": "#16877A", "incorrect": "#C44E52"}
LABELS = {"correct": "自动判为正确", "incorrect": "自动判为错误"}
INK, GRAY = "#19334D", "#63758A"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def mean_at_positions(values, max_length):
    counts = np.zeros(max_length, dtype=np.int64)
    sums = np.zeros(max_length)
    for item in values:
        observed = np.isfinite(item)
        sums[:len(item)] += np.where(observed, item, 0)
        counts[:len(item)] += observed
    mean = np.full(max_length, np.nan)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean, counts


def build_group(gaps, offsets, steps, grades, indices, scope, first, last):
    counts = Counter(str(grades[i]) for i in indices)
    values = {branch: [gaps[offsets[i]:offsets[i + 1]] for i in indices if grades[i] == branch]
              for branch in BRANCHES}
    length = max((len(item) for group in values.values() for item in group), default=0)
    curves = {branch: mean_at_positions(values[branch], length) for branch in BRANCHES}
    summary = {"scope": scope, "first_step": int(first), "last_step": int(last),
               "steps_observed": sorted(set(int(steps[i]) for i in indices)),
               "partial_five_step_group": scope != "all" and last - first + 1 < 5,
               "trajectories_total": len(indices), "label_counts": dict(counts),
               "included_counts": {branch: counts[branch] for branch in BRANCHES},
               "excluded_counts": {label: count for label, count in counts.items() if label not in BRANCHES},
               "max_response_length": length,
               "comparison_status": "both_groups_available" if all(counts[b] for b in BRANCHES) else "one_or_both_groups_absent",
               "single_trajectory_groups": [branch for branch in BRANCHES if counts[branch] == 1],
               "finite_gap_values": {branch: sum(int(np.isfinite(item).sum()) for item in values[branch]) for branch in BRANCHES},
               "unavailable_gap_values": {branch: sum(int((~np.isfinite(item)).sum()) for item in values[branch]) for branch in BRANCHES}}
    return {"summary": summary, "curves": curves}


def y_limits(curves, start):
    chunks = [mean[start:][np.isfinite(mean[start:])] for mean, _ in curves.values()]
    values = np.concatenate(chunks)
    if not len(values):
        return 0, 1
    low, high = min(0., float(values.min())), max(.01, float(values.max()))
    return low - (high - low) * .015, high + (high - low) * .10


def configure():
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "path.simplify": False, "savefig.facecolor": "white"})


def draw_axis(ax, group, start=0, coverage=False, compact=False, limits=None):
    info, curves = group["summary"], group["curves"]
    length = info["max_response_length"]
    x = np.arange(1, length + 1)
    for branch in BRANCHES:
        mean, n = curves[branch]
        label = f"{LABELS[branch]}（{info['included_counts'][branch]} 条）"
        if coverage:
            ax.step(x, n, where="mid", color=COLORS[branch], lw=1.2, label=label)
        else:
            ax.plot(x[start:], mean[start:], color=COLORS[branch], lw=.9 if compact else 1.15, label=label,
                    marker="." if length - start == 1 else None)
    ax.set_xlim((1, length) if length > 1 else (.5, 1.5))
    if coverage:
        ax.set_ylim(0, max(1, max(info["included_counts"].values())) * 1.15)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax.set_ylim(*(limits or y_limits(curves, start)))
    if not length or (start and length < 2):
        ax.text(.5, .45, "没有可绘制的有效观测", transform=ax.transAxes, ha="center", color=GRAY)
    ax.set_xlabel("回答 token 位置（从 1 开始）", fontsize=9 if compact else 10, color=INK)
    ax.set_ylabel("轨迹数" if coverage else "平均 gap / nat", fontsize=9 if compact else 10, color=INK)
    ax.grid(axis="y", color="#E4EBF1", lw=.65)
    ax.set_axisbelow(True)
    ax.spines[["left", "bottom"]].set_color("#B5C3D0")
    ax.tick_params(colors=GRAY, labelsize=8 if compact else 9)
    ax.legend(frameon=False, fontsize=7.5 if compact else 9, loc="upper right")


def figure(group, snapshot_time):
    info = group["summary"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.1, 1.25, .7]})
    fig.subplots_adjust(left=.085, right=.97, top=.78, bottom=.14, hspace=.67)
    suffix = "全部已完成训练步" if info["scope"] == "all" else f"step {info['first_step']}–{info['last_step']}"
    if info["partial_five_step_group"]:
        suffix += "（末组不足 5 步）"
    fig.text(.085, .95, "训练轨迹：正确与错误回答的平均师生 gap", fontsize=23, weight="bold", color=INK)
    fig.text(.085, .902, f"{suffix} · 正确 {info['included_counts']['correct']} 条 / 错误 {info['included_counts']['incorrect']} 条 · 最长 {info['max_response_length']:,} token", fontsize=11.5, color=GRAY)
    fig.text(.085, .856, f"快照 {snapshot_time} · 按原始 token 位置分别求两组均值；短轨迹结束后不补零。", fontsize=10.5, color=GRAY)
    draw_axis(axes[0], group)
    draw_axis(axes[1], group, start=1)
    draw_axis(axes[2], group, coverage=True)
    for ax, title in zip(axes, ["完整线性纵轴：包括首 token 的所有位置", "第 2 个 token 至末尾的纵轴放大", "各位置实际参与平均的轨迹数"]):
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    excluded = sum(info["excluded_counts"].values())
    flags = []
    if info["comparison_status"] != "both_groups_available":
        flags.append("本组缺少一个或两个类别，无法形成完整正误对照")
    if info["single_trajectory_groups"]:
        flags.append("某类别仅 1 条轨迹，曲线为单次观测")
    note = "；".join(flags) if flags else f"数值答案自动判分；另有 {excluded} 条截断、非数值、未判定等轨迹未纳入；无观测的尾部留空。"
    fig.text(.085, .082, note, fontsize=9.3, color=GRAY)
    fig.text(.085, .053, "两组题目及训练步组成可能不同，仅为描述性比较；原始 token 位置并不代表相同语义步骤。", fontsize=9.5, color=GRAY)
    fig.text(.085, .025, "gap = 未裁剪 KL（教师 ‖ 学生）；自然对数。均值未平滑，未将截断或未判定轨迹视作错误。", fontsize=9.5, color=GRAY)
    return fig


def run(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    source_path = input_dir / "flattened_gap.npz"
    npz = np.load(source_path, allow_pickle=False)
    gaps = np.asarray(npz["gaps"], dtype=float)
    offsets = np.asarray(npz["offsets"], dtype=np.int64)
    steps = np.asarray(npz["steps"], dtype=np.int64)
    grades = np.asarray(npz["grades"], dtype=str)
    snapshot_time = str(npz["snapshot_time"].item())
    metadata = json.loads((input_dir / "rows.json").read_text())
    if isinstance(metadata, dict) and "rows" in metadata:
        metadata = metadata["rows"]
    if gaps.ndim != 1 or offsets.ndim != 1 or steps.ndim != 1 or grades.ndim != 1:
        raise ValueError("All flattened input arrays must be one-dimensional")
    if not len(steps) or len(offsets) != len(steps) + 1 or len(grades) != len(steps) or len(metadata) != len(steps):
        raise ValueError("Trajectory metadata and array sizes do not match")
    if offsets[0] != 0 or offsets[-1] != len(gaps) or np.any(np.diff(offsets) <= 0) or np.isinf(gaps).any():
        raise ValueError("Invalid trajectory offsets or infinite gap values")
    first, last = int(steps.min()), int(steps.max())
    overall = build_group(gaps, offsets, steps, grades, np.arange(len(steps)), "all", first, last)
    groups = []
    for start in range((first - 1) // 5 * 5 + 1, last + 1, 5):
        end = min(start + 4, last)
        indices = np.flatnonzero((steps >= start) & (steps <= end))
        if len(indices):
            groups.append(build_group(gaps, offsets, steps, grades, indices, f"steps_{start:03d}_{end:03d}", start, end))
    output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    fig = figure(overall, snapshot_time)
    fig.savefig(output_dir / "correct_wrong_all_steps.png", dpi=180)
    fig.savefig(output_dir / "correct_wrong_all_steps.pdf")
    plt.close(fig)
    with PdfPages(output_dir / "every_5_steps_correct_wrong_gap.pdf") as pdf:
        for group in groups:
            fig = figure(group, snapshot_time)
            stem = group["summary"]["scope"] + "_correct_wrong_gap"
            fig.savefig(output_dir / (stem + ".png"), dpi=180)
            fig.savefig(output_dir / (stem + ".pdf"))
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(len(groups), 2, figsize=(16, 2.7 * len(groups) + 2), squeeze=False)
    fig.subplots_adjust(left=.075, right=.965, top=.92, bottom=.06, hspace=.78, wspace=.23)
    fig.text(.075, .98, "每 5 个训练步：正确与错误轨迹的平均 gap", fontsize=23, weight="bold", color=INK)
    fig.text(.075, .952, "原始 token 位置对齐；正确、错误分别平均，短轨迹不补零；两组题目和步数组成可能不同。", fontsize=11, color=GRAY)
    shared = [y_limits({str(i) + b: g["curves"][b] for i, g in enumerate(groups) for b in BRANCHES}, start) for start in [0, 1]]
    for row, group in enumerate(groups):
        info = group["summary"]
        for col in range(2):
            draw_axis(axes[row, col], group, start=col, compact=True, limits=shared[col])
            suffix = "全范围" if col == 0 else "位置 2 起放大"
            tail = " · 不足 5 步" if info["partial_five_step_group"] else ""
            axes[row, col].set_title(f"step {info['first_step']}–{info['last_step']}{tail} · {suffix}", loc="left", color=INK, fontsize=11, pad=10)
    fig.text(.075, .026, "数值答案自动判分，各组纵轴一致；截断、非数值和未判定等标签另列覆盖统计，不进入错误组。", fontsize=10, color=GRAY)
    fig.savefig(output_dir / "all_groups_overview.png", dpi=160)
    fig.savefig(output_dir / "all_groups_overview.pdf")
    plt.close(fig)
    with (output_dir / "position_means.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["scope", "first_step", "last_step", "token_position_1based", "correct_n", "correct_mean_gap", "incorrect_n", "incorrect_mean_gap"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in [overall, *groups]:
            info = group["summary"]
            for position in range(info["max_response_length"]):
                row = {field: info[field] for field in ["scope", "first_step", "last_step"]}
                row["token_position_1based"] = position + 1
                for branch in BRANCHES:
                    mean, counts = group["curves"][branch]
                    row[branch + "_n"] = int(counts[position])
                    row[branch + "_mean_gap"] = float(mean[position]) if np.isfinite(mean[position]) else None
                writer.writerow(row)
    summary = {"snapshot_time": snapshot_time, "overall": overall["summary"],
               "groups": [g["summary"] for g in groups],
               "definitions": {"gap": "Unclipped KL(teacher || student), natural log",
                               "x": "Original one-based response token position through maximum included trajectory length",
                               "y": "Arithmetic mean of finite observed trajectory gap values at each position, separately by correct/incorrect grade",
                               "missing_suffix": "Excluded from denominator; no zero padding",
                               "excluded_labels": "All labels except correct and incorrect, including truncated and ungraded",
                               "comparison": "Descriptive training-pool comparison; question and optimization-step composition may differ between groups",
                               "smoothing": "None"},
               "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_path, input_dir / "rows.json"]}}
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.input_dir, args.output_dir)
    print(json.dumps({"overall": summary["overall"], "groups": len(summary["groups"]), "output": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
