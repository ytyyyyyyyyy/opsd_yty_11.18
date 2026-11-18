"""CPU-only numerical and gradient checks for the independent auxiliary loss."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from reproduction import position_supervision
from reproduction.gap_smoothness import _auxiliary_terms, routed_smooth_forward_kl


def run_checks():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(20261113)
    source_s = torch.randn(3, 9, 8, generator=generator) * 1.7
    source_t = torch.randn(3, 9, 8, generator=generator) * 1.3
    labels = torch.tensor([
        [-100, 1, 2, 3, -100, 4, 2, 6, 7],
        [1, 2, 3, 4, 2, 3, 6, -100, -100],
        [-100, 1, 6, 7, -100, -100, -100, -100, -100],
    ])
    expected_mask = torch.tensor([
        [False, False, True, True, False, True, True, False, False],
        [False, True, True, True, True, True, False, False, False],
        [False, False, False, False, False, False, False, False, False],
    ])
    options = dict(mode="all", fraction=1., seed=91, temperature=1.1,
                   token_clip=.05, chunk_tokens=2, auxiliary_weight=.05,
                   smooth_alpha=.25, smooth_beta=.1, excluded_token_ids=(6, 7))
    results = []

    def check(name, **details):
        results.append({"name": name, "passed": True, **details})

    def original_kwargs(opts):
        return {key: opts[key] for key in (
            "mode", "fraction", "seed", "temperature", "token_clip", "chunk_tokens"
        )}

    def evaluate(arm, **overrides):
        opts = {**options, **overrides}
        student = source_s.clone().requires_grad_()
        teacher = source_t.clone().requires_grad_()
        loss, stats = routed_smooth_forward_kl(student, teacher, labels, arm=arm, **opts)
        loss.backward()
        assert teacher.grad is None
        return loss.detach(), student.grad.clone(), stats

    for chunk_size in (1, 2, 7, 100):
        opts = {**options, "chunk_tokens": chunk_size}
        student = source_s.clone().requires_grad_()
        teacher = source_t.clone().requires_grad_()
        reference, reference_stats = position_supervision.routed_forward_kl(
            student, teacher, labels, **original_kwargs(opts)
        )
        reference.backward()
        actual, gradient, stats = evaluate("baseline", chunk_tokens=chunk_size)
        assert torch.equal(actual, reference.detach())
        assert torch.equal(gradient, student.grad)
        assert teacher.grad is None
        assert all(stats[key] == value for key, value in reference_stats.items())
    check("baseline_loss_gradient_and_original_stats_exact", chunk_sizes=[1, 2, 7, 100])

    with patch.object(position_supervision, "routed_forward_kl", side_effect=AssertionError("recursive call")):
        evaluate("baseline")
    check("baseline_keeps_original_alias_after_external_monkeypatch")

    def reference(arm, token_clip=.05):
        student = source_s.clone().requires_grad_()
        teacher = source_t.clone().requires_grad_()
        s = F.log_softmax(student.float() / 1.1, dim=-1)
        t = F.log_softmax(teacher.detach().float() / 1.1, dim=-1)
        terms = t.exp() * (t - s)
        g = terms.sum(-1)
        main_terms = terms if token_clip is None else terms.clamp(max=token_clip)
        main = main_terms.sum(-1)[labels.ne(-100)].mean()
        # Independent explicit per-row expression verifies denominator and
        # original-position adjacency, including the wholly empty third row.
        means, smooths = [], []
        absolute = []
        for row in range(labels.shape[0]):
            positions = expected_mask[row].nonzero().flatten().tolist()
            n = len(positions)
            if not n:
                means.append(g[row].sum() * 0)
                smooths.append(g[row].sum() * 0)
                continue
            means.append(sum(g[row, pos] for pos in positions) / n)
            penalties = []
            for pos in range(labels.shape[1] - 1):
                if pos in positions and pos + 1 in positions:
                    delta = g[row, pos + 1] - g[row, pos]
                    absolute.append(delta.abs())
                    # SmoothL1's explicit formula, not the implementation helper.
                    magnitude = delta.abs()
                    penalties.append(torch.where(magnitude < .1,
                                                 delta.square() / .2, magnitude - .05))
            smooths.append(sum(penalties, g[row].sum() * 0) / n)
        mean = torch.stack(means).mean()
        smooth = torch.stack(smooths).mean()
        objective = main + .05 * (mean + (.25 * smooth if arm == "smooth" else 0))
        objective.backward()
        assert teacher.grad is None
        return objective.detach(), student.grad, float(main.detach()), float(mean.detach()), float(smooth.detach()), float(torch.stack(absolute).mean().detach())

    for arm in ("mean_kl", "smooth"):
        for clip in (.05, None):
            actual, gradient, stats = evaluate(arm, token_clip=clip)
            expected, expected_gradient, base, mean, smooth, roughness = reference(arm, clip)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-7)
            for key, value in (("base", base), ("aux_mean", mean), ("aux_smooth", smooth), ("roughness", roughness)):
                assert abs(stats[key] - value) < 3e-7, (key, stats[key], value)
            assert stats["token_counts_per_trajectory"] == [4, 5, 0]
            assert stats["pair_counts_per_trajectory"] == [2, 4, 0]
            assert stats["token_count"] == 9 and stats["pair_count"] == 6
            assert stats["empty_auxiliary_trajectories"] == 1
    check("mean_and_smooth_formula_gradient_ragged_normalization_and_teacher_detach",
          token_counts=[4, 5, 0], pair_counts=[2, 4, 0], tested_clip=[.05, None])

    base, base_gradient, _ = evaluate("baseline")
    for arm in ("mean_kl", "smooth"):
        loss, gradient, stats = evaluate(arm)
        torch.testing.assert_close((gradient - base_gradient)[~expected_mask],
                                   torch.zeros_like(gradient[~expected_mask]), atol=2e-7, rtol=0)
    check("first_token_eos_pad_and_mask_holes_have_no_auxiliary_gradient")

    baseline_by_chunk = {}
    for arm in ("mean_kl", "smooth"):
        expected, expected_gradient, _ = evaluate(arm, chunk_tokens=100)
        max_difference = 0.
        for chunk_size in (1, 2, 3, 4, 7):
            actual, gradient, stats = evaluate(arm, chunk_tokens=chunk_size)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-7)
            assert stats["pair_count"] == 6
            max_difference = max(max_difference, float((gradient - expected_gradient).abs().max()))
        baseline_by_chunk[arm] = max_difference
    check("adjacent_pairs_and_gradients_survive_chunk_boundaries",
          max_abs_gradient_difference=baseline_by_chunk)

    loss_b, grad_b, stats_b = evaluate("mean_kl")
    loss_c, grad_c, stats_c = evaluate("smooth")
    assert abs(float(loss_c - loss_b) - .05 * .25 * stats_c["aux_smooth"]) < 2e-7
    loss_c0, grad_c0, _ = evaluate("smooth", smooth_alpha=0)
    assert torch.equal(loss_c0, loss_b)
    assert torch.equal(grad_c0, grad_b)
    check("smooth_minus_mean_is_exactly_the_normalized_adjacent_term_and_alpha_zero_matches")

    for arm in ("mean_kl", "smooth"):
        actual, gradient, _ = evaluate(arm, auxiliary_weight=0)
        torch.testing.assert_close(actual, base, atol=2e-7, rtol=0)
        torch.testing.assert_close(gradient, base_gradient, atol=2e-7, rtol=0)
        actual, gradient, stats = evaluate(arm, excluded_token_ids=tuple(range(8)))
        torch.testing.assert_close(actual, base, atol=2e-7, rtol=0)
        torch.testing.assert_close(gradient, base_gradient, atol=2e-7, rtol=0)
        assert stats["aux_mean"] == stats["aux_smooth"] == stats["roughness"] == 0
        assert stats["token_count"] == stats["pair_count"] == 0
    check("zero_auxiliary_weight_and_empty_auxiliary_recover_base")

    # Analytic coordinate monotonicity, including alpha=.5 and masked holes.
    raw = torch.tensor([[50., 0., 2., 0., 3., 0., 50.],
                        [50., 2., 0., 4., 0., 2., 50.]], dtype=torch.float64)
    mask = torch.tensor([[False, True, True, True, True, True, False],
                         [False, True, True, False, True, True, False]])
    minimums = {}
    for alpha in (0., .25, .5):
        gaps = raw.clone().requires_grad_()
        mean, smooth, _, counts, _ = _auxiliary_terms(gaps, mask, .1)
        (mean + alpha * smooth).backward()
        lower = (1 - 2 * alpha) / (counts[:, None].to(raw.dtype) * raw.shape[0])
        assert bool((gaps.grad[mask] >= lower.expand_as(gaps)[mask] - 1e-12).all())
        assert bool((gaps.grad[~mask] == 0).all())
        minimums[str(alpha)] = float(gaps.grad[mask].min())
    check("auxiliary_gap_coordinate_derivatives_nonnegative", minimum_derivatives=minimums)

    for bad in ({"mode": "random"}, {"smooth_alpha": .50001}, {"smooth_alpha": -.1},
                {"smooth_beta": 0}, {"auxiliary_weight": -.1}, {"chunk_tokens": 0},
                {"temperature": 0}, {"arm": "unknown"}, {"excluded_token_ids": (1.5,)}):
        opts = {**options, "arm": "smooth", **bad}
        try:
            routed_smooth_forward_kl(source_s, source_t, labels, **opts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected rejection: {bad}")
    check("invalid_configuration_rejected")
    return {
        "passed": True, "device": "cpu", "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": results,
        "example_losses": {"baseline": float(base), "mean_kl": float(loss_b), "smooth": float(loss_c)},
        "example_smooth_stats": stats_c,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON check report")
    args = parser.parse_args()
    report = run_checks()
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
