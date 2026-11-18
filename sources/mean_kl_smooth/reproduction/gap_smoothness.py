"""All-position OPSD with an optional mean-KL / local-smoothness auxiliary.

The auxiliary uses unclipped FP32 forward KL at each response position.  Its
mean and adjacent-difference terms share the *same* per-response denominator.
With 0 <= smooth_alpha <= .5, its derivative with respect to each eligible
gap is nonnegative.  This is a coordinate-wise property of the auxiliary;
shared model parameters do not guarantee that each gap or roughness decreases.

The original clipped objective always retains its original positions.  Only
the auxiliary excludes the first valid response token and explicitly supplied
EOS/pad IDs.  Callers must either mask EOS/pad labels with -100 or supply their
IDs: token identities cannot be inferred from logits alone.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable

import torch
import torch.nn.functional as F

# Capture this before a training entry point replaces the routing function.
# A dynamic import inside the wrapper would recurse after that replacement.
from reproduction.position_supervision import (
    position_weights,
    routed_forward_kl as _original_routed_forward_kl,
)


def _auxiliary_mask(labels: torch.Tensor, excluded_ids: tuple[int, ...]):
    original = labels.ne(-100)
    mask = original.clone()
    first_count = 0
    for row in range(labels.shape[0]):
        valid = original[row].nonzero().flatten()
        if valid.numel():
            mask[row, valid[0]] = False
            first_count += 1
    excluded = torch.zeros_like(original)
    for token_id in excluded_ids:
        excluded |= original & labels.eq(token_id)
    mask &= ~excluded
    return mask, first_count, int(excluded.sum())


def _auxiliary_terms(gaps: torch.Tensor, mask: torch.Tensor, beta: float):
    """Return per-response-normalized terms; empty responses contribute zero.

    Adjacency uses original positions, not a compacted list of valid tokens.
    Building this from the assembled differentiable gap tensor also retains
    adjacent differences across numerical chunk boundaries.
    """
    counts = mask.sum(-1)
    denominators = counts.clamp_min(1).to(gaps.dtype)
    mean = (gaps.masked_fill(~mask, 0).sum(-1) / denominators).mean()
    pairs = mask[:, 1:] & mask[:, :-1]
    differences = gaps[:, 1:] - gaps[:, :-1]
    penalties = F.smooth_l1_loss(
        differences, torch.zeros_like(differences), reduction="none", beta=beta
    )
    smooth = (penalties.masked_fill(~pairs, 0).sum(-1) / denominators).mean()
    absolute = differences.abs().masked_fill(~pairs, 0)
    # This diagnostic uses the usual mean absolute difference per valid pair.
    # The training smooth term above deliberately divides by token count.
    roughness = absolute.sum() / pairs.sum().clamp_min(1)
    return mean, smooth, roughness, counts, pairs.sum(-1)


def _validate(student, teacher, labels, mode, fraction, temperature, token_clip,
              chunk_tokens, auxiliary_weight, smooth_alpha, smooth_beta, arm,
              excluded_token_ids):
    if (student.ndim != 3 or student.shape != teacher.shape
            or student.shape[:2] != labels.shape or labels.ndim != 2
            or student.shape[0] < 1 or student.shape[-1] < 1):
        raise ValueError("Expected matching [batch, positions, vocabulary] logits and [batch, positions] labels")
    if student.device != teacher.device or student.device != labels.device:
        raise ValueError("Logits and labels must be on the same device")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("Logits must have floating-point dtype")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Labels must have integer dtype")
    if mode != "all":
        raise ValueError("Gap smoothness requires mode='all'")
    if arm not in ("baseline", "mean_kl", "smooth"):
        raise ValueError("arm must be baseline, mean_kl, or smooth")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if token_clip is not None and (not math.isfinite(token_clip) or token_clip <= 0):
        raise ValueError("token_clip must be None or finite and positive")
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, Integral) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("auxiliary_weight must be finite and nonnegative")
    if not math.isfinite(smooth_alpha) or not 0 <= smooth_alpha <= .5:
        raise ValueError("smooth_alpha must be in [0, .5]")
    if not math.isfinite(smooth_beta) or smooth_beta <= 0:
        raise ValueError("smooth_beta must be finite and positive")
    if not bool(labels.ne(-100).any()):
        # Match the original objective: a wholly empty main loss is invalid.
        # An empty auxiliary within an otherwise valid response is allowed.
        raise ValueError("No valid completion positions")
    excluded = tuple(excluded_token_ids) if excluded_token_ids is not None else ()
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in excluded):
        raise ValueError("excluded_token_ids must contain integer token IDs")
    return tuple(dict.fromkeys(int(value) for value in excluded))


def routed_smooth_forward_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: str,
    fraction: float = .5,
    seed: int = 20261025,
    temperature: float = 1.1,
    token_clip: float | None = .05,
    chunk_tokens: int = 32,
    auxiliary_weight: float = .05,
    smooth_alpha: float = .25,
    smooth_beta: float = .1,
    arm: str = "baseline",
    excluded_token_ids: Iterable[int] | None = None,
):
    """Return ``(loss, stats)`` with the original routing API plus auxiliaries.

    ``baseline`` delegates its loss to the captured original implementation.
    ``mean_kl`` adds auxiliary_weight * mean_i(sum_t g_it / N_i).
    ``smooth`` also adds auxiliary_weight * smooth_alpha *
    mean_i(sum_adjacent SmoothL1(delta_gap, beta=smooth_beta) / N_i).
    Empty auxiliary responses contribute zero to the batch mean.

    EOS/pad exclusions apply only to auxiliaries.  Pair terms never cross a
    mask hole or response boundary; numeric chunk size changes neither their
    membership nor their gradients.  ``aux_mean`` / ``aux_smooth`` in stats
    are unweighted diagnostics for every arm, including baseline; ``aux_loss``
    is the auxiliary actually added.  ``roughness`` is mean absolute gap
    difference over eligible pairs.  Original routing statistics are retained.
    """
    excluded = _validate(
        student, teacher, labels, mode, fraction, temperature, token_clip,
        chunk_tokens, auxiliary_weight, smooth_alpha, smooth_beta, arm,
        excluded_token_ids,
    )
    mask, excluded_first, excluded_ids_count = _auxiliary_mask(labels, excluded)
    valid_indices = labels.reshape(-1).ne(-100).nonzero().flatten()
    flat_s = student.reshape(-1, student.shape[-1])
    flat_t = teacher.reshape(-1, teacher.shape[-1])

    if arm == "baseline":
        base, stats = _original_routed_forward_kl(
            student, teacher, labels, mode=mode, fraction=fraction, seed=seed,
            temperature=temperature, token_clip=token_clip,
            chunk_tokens=chunk_tokens,
        )
        # Additional detached diagnostics do not change the baseline graph.
        # Only log-prob arithmetic is repeated; no model forward is performed.
        with torch.no_grad():
            chunks = []
            for chunk in valid_indices.split(chunk_tokens):
                s = F.log_softmax(flat_s[chunk].float() / temperature, dim=-1)
                t = F.log_softmax(flat_t[chunk].detach().float() / temperature, dim=-1)
                chunks.append((t.exp() * (t - s)).sum(-1))
            gaps = student.new_zeros(labels.numel(), dtype=torch.float32).index_copy(
                0, valid_indices, torch.cat(chunks)
            ).reshape(labels.shape)
    else:
        base_sum = student.new_zeros((), dtype=torch.float32)
        chunks = []
        entropy = student.new_zeros(labels.numel(), dtype=torch.float32)
        for chunk in valid_indices.split(chunk_tokens):
            s = F.log_softmax(flat_s[chunk].float() / temperature, dim=-1)
            t = F.log_softmax(flat_t[chunk].detach().float() / temperature, dim=-1)
            terms = t.exp() * (t - s)
            # Both branches use this same differentiable vocabulary tensor.
            chunks.append(terms.sum(-1))
            clipped = terms if token_clip is None else terms.clamp(max=token_clip)
            base_sum = base_sum + clipped.sum(-1).sum()
            with torch.no_grad():
                entropy[chunk] = -(s.detach().exp() * s.detach()).sum(-1)
        gaps = student.new_zeros(labels.numel(), dtype=torch.float32).index_copy(
            0, valid_indices, torch.cat(chunks)
        ).reshape(labels.shape)
        base = base_sum / len(valid_indices)
        # All-position routing has exactly unit weights.  Reuse the original
        # statistics routine without retaining any graph for its diagnostics.
        _, stats = position_weights(
            entropy.reshape(labels.shape), gaps.detach().clamp_min(0), labels,
            mode, fraction, seed,
        )

    mean, smooth, roughness, counts, pairs = _auxiliary_terms(gaps, mask, smooth_beta)
    if arm == "baseline":
        loss = base
        auxiliary = base.detach().new_zeros(())
    else:
        auxiliary = auxiliary_weight * (
            mean + smooth_alpha * smooth if arm == "smooth" else mean
        )
        loss = base + auxiliary
    stats = dict(stats)
    stats.update(
        arm=arm, base=float(base.detach()),
        aux_mean=float(mean.detach()), aux_smooth=float(smooth.detach()),
        aux_loss=float(auxiliary.detach()), roughness=float(roughness.detach()),
        token_count=int(counts.sum()), pair_count=int(pairs.sum()),
        token_counts_per_trajectory=counts.detach().cpu().tolist(),
        pair_counts_per_trajectory=pairs.detach().cpu().tolist(),
        empty_auxiliary_trajectories=int(counts.eq(0).sum()),
        excluded_first_positions=excluded_first,
        excluded_id_positions=excluded_ids_count,
        excluded_token_ids=list(excluded),
        auxiliary_weight=float(auxiliary_weight),
        smooth_alpha=float(smooth_alpha), smooth_beta=float(smooth_beta),
    )
    return loss, stats
