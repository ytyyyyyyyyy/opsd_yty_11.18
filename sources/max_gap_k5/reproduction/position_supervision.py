"""Full-vocabulary forward-KL routing; preserve the first-position weight.

All arms compute the same detached scores. The random arm has an independent,
content-keyed RNG, so selection never consumes the rollout RNG stream.
"""
import hashlib
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def position_weights(entropy, gap, labels, mode, fraction, seed):
    if mode not in ('all', 'random', 'state') or not 0 < fraction <= 1:
        raise ValueError('Invalid position selection configuration')
    valid = labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError('No valid completion positions')
    weights = torch.zeros_like(entropy, dtype=torch.float32)
    selected = torch.zeros_like(valid)
    protected = torch.zeros_like(valid)
    for batch in range(labels.shape[0]):
        indices = valid[batch].nonzero().flatten()
        if not len(indices):
            continue
        first, content = indices[0], indices[1:]
        protected[batch, first] = True
        selected[batch, first] = True
        weights[batch, first] = 1
        n = len(content)
        if not n:
            continue
        keep = n if mode == 'all' else max(1, math.ceil(n * fraction))
        if mode == 'random':
            raw = labels[batch, indices].detach().cpu().contiguous().numpy().tobytes()
            key = (int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), 'little') + seed) % (2**63 - 1)
            generator = torch.Generator(device='cpu').manual_seed(key)
            chosen = torch.randperm(n, generator=generator)[:keep].to(content.device)
        elif mode == 'state':
            def normalize(values):
                values = values.clamp(max=torch.quantile(values, .98))
                return (values - values.min()) / (values.max() - values.min()).clamp_min(1e-12)
            h = normalize(entropy[batch, content])
            d = normalize(gap[batch, content])
            score = h + d - h * d
            chosen = torch.argsort(score, descending=True, stable=True)[:keep]
        else:
            chosen = torch.arange(n, device=content.device)
        keep_indices = content[chosen]
        weights[batch, keep_indices] = n / keep
        selected[batch, keep_indices] = True
    # Each example keeps its total weight and the initial position's weight.
    assert torch.allclose(weights.sum(-1), valid.sum(-1).float(), atol=2e-4, rtol=1e-6)
    content = valid & ~protected
    chosen_content = selected & content
    stats = {
        'valid_positions': int(valid.sum()), 'protected_positions': int(protected.sum()),
        'content_positions': int(content.sum()), 'selected_content_positions': int(chosen_content.sum()),
        'weight_sum': float(weights.sum()),
        'selected_entropy_mean': float(entropy[chosen_content].mean()) if bool(chosen_content.any()) else None,
        'selected_gap_mean': float(gap[chosen_content].mean()) if bool(chosen_content.any()) else None,
    }
    return weights, stats


def routed_forward_kl(student, teacher, labels, *, mode, fraction=.5, seed=20261025,
                      temperature=1.1, token_clip=.05, chunk_tokens=32):
    """FP32 exact vocabulary terms; clipping precedes vocabulary summation.

First positions carry weight 1. Selected content carries weight n/k; all
positions' weight sum equals the original token count. This preserves the
boundary's contribution and makes uniform random routing unbiased for the
all-position loss conditional on fixed logits.
"""
    if student.shape != teacher.shape or student.shape[:2] != labels.shape:
        raise ValueError('Logit/label shape mismatch')
    if temperature <= 0 or chunk_tokens < 1:
        raise ValueError('Invalid loss configuration')
    shape = labels.shape
    flat_s, flat_t = student.reshape(-1, student.shape[-1]), teacher.reshape(-1, teacher.shape[-1])
    valid_indices = labels.reshape(-1).ne(-100).nonzero().flatten()
    entropy = torch.zeros(shape, dtype=torch.float32, device=student.device)
    gap = torch.zeros_like(entropy)
    with torch.no_grad():
        for chunk in valid_indices.split(chunk_tokens):
            s = F.log_softmax(flat_s[chunk].float() / temperature, dim=-1)
            t = F.log_softmax(flat_t[chunk].float() / temperature, dim=-1)
            entropy.reshape(-1)[chunk] = -(s.exp() * s).sum(-1)
            gap.reshape(-1)[chunk] = (t.exp() * (t-s)).sum(-1).clamp_min(0)
    weights, stats = position_weights(entropy, gap, labels, mode, fraction, seed)
    loss = student.new_zeros((), dtype=torch.float32)
    for chunk in valid_indices.split(chunk_tokens):
        s = F.log_softmax(flat_s[chunk].float() / temperature, dim=-1)
        t = F.log_softmax(flat_t[chunk].detach().float() / temperature, dim=-1)
        terms = t.exp() * (t-s)
        if token_clip is not None:
            terms = terms.clamp(max=token_clip)
        loss = loss + (terms.sum(-1) * weights.reshape(-1)[chunk]).sum()
    return loss / len(valid_indices), stats
