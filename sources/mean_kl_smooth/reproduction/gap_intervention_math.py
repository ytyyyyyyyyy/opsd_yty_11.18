"""Memory-bounded, unclipped full-vocabulary gap diagnostics (no gradients)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def full_vocabulary_gap(student_logits, teacher_logits, temperature=1.1, chunk_tokens=16):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Expected equally shaped [response_tokens, vocabulary] logits")
    if temperature <= 0 or chunk_tokens < 1:
        raise ValueError("Invalid gap configuration")
    gaps = []
    entropies = []
    for start in range(0, len(student_logits), chunk_tokens):
        s = F.log_softmax(student_logits[start:start + chunk_tokens].float() / temperature, dim=-1)
        t = F.log_softmax(teacher_logits[start:start + chunk_tokens].float() / temperature, dim=-1)
        gap = (t.exp() * (t - s)).sum(-1).clamp_min(0)
        if not bool(torch.isfinite(gap).all()):
            raise ValueError("Non-finite full-vocabulary gap")
        gaps.extend(gap.cpu().tolist())
        entropies.extend((-(s.exp() * s).sum(-1)).cpu().tolist())
    return gaps, entropies
