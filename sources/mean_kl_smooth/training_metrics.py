"""Durable, token-weighted OPSD diagnostics. Never used to construct the loss."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import torch
import torch.distributed as dist
from transformers import TrainerCallback


DISTRIBUTION_SUMS = (
    'kl_teacher_student', 'kl_student_teacher', 'entropy_student', 'entropy_teacher',
    'objective_unclipped', 'objective_clipped', 'clipped_entries',
    'clipped_positions', 'objective_entries', 'nonfinite_positions', 'valid_positions',
)


@torch.no_grad()
def distribution_sums(student, teacher, labels, temperature, beta, token_clip, top_k=None, chunk_tokens=32, token_records=None):
    """Full-vocabulary diagnostics; optional top-k applies ONLY to objective metrics.

    Convert detached logits to FP32 in bounded chunks. Token means are reduced
    later using valid_positions, including across microbatches and ranks.
    """
    if chunk_tokens < 1 or temperature <= 0:
        raise ValueError('chunk_tokens and temperature must be positive')
    if student.shape != teacher.shape or labels.shape != student.shape[:2]:
        raise ValueError('Diagnostic logits / labels shape mismatch')
    totals = torch.zeros(len(DISTRIBUTION_SUMS), dtype=torch.float64, device=student.device)
    for batch in range(student.shape[0]):
        record = {key: [None] * student.shape[1] for key in DISTRIBUTION_SUMS[:4]}
        if token_records is not None:
            token_records.append(record)
        positions = (labels[batch] != -100).nonzero(as_tuple=True)[0]
        for offset in range(0, len(positions), chunk_tokens):
            indices = positions[offset:offset+chunk_tokens]
            s = student[batch, indices].detach().float() / temperature
            t = teacher[batch, indices].detach().float() / temperature
            finite = torch.isfinite(s).all(-1) & torch.isfinite(t).all(-1)
            totals[-2] += (~finite).sum()
            s, t = s[finite], t[finite]
            if not s.shape[0]:
                continue
            ls, lt = s.log_softmax(-1), t.log_softmax(-1)
            ps, pt = ls.exp(), lt.exp()
            per_token = [(pt * (lt-ls)).sum(-1), (ps * (ls-lt)).sum(-1),
                         -(ps * ls).sum(-1), -(pt * lt).sum(-1)]
            values = [v.sum() for v in per_token]
            if token_records is not None:
                valid_indices = indices[finite].cpu().tolist()
                for key, value in zip(DISTRIBUTION_SUMS[:4], per_token):
                    for index, number in zip(valid_indices, value.cpu().tolist()):
                        record[key][index] = number
            if top_k is not None and top_k > 0:
                selected = t.topk(min(top_k, t.shape[-1]), dim=-1).indices
                ls = s.gather(-1, selected).log_softmax(-1)
                lt = t.gather(-1, selected).log_softmax(-1)
                ps, pt = ls.exp(), lt.exp()
            if beta == 0:
                terms = pt * (lt-ls)
            elif beta == 1:
                terms = ps * (ls-lt)
            else:
                mixture = torch.logaddexp(ls + math.log1p(-beta), lt + math.log(beta))
                terms = beta * pt * (lt-mixture) + (1-beta) * ps * (ls-mixture)
            clipped = terms.clamp(max=token_clip) if token_clip is not None else terms
            mask = terms > token_clip if token_clip is not None else torch.zeros_like(terms, dtype=torch.bool)
            values += [terms.sum(), clipped.sum(), mask.sum(), mask.any(-1).sum(),
                       terms.new_tensor(terms.numel()), terms.new_zeros(()), terms.new_tensor(s.shape[0])]
            totals += torch.stack([v.double() for v in values])
    return dict(zip(DISTRIBUTION_SUMS, totals.cpu().tolist()))


def completion_info(token_ids, eos_token_id, pad_token_id, max_new_tokens,
                    finish_reason=None, stop_reason=None, source='inferred'):
    """EOS before padding; length-cap hits and actual length stops are distinct."""
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]) - {None}
    ids = list(token_ids)
    eos_hit = next((i for i, token in enumerate(ids) if token in eos), None)
    if eos_hit is not None:
        ids = ids[:eos_hit+1]
    elif source != 'vllm':
        while ids and pad_token_id is not None and ids[-1] == pad_token_id:
            ids.pop()
    if finish_reason is None:
        finish_reason = 'eos' if eos_hit is not None else ('length' if len(ids) >= max_new_tokens else 'unknown')
    return {'completion_token_ids': ids, 'length': len(ids), 'eos': eos_hit is not None,
            'length_cap_hit': len(ids) >= max_new_tokens, 'truncated': finish_reason == 'length',
            'finish_reason': finish_reason, 'stop_reason': stop_reason, 'finish_reason_source': source}


def _mean(total, count):
    return total / count if count else None


def merge_windows(windows):
    """Merge sufficient statistics, not averages of unequal microbatches/ranks."""
    sums = {key: sum(w['distribution'].get(key, 0) for w in windows) for key in DISTRIBUTION_SUMS}
    tokens = sums['valid_positions']
    lengths = [n for w in windows for n in w['lengths']]
    samples = len(lengths)
    result = {'diagnostics/valid_tokens': tokens, 'diagnostics/nonfinite_tokens': sums['nonfinite_positions']}
    for key in DISTRIBUTION_SUMS[:6]:
        prefix = 'objective/' if key.startswith('objective_') else 'distribution/'
        name = key.removeprefix('objective_') if prefix == 'objective/' else key
        result[prefix + name] = _mean(sums[key], tokens)
    result.update({
        'clip/entry_fraction': _mean(sums['clipped_entries'], sums['objective_entries']),
        'clip/token_fraction': _mean(sums['clipped_positions'], tokens),
        'clip/removed_per_token': _mean(sums['objective_unclipped']-sums['objective_clipped'], tokens),
        'rollout/samples': samples, 'rollout/tokens': sum(lengths),
        'rollout/length_mean': _mean(sum(lengths), samples),
        'rollout/length_min': min(lengths) if lengths else None,
        'rollout/length_max': max(lengths) if lengths else None,
    })
    for name in ('eos', 'length_cap_hit', 'truncated', 'unknown_finish'):
        result[f'rollout/{name}_rate'] = _mean(sum(w[name] for w in windows), samples)
    for quantile in (.5, .95):
        value = float(torch.tensor(lengths, dtype=torch.float64).quantile(quantile)) if lengths else None
        result[f'rollout/length_p{int(quantile*100)}'] = value
    seconds = max(w['step_seconds'] for w in windows)
    generation_seconds = max(w['generation_seconds'] for w in windows)
    gradient_maxima = [w['gradient'].get('max_abs') for w in windows
                      if w['gradient'].get('max_abs') is not None]
    nonfinite_gradients = sum(w['gradient'].get('nonfinite_tensors', 0) for w in windows)
    result.update({
        'time/step_seconds_max_rank': seconds,
        'time/generation_seconds_max_rank': generation_seconds,
        'time/diagnostics_seconds_max_rank': max(w['diagnostics_seconds'] for w in windows),
        'time/elapsed_seconds_max_rank': max(w['elapsed_seconds'] for w in windows),
        'throughput/samples_per_second': _mean(samples, seconds),
        'throughput/generated_tokens_per_second': _mean(sum(lengths), seconds),
        'throughput/rollout_tokens_per_second': _mean(sum(lengths), generation_seconds),
        'grad/max_abs_after_clip': max(gradient_maxima) if gradient_maxima and not nonfinite_gradients else None,
        'grad/nonfinite_tensors': nonfinite_gradients,
        'grad/tensors_per_rank_min': min(w['gradient'].get('tensors', 0) for w in windows),
        'grad/tensors_per_rank_max': max(w['gradient'].get('tensors', 0) for w in windows),
    })
    for key in ('allocated_mib', 'reserved_mib', 'peak_allocated_mib', 'peak_reserved_mib'):
        values = [w['memory'].get(key) for w in windows if w['memory'].get(key) is not None]
        result[f'memory/{key}_max_rank'] = max(values) if values else None
    return result


class TrainingMetrics(TrainerCallback):
    def __init__(self, trainer, chunk_tokens=32):
        self.trainer = trainer
        self.chunk_tokens = chunk_tokens
        self.output = Path(trainer.args.output_dir) / 'metrics'
        self.session_id = None
        self.started = self.previous_step_end = None
        self.gradient = {}
        self.pending_step = None
        self.trace_index = 0
        self.pending_traces = []
        self.reset_window()

    def reset_window(self):
        self.distribution = defaultdict(float)
        self.lengths = []
        self.eos = self.length_cap_hit = self.truncated = self.unknown_finish = 0
        self.generation_seconds = self.diagnostics_seconds = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_window()
        self.gradient = {}
        self.pending_step = None
        self.trace_index = 0
        self.pending_traces = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = self.previous_step_end = time.perf_counter()
        self.session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        if dist.is_available() and dist.is_initialized():
            ids = [self.session_id]
            dist.broadcast_object_list(ids, src=0)
            self.session_id = ids[0]
        if self.trainer.accelerator.is_main_process:
            root = Path(__file__).parent
            metadata = {
                'schema_version': 2, 'session_id': self.session_id, 'start_step': state.global_step,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'world_size': args.world_size,
                'model': self.trainer.model_name_or_path, 'seed': args.seed,
                'temperature': self.trainer.temperature, 'beta': self.trainer.beta,
                'pointwise_clip': self.trainer.jsd_token_clip, 'top_k_loss': self.trainer.top_k_loss,
                'fixed_teacher': self.trainer.fixed_teacher, 'ema_teacher': self.trainer.use_ema_teacher,
                'sampled_token_objective': self.trainer.use_thinking_machines_loss,
                'distribution_metrics_available': not self.trainer.use_thinking_machines_loss,
                'micro_batch': args.per_device_train_batch_size, 'gradient_accumulation': args.gradient_accumulation_steps,
                'generation_config': self.trainer.generation_config.to_dict(), 'chunk_tokens': self.chunk_tokens,
                'torch_version': torch.__version__,
                'definitions': {
                    'distribution': 'Full vocabulary, temperature-scaled FP32 probabilities; natural log; no top-k/top-p sampling filter.',
                    'weighting': 'Valid completion token weighted across microbatches and ranks; padding excluded. Nonfinite positions counted separately and excluded.',
                    'objective': 'Recomputed FP32 diagnostic of the configured KL/JSD, including top_k_loss if set; separate from the actual mixed-precision optimizer loss.',
                    'clip': 'Pointwise across vocabulary, not clipping the vocabulary-summed KL. Entry and token fractions use separate denominators.',
                    'grad_norm': 'Trainer grad_norm is before clipping; max_abs_after_clip is maximum gradient element across ranks.',
                    'memory': 'PyTorch training process only; peak is cumulative, not whole-GPU/vLLM memory.',
                    'time': 'Step-end intervals include previous logging/checkpoint overhead; global throughput divides global work by maximum rank duration.',
                    'truncation': 'Backend length stop when available, otherwise EOS/cap inference with explicit source. Cap hit alone is not truncation.',
                    'resume': 'Append-only, session_id disambiguates repeated steps after checkpoint resume; no silent overwrite.',
                    'unavailable': 'Missing measurements are null, never fabricated zeros. Sampled-token objective does not retain full-distribution diagnostics.',
                    'token_metrics': 'token_metrics.rankN.jsonl joins generation_trace by session_id/rank/sequence_index. Arrays align with scored_completion_token_ids; masked/nonfinite positions are null. Prediction for response j uses prefix through j-1, before optimizer update. Full vocabulary at training temperature, no sampling filters or clipping.',
                    'accuracy': 'Independent evaluation only; never inferred from training loss or generated answer formatting.',
                },
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [root/'training_metrics.py', root/'opsd_trainer.py', root/'opsd_train.py', root/'data_collator.py']},
            }
            self.append_json(self.output / 'sessions.jsonl', metadata)

    @staticmethod
    def append_json(path, row):
        with path.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()

    def observe_logits(self, student, teacher, labels):
        start = time.perf_counter()
        tr = self.trainer
        token_records = [] if self.session_id is not None else None
        sums = distribution_sums(student, teacher, labels, tr.temperature, tr.beta,
                                 tr.jsd_token_clip, tr.top_k_loss, self.chunk_tokens,
                                 token_records=token_records)
        if token_records is not None:
            if len(token_records) != len(self.pending_traces):
                raise ValueError('Token diagnostics / generation trace batch mismatch')
            rank = tr.accelerator.process_index
            path = self.output / f'token_metrics.rank{rank}.jsonl'
            for trace, record in zip(self.pending_traces, token_records):
                if len(record['entropy_student']) != trace['scored_length']:
                    raise ValueError('Token diagnostics / generation trace length mismatch')
                self.append_json(path, {**trace, 'temperature': tr.temperature,
                                        'measurement': 'training_forward_before_update', **record})
            self.pending_traces = []
        for key, value in sums.items():
            self.distribution[key] += value
        self.diagnostics_seconds += time.perf_counter() - start

    def observe_rollout(self, infos, seconds):
        self.generation_seconds += seconds
        for info in infos:
            self.lengths.append(info['length'])
            self.eos += info['eos']
            self.length_cap_hit += info['length_cap_hit']
            self.truncated += info['truncated']
            self.unknown_finish += info['finish_reason'] == 'unknown'

    def save_traces(self, inputs, infos, scoring_labels):
        if self.session_id is None:
            return
        rank = self.trainer.accelerator.process_index
        path = self.output / f'generation_trace.rank{rank}.jsonl'
        student_ids = inputs['student_input_ids'][:, :inputs['student_prompt_length']].detach().cpu().tolist()
        student_mask = inputs['student_attention_mask'][:, :inputs['student_prompt_length']].detach().cpu().tolist()
        teacher_ids = inputs['teacher_prompts'].detach().cpu().tolist()
        teacher_mask = inputs['teacher_attention_mask'][:, :inputs['teacher_prompt_length']].detach().cpu().tolist()
        labels = scoring_labels.detach().cpu().tolist()
        # Actual scoring IDs retain padding and any backend normalization/truncation.
        scored_ids = inputs['student_input_ids'][:, inputs['student_prompt_length']:].detach().cpu().tolist()
        if len(infos) != len(scored_ids) or any(len(ids) != len(mask) for ids, mask in zip(scored_ids, labels)):
            raise ValueError('Generation trace / scoring mask shape mismatch')
        with path.open('a') as handle:
            self.pending_traces = []
            for i, info in enumerate(infos):
                row = {'session_id': self.session_id, 'rank': rank, 'sequence_index': self.trace_index,
                       'model_step': self.trainer.state.global_step, 'optimizer_step': self.trainer.state.global_step+1,
                       'student_prompt_token_ids': student_ids[i], 'student_prompt_attention_mask': student_mask[i],
                       'teacher_prompt_token_ids': teacher_ids[i], 'teacher_prompt_attention_mask': teacher_mask[i],
                       'scored_completion_token_ids': scored_ids[i], 'completion_loss_mask': [x != -100 for x in labels[i]],
                       **info}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                self.pending_traces.append({key: row[key] for key in
                    ('session_id', 'rank', 'sequence_index', 'model_step', 'optimizer_step')})
                self.pending_traces[-1]['scored_length'] = len(scored_ids[i])
                self.trace_index += 1
            handle.flush()

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.numel()]
        if gradients:
            finite = torch.stack([torch.isfinite(g).all() for g in gradients])
            maximum = torch.stack([g.detach().abs().max().float() for g in gradients]).max().item()
            self.gradient = {'tensors': len(gradients), 'nonfinite_tensors': int((~finite).sum().item()),
                             'max_abs': maximum if math.isfinite(maximum) else None}
        else:
            self.gradient = {'tensors': 0, 'nonfinite_tensors': 0, 'max_abs': None}

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.step_seconds = now - self.previous_step_end
        self.previous_step_end = now
        self.elapsed_seconds = now - self.started
        self.pending_step = state.global_step
        control.should_log = True
        return control

    def flush(self, logs, step):
        if self.pending_step != step:
            return {}
        device = self.trainer.accelerator.device
        memory = {}
        if device.type == 'cuda':
            memory = {name: fn(device) / 1024**2 for name, fn in {
                'allocated_mib': torch.cuda.memory_allocated, 'reserved_mib': torch.cuda.memory_reserved,
                'peak_allocated_mib': torch.cuda.max_memory_allocated,
                'peak_reserved_mib': torch.cuda.max_memory_reserved}.items()}
        local = {'distribution': dict(self.distribution), 'lengths': self.lengths,
                 'eos': self.eos, 'length_cap_hit': self.length_cap_hit, 'truncated': self.truncated,
                 'unknown_finish': self.unknown_finish, 'generation_seconds': self.generation_seconds,
                 'diagnostics_seconds': self.diagnostics_seconds, 'step_seconds': self.step_seconds,
                 'elapsed_seconds': self.elapsed_seconds, 'gradient': self.gradient, 'memory': memory}
        windows = [local]
        if dist.is_available() and dist.is_initialized():
            windows = [None] * dist.get_world_size()
            dist.all_gather_object(windows, local)
        metrics = merge_windows(windows)
        if self.trainer.accelerator.is_main_process:
            row = {'schema_version': 1, 'session_id': self.session_id, 'step': step,
                   'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                   'epoch': self.trainer.state.epoch,
                   **{k: logs.get(k) for k in ('loss', 'on_policy_loss', 'grad_norm', 'learning_rate')}, **metrics}
            row = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in row.items()}
            self.append_json(self.output / 'training_metrics.jsonl', row)
            path = self.output / 'training_metrics.csv'
            existing = path.exists() and path.stat().st_size > 0
            if existing:
                with path.open(newline='') as f:
                    if next(csv.reader(f)) != list(row):
                        raise ValueError('Existing metrics CSV uses a different schema; choose a new run output directory')
            with path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not existing:
                    writer.writeheader()
                writer.writerow(row)
        self.reset_window()
        self.pending_step = None
        return {k: v for k, v in metrics.items() if v is not None}
