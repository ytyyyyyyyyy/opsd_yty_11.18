"""Join completed training token metrics to conservative final-answer labels, on CPU."""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer
from math_verify import parse, verify
from math_verify.errors import TimeoutException


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--training-subdir', default='opsd')
    p.add_argument('--expected-start-step', type=int, default=1)
    args = p.parse_args()
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    training = run / 'training' / args.training_subdir
    config = json.loads((run / 'config.json').read_text())
    sources = {}

    def rows(path):
        with path.open('rb') as f:
            data = f.read(path.stat().st_size)
        data = data[:data.rfind(b'\n') + 1]
        sources[str(path)] = {'complete_prefix_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for line in data.splitlines():
            if line.strip():
                yield json.loads(line)

    def key(row):
        return row['session_id'], row.get('rank', 0), row['sequence_index']

    metrics = list(rows(training / 'metrics/training_metrics.jsonl'))
    completed = {(r['session_id'], r['step']) for r in metrics}
    assert len(completed) == len(metrics) == len({r['step'] for r in metrics})
    assert sorted(r['step'] for r in metrics) == list(range(args.expected_start_step, max(r['step'] for r in metrics) + 1))
    assert all(s['world_size'] == 1 for s in rows(training / 'metrics/sessions.jsonl'))
    records = {}
    for r in rows(training / 'intervention_events.jsonl'):
        if (r['session_id'], r['optimizer_step']) not in completed:
            continue
        assert r['arm'] == 'opsd' and not r['applied']
        assert r['initial_ids'] == r['final_ids'] and r['initial_correct'] == r['final_correct']
        k = key(r)
        assert k not in records
        records[k] = {'session_id': k[0], 'rank': k[1], 'sequence_index': k[2],
                      'optimizer_step': r['optimizer_step'], 'problem': r['problem'],
                      'saved_final_correct': r['final_correct'],
                      'reference_answer_extractable': r['reference_answer_extractable'],
                      'ids': np.array(r['final_ids'], dtype=np.int32)}
    assert len(records) == sum(r['rollout/samples'] for r in metrics)
    print('COMPLETED_EVENTS', len(records), flush=True)
    tokenizer = Tokenizer.from_file(str(Path(config['model']) / 'tokenizer.json'))
    extraction_source = run / 'source/eval/evaluate_math.py'
    node = next(n for n in ast.parse(extraction_source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'extract_boxed_answer')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(extraction_source), 'exec'), namespace)
    extract = namespace['extract_boxed_answer']
    matched = set()
    for trace in rows(training / 'metrics/generation_trace.rank0.jsonl'):
        k = key(trace)
        if k not in records:
            continue
        assert k not in matched
        r = records[k]
        assert trace['optimizer_step'] == r['optimizer_step']
        assert np.array_equal(r['ids'], trace['scored_completion_token_ids'])
        assert np.array_equal(r['ids'], trace['completion_token_ids'])
        text = tokenizer.decode(r['ids'].tolist(), skip_special_tokens=True)
        r.update(prediction=extract(text), length=len(r['ids']),
                 eos=trace['eos'], truncated=trace['truncated'], finish_reason=trace['finish_reason'],
                 mask=np.array(trace['completion_loss_mask'], dtype=bool))
        matched.add(k)
    assert len(matched) == len(records)
    matched = set()
    for token in rows(training / 'metrics/token_metrics.rank0.jsonl'):
        k = key(token)
        if k not in records:
            continue
        assert k not in matched
        r = records[k]
        assert token['measurement'] == 'training_forward_before_update'
        assert token['optimizer_step'] == r['optimizer_step'] and token['scored_length'] == r['length']
        values = np.array([np.nan if v is None else v for v in token['kl_teacher_student']], dtype=float)
        assert len(values) == r['length'] == len(r['mask'])
        values[~r['mask']] = np.nan
        r['gap'] = values
        matched.add(k)
    assert len(matched) == len(records)
    problems = {r['problem'] for r in records.values()}
    references = defaultdict(set)
    dataset = load_from_disk(str(run / 'train_dataset'))
    for item in dataset:
        if item['problem'] in problems:
            references[item['problem']].add(extract(item['solution']))
    assert len(references) == len(problems)

    @lru_cache(maxsize=4096)
    def parsed(value):
        if not isinstance(value, str) or not value.strip():
            return []
        expression = value if '$' in value else f'${value}$'
        return parse(expression, fallback_mode='no_fallback', parsing_timeout=3, raise_on_error=True)

    ordered = sorted(records.values(), key=lambda r: (r['optimizer_step'], r['session_id'], r['rank'], r['sequence_index']))
    for i, r in enumerate(ordered):
        target_set = references[r['problem']]
        r['target'] = next(iter(target_set)) if len(target_set) == 1 else None
        r['grade'], r['grade_reason'] = 'ungraded', 'not_classified'
        if r['truncated']:
            r.update(grade='truncated', grade_reason='Length-capped response excluded even when a boxed answer is present')
        elif not r['eos']:
            r.update(grade='unknown_finish', grade_reason='No confirmed EOS')
        elif len(target_set) != 1:
            r['grade_reason'] = 'Conflicting reference answers for duplicate problem'
        elif not r['reference_answer_extractable'] or not r['target'] or not r['prediction']:
            r['grade_reason'] = 'Missing reference or completed boxed prediction'
        else:
            try:
                pred, target = parsed(r['prediction']), parsed(r['target'])
                if not pred or not target:
                    r['grade_reason'] = 'Prediction or reference cannot be parsed without fallback'
                elif not all(getattr(value, 'is_number', False) is True
                             and getattr(value, 'is_finite', False) is True
                             and not getattr(value, 'free_symbols', set()) for value in pred + target):
                    r['grade_reason'] = 'Non-numeric or ambiguous symbolic/choice/set answer excluded symmetrically from both groups'
                else:
                    verified = bool(verify(target, pred, timeout_seconds=3, raise_on_error=True))
                    r['rechecked_correct'] = verified
                    if not isinstance(r['saved_final_correct'], bool) or verified != r['saved_final_correct']:
                        r.update(grade='grading_disagreement', grade_reason='Recheck and saved label disagree')
                    else:
                        r.update(grade='correct' if verified else 'incorrect', grade_reason='Completed EOS; parsed final answer; recheck agrees with saved training label')
            except (Exception, TimeoutException) as exc:
                r['grade_reason'] = f'Grading exception: {type(exc).__name__}'
        if (i + 1) % 100 == 0:
            print('LABEL_PROGRESS', i + 1, len(ordered), flush=True)
    review_file = out.parent / 'label_review_exclusions.json'
    if review_file.exists():
        reviewed = json.loads(review_file.read_text())
        excluded = {(x['session_id'], x.get('rank', 0), x['sequence_index']): x
                    for x in reviewed['exclusions']}
        for r in ordered:
            item = excluded.get((r['session_id'], r['rank'], r['sequence_index']))
            if item is not None and r['grade'] not in ('truncated', 'unknown_finish'):
                r.update(grade='needs_review', grade_reason=item['reason'], review_evidence=item.get('evidence'))
        sources[str(review_file)] = {'sha256': hashlib.sha256(review_file.read_bytes()).hexdigest()}
    for metric in metrics:
        group = [r for r in ordered if r['session_id'] == metric['session_id'] and r['optimizer_step'] == metric['step']]
        a = np.concatenate([r['gap'] for r in group])
        assert len(group) == metric['rollout/samples']
        assert np.isfinite(a).sum() == metric['diagnostics/valid_tokens']
        assert np.isclose(np.nanmean(a), metric['distribution/kl_teacher_student'], atol=1e-6)
    offsets = np.cumsum([0] + [r['length'] for r in ordered])
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    np.savez_compressed(out / 'flattened_gap.npz', gaps=np.concatenate([r['gap'] for r in ordered]),
                        offsets=offsets, steps=np.array([r['optimizer_step'] for r in ordered]),
                        grades=np.array([r['grade'] for r in ordered]), snapshot_time=np.array(now))
    metadata = [{k: v for k, v in r.items() if k not in ('ids', 'gap', 'mask')} for r in ordered]
    (out / 'rows.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    report = {'snapshot_time_beijing': now, 'last_completed_step': max(r['step'] for r in metrics),
              'trajectories': len(ordered), 'tokens': int(offsets[-1]),
              'grade_counts': dict(Counter(r['grade'] for r in ordered)),
              'grade_reason_counts': dict(Counter(r['grade_reason'] for r in ordered)),
              'definition': 'Original training trajectories, session/step joined to completed optimizer updates; automated final-answer correctness on completed, parsable finite numeric scalar answers only. Same eligibility criteria for both groups. Truncated, symbolic, unparseable, uncertain or inconsistent grades excluded from correct/incorrect curves. Reference answers have not all been independently solved.',
              'gap_source': 'training_forward_before_update full-vocabulary unclipped KL(teacher||student), temperature 1.1',
              'sources': sources, 'independent_sampling_started': False,
              'per_step_means_verified': True}
    (out / 'snapshot_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['last_completed_step', 'trajectories', 'tokens', 'grade_counts']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
