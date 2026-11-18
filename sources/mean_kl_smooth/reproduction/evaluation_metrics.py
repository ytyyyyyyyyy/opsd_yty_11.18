"""CPU-only summary/export helpers for the durable independent evaluation."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def summarize_evaluation(rows, expected_questions, n, probe=False, wall_seconds=None):
    indices = [row['index'] for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate evaluation question indices')
    if any(len(row['samples']) != n for row in rows):
        raise ValueError('Every completed question must contain exactly n samples')
    samples = [sample for row in rows for sample in row['samples']]
    count = len(samples)
    correct = sum(s['correct'] for s in samples)
    lengths = sorted(s['tokens'] for s in samples)
    truncated = sum(s['truncated'] for s in samples)
    boxed = sum(s['has_boxed_answer'] for s in samples)
    def quantile(q):
        if not lengths:
            return None
        pos = (len(lengths)-1)*q
        low = int(pos)
        return lengths[low] + (lengths[min(low+1, len(lengths)-1)]-lengths[low])*(pos-low)
    return {
        'wall_seconds': wall_seconds,  # Current invocation, not total across resumes.
        'questions': len(rows), 'expected_questions': expected_questions,
        'samples': count, 'correct': correct,
        f'avg_at_{n}': correct/count if count else None,
        f'pass_at_{n}': sum(any(s['correct'] for s in row['samples']) for row in rows)/len(rows) if rows else None,
        'truncated': truncated, 'truncated_rate': truncated/count if count else None,
        'boxed_answers': boxed, 'boxed_answer_rate': boxed/count if count else None,
        'generated_tokens': sum(lengths), 'length_mean': sum(lengths)/count if count else None,
        'length_p50': quantile(.5), 'length_p95': quantile(.95),
        'length_max': max(lengths) if lengths else None,
        'complete': len(rows) == expected_questions, 'probe': probe,
    }


def export_evaluation_metrics(output, results):
    """Append progress JSONL and replace a one-row CSV; full JSON is authoritative."""
    output = Path(output)
    protocol = results['protocol']
    adapter = protocol.get('adapter')
    step = 0 if adapter is None else None
    if adapter:
        state_path = Path(adapter)/'trainer_state.json'
        if state_path.is_file():
            step = json.loads(state_path.read_text()).get('global_step')
    row = {
        'schema_version': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'result_file': str(output.resolve()), 'checkpoint_step': step,
        **results['summary'],
        # Preserve the exact protocol (dataset revision, adapter and all sampling parameters).
        'protocol_json': json.dumps(protocol, ensure_ascii=False, sort_keys=True),
        'runtime_sessions_json': json.dumps(results.get('runtime_sessions',[]), ensure_ascii=False, sort_keys=True),
    }
    with output.with_suffix('.metrics.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    path = output.with_suffix('.metrics.csv')
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)
