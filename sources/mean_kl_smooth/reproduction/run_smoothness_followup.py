"""Serial long-budget evaluation and bounded full-solution trajectory diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.evaluate_smoothness_long import file_hash, load_grader, parse_numeric, save_json

ARMS = ('baseline', 'mean_kl', 'smooth')
FILES = ('reproduction/evaluate_smoothness_long.py',
         'reproduction/generate_smoothness_diagnostics.py',
         'reproduction/score_smoothness_trajectories.py',
         'reproduction/run_smoothness_followup.py',
         'reproduction/analyze_smoothness_followup.py')


def status(run, stage, **fields):
    record = {'stage': stage, 'time_utc': datetime.now(timezone.utc).isoformat(), **fields}
    save_json(run / 'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def inventory(path):
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*'))
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}


def prepare(run, parent):
    if (run / 'protocol.json').exists():
        verify(run)
        print('ALREADY_PREPARED', run)
        return
    if run.exists() and any(run.iterdir()):
        raise RuntimeError('Nonempty partial preparation; inspect before retrying')
    for name in FILES:
        if not (ROOT / name).is_file():
            raise ValueError(f'Missing source: {name}')
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer
    from data_collator import SelfDistillationDataCollator
    from reproduction.gap_intervention_core import canonical_problem

    old = json.loads((parent / 'config.json').read_text())
    if file_hash(ROOT / 'data_collator.py') != file_hash(parent / 'source/data_collator.py'):
        raise ValueError('Current collator differs from frozen training prompts')
    origin = Path(old['pilot']['origin'])
    pool = load_from_disk(str(parent / 'train_dataset'))
    seen = set()
    event_paths = [origin / 'training/opsd/intervention_events.jsonl'] + [
        parent / 'training' / arm / 'intervention_events.jsonl' for arm in ARMS]
    for path in event_paths:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    seen.add(canonical_problem(json.loads(line)['problem']))
    tokenizer = AutoTokenizer.from_pretrained(old['model'], local_files_only=True)
    collator = SelfDistillationDataCollator(tokenizer, max_length=40960, reason_first=False,
                                          student_thinking=False, teacher_thinking=True)
    grader = load_grader(parent / 'source')
    indices = list(range(len(pool)))
    random.Random(20261113).shuffle(indices)
    selected, answers, questions, exclusions = [], [], [], Counter()
    selected_keys = set()
    for source_index in indices:
        row = pool[source_index]
        key = canonical_problem(row['problem'])
        if key in seen or key in selected_keys:
            exclusions['seen_or_duplicate'] += 1
            continue
        answer = grader['extract'](row['solution'])
        try:
            parsed, error = parse_numeric(answer, grader) if answer else (None, 'missing_reference_box')
        except (Exception, grader['TimeoutException']):
            parsed, error = None, 'reference_parse_error'
        if error:
            exclusions[error] += 1
            continue
        if any(x in row['problem'].lower() for x in ('[asy]', '<image', '<svg', 'in the diagram', 'in the figure')):
            exclusions['visual_question'] += 1
            continue
        batch = collator([row])
        student = batch['student_prompts'][0].tolist()
        teacher = batch['teacher_prompts'][0].tolist()
        if max(len(student), len(teacher)) > 12000:
            exclusions['prompt_budget'] += 1
            continue
        index = len(selected)
        questions.append({'index': index, 'id': str(index), 'source_index': source_index,
                          'problem': row['problem'], 'ground_truth': answer,
                          'prompt': tokenizer.decode(student, skip_special_tokens=False),
                          'prompt_token_ids': student, 'teacher_prompt_token_ids': teacher})
        selected.append(source_index)
        answers.append(answer)
        selected_keys.add(key)
        if len(selected) == 64:
            break
    if len(selected) != 64:
        raise ValueError('Could not prepare 64 eligible questions')
    run.mkdir(parents=True, exist_ok=True)
    # Materialize in memory so adding metadata cannot create caches in the frozen parent.
    selected_rows = [pool[index] for index in selected]
    diagnostic = Dataset.from_dict({name: [row[name] for row in selected_rows]
                                    for name in pool.column_names})
    diagnostic = diagnostic.add_column('answer', answers).add_column('id', list(map(str, range(64)))).add_column('source_index', selected)
    diagnostic.save_to_disk(str(run / 'diagnostic_dataset'))
    save_json(run / 'diagnostic_questions.json', questions)
    save_json(run / 'pool_selection.json', {
        'seed': 20261113, 'source_indices': selected, 'known_seen_problems_excluded': len(seen),
        'selection_exclusions': dict(exclusions),
        'selection': 'Seeded scan, finite numeric reference answers, full original solutions, no rollout/outcome/gap selection',
        'sources': {str(p): file_hash(p) for p in event_paths},
        'scope': 'Same training-data repository, but excluded questions observed in the recorded original and pilot rollouts; not claimed pretraining-unseen',
    })
    shutil.copytree(parent / 'source', run / 'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in FILES:
        shutil.copy2(ROOT / name, run / 'source' / name)
    checkpoint_sources = {'start_step35': parent / 'initial/checkpoint-35', **{
        arm: parent / 'training' / arm / 'checkpoint-40' for arm in ARMS}}
    adapter_hashes = {}
    for label, source in checkpoint_sources.items():
        destination = run / 'adapters' / label
        destination.mkdir(parents=True)
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json'):
            shutil.copy2(source / name, destination / name)
        adapter_hashes[label] = inventory(destination)
    (run / 'validation_dataset').symlink_to((parent / 'validation_dataset').resolve(), target_is_directory=True)
    cfg = {
        'project_root': str(ROOT), 'parent_run': str(parent), 'model': old['model'],
        'gpu': 0, 'gpu_memory_gib': 20., 'scoring_allocator_gib': 16.,
        'resource_policy': 'At most one GPU occupied by this user, all jobs serial; other users are never stopped',
        'wait_for': 'Parent pilot C evaluation completed and its GPU process released',
        'no_training': True,
        'evaluation': {'labels': ['start_step35', *ARMS], 'n': 2, 'max_new_tokens': 38912,
                       'seed': 42, 'thinking': True, 'temperature': 1., 'max_active_questions': 1},
        'diagnostics': {'arms': list(ARMS), 'questions': 64, 'n_per_round': 2,
                        'round_seeds': [20261113 + 10000 * r for r in range(4)],
                        'max_rounds': 4, 'max_new_tokens': 16384,
                        'student_thinking': False, 'teacher_thinking': True,
                        'temperature': 1.1, 'top_p': .95, 'top_k': 20,
                        'min_complete_per_class_per_arm': 30, 'min_distinct_questions_per_class_per_arm': 10,
                        'stop_policy': 'Finish the same full round for all three arms; stop once every arm meets both quotas, or after four rounds; retain and report all outcomes',
                        'teacher': 'Frozen same base, adapter disabled, full reference solution visible',
                        'grade': 'Only EOS-completed strictly graded correct/incorrect enter comparison; other outcomes separately counted'},
        'interpretation': 'Long AIME evaluation has a fixed denominator. Adaptive diagnostic sampling is separate and is never used as a benchmark accuracy estimate.',
    }
    save_json(run / 'config.json', cfg)
    save_json(run / 'protocol.json', {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': file_hash(run / 'config.json'), 'source_sha256': inventory(run / 'source'),
        'adapter_sha256': adapter_hashes, 'questions_sha256': file_hash(run / 'diagnostic_questions.json'),
        'dataset_sha256': inventory(run / 'diagnostic_dataset'),
        'parent_protocol_sha256': file_hash(parent / 'protocol.json'),
    })
    status(run, 'prepared', parent_run=str(parent), max_simultaneous_gpus=1)


def verify(run):
    cfg = json.loads((run / 'config.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    if file_hash(run / 'config.json') != protocol['config_sha256'] or inventory(run / 'source') != protocol['source_sha256']:
        raise ValueError('Frozen configuration/source changed')
    if file_hash(run / 'diagnostic_questions.json') != protocol['questions_sha256']:
        raise ValueError('Question pool changed')
    if inventory(run / 'diagnostic_dataset') != protocol['dataset_sha256']:
        raise ValueError('Full-reference diagnostic dataset changed')
    for label, expected in protocol['adapter_sha256'].items():
        if inventory(run / 'adapters' / label) != expected:
            raise ValueError(f'Frozen adapter changed: {label}')
    return cfg


def coverage(run, rounds):
    per_arm = {}
    for arm in ARMS:
        count, distinct = Counter(), defaultdict(set)
        for r in range(rounds):
            path = run / 'diagnostics' / arm / f'round_{r}' / 'generations/results.json'
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if not data['summary']['complete']:
                raise ValueError(f'Partial diagnostic round: {arm}/{r}')
            for row in data['rows']:
                for sample in row['samples']:
                    count[sample['status']] += 1
                    distinct[sample['status']].add(row['index'])
        per_arm[arm] = {'counts': dict(count), 'distinct_questions': {k: len(v) for k, v in distinct.items()}}
    target = json.loads((run / 'config.json').read_text())['diagnostics']
    reached = all(v['counts'].get(g, 0) >= target['min_complete_per_class_per_arm']
                  and v['distinct_questions'].get(g, 0) >= target['min_distinct_questions_per_class_per_arm']
                  for v in per_arm.values() for g in ('correct', 'incorrect'))
    record = {'completed_rounds': rounds, 'per_arm': per_arm, 'quotas_met': reached,
              'accuracy_role': 'Diagnostic sampling with outcome-based stopping; not a benchmark estimate'}
    save_json(run / 'coverage.json', record)
    return reached


def own_gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    found = []
    for line in output.splitlines():
        pid, gpu = [x.strip() for x in line.split(',')]
        try:
            if Path(f'/proc/{int(pid)}').stat().st_uid == os.getuid():
                found.append({'pid': int(pid), 'gpu_uuid': gpu})
        except FileNotFoundError:
            pass
    return found


def execute(run):
    cfg = verify(run)
    lock = (run / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'STOP').exists():
        raise RuntimeError('STOP exists; do not silently restart')
    (run / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    signal.signal(signal.SIGTERM, lambda *args: (run / 'STOP').write_text('SIGTERM\n'))
    signal.signal(signal.SIGINT, lambda *args: (run / 'STOP').write_text('SIGINT\n'))
    root = Path(cfg['project_root'])
    python = str(root / '.env-vllm/bin/python')
    finished_path = run / 'completed_jobs.json'
    finished = json.loads(finished_path.read_text()) if finished_path.exists() else []

    def wait_ready(label):
        while not (run / 'STOP').exists():
            parent = Path(cfg['parent_run'])
            endpoint = json.loads((parent / 'evaluation/smooth/progress.json').read_text())
            if not endpoint.get('complete'):
                status(run, 'waiting_for_C', completed_questions=endpoint.get('completed_questions'),
                       total_questions=endpoint.get('planned_questions'), next_job=label)
                time.sleep(20)
                continue
            active = own_gpu_processes()
            if active:
                status(run, 'waiting_for_single_gpu_slot', active_user_processes=active, next_job=label)
                time.sleep(20)
                continue
            free = int(subprocess.check_output(['nvidia-smi', f'--id={cfg["gpu"]}',
                       '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free < (cfg['gpu_memory_gib'] + 1) * 1024:
                status(run, 'waiting_for_gpu_memory', free_mib=free, next_job=label)
                time.sleep(20)
                continue
            return True
        return False

    def job(label, command, gpu=True):
        if label in finished:
            return True
        while not (run / 'STOP').exists():
            if gpu and not wait_ready(label):
                return False
            log = run / 'logs' / f'{label}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(cfg['gpu']) if gpu else '-1',
                   'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_HUB_OFFLINE': '1',
                   'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'WANDB_MODE': 'disabled', 'OPSD_REQUIRE_IDLE_GPU': '0',
                   'PYTHONPATH': str(run / 'source')}
            started = time.monotonic()
            log_start = log.stat().st_size if log.exists() else 0
            with log.open('a') as stream:
                proc = subprocess.Popen(['bash', str(run / 'source/reproduction/with_acceleration.sh'), *command],
                                        cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
                status(run, 'running' if gpu else 'analyzing', job=label, pid=proc.pid, log=str(log), command=command)
                heartbeat = time.monotonic()
                while proc.poll() is None:
                    if time.monotonic() - heartbeat > 60:
                        print('JOB_HEARTBEAT', label, int(time.monotonic() - started), flush=True)
                        heartbeat = time.monotonic()
                    time.sleep(2)
            with (run / 'runtime.jsonl').open('a') as f:
                f.write(json.dumps({'job': label, 'exit_code': proc.returncode,
                                    'elapsed_seconds': time.monotonic() - started}) + '\n')
            with log.open('rb') as stream:
                # Restrict classification to this attempt; older failures must not
                # turn a later code/protocol error into an endless resource retry.
                stream.seek(max(log_start, log.stat().st_size - 256 * 1024))
                attempt_tail = stream.read().decode('utf-8', errors='replace')
            startup_memory_race = (proc.returncode != 0
                                   and 'Free memory on device' in attempt_tail
                                   and 'on startup is less than desired GPU memory utilization' in attempt_tail)
            if proc.returncode == 75 or startup_memory_race:
                status(run, 'waiting_for_gpu_memory', next_job=label,
                       reason='GPU headroom changed during initialization' if startup_memory_race
                              else 'Worker resource preflight requested retry', log=str(log))
                time.sleep(20)
                continue
            if proc.returncode:
                raise RuntimeError(f'{label} exited {proc.returncode}; see {log}')
            finished.append(label)
            save_json(finished_path, finished)
            return not (run / 'STOP').exists()
        return False

    try:
        for label in cfg['evaluation']['labels']:
            command = [python, '-u', str(run / 'source/reproduction/evaluate_smoothness_long.py'),
                       '--model', cfg['model'], '--adapter', str(run / 'adapters' / label),
                       '--dataset', str(run / 'validation_dataset'), '--source-dir', str(run / 'source'),
                       '--output-dir', str(run / 'evaluation' / label), '--label', label,
                       '--n', '2', '--seed', '42', '--max-new-tokens', '38912',
                       '--gpu-memory-gib', '20', '--max-num-seqs', '12', '--max-active-questions', '1']
            if not job(f'evaluate_long_{label}', command):
                return
        for r, seed in enumerate(cfg['diagnostics']['round_seeds']):
            for arm in ARMS:
                destination = run / 'diagnostics' / arm / f'round_{r}'
                command = [python, '-u', str(run / 'source/reproduction/generate_smoothness_diagnostics.py'),
                           '--model', cfg['model'], '--adapter', str(run / 'adapters' / arm),
                           '--dataset', str(run / 'diagnostic_dataset'),
                           '--questions-json', str(run / 'diagnostic_questions.json'),
                           '--source-dir', str(run / 'source'), '--output-dir', str(destination / 'generations'),
                           '--label', arm, '--round-index', str(r), '--seed', str(seed),
                           '--gpu-memory-gib', '20']
                if not job(f'generate_{arm}_round{r}', command):
                    return
            if coverage(run, r + 1):
                break
        rounds = json.loads((run / 'coverage.json').read_text())['completed_rounds']
        for arm in ARMS:
            for r in range(rounds):
                destination = run / 'diagnostics' / arm / f'round_{r}'
                command = [python, '-u', str(run / 'source/reproduction/score_smoothness_trajectories.py'),
                           '--results', str(destination / 'generations/results.json'),
                           '--dataset', str(run / 'diagnostic_dataset'), '--model', cfg['model'],
                           '--adapter', str(run / 'adapters' / arm), '--output-dir', str(destination / 'scores'),
                           '--source-dir', str(run / 'source'), '--temperature', '1.1',
                           '--checkpoint-step', '40',
                           '--chunk-size', '128', '--vocab-chunk-size', '8192',
                           '--allocator-limit-gib', '16', '--resume']
                if not job(f'score_{arm}_round{r}', command):
                    return
        command = [python, str(run / 'source/reproduction/analyze_smoothness_followup.py'), '--run-dir', str(run)]
        if job('analyze_followup', command, gpu=False):
            covered = json.loads((run / 'coverage.json').read_text())
            status(run, 'complete' if covered['quotas_met'] else 'sample_target_not_reached', coverage=covered,
                   report=str(run / 'analysis/summary.json'))
    except Exception as exc:
        status(run, 'failed', error=str(exc))
        raise
    finally:
        if (run / 'STOP').exists():
            status(run, 'stopped', completed_jobs=finished)
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'check'])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--parent-run', type=Path, default=ROOT / 'runs/opsd_smoothness_pilot_20261113')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.run_dir.resolve(), a.parent_run.resolve())
    elif a.action == 'run':
        execute(a.run_dir.resolve())
    else:
        verify(a.run_dir.resolve())
        print('FOLLOWUP_PROTOCOL_VERIFIED')


if __name__ == '__main__':
    main()
