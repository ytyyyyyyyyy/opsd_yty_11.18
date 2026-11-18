"""Freeze and run a bounded three-arm OPSD smoothness pilot on an idle GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import run_gap_intervention as runner
from reproduction.gap_intervention_core import atomic_json, complete_checkpoint, sha256

NEW_SOURCES = (
    'reproduction/gap_smoothness.py', 'reproduction/check_gap_smoothness.py',
    'reproduction/smoothness_pilot_train.py', 'reproduction/run_smoothness_pilot.py',
    'reproduction/evaluate_smoothness_pilot.py', 'reproduction/analyze_smoothness_pilot.py',
    'reproduction/export_training_correct_wrong_gap.py',
    'reproduction/plot_training_correct_wrong_gap.py',
)
ARMS = ('baseline', 'mean_kl', 'smooth')


def inventory(path):
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*'))
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}


def prepare(run, origin, gpu):
    if (run / 'protocol.json').exists():
        verify(run)
        print('ALREADY_PREPARED', run)
        return
    if run.exists() and any(run.iterdir()):
        raise RuntimeError('Partial preparation exists; inspect it before reusing this directory')
    checkpoint = origin / 'training/opsd/checkpoint-35'
    if not complete_checkpoint(checkpoint, 35):
        raise ValueError('The pinned full checkpoint-35 is unavailable')
    for name in NEW_SOURCES:
        if not (ROOT / name).is_file():
            raise ValueError(f'Missing pilot source: {name}')
    run.mkdir(parents=True, exist_ok=True)
    shutil.copytree(origin / 'source', run / 'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in NEW_SOURCES:
        shutil.copy2(ROOT / name, run / 'source' / name)
    # Own an immutable copy of the complete optimizer/scheduler/RNG checkpoint.
    start = run / 'initial/checkpoint-35'
    shutil.copytree(checkpoint, start)
    for name in ('train_dataset', 'validation_dataset'):
        (run / name).symlink_to((origin / name).resolve(), target_is_directory=True)
    cfg = json.loads((origin / 'config.json').read_text())
    cfg.update(gpu=gpu, arms=list(ARMS),
               schedule='Bounded pilot: common start evaluation, five updates per arm, endpoint evaluation; no periodic evaluation')
    cfg['training']['train_on'] = 'Unmodified student on-policy trajectories; isolated A/B/C auxiliary objectives'
    cfg['validation'].update(n=2, max_new_tokens=8192, every_optimizer_updates=None,
                             gpu_memory_gib=24, max_num_seqs=12,
                             metric='Completed correct generations / 60; capped pilot, not comparable to 38k Avg@12')
    cfg['pilot'] = {
        'origin': str(origin), 'origin_checkpoint': str(checkpoint), 'checkpoint': str(start),
        'start_step': 35, 'target_step': 40, 'updates_per_arm': 5,
        'expected_trajectories_per_arm': 160,
        'auxiliary': {'auxiliary_weight': 0.05, 'smooth_alpha': 0.25, 'smooth_beta': 0.1},
        'objective': '(sum unclipped content KL + alpha * sum SmoothL1(adjacent KL differences)) / content_token_count; trajectory mean',
        'arms': {'baseline': 'Exact frozen original loss', 'mean_kl': 'Original + lambda * content mean KL',
                 'smooth': 'Original + lambda * (content mean KL + alpha * SmoothL1 sum / content count)'},
        'auxiliary_exclusions': 'First response token, pad and EOS ids; no crossing masked holes or trajectory boundaries',
        'grading_used_for_loss': False, 'full_optimizer_scheduler_rng_resume': True,
        'periodic_validation': False, 'source_collection_continues_independently': False,
        'resource_policy': 'At most one GPU occupied by this user; pause original collection at a saved optimizer boundary; serialize all pilot jobs on GPU 0',
        'interpretation': 'Single-seed short pilot; accuracy and curve changes are preliminary, not significance claims',
    }
    atomic_json(run / 'config.json', cfg)
    command = [str(ROOT / '.env-vllm/bin/python'), str(run / 'source/reproduction/check_gap_smoothness.py')]
    checked = subprocess.run(['bash', str(run / 'source/reproduction/with_acceleration.sh'), *command],
                             cwd=ROOT, text=True, capture_output=True,
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '2'})
    (run / 'cpu_preflight.log').write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError('Loss preflight failed; see cpu_preflight.log')
    protocol = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha256(run / 'config.json'), 'source_sha256': inventory(run / 'source'),
        'checkpoint_sha256': inventory(start), 'origin_checkpoint_sha256': inventory(checkpoint),
        'train_dataset_sha256': inventory((run / 'train_dataset').resolve()),
        'validation_dataset_sha256': inventory((run / 'validation_dataset').resolve()),
        'cpu_preflight_passed': True, 'cpu_preflight_log_sha256': sha256(run / 'cpu_preflight.log'),
        'original_frozen_protocol_sha256': sha256(origin / 'protocol.json'),
    }
    assert protocol['checkpoint_sha256'] == protocol['origin_checkpoint_sha256']
    atomic_json(run / 'protocol.json', protocol)
    runner.status(run, 'prepared', gpu=gpu, start_step=35, target_step=40, arms=list(ARMS))


def verify(run):
    cfg = json.loads((run / 'config.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    if sha256(run / 'config.json') != protocol['config_sha256']:
        raise ValueError('Frozen pilot configuration changed')
    if inventory(run / 'source') != protocol['source_sha256']:
        raise ValueError('Frozen pilot source changed')
    if inventory(Path(cfg['pilot']['checkpoint'])) != protocol['checkpoint_sha256']:
        raise ValueError('Initial resume checkpoint changed')
    if not protocol['cpu_preflight_passed']:
        raise ValueError('CPU preflight did not pass')
    return cfg


def eval_command(run, cfg, label, checkpoint):
    return [str(ROOT / '.env-vllm/bin/python'), '-u',
            str(run / 'source/reproduction/evaluate_smoothness_pilot.py'),
            '--model', cfg['model'], '--adapter', str(checkpoint),
            '--dataset', str(run / 'validation_dataset'), '--output-dir', str(run / 'evaluation' / label),
            '--source-dir', str(run / 'source'), '--label', label,
            '--n', '2', '--max-new-tokens', '8192', '--seed', '42',
            '--gpu-memory-gib', '24', '--max-num-seqs', '12']


def execute(run):
    cfg = verify(run)
    handle = (run / 'run.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'STOP').exists():
        raise RuntimeError('STOP exists; explicitly remove it after reviewing before resuming')
    prior = json.loads((run / 'status.json').read_text())
    pid = prior.get('pid')
    if prior.get('stage') == 'running' and pid and Path(f'/proc/{pid}').exists():
        raise RuntimeError('An existing child may still be running; refusing duplicate GPU work')
    (run / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    os.environ.update(SMOOTHNESS_PILOT_RUN=str(run), OPSD_REQUIRE_IDLE_GPU='1')
    signal.signal(signal.SIGTERM, lambda *args: (run / 'STOP').write_text('SIGTERM\n'))
    signal.signal(signal.SIGINT, lambda *args: (run / 'STOP').write_text('SIGINT\n'))
    start, target = Path(cfg['pilot']['checkpoint']), cfg['pilot']['target_step']
    done = run / 'completed_jobs.json'
    finished = json.loads(done.read_text()) if done.exists() else []

    def wait_for_single_gpu_slot():
        # Check all physical GPUs, not just the selected device. This also
        # waits for the old collection trainer to release its CUDA context.
        while not (run / 'STOP').exists():
            output = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                text=True, timeout=15)
            own = []
            for line in output.splitlines():
                fields = [x.strip() for x in line.split(',')]
                try:
                    pid = int(fields[0])
                    if Path(f'/proc/{pid}').stat().st_uid == os.getuid():
                        own.append({'pid': pid, 'gpu_uuid': fields[1]})
                except FileNotFoundError:
                    pass
            if not own:
                origin_status_path = Path(cfg['pilot']['origin']) / 'status.json'
                origin_status = json.loads(origin_status_path.read_text())
                if origin_status.get('stage') == 'running':
                    # Require the original supervisor to finish checkpoint
                    # verification as well as releasing its model process.
                    runner.status(run, 'waiting_for_original_checkpoint', original_status=origin_status.get('stage'))
                    time.sleep(10)
                    continue
                if origin_status.get('stage') not in ('manually_stopped', 'stopped', 'complete'):
                    raise RuntimeError(f'Original collection did not stop cleanly: {origin_status.get("stage")}')
                return True
            runner.status(run, 'waiting_for_single_gpu_slot', active_user_gpu_processes=own,
                          max_simultaneous_gpus=1, next_gpu=cfg['gpu'])
            time.sleep(15)
        return False

    def job(label, command, arm='opsd', gpu=True):
        if label in finished:
            return True
        if (run / 'STOP').exists():
            return False
        os.environ['SMOOTHNESS_PILOT_ARM'] = arm
        if gpu:
            while True:
                if not wait_for_single_gpu_slot():
                    return False
                try:
                    code = runner.run_child(run, cfg, label, command, arm)
                    if code is None:
                        return False
                    break
                except RuntimeError:
                    journal = run / 'runtime.jsonl'
                    last = json.loads(journal.read_text().splitlines()[-1]) if journal.exists() else {}
                    if last.get('job') != label or last.get('exit_code') != 75:
                        raise
                    runner.status(run, 'waiting_for_idle_gpu', job=label, requeue_after_exit=75)
                    time.sleep(15)
        else:
            log = run / 'logs' / f'{label}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            runner.status(run, 'analyzing', job=label, log=str(log))
            with log.open('a') as stream:
                subprocess.run(['bash', str(run / 'source/reproduction/with_acceleration.sh'), *command],
                               cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, check=True)
        finished.append(label)
        atomic_json(done, finished)
        return not (run / 'STOP').exists()

    try:
        if not job('evaluate_start', eval_command(run, cfg, 'start_step35', start)):
            return
        for arm in ARMS:
            checkpoint = run / 'training' / arm / f'checkpoint-{target}'
            if not complete_checkpoint(checkpoint, target):
                output = run / 'training' / arm
                if (output / 'smoothness_metrics.jsonl').exists():
                    raise RuntimeError(f'Partial {arm} training exists; inspect before replaying from step35')
                command = runner.train_command(run, arm, target, start)
                command[2] = str(run / 'source/reproduction/smoothness_pilot_train.py')
                if not job(f'train_{arm}_35_to40', command, arm):
                    return
            if not complete_checkpoint(checkpoint, target):
                raise RuntimeError(f'{arm} did not save the complete endpoint')
            if not job(f'evaluate_{arm}', eval_command(run, cfg, arm, checkpoint), arm):
                return
            export = [str(ROOT / '.env-vllm/bin/python'),
                      str(run / 'source/reproduction/export_training_correct_wrong_gap.py'),
                      '--run-dir', str(run), '--training-subdir', arm, '--expected-start-step', '36',
                      '--output-dir', str(run / 'figures' / arm / 'input')]
            if not job(f'export_{arm}', export, arm, gpu=False):
                return
            plot = [str(ROOT / '.env-vllm/bin/python'),
                    str(run / 'source/reproduction/plot_training_correct_wrong_gap.py'),
                    '--input-dir', str(run / 'figures' / arm / 'input'),
                    '--output-dir', str(run / 'figures' / arm)]
            if not job(f'plot_{arm}', plot, arm, gpu=False):
                return
        report = [str(ROOT / '.env-vllm/bin/python'),
                  str(run / 'source/reproduction/analyze_smoothness_pilot.py'), '--run-dir', str(run)]
        if job('compare_arms', report, gpu=False):
            runner.status(run, 'complete', arms=list(ARMS),
                          report=str(run / 'analysis/summary.json'), figures=str(run / 'figures'))
    except Exception as exc:
        runner.status(run, 'failed', error=str(exc))
        raise
    finally:
        if (run / 'STOP').exists():
            runner.status(run, 'stopped', completed_jobs=finished)
        handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'check'])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--origin', type=Path, default=ROOT / 'runs/opsd_token_diagnostics_20261110')
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    run = a.run_dir.resolve()
    if a.action == 'prepare':
        prepare(run, a.origin.resolve(), a.gpu)
    elif a.action == 'run':
        execute(run)
    else:
        verify(run)
        print('PILOT_PROTOCOL_VERIFIED', run)


if __name__ == '__main__':
    main()
