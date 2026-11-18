"""Install the isolated A/B/C auxiliary objective on the frozen OPSD trainer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install():
    import torch
    import opsd_trainer
    from reproduction import gap_intervention_train, position_supervision
    # Import before monkeypatching: baseline captures the original function.
    from reproduction.gap_smoothness import routed_smooth_forward_kl
    from transformers import TrainerCallback

    run = Path(os.environ['SMOOTHNESS_PILOT_RUN']).resolve()
    arm = os.environ['SMOOTHNESS_PILOT_ARM']
    cfg = json.loads((run / 'config.json').read_text())
    if arm not in ('baseline', 'mean_kl', 'smooth'):
        raise ValueError(f'Unknown pilot arm: {arm}')
    os.environ['GAP_INTERVENTION_RUN'] = str(run)
    os.environ['GAP_INTERVENTION_ARM'] = 'opsd'
    parent = gap_intervention_train.install()
    active = {}

    class BranchWitness(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            expected = cfg['pilot']['start_step']
            if state.global_step != expected:
                raise RuntimeError(f'Expected exact full resume at {expected}, got {state.global_step}')
            trainer = active['trainer']
            actual = {
                'arm': arm, 'start_step': state.global_step,
                'target_step': cfg['pilot']['target_step'],
                'planned_max_steps': state.max_steps,
                'micro_batch': args.per_device_train_batch_size,
                'gradient_accumulation': args.gradient_accumulation_steps,
                'dataset_rows': len(trainer.train_dataset),
                'optimizer_state_entries': len(trainer.optimizer.state),
                'scheduler_state': trainer.lr_scheduler.state_dict(),
                'auxiliary': cfg['pilot']['auxiliary'],
                'excluded_auxiliary_token_ids': active['excluded'],
                'start_checkpoint': cfg['pilot']['checkpoint'],
                'diagnostic_gap': 'unclipped KL(teacher||student) from training forward',
                'grading_used_for_loss': False,
            }
            if actual['gradient_accumulation'] != 32 or actual['micro_batch'] != 1:
                raise ValueError('Pilot must preserve the frozen effective batch')
            if not actual['optimizer_state_entries']:
                raise ValueError('Optimizer state was not restored')
            path = Path(args.output_dir) / 'branch_start.json'
            path.write_text(json.dumps(actual, indent=2) + '\n')
            print('SMOOTHNESS_BRANCH_STARTED', json.dumps(actual), flush=True)
            return control

    class SmoothnessPilotTrainer(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active['trainer'] = self
            excluded = {self.processing_class.pad_token_id, self.processing_class.eos_token_id}
            generation_eos = self.generation_config.eos_token_id
            excluded.update([generation_eos] if isinstance(generation_eos, int) else (generation_eos or []))
            active['excluded'] = sorted(x for x in excluded if x is not None)
            self.add_callback(BranchWitness())

    def objective(student, teacher, labels, **kwargs):
        trainer = active['trainer']
        loss, stats = routed_smooth_forward_kl(
            student, teacher, labels, **kwargs, arm=arm,
            excluded_token_ids=active['excluded'],
            **cfg['pilot']['auxiliary'])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite pilot loss')
        if trainer._collecting_train_metrics and trainer.accelerator.is_main_process:
            metrics = trainer.training_metrics
            record = {
                'arm': arm, 'optimizer_step': trainer.state.global_step + 1,
                'session_id': metrics.session_id,
                'sequence_index': metrics.trace_index - labels.shape[0],
                'micro_batch': labels.shape[0],
                'actual_total_loss': float(loss.detach()), **stats,
            }
            with (Path(trainer.args.output_dir) / 'smoothness_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        return loss, stats

    position_supervision.routed_forward_kl = objective
    opsd_trainer.OPSDTrainer = SmoothnessPilotTrainer


if __name__ == '__main__':
    install()
    runpy.run_path(str(ROOT / 'opsd_train.py'), run_name='__main__')
