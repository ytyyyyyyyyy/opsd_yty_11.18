"""Monitor real optimizer updates and stop without shortening the official LR plan."""
import json
import time
from pathlib import Path
import torch
from transformers import TrainerCallback

class StopAndMonitor(TrainerCallback):
    def __init__(self, steps, output_dir):
        if steps < 1:
            raise ValueError("stop_after_steps must be positive")
        self.steps = steps
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.gradient_record = {}

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients:
            raise RuntimeError("No trainable gradients at optimizer step")
        finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all().item())
        maximum = float(torch.stack([g.detach().abs().max().float() for g in gradients]).max().item())
        if not finite or maximum == 0:
            raise RuntimeError(f"Invalid gradients: finite={finite}, max={maximum}")
        self.gradient_record = {"finite_gradients": finite, "gradient_tensors": len(gradients),
                                "max_abs_clipped_gradient": maximum}

    def on_step_end(self, args, state, control, **kwargs):
        record = {"step": state.global_step, "planned_steps": state.max_steps,
                  "micro_batch": args.per_device_train_batch_size,
                  "gradient_accumulation": args.gradient_accumulation_steps,
                  "elapsed_seconds": time.monotonic() - self.started,
                  "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                  **self.gradient_record}
        with (self.output / "optimizer_witness.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print("OPTIMIZER_WITNESS", json.dumps(record), flush=True)
        control.should_log = True
        if state.global_step >= self.steps:
            control.should_training_stop = True
            control.should_save = True
        return control
