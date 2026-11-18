"""CPU-only protocol and real generation-path checks, with a deterministic toy model."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from reproduction.gap_intervention_core import (
    ARMS, KS, atomic_json, candidate_positions, canonical_problem, choose_positions,
    gap_window, parse_arm, prediction_positions, record_validation, sha256,
)
from reproduction.gap_intervention_math import full_vocabulary_gap
from reproduction.position_supervision import routed_forward_kl


class ProtocolTests(unittest.TestCase):
    def test_response_indices_and_shared_candidates(self):
        ids = list(range(10, 25)) + [99]
        positions = candidate_positions(ids, (0, 99))
        gap = [0.] * len(ids)
        gap[2] = 10.
        self.assertEqual(choose_positions(gap, positions, "max_gap", 42)["replacement_start"], 2)
        self.assertEqual(choose_positions(gap, positions, "pre_max_gap", 42)["replacement_start"], 1)
        self.assertEqual(prediction_positions(5, 3), [4, 5, 6])
        self.assertTrue(all(i >= 1 and i + 10 <= 15 for i in positions))
        self.assertEqual(candidate_positions([10] * 10, ()), [])
        self.assertEqual(candidate_positions([10, 11, 99], (99,)), [])
        self.assertEqual(len(ARMS), 13)
        self.assertEqual(len(set(ARMS)), 13)
        self.assertEqual({parse_arm(a)[1] for a in ARMS if a != "opsd"}, set(KS))

    def test_window_excludes_teacher_and_handles_eos(self):
        old, new = [1., 1., 20., 20., 3., 3.], [1., 1., 0., 0., 3., 3.]
        self.assertEqual(gap_window(old, new, 4, 64)["delta"], 0.)
        self.assertEqual(gap_window(old, new[:3], 4, 64)["count"], 0)
        self.assertIsNone(gap_window(old, new[:3], 4, 64)["delta"])

    def test_unclipped_gap_is_the_full_distribution_not_sampled_token_score(self):
        student = torch.tensor([[1., -2., .4], [.3, .8, -1.]])
        teacher = torch.tensor([[-1., 2., .2], [.3, .8, -1.]])
        gap, _ = full_vocabulary_gap(student, teacher, temperature=1.1, chunk_tokens=1)
        expected = F.kl_div(F.log_softmax(student / 1.1, -1),
                            F.softmax(teacher / 1.1, -1), reduction="none").sum(-1)
        self.assertTrue(torch.allclose(torch.tensor(gap), expected.clamp_min(0), atol=1e-6))
        self.assertGreater(gap[0], .05)
        self.assertAlmostEqual(gap[1], 0., places=6)

    def test_original_loss_value_and_gradients_include_replacement_positions(self):
        s = torch.tensor([[[1., -1., .3], [0., 1., -1.], [2., 0., -2.]]], requires_grad=True)
        t = torch.tensor([[[0., 1., .3], [1., -1., 0.], [-1., 0., 2.]]], requires_grad=True)
        labels = torch.tensor([[10, 210, -100]])  # 210 represents a teacher-inserted token
        loss, _ = routed_forward_kl(s, t, labels, mode="all", temperature=1.1,
                                    token_clip=.05, chunk_tokens=1)
        terms = F.kl_div(F.log_softmax(s / 1.1, -1),
                         F.log_softmax(t.detach() / 1.1, -1),
                         reduction="none", log_target=True).clamp(max=.05)
        expected = terms[labels != -100].sum() / 2
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        loss.backward()
        self.assertGreater(s.grad[0, 1].abs().sum().item(), 0.)
        self.assertEqual(s.grad[0, 2].abs().sum().item(), 0.)
        self.assertIsNone(t.grad)

    def test_real_intervention_generation_path_for_every_arm(self):
        import reproduction.gap_intervention_train as intervention
        from transformers import GenerationConfig

        initial = list(range(10, 25)) + [99]

        class Tokenizer:
            all_special_ids = [0, 99]
            def decode(self, ids, **kwargs):
                return " ".join(map(str, ids))

        class Model:
            def __init__(self):
                self.config = SimpleNamespace(use_cache=False)
                self.training = True
                self.teacher = False
                self.calls = []
            def eval(self):
                self.training = False
            def train(self, enabled=True):
                self.training = enabled
            @contextmanager
            def disable_adapter(self):
                previous = self.teacher
                self.teacher = True
                try:
                    yield
                finally:
                    self.teacher = previous
            def generate(self, input_ids, generation_config, **kwargs):
                prefix = input_ids[0].tolist()
                self.calls.append((self.teacher, prefix))
                if self.teacher:
                    output = list(range(210, 210 + generation_config.max_new_tokens))
                elif len(prefix) == 2:
                    output = initial
                else:
                    output = [30, 31, 32, 33, 34, 35, 99][:generation_config.max_new_tokens]
                return torch.cat([input_ids, torch.tensor([output])], dim=1)
            def __call__(self, input_ids, logits_to_keep, **kwargs):
                prompt_length = 4 if input_ids[0, 0].item() == 110 else 2
                logits = torch.zeros(1, len(logits_to_keep), 256)
                for j, position in enumerate(logits_to_keep.tolist()):
                    response_index = position - prompt_length + 1
                    if self.teacher and response_index == 2 and input_ids[0, prompt_length + 1].item() == 11:
                        logits[0, j, 12] = 5.
                return SimpleNamespace(logits=logits)

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            atomic_json(run / "config.json", {"ks": list(KS), "gap": {"chunk_tokens": 2, "horizon": 64}})
            os.environ["GAP_INTERVENTION_RUN"] = str(run)
            outcomes = {}
            for arm in ARMS:
                os.environ["GAP_INTERVENTION_ARM"] = arm
                cls = intervention.install(base_class=object)
                trainer = cls.__new__(cls)
                trainer.args = SimpleNamespace(seed=42)
                trainer.state = SimpleNamespace(global_step=0)
                trainer.processing_class = Tokenizer()
                trainer.temperature = 1.1
                trainer.training_metrics = None
                trainer.gap_events = run / f"{arm}.jsonl"
                model = Model()
                batch = {"student_prompts": torch.tensor([[100, 101]]),
                         "teacher_prompts": torch.tensor([[110, 111, 112, 113]]),
                         "gap_problem": ["Toy prefix problem"], "gap_reference": [r"\boxed{1}"]}
                settings = GenerationConfig(max_new_tokens=32, eos_token_id=99, pad_token_id=0)
                rng = torch.get_rng_state().clone()
                seq, mask, labels = trainer.generate_on_policy_outputs(model, batch, settings, 0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(model.training)
                self.assertFalse(model.teacher)
                self.assertFalse(model.config.use_cache)
                row = json.loads(trainer.gap_events.read_text())
                response = seq[0, 2:].tolist()
                strategy, k = parse_arm(arm)
                if arm == "opsd":
                    self.assertEqual(response, initial)
                else:
                    start = row["replacement_start"]
                    if strategy != "random":
                        self.assertEqual(start, 1 if strategy == "pre_max_gap" else 2)
                    self.assertEqual(response[:start], initial[:start])
                    self.assertEqual(response[start:start+k], list(range(210, 210+k)))
                    self.assertTrue(bool((labels[0, 2+start:2+start+k] != -100).all()))
                    teacher_call = next(prefix for is_teacher, prefix in model.calls if is_teacher)
                    self.assertEqual(teacher_call, [110, 111, 112, 113] + initial[:start])
                    if strategy != "random":
                        self.assertEqual(row["common_window"]["start"], 2+k)
                    self.assertEqual(row["actual_k"], k)
                outcomes[arm] = row
            self.assertGreater(outcomes["max_gap_k1"]["peak_gap_modified"], 0.)
            self.assertEqual(outcomes["pre_max_gap_k1"]["peak_gap_modified"], 0.)

            class EarlyEOSModel(Model):
                def generate(self, input_ids, generation_config, **kwargs):
                    if self.teacher:
                        self.calls.append((True, input_ids[0].tolist()))
                        return torch.cat([input_ids, torch.tensor([[99]])], dim=1)
                    return super().generate(input_ids, generation_config, **kwargs)

            os.environ["GAP_INTERVENTION_ARM"] = "pre_max_gap_k3"
            cls = intervention.install(base_class=object)
            eos_trainer = cls.__new__(cls)
            eos_trainer.__dict__.update(trainer.__dict__)
            eos_trainer.gap_events = run / "early_eos.jsonl"
            model = EarlyEOSModel()
            seq, _, _ = eos_trainer.generate_on_policy_outputs(model, batch, settings, 0)
            event = json.loads(eos_trainer.gap_events.read_text())
            self.assertEqual(seq[0, 2:].tolist(), [10, 99])
            self.assertEqual(event["actual_k"], 1)
            self.assertTrue(event["teacher_ended_early"])
            self.assertEqual(event["common_window"]["count"], 0)
            self.assertIsNone(event["common_window"]["delta"])
            self.assertEqual(len(model.calls), 2)  # no continuation after teacher EOS

    def test_queue_validates_each_five_updates_without_launching_a_process(self):
        import signal
        import reproduction.run_gap_intervention as queue
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            atomic_json(run / "protocol.json", {"synthetic": True})
            atomic_json(run / "preflight.json", {"passed": True,
                        "protocol_sha256": sha256(run / "protocol.json")})
            order = []

            def fake_child(run, config, label, command, arm):
                target = command[1]
                checkpoint = run / "training" / arm / f"checkpoint-{target}"
                checkpoint.mkdir(parents=True)
                for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth",
                             "adapter_model.safetensors", "adapter_config.json"):
                    (checkpoint / name).write_text("synthetic CPU test only")
                atomic_json(checkpoint / "trainer_state.json", {"global_step": target})
                order.append(("train", arm, target))
                return 0

            def fake_validate(run, config, arm, step):
                path = run / "validation_points.json"
                points = json.loads(path.read_text()) if path.exists() else []
                points.append({"arm": arm, "step": step})
                atomic_json(path, points)
                order.append(("validate", arm, step))
                if len(points) == 14:
                    (run / "STOP").write_text("synthetic manual stop")

            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                with patch.object(queue, "verify_prepared", return_value={"validation": {"every_optimizer_updates": 5}}), \
                     patch.object(queue, "train_command", side_effect=lambda run, arm, target, previous: [arm, target]), \
                     patch.object(queue, "run_child", side_effect=fake_child), \
                     patch.object(queue, "validate", side_effect=fake_validate):
                    queue.execute(run)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            expected = [("validate", "base", 0)]
            for arm in ARMS:
                expected.extend([("train", arm, 5), ("validate", arm, 5)])
            self.assertEqual(order, expected)
            self.assertEqual(json.loads((run / "status.json").read_text())["stage"], "manually_stopped")

    def test_validation_and_plot_pipeline_rejects_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [{"index": i, "samples": [{"correct": j % 2 == 0} for j in range(12)]}
                    for i in range(30)]
            value = {"protocol": {"n": 12, "adapter": None}, "rows": rows,
                     "summary": {"complete": True, "probe": False, "questions": 30,
                                 "samples": 360, "correct": 180,
                                 "truncated_rate": 0., "generated_tokens": 3600}}
            path = run / "synthetic_validation.json"
            atomic_json(path, value)
            point = record_validation(run, "base", 0, path)
            self.assertEqual(point["accuracy"], .5)
            self.assertGreater((run / "validation_accuracy.png").stat().st_size, 1000)
            self.assertTrue((run / "validation_accuracy.csv").is_file())
            record_validation(run, "base", 0, path)
            self.assertEqual(len(json.loads((run / "validation_points.json").read_text())), 1)
            value["summary"]["complete"] = False
            atomic_json(path, value)
            with self.assertRaises(ValueError):
                record_validation(run, "base", 0, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("Preflight must run with CUDA_VISIBLE_DEVICES=-1")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    report = {"passed": result.wasSuccessful(), "cpu_only": True, "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "gpu_model_loaded": False, "training_started": False,
              "validation_generation_started": False}
    if args.run_dir and result.wasSuccessful():
        from datasets import load_from_disk
        training = load_from_disk(str(args.run_dir / "train_dataset"))
        validation = load_from_disk(str(args.run_dir / "validation_dataset"))
        heldout = {canonical_problem(row["problem"]) for row in validation}
        overlap = sum(canonical_problem(row["problem"]) in heldout for row in training)
        if overlap or len(validation) != 30:
            raise ValueError("Prepared training/validation overlap or wrong validation size")
        report.update(training_rows=len(training), validation_rows=len(validation), normalized_exact_overlap=overlap)
        report["protocol_sha256"] = sha256(args.run_dir / "protocol.json")
    if args.run_dir:
        atomic_json(args.run_dir / "preflight.json", report)
    print("CPU_PREFLIGHT", json.dumps(report), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
