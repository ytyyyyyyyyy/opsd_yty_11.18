"""Frozen-policy diagnostic rollouts on a fixed full-solution question pool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.evaluate_smoothness_long import (
    QuestionWindow, file_hash, canonical_hash, save_json, load_grader,
    parse_numeric, grade_sample, summarize, gpu_memory_fraction,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'adapter', 'dataset', 'questions-json', 'source-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--round-index', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=16384)
    p.add_argument('--max-num-seqs', type=int, default=12)
    p.add_argument('--max-active-questions', type=int, default=2)
    p.add_argument('--gpu-memory-gib', type=float, default=20.)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    from datasets import load_from_disk
    dataset = load_from_disk(str(a.dataset))
    questions = json.loads(a.questions_json.read_text())
    if a.n != 2 or len(questions) != 64 or len(dataset) != 64 or a.max_new_tokens != 16384:
        raise ValueError('Diagnostic protocol requires 64 fixed questions, n=2, cap=16384')
    grader = load_grader(a.source_dir)
    gold = {}
    for i, q in enumerate(questions):
        if q['index'] != i or q['problem'] != dataset[i]['problem'] or q['ground_truth'] != dataset[i]['answer']:
            raise ValueError('Question/dataset alignment changed')
        gold[i], error = parse_numeric(q['ground_truth'], grader)
        if error:
            raise ValueError(f'Unparseable frozen reference: {i}')
        if max(len(q['prompt_token_ids']), len(q['teacher_prompt_token_ids'])) + a.max_new_tokens > 40960:
            raise ValueError(f'Context budget overflow: {i}')
    context = ((max(len(q['prompt_token_ids']) for q in questions) + a.max_new_tokens + 255) // 256) * 256
    protocol = {
        'schema_version': 1, 'label': a.label, 'round_index': a.round_index,
        'model': str(a.model.resolve()), 'adapter': str(a.adapter.resolve()),
        'adapter_sha256': file_hash(a.adapter / 'adapter_model.safetensors'),
        'adapter_config_sha256': file_hash(a.adapter / 'adapter_config.json'),
        'dataset': str(a.dataset.resolve()), 'dataset_fingerprint': dataset._fingerprint,
        'questions_sha256': canonical_hash(questions), 'n': a.n, 'seed': a.seed,
        'question_seed': 'seed + n * fixed_question_index',
        'max_new_tokens': a.max_new_tokens, 'temperature': 1.1, 'top_p': .95, 'top_k': 20,
        'student_thinking': False, 'teacher_thinking': True,
        'teacher_reference': 'Full original reference solution, not answer-only',
        'max_model_len': context, 'max_num_seqs': a.max_num_seqs,
        'max_active_questions': a.max_active_questions,
        'gpu_memory_budget_gib': a.gpu_memory_gib,
        'source_sha256': file_hash(Path(__file__)), 'grader_sha256': grader['source_sha256'],
        'scope': 'Exploratory fixed-pool trajectory diagnostics; outcome quotas are not an accuracy benchmark',
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    path = a.output_dir / 'results.json'
    result = {'protocol': protocol, 'questions': questions, 'rows': [], 'runtime_sessions': []}
    if path.exists():
        result = json.loads(path.read_text())
        if result['protocol'] != protocol or result['questions'] != questions:
            raise ValueError('Frozen diagnostic generation protocol changed')
    done = {r['index'] for r in result['rows']}
    if len(done) != len(result['rows']):
        raise ValueError('Duplicate saved diagnostic question')
    for r in result['rows']:
        if len(r['samples']) != a.n or {s['sample_index'] for s in r['samples']} != set(range(a.n)):
            raise ValueError('Incomplete saved question')

    def save():
        result['rows'].sort(key=lambda r: r['index'])
        result['summary'] = summarize(result['rows'], len(questions), a.n)
        result['complete'] = result['summary']['complete']
        save_json(path, result)
        save_json(a.output_dir / 'progress.json', {'label': a.label, **result['summary']})

    save()
    if a.prepare_only or result['complete']:
        print('DIAGNOSTIC_PREPARED_OR_COMPLETE', path, flush=True)
        return
    fraction = gpu_memory_fraction(a)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    config = dict(model=str(a.model), dtype='bfloat16', max_model_len=context,
                  gpu_memory_utilization=fraction, tensor_parallel_size=1,
                  enforce_eager=True, max_num_seqs=a.max_num_seqs,
                  max_num_batched_tokens=2048, enable_chunked_prefill=True,
                  enable_lora=True, max_lora_rank=64, max_loras=1, seed=a.seed)
    runtime = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'engine_config': config, 'completed_before': sorted(done)}
    result['runtime_sessions'].append(runtime)
    save()
    engine = None
    started = time.monotonic()
    try:
        engine = LLM(**config)
        adapter = LoRARequest('smoothness_diagnostic', 1, str(a.adapter.resolve()))
        window = QuestionWindow([q for q in questions if q['index'] not in done], a.max_active_questions)

        def submit(q):
            params = SamplingParams(n=a.n, temperature=1.1, top_p=.95, top_k=20,
                                    max_tokens=a.max_new_tokens, seed=a.seed + a.n * q['index'],
                                    output_kind=RequestOutputKind.FINAL_ONLY)
            engine.llm_engine.add_request(str(q['index']), {'prompt_token_ids': q['prompt_token_ids']},
                                          params, lora_request=adapter)

        window.fill(submit)
        heartbeat = time.monotonic()
        while engine.llm_engine.has_unfinished_requests():
            for output in engine.llm_engine.step():
                if not output.finished:
                    continue
                i = int(output.request_id)
                if i in done:
                    raise ValueError('Repeated diagnostic output')
                q = questions[i]
                samples = [grade_sample(c, q['ground_truth'], gold[i], grader)
                           for c in sorted(output.outputs, key=lambda c: c.index)]
                if len(samples) != a.n:
                    raise ValueError('Missing diagnostic samples')
                result['rows'].append({'index': i, 'id': str(q['id']), 'problem': q['problem'],
                                       'ground_truth': q['ground_truth'], 'samples': samples,
                                       'runtime_session': runtime['started_at_utc']})
                done.add(i)
                window.complete(i)
                save()
                print('DIAGNOSTIC_PROGRESS', a.label, a.round_index, len(done), '/64', flush=True)
            window.fill(submit)
            if time.monotonic() - heartbeat > 30:
                runtime['elapsed_seconds'] = time.monotonic() - started
                save()
                print('DIAGNOSTIC_HEARTBEAT', window.status(), flush=True)
                heartbeat = time.monotonic()
        if len(done) != 64 or not window.finished:
            raise ValueError('Diagnostic queue stopped early')
        runtime['elapsed_seconds'] = time.monotonic() - started
        save()
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
