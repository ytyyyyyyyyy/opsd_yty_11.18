"""Resumable long-budget AIME25 evaluation with a bounded question queue.

Run once per adapter with identical sampling arguments. The primary score is
completed, strictly graded correct answers / all 60 scheduled generations.
Truncations and grading failures are separate categories, never incorrect labels.
The archived official grader is also recorded for compatibility, not used to
silently replace strict grades. All four checkpoints must use the same prompt,
seed, token cap, and queue settings. Use a new output directory for each seed
round. Default: 38912 tokens, n=2, one active question (two output sequences).
The engine retains >=9 sequence slots for pinned FlashInfer warmup, but only
max_active_questions * 2 sequences can be submitted for actual generation.
This small n=2 evaluation is not the full Avg@12 benchmark.
"""
import argparse
import ast
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


class QuestionWindow:
    """Track bounded parent requests; each parent owns both n=2 samples.

    A slot is released only after the engine returns all final samples for that
    question and the caller has handled them. No full-dataset engine backlog is
    created, even though max_num_seqs remains large enough for engine warmup.
    """
    def __init__(self, questions, limit):
        questions = list(questions)
        if limit < 1 or len({q['index'] for q in questions}) != len(questions):
            raise ValueError('Positive queue limit and unique question indices required')
        self.waiting = deque(questions)
        self.limit = limit
        self.active = set()
        self.completed = set()
        self.submitted = 0
        self.peak_active = 0

    def fill(self, submit):
        while self.waiting and len(self.active) < self.limit:
            question = self.waiting[0]
            submit(question)
            self.waiting.popleft()
            self.active.add(question['index'])
            self.submitted += 1
            self.peak_active = max(self.peak_active, len(self.active))

    def complete(self, index):
        if index not in self.active:
            raise ValueError(f'Unexpected or repeated completed request: {index}')
        self.active.remove(index)
        self.completed.add(index)

    @property
    def finished(self):
        return not self.waiting and not self.active

    def status(self):
        return {'active_question_indices': sorted(self.active),
                'waiting_questions': len(self.waiting),
                'submitted_questions_this_session': self.submitted,
                'completed_questions_this_session': len(self.completed),
                'max_active_questions_observed': self.peak_active}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_grader(source_dir):
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    import sympy

    source = Path(source_dir) / 'eval/evaluate_math.py'
    names = {'extract_boxed_answer', 'grade_answer'}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError(f'Missing official grader functions: {source}')
    namespace = {'parse': parse, 'verify': verify}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return {'extract': namespace['extract_boxed_answer'], 'official': namespace['grade_answer'],
            'parse': parse, 'verify': verify, 'TimeoutException': TimeoutException,
            'Expr': sympy.Expr, 'source_sha256': file_hash(source)}


def parse_numeric(answer, grader):
    text = str(answer)
    wrapped = text if '$' in text else f'${text}$'
    parsed = grader['parse'](wrapped, fallback_mode='no_fallback',
                             parsing_timeout=5, raise_on_error=True)
    if not parsed:
        return None, 'unparsed'
    if (len(parsed) != 1 or not isinstance(parsed[0], grader['Expr'])
            or parsed[0].free_symbols or parsed[0].is_number is not True
            or parsed[0].is_real is not True or parsed[0].is_finite is not True):
        return None, 'non_numeric'
    return parsed, None


def grade_sample(candidate, answer, gold, grader):
    """Keep generation stopping, strict grading, and official matching separate."""
    predicted = grader['extract'](candidate.text)
    sample = {'sample_index': candidate.index, 'generation': candidate.text,
              'token_ids': list(candidate.token_ids), 'tokens': len(candidate.token_ids),
              'prediction': predicted, 'has_boxed_answer': predicted is not None,
              'finish_reason': candidate.finish_reason, 'stop_reason': candidate.stop_reason,
              'truncated': candidate.finish_reason == 'length',
              'official_answer_match': None, 'official_grade_error': None,
              'answer_match': None, 'answer_status': None, 'grading_error': None}
    exceptions = (Exception, grader['TimeoutException'])
    try:
        sample['official_answer_match'] = bool(grader['official'](predicted, answer))
    except exceptions as exc:
        sample['official_grade_error'] = f'{type(exc).__name__}: {exc}'
    if predicted is None:
        sample['answer_status'] = 'missing_box'
    else:
        try:
            parsed, reason = parse_numeric(predicted, grader)
            if reason:
                sample['answer_status'] = reason
            else:
                sample['parsed_prediction'] = str(parsed[0])
                match = bool(grader['verify'](gold, parsed, timeout_seconds=5, raise_on_error=True))
                sample['answer_match'] = match
                sample['answer_status'] = 'correct' if match else 'incorrect'
        except exceptions as exc:
            sample['answer_status'] = 'grading_error'
            sample['grading_error'] = f'{type(exc).__name__}: {exc}'
    if sample['truncated']:
        sample['status'] = 'truncated'
    elif candidate.finish_reason != 'stop':
        sample['status'] = 'unknown_finish'
    else:
        sample['status'] = sample['answer_status']
    # None means not adjudicated as a completed mathematical answer.
    sample['correct'] = (sample['status'] == 'correct'
                         if sample['status'] in ('correct', 'incorrect') else None)
    sample['strict_correct'] = sample['status'] == 'correct'
    return sample


def summarize(rows, question_count, n):
    samples = [sample for row in rows for sample in row['samples']]
    counts = Counter(sample['status'] for sample in samples)
    expected = question_count * n
    graded = counts['correct'] + counts['incorrect']
    complete = len(rows) == question_count and len(samples) == expected
    official_available = sum(s['official_answer_match'] is not None for s in samples)
    official_correct = sum(s['official_answer_match'] is True for s in samples)
    return {'complete': complete, 'completed_questions': len(rows),
            'planned_questions': question_count, 'observed_generations': len(samples),
            'planned_generations': expected, 'strict_correct': counts['correct'],
            'completed_incorrect': counts['incorrect'], 'truncated': counts['truncated'],
            'ungraded_or_unknown_finish': len(samples) - graded - counts['truncated'],
            'status_counts': dict(sorted(counts.items())),
            'answer_status_counts_including_truncated': dict(sorted(Counter(
                s['answer_status'] for s in samples).items())),
            'accuracy': counts['correct'] / expected if complete else None,
            'strict_correct_over_planned': counts['correct'] / expected,
            'strict_accuracy_observed': counts['correct'] / len(samples) if samples else None,
            'completed_gradable_accuracy': counts['correct'] / graded if graded else None,
            'official_answer_matches_including_truncated': official_correct,
            'official_grade_errors': len(samples) - official_available,
            'official_match_rate_all': (official_correct / expected
                                        if complete and official_available == expected else None),
            'truncated_with_matching_boxed_answer': sum(
                s['truncated'] and s['answer_match'] is True for s in samples),
            'questions_with_at_least_one_strict_correct': sum(
                any(s['strict_correct'] for s in row['samples']) for row in rows),
            'generated_tokens': sum(s['tokens'] for s in samples),
            'interpretation': 'accuracy = completed strict correct / all planned samples; '
                              'non-completions receive no credit but are not incorrect labels. '
                              'Conditional accuracy is diagnostic only; n=2 remains a small evaluation.'}


def prepare(args):
    """CPU-only data, reference, prompt, adapter, and protocol validation."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    if args.n != 2 or not 8192 <= args.max_new_tokens <= 38912 or args.max_num_seqs < 9:
        raise ValueError('Use n=2, 8192..38912 tokens, and max-num-seqs >= 9 for pinned FlashInfer warmup')
    if args.max_active_questions < 1 or args.max_active_questions * args.n > args.max_num_seqs:
        raise ValueError('Active questions must be positive and active_questions * n <= max_num_seqs')
    if not 0 <= args.seed < 2**32 - 2 * 30:
        raise ValueError('Seed must be nonnegative and leave space for all question seeds')
    if args.gpu_memory_gib <= 2:
        raise ValueError('GPU budget must exceed 2 GiB')
    adapter_config_path = args.adapter / 'adapter_config.json'
    weights_path = args.adapter / 'adapter_model.safetensors'
    if not weights_path.is_file() or not adapter_config_path.is_file():
        raise ValueError('Missing adapter config/weights; never silently evaluate the base model')
    adapter_config = json.loads(adapter_config_path.read_text())
    if int(adapter_config.get('r', 0)) > 64:
        raise ValueError('This evaluation supports the existing rank-64 adapters only')
    dataset = load_from_disk(str(args.dataset))
    if len(dataset) != 30 or not {'problem', 'answer'}.issubset(dataset.column_names):
        raise ValueError('Expected the complete local 30-question AIME25 dataset')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    grader = load_grader(args.source_dir)
    questions, gold = [], {}
    instruction = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    for index, row in enumerate(dataset):
        answer = str(row['answer'])
        parsed, reason = parse_numeric(answer, grader)
        if reason or parsed[0].is_integer is not True or not 0 <= int(parsed[0]) <= 999:
            raise ValueError(f'Invalid AIME reference at row {index}: {answer!r}, {reason}')
        gold[index] = parsed
        messages = [{'role': 'user', 'content': row['problem'] + instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        questions.append({'index': index, 'id': str(row.get('id', index)),
                          'problem': row['problem'], 'ground_truth': answer, 'messages': messages,
                          'prompt': prompt, 'prompt_token_ids': ids, 'prompt_tokens': len(ids),
                          'seed': args.seed + args.n * index})
    if len({q['problem'] for q in questions}) != 30:
        raise ValueError('Duplicate holdout problems')
    context = int(math.ceil((max(q['prompt_tokens'] for q in questions)
                             + args.max_new_tokens) / 256) * 256)
    model_identity = {name: file_hash(args.model / name) for name in
                      ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json')
                      if (args.model / name).exists()}
    dependencies = {}
    for package in ('vllm', 'torch', 'transformers', 'datasets', 'math-verify', 'sympy'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    protocol = {'schema_version': 1, 'label': args.label, 'model': str(args.model.resolve()),
                'model_config_tokenizer_sha256': model_identity,
                'adapter': str(args.adapter.resolve()), 'adapter_sha256': file_hash(weights_path),
                'adapter_config_sha256': file_hash(adapter_config_path),
                'dataset': str(args.dataset.resolve()), 'dataset_name': 'aime25',
                'dataset_fingerprint': dataset._fingerprint,
                'questions_sha256': canonical_hash(questions), 'question_count': 30,
                'source_dir': str(args.source_dir.resolve()),
                'official_grader_sha256': grader['source_sha256'],
                'evaluator_sha256': file_hash(Path(__file__)), 'dependencies': dependencies,
                'n': args.n, 'seed': args.seed, 'question_seed': 'seed + n * dataset_row_index',
                'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0,
                'presence_penalty': 0.0, 'max_new_tokens': args.max_new_tokens,
                'thinking': True, 'ignore_eos': False, 'custom_stop_strings': [],
                'max_model_len': context, 'max_num_seqs': args.max_num_seqs,
                'max_active_questions': args.max_active_questions,
                'max_active_output_sequences': args.max_active_questions * args.n,
                'scheduling': 'Bounded parent-question queue; refill only after all n final samples '
                              'for a question are graded and persisted; no full-dataset engine backlog',
                'seed_round_policy': 'One seed per independent round; keep identical across all four '
                                     'checkpoints and use a separate output directory for each round',
                'max_num_batched_tokens': 2048, 'gpu_memory_budget_gib': args.gpu_memory_gib,
                'gpu_overhead_reserve_gib': 1.5, 'dtype': 'bfloat16',
                'strict_grade': 'last boxed answer; numeric finite real scalar; math_verify '
                                'no fallback, raise_on_error; length/unknown stops never credited',
                'official_grade': 'archived evaluate_math.py grade_answer; exceptions including '
                                  'TimeoutException are separately recorded',
                'score': 'completed strict correct / 60 (with n=2); no credit for ungraded or '
                         'truncated, without relabeling these as mathematical errors',
                'scope': 'Long-budget n=2 fixed-holdout recheck; not Avg@12 and not a significance claim'}
    return protocol, questions, gold, grader


def gpu_memory_fraction(args):
    """Inspect the selected GPU in subprocesses before vLLM creates its worker."""
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(selected) != 1 or not selected[0].strip():
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    gpu = selected[0].strip()
    if os.environ.get('OPSD_REQUIRE_IDLE_GPU') == '1':
        info = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
        used, utilization = [int(x.strip()) for x in info.strip().split(',')]
        processes = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
        if used >= 500 or utilization >= 5 or processes:
            print('GPU_BUSY_REQUEUE', info.strip(), processes, flush=True)
            raise SystemExit(75)
    free, total = json.loads(subprocess.check_output([sys.executable, '-c',
        'import json,torch; print(json.dumps(torch.cuda.mem_get_info()))'], text=True))
    budget = int(args.gpu_memory_gib * 1024**3)
    if budget >= total or free < budget + 1024**3:
        print('GPU_BUSY_REQUEUE insufficient free memory', free, budget, total, flush=True)
        raise SystemExit(75)
    return (budget - int(1.5 * 1024**3)) / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'adapter', 'dataset', 'output-dir', 'source-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--label', default='checkpoint')
    parser.add_argument('--n', type=int, choices=(2,), default=2)
    parser.add_argument('--max-new-tokens', type=int, default=38912,
                        help='8192..38912; keep the same cap across all checkpoints in a round')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-memory-gib', type=float, default=20.0)
    parser.add_argument('--max-num-seqs', type=int, default=12)
    parser.add_argument('--max-active-questions', type=int, default=1,
                        help='Actual parent requests admitted to vLLM; each owns two sequences. Default 1.')
    parser.add_argument('--prepare-only', action='store_true', help='Validate/save protocol and prompts on CPU; never import vLLM')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    protocol, questions, gold, grader = prepare(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / 'results.json'
    result = {'protocol': protocol, 'questions': questions, 'rows': [], 'runtime_sessions': []}
    if output_path.exists():
        result = json.loads(output_path.read_text())
        if result['protocol'] != protocol or result['questions'] != questions:
            raise ValueError('Resume protocol, checkpoint, evaluator, dependencies, or prompt mismatch')
    done = {row['index'] for row in result['rows']}
    if len(done) != len(result['rows']) or not done.issubset(range(len(questions))):
        raise ValueError('Duplicate or invalid saved question rows')
    for row in result['rows']:
        if len(row['samples']) != args.n or {s['sample_index'] for s in row['samples']} != set(range(args.n)):
            raise ValueError('Incomplete/duplicate saved samples')

    def save():
        result['rows'].sort(key=lambda row: row['index'])
        result['summary'] = summarize(result['rows'], len(questions), args.n)
        result['complete'] = result['summary']['complete']
        save_json(output_path, result)
        save_json(args.output_dir / 'progress.json', {'label': args.label, **result['summary']})

    save()
    if args.prepare_only or result['complete']:
        print('EVAL_PREPARED' if args.prepare_only else 'EVAL_ALREADY_COMPLETE', output_path, flush=True)
        return
    fraction = gpu_memory_fraction(args)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind

    kw = {'model': str(args.model.resolve()), 'dtype': 'bfloat16',
          'max_model_len': protocol['max_model_len'], 'gpu_memory_utilization': fraction,
          'tensor_parallel_size': 1, 'enforce_eager': True, 'max_num_seqs': args.max_num_seqs,
          'max_num_batched_tokens': protocol['max_num_batched_tokens'],
          'enable_chunked_prefill': True, 'enable_lora': True, 'max_lora_rank': 64,
          'max_loras': 1, 'seed': args.seed}
    runtime = {'session_id': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'vllm_attention_backend': os.environ.get('VLLM_ATTENTION_BACKEND'),
               'engine_config': kw, 'completed_indices_before_resume': sorted(done),
               'max_active_questions': args.max_active_questions,
               'max_active_output_sequences': args.max_active_questions * args.n}
    result['runtime_sessions'].append(runtime)
    save()
    print('EVAL_RUNTIME', json.dumps(runtime), flush=True)
    engine = None
    start = time.monotonic()
    try:
        try:
            engine = LLM(**kw)
        except Exception as exc:
            if ('Free memory on device' in str(exc)
                    and 'desired GPU memory utilization' in str(exc)):
                print('GPU_BUSY_REQUEUE', str(exc), flush=True)
                raise SystemExit(75) from exc
            raise
        runtime['engine_initialization_seconds'] = time.monotonic() - start
        lora = LoRARequest('smoothness_long', 1, str(args.adapter.resolve()))
        pending = {q['index']: q for q in questions if q['index'] not in done}
        window = QuestionWindow(pending.values(), args.max_active_questions)
        queued_at = {}

        def submit(question):
            params = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, top_k=-1,
                min_p=0.0, presence_penalty=0.0, max_tokens=args.max_new_tokens,
                seed=question['seed'], ignore_eos=False, output_kind=RequestOutputKind.FINAL_ONLY)
            engine.llm_engine.add_request(str(question['index']),
                {'prompt_token_ids': question['prompt_token_ids']}, params, lora_request=lora)
            queued_at[question['index']] = time.monotonic()

        window.fill(submit)
        runtime.update(window.status())
        save()
        heartbeat = time.monotonic()
        while not window.finished:
            if not engine.llm_engine.has_unfinished_requests():
                raise ValueError('Engine stopped with active question requests still incomplete')
            for output in engine.llm_engine.step():
                if not output.finished:
                    continue
                index = int(output.request_id)
                if index not in pending or index in done or index not in window.active:
                    raise ValueError('Unexpected or repeated request output')
                question = pending[index]
                samples = [grade_sample(candidate, question['ground_truth'], gold[index], grader)
                           for candidate in sorted(output.outputs, key=lambda item: item.index)]
                if len(samples) != args.n or {s['sample_index'] for s in samples} != set(range(args.n)):
                    raise ValueError('Missing or repeated generated samples')
                stats = summarize([{'samples': samples}], 1, args.n)
                result['rows'].append({'index': index, 'id': question['id'],
                    'problem': question['problem'], 'ground_truth': question['ground_truth'],
                    'samples': samples, 'question_statistics': stats,
                    'elapsed_since_queued_seconds': time.monotonic() - queued_at[index],
                    'runtime_session': runtime['session_id']})
                done.add(index)
                window.complete(index)
                runtime['elapsed_seconds'] = time.monotonic() - start
                runtime.update(window.status())
                save()
                print('LONG_EVAL_PROGRESS', args.label, index, json.dumps(result['summary']), flush=True)
            window.fill(submit)
            runtime.update(window.status())
            if time.monotonic() - heartbeat >= 15:
                runtime['elapsed_seconds'] = time.monotonic() - start
                save_json(args.output_dir / 'progress.json', {
                    'label': args.label, **result['summary'], **window.status(),
                    'elapsed_seconds': runtime['elapsed_seconds']})
                print('EVAL_HEARTBEAT', args.label, len(done), '/ 30',
                      json.dumps(window.status()), flush=True)
                heartbeat = time.monotonic()
        if len(done) != len(questions):
            raise ValueError('Engine stopped before all questions completed')
        runtime['elapsed_seconds'] = time.monotonic() - start
        save()
        print('LONG_EVAL_COMPLETE', args.label, json.dumps(result['summary']), flush=True)
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            print('VLLM_ENGINE_SHUTDOWN', flush=True)


if __name__ == '__main__':
    main()
