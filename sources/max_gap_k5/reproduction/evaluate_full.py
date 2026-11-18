"""Durable full Avg@12 eval, official prompt/grader, pinned default splits."""
import argparse,ast,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from math_verify import parse,verify
from vllm import LLM,SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.lora.request import LoRARequest
from evaluation_metrics import summarize_evaluation, export_evaluation_metrics
ROOT=Path(__file__).resolve().parents[1]

def require_idle_gpu():
    if os.environ.get('OPSD_REQUIRE_IDLE_GPU') != '1':
        return
    try:
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        info=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        used,utilization=[int(x.strip()) for x in info.strip().split(',')]
        processes=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        if used < 500 and utilization < 5 and not processes:
            return
        reason=f'used_mib={used}, utilization={utilization}, pids={processes}'
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        reason=str(exc)
    print('GPU_BUSY_REQUEUE',reason,flush=True)
    raise SystemExit(75)

def check_resume_protocol(previous,current,allow_runtime_change=False):
    if previous == current:
        return
    assert allow_runtime_change, 'Resume protocol mismatch; runtime change requires explicit opt-in'
    assert {k:v for k,v in previous.items() if k!='scheduling'} == {k:v for k,v in current.items() if k!='scheduling'}, 'Resume sampling/model/dataset protocol mismatch'

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--dataset',choices=['aime24','aime25','hmmt25'],required=True);p.add_argument('--probe',action='store_true');p.add_argument('--stop_after_questions',type=int);p.add_argument('--gpu-memory-gib',type=float);p.add_argument('--max-num-seqs',type=int,default=12);p.add_argument('--allow-runtime-change',action='store_true');p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--completed-from',type=Path);a=p.parse_args()
assert a.num_shards>=1 and 0<=a.shard_index<a.num_shards
assert a.max_num_seqs>=9, 'Pinned vLLM FlashInfer warmup needs capacity for its 9-request mixed batch'
assert a.gpu_memory_gib is None or a.gpu_memory_gib>1
a.output.parent.mkdir(parents=True,exist_ok=True)
nodes=[x for x in ast.parse((ROOT/'eval/evaluate_math.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name in {'extract_boxed_answer','grade_answer'}]
assert len(nodes)==2
exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_grader','exec'))
assert grade_answer('2','2') and not grade_answer('3','2')
manifest=json.loads((ROOT/'reproduction/full_eval_manifest.json').read_text())
d=load_from_disk(manifest[a.dataset]['path'])
t=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
config={'dataset':manifest[a.dataset],'adapter':str(a.adapter.resolve()) if a.adapter else None,'n':12,'seed':42,'temperature':1.,'top_p':1.,'top_k':-1,'min_p':0.,'presence_penalty':0.,'max_new_tokens':16 if a.probe else 38912,'thinking':True,'probe':a.probe,'backend':'vLLM 0.11.0 FLASHINFER','question_seed':'42 + 12 * row index','scheduling':f'all remaining questions queued; max{a.max_num_seqs} active sequences'}
target=[i for i in range(min(3,len(d)) if a.probe else len(d)) if i%a.num_shards==a.shard_index]
if a.num_shards>1:
    config['scheduling']=f'questions partitioned across {a.num_shards} workers; max{a.max_num_seqs} active sequences per worker'
results={'protocol':config,'rows':[]}
if a.output.exists():
    results=json.loads(a.output.read_text())
    check_resume_protocol(results['protocol'],config,a.allow_runtime_change)
    if results['protocol']!=config:
        results.setdefault('protocol_history',[]).append({'protocol':results['protocol'],'completed_indices':sorted(r['index'] for r in results['rows'])})
        results['protocol']=config
if a.completed_from and a.completed_from.exists():
    previous=json.loads(a.completed_from.read_text())
    check_resume_protocol(previous['protocol'],config,a.allow_runtime_change)
    saved={row['index']:row for row in results['rows']}
    for row in previous['rows']:
        if row['index'] in target:
            if row['index'] in saved: assert saved[row['index']]==row, 'Conflicting saved answer'
            else: results['rows'].append(row);saved[row['index']]=row
    sessions={s['session_id']:s for s in results.get('runtime_sessions',[])}
    for session in previous.get('runtime_sessions',[]): sessions.setdefault(session['session_id'],session)
    results['runtime_sessions']=list(sessions.values())
completed={r['index'] for r in results['rows']}
assert completed<=set(target) and len(completed)==len(results['rows']), 'Invalid shard rows'
if a.num_shards>1:
    results['partition']={'num_shards':a.num_shards,'shard_index':a.shard_index,'target_indices':target}
if all(i in completed for i in target):
    results['summary']=summarize_evaluation(results['rows'],len(target),config['n'],a.probe)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n');tmp.replace(a.output)
    print('EVAL_ALREADY_COMPLETE',a.output);raise SystemExit(0)
kw=dict(model=str(ROOT/'models/Qwen3-1.7B'),dtype='bfloat16',max_model_len=40960,gpu_memory_utilization=.85,tensor_parallel_size=1,enforce_eager=True,max_num_seqs=a.max_num_seqs,max_num_batched_tokens=4096,seed=42)
if a.gpu_memory_gib is not None:
    # Query CUDA in a short-lived process so the evaluator does not initialize
    # CUDA before vLLM forks its worker. Leave 1.5 GiB for runtime overhead.
    free,total=json.loads(subprocess.check_output([sys.executable,'-c','import json,torch; print(json.dumps(torch.cuda.mem_get_info()))'],text=True))
    budget=int(a.gpu_memory_gib*1024**3)
    assert budget<total, 'GPU budget must be smaller than device capacity'
    assert free>=budget+1024**3, 'Insufficient free GPU memory for requested budget plus startup headroom'
    kw['gpu_memory_utilization']=(budget-1536*1024**2)/total
lora=None
if a.adapter:
    assert (a.adapter/'adapter_model.safetensors').is_file(),'Missing adapter; never silently use base'
    kw.update(enable_lora=True,max_lora_rank=64)
    lora=LoRARequest('trained',1,str(a.adapter.resolve()))
require_idle_gpu()
runtime={'session_id':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'gpu_memory_budget_gib':a.gpu_memory_gib,'overhead_reserve_gib':1.5 if a.gpu_memory_gib is not None else 0,'engine_config':kw,'completed_indices_before_resume':sorted(completed),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
runtime.update(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),num_shards=a.num_shards,shard_index=a.shard_index,target_indices=target)
with a.output.with_suffix('.runtime.jsonl').open('a') as journal:
    journal.write(json.dumps(runtime)+'\n')
print('EVAL_RUNTIME',json.dumps(runtime),flush=True)
try:
    engine=LLM(**kw)
except Exception as exc:
    # vLLM can observe a competing allocation after the pre-load check.
    if os.environ.get('OPSD_REQUIRE_IDLE_GPU') == '1' and 'Free memory on device' in str(exc) and 'desired GPU memory utilization' in str(exc):
        print('GPU_BUSY_REQUEUE',str(exc),flush=True)
        raise SystemExit(75)
    raise
try:
    results.setdefault('runtime_sessions',[]).append(runtime)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n');tmp.replace(a.output)
    pending=[i for i in target if i not in completed]
    if a.stop_after_questions: pending=pending[:a.stop_after_questions]
    started={}
    evaluation_started=time.monotonic()
    for i in pending:
        row=d[i]
        text=t.apply_chat_template([{'role':'user','content':row['problem']+'\n\nPlease reason step by step, and put your final answer within \\boxed{}.'}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert len(t.encode(text))+config['max_new_tokens']<=40960
        started[i]=time.monotonic()
        params=SamplingParams(n=config['n'],temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,max_tokens=config['max_new_tokens'],seed=42+12*i,output_kind=RequestOutputKind.FINAL_ONLY)
        engine.llm_engine.add_request(str(i),text,params,lora_request=lora)
    seen=set()
    heartbeat=time.monotonic()
    while engine.llm_engine.has_unfinished_requests():
        for output in engine.llm_engine.step():
            if not output.finished: continue
            i=int(output.request_id)
            assert i in pending and i not in seen
            seen.add(i)
            row=d[i];outputs=output.outputs
            assert len(outputs)==config['n']
            samples=[]
            for o in outputs:
                pred=extract_boxed_answer(o.text)
                samples.append({'sample_index':o.index,'generation':o.text,'token_ids':list(o.token_ids),'prediction':pred,'correct':bool(grade_answer(pred,row['answer'])),'tokens':len(o.token_ids),'finish_reason':o.finish_reason,'stop_reason':o.stop_reason,'truncated':o.finish_reason=='length','has_boxed_answer':pred is not None})
            results['rows'].append({'index':i,'id':row['id'],'problem':row['problem'],'ground_truth':row['answer'],'elapsed_since_queued_seconds':time.monotonic()-started[i],'samples':samples,'runtime_session':runtime['session_id']})
            results['summary']=summarize_evaluation(results['rows'],len(target),config['n'],a.probe,time.monotonic()-evaluation_started)
            tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n');tmp.replace(a.output)
            export_evaluation_metrics(a.output,results)
            print('FULL_EVAL_PROGRESS',a.dataset,i,json.dumps(results['summary']),flush=True)
        if time.monotonic()-heartbeat>=15:
            progress={'dataset':a.dataset,'completed_questions':len(results['rows']),'pending_questions':len(pending)-len(seen),'wall_seconds':time.monotonic()-evaluation_started}
            a.output.with_suffix('.progress.json').write_text(json.dumps(progress)+'\n')
            print('EVAL_HEARTBEAT',json.dumps(progress),flush=True)
            heartbeat=time.monotonic()
    assert seen==set(pending), 'Missing final outputs'
finally:
    engine.llm_engine.engine_core.shutdown()
    print('VLLM_ENGINE_SHUTDOWN',flush=True)
