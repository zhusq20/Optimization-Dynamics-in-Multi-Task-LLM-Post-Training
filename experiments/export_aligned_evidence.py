"""Freeze existing aligned evidence; no generation, training, or source mutation.

Run with --source /path/to/slime_opd_geometry. Plotting uses only the frozen bundle.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'experiments/aligned_evidence_20260910'
DATASETS = {'Math': 'math500_pass1', 'Code': 'livecodebench_postcutoff_pass1',
            'IF': 'ifbench_strict', 'GPQA': 'gpqa_diamond_avg4'}
COUNTS = {'Math': 500, 'Code': 128, 'IF': 300, 'GPQA': 792}
RUNS = {
    'M-PG': 'outputs/mopd_qwen3_aligned_m_pg_20260909/m-pg-s42-g123',
    'M-I64-DR': 'outputs/mopd_qwen3_aligned_20260909_sn4622128200/m-intersection64-dr-s42',
    'M-I64-DT': 'outputs/mopd_qwen3_aligned_20260909_sn4622128200/m-intersection64-dt-s42',
    'M-I64-GT': 'outputs/mopd_qwen3_aligned_20260909_sn4622128200/m-intersection64-gt-s42',
    'Initial': 'outputs/mopd_qwen3_aligned_capability_20260909/initial_student',
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--capability-only', action='store_true',
                    help='Add newly completed suites while preserving frozen training/mechanism records.')
    args = ap.parse_args(); src = args.source.resolve()
    OUT.mkdir(exist_ok=True); (OUT / 'raw').mkdir(exist_ok=True)
    previous = json.loads((OUT/'manifest.json').read_text()) if args.capability_only else None
    manifest = list(previous['sources']) if previous else []

    def freeze(rel, name=None):
        path = src / rel; data = path.read_bytes()
        name = name or path.name
        target = OUT / 'raw' / name
        target.write_bytes(data)
        manifest[:] = [r for r in manifest if r['source'] != rel]
        manifest.append({'source': rel, 'frozen': str(target.relative_to(OUT)),
                         'sha256': sha(data), 'bytes': len(data)})
        return data

    gpqa_spec = importlib.util.spec_from_file_location('paper_gpqa', src / 'slime/rollout/rm_hub/gpqa.py')
    gpqa = importlib.util.module_from_spec(gpqa_spec); gpqa_spec.loader.exec_module(gpqa)
    freeze('slime/rollout/rm_hub/gpqa.py', 'gpqa_scorer.py')
    suites = json.loads((OUT/'capability.json').read_text()) if previous else []
    prompt_rows = [json.loads(l) for l in (OUT/'prompt_scores.jsonl').read_text().splitlines()] if previous else []
    old_audit = json.loads((OUT/'gpqa_score_audit.json').read_text()) if previous else {'responses':0,'disagreements':[]}
    assert not old_audit['disagreements']
    gpqa_audit = []
    eval_runs = {
        **RUNS,
        'S-PG': 'outputs/mopd_qwen3_aligned_capability_20260910_sn4622128200/s-pg-s42',
        'S-I64': 'outputs/mopd_qwen3_aligned_capability_20260910_sn4622128200/s-intersection64-s42',
    }
    for model, rel in eval_runs.items():
        for marker in sorted((src / rel / 'capability_eval').glob('step_*/run_complete.json')):
            base = marker.parent; step = int(base.name.split('_')[1]); tag = f'{model}_{step}'
            if any(r['model']==model and r['step']==step for r in suites): continue
            complete = json.loads(freeze(str(marker.relative_to(src)), tag + '_complete.json'))
            assert complete['status'] == 'complete' and complete['final_num_updates'] == step
            for name in ['job.json', 'verified_summary.json']:
                if (base / name).is_file():
                    freeze(str((base / name).relative_to(src)), tag + '_' + name)
            metrics = {}
            for line in freeze(str((base / 'metrics/eval.jsonl').relative_to(src)), tag + '_eval.jsonl').splitlines():
                metrics.update(json.loads(line)['metrics'])
            index = json.loads(freeze(str((base / 'artifacts/index.jsonl').relative_to(src)), tag + '_index.jsonl').splitlines()[-1])
            suite = {'model': model, 'step': step, 'source': rel, 'verification': 'raw_artifacts',
                     'completed_at': complete['completed_at_utc'], 'scores': {}, 'truncation': {},
                     'repetition': {}, 'healthy_reward': {}, 'response_counts': COUNTS.copy()}
            for domain, ds in DATASETS.items():
                info = index['datasets'][ds]; path = base / info['run_relative_path']
                digest = hashlib.sha256(); records = []
                with path.open('rb') as handle:
                    for line in handle:
                        digest.update(line); r = json.loads(line)
                        assert r['num_updates'] == step and r['dataset'] == ds
                        # This post-generation field describes the answer, not the question.
                        question_metadata = {k:v for k,v in r['metadata'].items()
                                             if k not in {'mopd_response_diagnostics', 'sandbox_eval'}}
                        if domain == 'Code':
                            assert r['metadata']['sandbox_eval']['infrastructure_errors'] == 0
                        identity = sha(json.dumps([r['prompt'], r['label'], question_metadata],
                                                 sort_keys=True, ensure_ascii=False).encode())
                        records.append({'model': model, 'step': step, 'domain': domain,
                                        'prompt_index': r['prompt_index'], 'sample': r['sample_within_prompt'],
                                        'identity': identity, 'reward': float(r['reward']),
                                        'status': r['status'], 'response_length': r['response_length']})
                        if domain == 'GPQA':
                            rescored = gpqa.compute_gpqa_reward(r['response'], r['label'], r['metadata'])
                            gpqa_audit.append({'model': model, 'step': step,
                                               'prompt': r['prompt_index'], 'sample': r['sample_within_prompt'],
                                               'original': r['reward'], 'rescored': rescored})
                assert digest.hexdigest() == info['sha256'], path
                assert len(records) == info['samples'] == COUNTS[domain]
                assert len({(r['prompt_index'], r['sample']) for r in records}) == len(records)
                groups = defaultdict(list)
                for r in records: groups[r['prompt_index']].append(r['reward'])
                assert len(groups) == (198 if domain == 'GPQA' else COUNTS[domain])
                assert all(len(g) == (4 if domain == 'GPQA' else 1) for g in groups.values())
                score = float(np.mean([np.mean(g) for g in groups.values()]))
                assert abs(score - metrics[f'eval/capability/{ds}']) < 1e-12
                assert metrics[f'eval/capability/{ds}/responses'] == COUNTS[domain]
                suite['scores'][domain] = score
                for dest, key in [('truncation','truncation_rate'), ('repetition','repetition_rate'),
                                  ('healthy_reward','completed_nonrepetitive_reward_mean')]:
                    suite[dest][domain] = metrics.get(f'eval/capability/{ds}/{key}')
                prompt_rows.extend(records)
                manifest.append({'source': str(path.relative_to(src)), 'sha256': digest.hexdigest(),
                                 'rows': len(records), 'frozen': 'prompt_scores.jsonl',
                                 'note': 'compact audit rows; response text remains in source workspace'})
            suites.append(suite)
    # Remote verified summary is retained with its evidence level, not assigned synthetic prompt CIs.
    remote = json.loads(freeze('local/aligned_capability_20260909_a6000/results_summary.json', 'single_capability.json'))
    for r in remote['runs']:
        if any(x['model']==r['model'] and x['step']==r['step'] for x in suites): continue
        rename = {'MATH':'Math','Code':'Code','IFBench':'IF','GPQA':'GPQA'}
        suites.append({'model':r['model'], 'step':r['step'], 'source':r['output'],
                       'verification':'remote_verified_summary', 'completed_at':r['completed_at_utc'],
                       **{dest:{rename[k]:v for k,v in r[key].items()} for dest,key in
                          [('scores','scores'),('truncation','truncation'),('repetition','repetition'),('response_counts','responses')]},
                       'healthy_reward':{d:None for d in DATASETS}})
    with (OUT / 'prompt_scores.jsonl').open('w') as f:
        for r in prompt_rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    dump(OUT/'capability.json', sorted(suites,key=lambda r:(r['model'],r['step'])))
    # Re-score only for audit: any discrepancy must be resolved before manuscript use.
    dump(OUT/'gpqa_score_audit.json', {'scorer':gpqa.GPQA_SCORER_VERSION,'responses':old_audit['responses']+len(gpqa_audit),
         'disagreements':[r for r in gpqa_audit if r['original'] != r['rescored']]})
    assert all(r['original']==r['rescored'] for r in gpqa_audit), 'GPQA scorer mismatch'
    # Conditional paired prompt-cluster bootstrap: all four GPQA samples stay together.
    grouped = defaultdict(dict)
    for r in prompt_rows:
        key = (r['model'],r['step'],r['domain'])
        grouped[key].setdefault(r['prompt_index'], []).append(r)
    comparisons=[]
    pairs=[(('Initial',0),(r['model'],r['step'])) for r in suites
           if r['model']!='Initial' and r['verification']=='raw_artifacts']
    pairs += [(('M-I64-DR',50),(m,50)) for m in ['M-I64-DT','M-I64-GT']]
    for left, right in [('S-PG', 'S-I64'), ('M-PG', 'M-I64-DR')]:
        left_steps = {r['step'] for r in suites if r['model'] == left and r['verification'] == 'raw_artifacts'}
        right_steps = {r['step'] for r in suites if r['model'] == right and r['verification'] == 'raw_artifacts'}
        pairs += [((left, step), (right, step)) for step in sorted(left_steps & right_steps)]
    for (left,ls),(right,rs) in pairs:
        for domain in DATASETS:
            a,b=grouped[(left,ls,domain)],grouped[(right,rs,domain)]
            assert a.keys()==b.keys()
            delta=[]
            for pid in sorted(a):
                assert {r['identity'] for r in a[pid]} == {r['identity'] for r in b[pid]}, (left, ls, right, rs, domain, pid)
                delta.append(np.mean([r['reward'] for r in b[pid]])-np.mean([r['reward'] for r in a[pid]]))
            delta=np.asarray(delta);rng=np.random.default_rng(1042)
            draws=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            comparisons.append({'left':left,'left_step':ls,'right':right,'right_step':rs,'domain':domain,
                                 'delta_pp':100*float(delta.mean()),'lo_pp':100*float(np.quantile(draws,.025)),
                                 'hi_pp':100*float(np.quantile(draws,.975)),'prompts':len(delta)})
    dump(OUT/'paired_comparisons.json',comparisons)
    # Preserve model conversion and evaluation identities for the newly available endpoint.
    single_protocol = json.loads(freeze('local/single_aligned_20260909_a6000/generated/protocol.json', 'single_protocol.json'))
    shared_protocol = json.loads((OUT/'raw/protocol.json').read_text()) if previous else json.loads((src/'local/m_pg_aligned_20260909/generated/protocol.json').read_text())
    for key in ['initialization','student','teachers','prompt_format','response_semantics','evaluation','datasets']:
        assert single_protocol[key] == shared_protocol[key], key
    for domain, splits in single_protocol['splits'].items():
        for split, record in splits.items():
            assert record['sha256'] == shared_protocol['splits'][domain][split]['sha256']
    freeze('local/single_aligned_20260909_a6000/generated/capability_eval.yaml', 'single_capability_eval.yaml')
    endpoint = eval_runs['S-PG']+'/capability_eval/step_500/'
    freeze(endpoint+'job.json', 'S-PG_500_job.json')
    freeze(endpoint+'verified_summary.json', 'S-PG_500_verified_summary.json')
    freeze('local/recovery_blackwell_20260910_sn4622128200/eval_weights/s-pg-s42/step_500/export_verified.json',
           'S-PG_500_export_verified.json')
    for model, run in [('S-PG', 's-pg-s42'), ('S-I64', 's-intersection64-s42')]:
        if any(r['model'] == model and r['step'] == 500 for r in suites):
            conversion = json.loads(freeze(
                f'local/recovery_blackwell_20260910_sn4622128200/eval_weights/{run}/step_500/export_verified.json',
                f'{model}_500_export_verified.json'))
            assert conversion['step'] == 500 and conversion['run'] == run
            assert conversion['exact_original_key_set'] and conversion['all_finite']
            assert conversion['serialized_tensors_equal_converted_native']
    reduction_protocol = json.loads(freeze('local/dt_gt_configuration_20260909/generated/protocol.json',
                                            'reduction_protocol.json'))
    for key in ['initialization','student','teachers','prompt_format','response_semantics','evaluation','datasets']:
        assert reduction_protocol[key] == shared_protocol[key], key
    for domain, splits in reduction_protocol['splits'].items():
        for split, record in splits.items():
            assert record['sha256'] == shared_protocol['splits'][domain][split]['sha256']
    if args.capability_only:
        previous['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        previous['sources'] = manifest
        previous['refresh_scope'] = 'New complete capability suites only; existing online and local records retained.'
        dump(OUT/'manifest.json', previous)
        print(json.dumps({'suites':len(suites),'raw_prompt_rows':len(prompt_rows),
                          'paired_comparisons':len(comparisons),'new_gpqa_rescored':len(gpqa_audit)}))
        return
    mechanism='local/aligned_mechanism_20260910_exx/'
    for name in ['measurements.jsonl','results_summary.json','coverage.json','online_context.json',
                 'input_manifest.json','verification.json','measure_complete.json','numerical_validation.json',
                 'student_verified.json','teacher_precision_verified.json']:
        freeze(mechanism+name,name)
    for name in ['results.json','run_complete.json','verification.json']:
        freeze(mechanism+'pg_variance/'+name,'pg_variance_'+name)
    for name in ['results.json','run_complete.json']:
        freeze(mechanism+'direct_updates/'+name,'direct_updates_'+name)
    measured=(OUT/'raw/measurements.jsonl').read_bytes()
    assert sha(measured)==json.loads((OUT/'raw/verification.json').read_text())['measurement_sha256']
    for model,rel in RUNS.items():
        if model=='Initial':continue
        for stream in ['rollout','mopd']:
            freeze(rel+'/metrics/'+stream+'.jsonl',model+'_'+stream+'.jsonl')
        freeze(rel+'/paper/measurements.jsonl',model+'_paper.jsonl')
    # Extract reproducible training arguments without copying the host environment.
    training = {}
    selected = {'--lr','--adam-beta1','--adam-beta2','--adam-eps','--weight-decay',
                '--clip-grad','--lr-decay-style','--rollout-temperature','--rollout-top-p',
                '--rollout-batch-size','--global-batch-size','--micro-batch-size',
                '--rollout-max-prompt-len','--rollout-max-response-len'}
    for model, rel in RUNS.items():
        if model == 'Initial': continue
        path = src / rel / 'provenance/run_manifest.json'
        data = path.read_bytes(); record = json.loads(data); command = record['command']
        training[model] = {key: command[i+1] for i,key in enumerate(command) if key in selected}
        training[model]['source_sha256'] = sha(data)
        training[model]['source'] = str(path.relative_to(src))
    dump(OUT/'training_configuration.json', training)
    for name in ['prepare.py','measure.py','pg_variance.py','direct_update_comparison.py',
                 'validate_numerics.py','paper_diagnostics_source.py','bf16_reference_source.py',
                 'imported_loss.py','imported_topk.py','imported_paper_diagnostics.py',
                 'imported_sampler.py','imported_sources.json','prepare_complete.json',
                 'cross_probe_consistency.json']:
        freeze(mechanism+name, 'mechanism_'+name)
    freeze('slime_plugins/mopd/response_diagnostics.py','response_diagnostics.py')
    freeze('local/m_pg_aligned_20260909/generated/protocol.json','protocol.json')
    freeze('local/m_pg_aligned_20260909/generated/capability_eval.yaml','capability_eval.yaml')
    # Only teacher references are admitted from the earlier evaluation campaign; never its Base student.
    refs=json.loads((src.parent/'Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/experiments/teacher_reference_results_20260906.json').read_text())
    teachers=[]
    for r in refs['records']:
        item={k:r[k] for k in ['teacher','dataset','prompts','responses','score','truncation_rate','artifact_sha256','source_artifacts']}
        path=src.parent/r['source_artifacts']
        item['verification']='previously_verified_reference'
        if path.exists():
            data=path.read_bytes(); assert sha(data)==r['artifact_sha256']
            if r['teacher']=='teacher_science':
                records=[json.loads(l) for l in data.splitlines()]
                values=[gpqa.compute_gpqa_reward(x['response'],x['label'],x['metadata']) for x in records]
                item['current_scorer_score']=float(np.mean(values))
                item['current_scorer_disagreements']=sum(v!=x['reward'] for v,x in zip(values,records))
        teachers.append(item)
    dump(OUT/'teacher_references.json',teachers)
    dump(OUT/'manifest.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'selection':'All complete aligned suites visible here, plus four remote verified single-teacher suites; compatible historical teacher references only.',
         'excluded':'Base student trajectories, Student64-to-I64 continuations, GPAS and superseded teacher-Top64 results; incomplete DT/100 geometry.',
         'sources':manifest,'bootstrap':{'replicates':10000,'seed':1042,'unit':'prompt cluster','training_seeds':1}})
    print(json.dumps({'suites':len(suites),'raw_prompt_rows':len(prompt_rows),'paired_comparisons':len(comparisons),
                      'gpqa_disagreements':sum(r['original']!=r['rescored'] for r in gpqa_audit)}))


if __name__=='__main__': main()
