#!/usr/bin/env python3
"""Freeze evidence from existing files. CPU/I/O only; never changes source runs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'slime_opd_geometry/outputs/mopd_qwen3'
SOURCES, ISSUES = {}, []
EXPECTED = {'math500_pass1': (500, 1), 'livecodebench_postcutoff_pass1': (128, 1),
            'ifbench_strict': (300, 1), 'gpqa_diamond_avg4': (198, 4)}
GROUPS = ('formal_20260906', 'student_top64_20260907', 'student_top64_intersection_20260907',
          'student_top64_single_20260907_launch', 'student_top64_single_intersection_20260907')

def relative(path):
    try: return str(path.relative_to(WORKSPACE))
    except ValueError: return str(path)

def read(path):
    data = path.read_bytes()
    SOURCES[relative(path)] = {'bytes_read': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        'mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    return data

def read_json(path): return json.loads(read(path))

def read_lines(path):
    if not path.exists(): return []
    result = []
    for i, line in enumerate(read(path).splitlines()):
        if not line.strip(): continue
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            ISSUES.append({'source': relative(path), 'line': i+1, 'issue': 'invalid_or_incomplete_line_excluded'})
    return result

def run_inventory():
    runs, rollouts, cumulative = [], [], []
    for group in GROUPS:
        for run in sorted((ROOT/group).iterdir()):
            allocation = run/'allocation.jsonl'
            rows = read_lines(allocation)
            if not rows: continue
            manifest = read_json(run/'provenance/run_manifest.json')
            cfg = manifest['protocol_cli']
            tp = run/'provenance/loss_transition.json'
            tr = read_json(tp) if tp.exists() else {}
            inherited = tr.get('inherited_optimizer_updates', 0)
            updates = max(r['optimizer_updates_after'] for r in rows)
            retained = sorted(int(p.stem.rsplit('_',1)[1]) for p in (run/'paper').glob('checkpoint_step_*.pt'))
            hf = []
            for p in sorted((run/'weights').glob('iter_*')):
                index = p/'model.safetensors.index.json'
                if index.exists():
                    shards = set(read_json(index)['weight_map'].values())
                    if all((p/s).is_file() and (p/s).stat().st_size > 0 for s in shards):
                        hf.append(int(p.name.split('_')[1])+1)
            marks = {}
            for name in ('run_complete.json', 'run_stopped_by_user.json', 'run_failed.json'):
                if (run/name).exists(): marks[name] = read_json(run/name).get('status')
            state = ('complete' if 'run_complete.json' in marks else
                     'continuation_incomplete_on_disk' if tr else
                     'stopped_or_paused' if 'run_stopped_by_user.json' in marks else
                     'handed_off' if manifest.get('continued_to') else 'incomplete_on_disk')
            recent = [r for r in rows if r['optimizer_updates_after'] > inherited][-20:]
            times = [r['feedback']['total_step_seconds'] for r in recent
                     if r['feedback'].get('total_step_seconds') is not None and not r.get('checkpoint_due')]
            runs.append({'run_id': run.name, 'source': relative(run), 'loss': cfg.get('mopd_loss'),
                'k': cfg.get('mopd_topk'), 'reduction': rows[-1]['aggregation'], 'seed': int(cfg.get('mopd_seed',42)),
                'status': state, 'updates': updates, 'target_updates': int(cfg['mopd_total_steps']),
                'allocation_rows': len(rows), 'inherited_updates': inherited,
                'first_intersection_update': tr.get('first_intersection_update'), 'parent_run': tr.get('parent_run'),
                'new_objective_updates': updates-inherited if inherited else None,
                'hf_steps_shards_present': sorted(hf), 'optimizer_snapshot_steps': retained,
                'recent_step_seconds_median': statistics.median(times) if times else None, 'recent_timing_n': len(times),
                'markers': marks, 'allocation_mtime_utc': SOURCES[relative(allocation)]['mtime_utc'],
                'total_generated_tokens': sum(r['feedback'].get('generated_tokens',0) for r in rows),
                'remaining_updates': max(0,int(cfg['mopd_total_steps'])-updates)})
            # Continuations contain copied allocation history. Do not sum parents and children.
            for r in read_lines(run/'metrics/rollout.jsonl'):
                m = r['metrics']
                for task in ('math','code','if','science'):
                    stem = f'mopd/task/{task}/'
                    if stem+'token_share' in m:
                        rollouts.append({'run_id': run.name, 'rollout_id': m.get('rollout/id'),
                            'num_updates_before': m.get('rollout/num_updates'), 'domain': task,
                            **{k: m.get(stem+k) for k in ('token_share','prompt_share','mean_response_length',
                                'valid_response_tokens','generated_tokens','truncation_rate','completion_rate','attempted_responses')}})
            for r in read_lines(run/'paper/measurements.jsonl'):
                if r.get('quantity') == 'delta_bf16':
                    cumulative.append({'run_id': run.name, 'step': r['step'], 'layer': r['layer'],
                        'recorded_loss': r['loss'], 'metrics': r['metrics'], 'source': relative(run/'paper/measurements.jsonl')})
    return runs, rollouts, cumulative

def evaluation_inventory():
    results, incomplete = [], []
    for root in sorted(ROOT.glob('*/*/capability_eval/step_*')):
        marker = root/'run_complete.json'
        if not marker.exists():
            incomplete.append(relative(root)); continue
        complete = read_json(marker)
        for record in read_lines(root/'artifacts/index.jsonl'):
            for dataset, info in record.get('datasets',{}).items():
                if dataset not in EXPECTED: continue
                artifact = root/info['run_relative_path']
                if not artifact.exists():
                    ISSUES.append({'source':relative(artifact),'issue':'completed_eval_missing_artifact'}); continue
                data = read_lines(artifact)
                n, repeats = EXPECTED[dataset]
                counts = Counter(r['prompt_index'] for r in data)
                valid_rewards = all(isinstance(r.get('reward'),(int,float)) and math.isfinite(r['reward'])
                                    and 0 <= r['reward'] <= 1 for r in data)
                score = statistics.mean(r['reward'] for r in data) if data and valid_rewards else None
                reported = record['metrics'].get(f'eval/capability/{dataset}')
                step = int(root.name.split('_')[1])
                checks = {'artifact_sha256': SOURCES[relative(artifact)]['sha256']==info['sha256'],
                    'counts': len(data)==n*repeats and len(counts)==n and set(counts.values())=={repeats},
                    'step': all(r['num_updates']==step for r in data), 'rewards': valid_rewards,
                    'terminal_status': all(r['status'] in {'completed','truncated'} for r in data),
                    'score_matches': score is not None and reported is not None and abs(score-reported)<1e-10}
                subject = root.parents[1].name
                family = 'M-PG' if subject.startswith('m-pg-') else 'S-PG' if subject.startswith('s-pg-') else subject
                row = {'subject':subject,'family':family,'step':step,'dataset':dataset,'score':score,
                    'responses':len(data),'prompts':len(counts),
                    'correct_responses':sum(r['reward'] for r in data) if valid_rewards else None,
                    'truncation_rate':sum(r['status']=='truncated' for r in data)/len(data) if data else None,
                    'completed_at_utc':complete['completed_at_utc'],'source':relative(artifact),
                    'checks':checks,'verified':all(checks.values())}
                results.append(row)
                if not row['verified']: ISSUES.append({'source':relative(artifact),'issue':'evaluation_check_failed','checks':checks})
    return results, incomplete

def scans_inventory():
    result = []
    roots = [ROOT/'bf16_checkpoint_scan_20260907_sn4622122392',ROOT/'mpg500_noncode_20260907_sn4622122392']
    for root in roots:
        for f in sorted(root.glob('*-bf16-vs-base/report.json')):
            r = read_json(f)
            if r.get('status') != 'complete': continue
            checkpoint = r['checkpoint']; actual_run = Path(checkpoint).parent.parent.name
            if f.parent.name == 'm-pg-s42-step500-bf16-vs-base': actual_run = 'm-pg-s42'
            if actual_run.startswith(('m-tk64','s-tk64')): actual_loss = 'student_topk'
            elif actual_run in {'m-pg-s42','s-pg-s42'}: actual_loss = 'sampled_reverse_kl'
            else: raise ValueError('Review new checkpoint lineage before assigning a loss: '+actual_run)
            result.append({'reported_run':r.get('run', actual_run),'actual_checkpoint_run':actual_run,'actual_loss_at_checkpoint':actual_loss,
                'step':r.get('optimizer_step', 500),'checkpoint':checkpoint,'global':r['global'],'layers':r['layers'],
                'source':relative(f),'baseline_hash':r.get('baseline_ordered_tensor_sha256'),
                'checkpoint_hash':r['checkpoint_ordered_tensor_sha256']})
    f = roots[0]/'summary.csv'
    for r in csv.DictReader(io.StringIO(read(f).decode())):
        if r['method']=='teacher_student_top64_intersection_for_entire_run':
            ISSUES.append({'source':relative(f),'run':r['run'],'step':int(r['optimizer_step']),
                'issue':'summary_method_conflicts_with_checkpoint_and_loss_transition',
                'derived_label':'student Top64 before the intersection switch'})
    return result

def diagnostics_inventory():
    normalizations, probes, pairs = [], [], []
    for f in sorted((ROOT/'normalization_cuda6_9_20260907_sn4622122392').glob('*/measurements.json')):
        r = read_json(f)
        if r.get('status')!='complete' or not (f.parent/'run_complete.json').exists(): continue
        manifest=read_json(f.parent/'probe_manifest.json'); banks=read_json(f.parent/'banks.json')
        weights=read_json(f.parent/'weights.json')
        normalizations.append({'job':r['job'],'step':r['snapshot_step'],'loss':r['loss'],'source':relative(f),
            'decomposition':r['decomposition'],'gradient_comparisons':r['gradient_comparisons'],'branches':r['branches'],
            'train_valid_tokens':r['train_valid_tokens'],'heldout_valid_tokens':r['heldout_valid_tokens'],
            'train_truncated_responses':r['train_truncated_responses'],'heldout_truncated_responses':r['heldout_truncated_responses'],
            'train_responses':len(banks['train']),'heldout_responses':len(banks['heldout']),
            'train_domains':[a['task'] for a in banks['train']],'weights':weights,
            'max_new_tokens':manifest['arguments']['max_new_tokens'],'elapsed_seconds':r['elapsed_seconds']})
    for root in (ROOT/'bf16_local_update_probe_20260907_sn4622122392',ROOT/'mpg500_teacher_overlap_20260907_sn4622122392'):
        for f in sorted(root.glob('*/measurements.json')):
            r=read_json(f)
            if r.get('status')!='complete' or not (f.parent/'run_complete.json').exists(): continue
            manifest=read_json(f.parent/'probe_manifest.json')
            probes.append({'job':r['job'],'source':relative(f),'snapshot':r['snapshot'],'branches':r['branches'],
                'pairs':r['pairwise_bf16_writebacks'],'teacher_distances':r['teacher_distances'],
                'arguments':{k:manifest['arguments'].get(k) for k in ('seed','max_new_tokens','prefixes_per_response','topk','losses')},
                'response_count':len(r['prefix_bank']),'elapsed_seconds':r['elapsed_seconds']})
    f=ROOT/'mpg500_teacher_overlap_20260907_sn4622122392/teacher_pairs.csv'
    for r in csv.DictReader(io.StringIO(read(f).decode())):
        pairs.append({k:float(v) if k not in {'job','mode','teacher_left','teacher_right','loss'} else v for k,v in r.items()})
    return normalizations,probes,pairs

def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'figure_audit_20260908')
    args=parser.parse_args(); started=datetime.now(timezone.utc).isoformat()
    runs,rollouts,cumulative=run_inventory(); evals,incomplete=evaluation_inventory()
    scans=scans_inventory(); norms,probes,pairs=diagnostics_inventory()
    data={'schema_version':1,'audit_started_utc':started,'audit_finished_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Shared filesystem only; remote process liveness not verified; source runs unchanged',
        'runs':runs,'evaluations':evals,'incomplete_eval_directories':incomplete,'bf16_scans':scans,
        'cumulative_measurements':cumulative,'rollout_domain_metrics':rollouts,'normalization_probes':norms,
        'local_probes':probes,'teacher_pairs':pairs,'issues':ISSUES,'sources':SOURCES}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'evidence_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    write_csv(args.output/'run_inventory.csv',runs);write_csv(args.output/'capability_inventory.csv',evals)
    write_csv(args.output/'bf16_inventory.csv',[{k:r[k] for k in ('reported_run','actual_checkpoint_run','actual_loss_at_checkpoint','step','source')}|r['global'] for r in scans])
    print(json.dumps({'audit_finished_utc':data['audit_finished_utc'],'runs':len(runs),
        'verified_eval_dataset_points':sum(r['verified'] for r in evals),'failed_eval_checks':sum(not r['verified'] for r in evals),
        'bf16_scans':len(scans),'normalization_probes':len(norms),'local_probes':len(probes),
        'teacher_pair_rows':len(pairs),'source_files_hashed':len(SOURCES),'issues':len(ISSUES)},indent=2))
if __name__=='__main__':main()
