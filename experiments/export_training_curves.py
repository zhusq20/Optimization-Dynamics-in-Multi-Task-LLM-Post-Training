#!/usr/bin/env python3
"""Freeze existing W&B scalar mirrors and verified evaluations; CPU and file I/O only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DATASETS = {
    'math500_pass1': ('math', 500, 1),
    'livecodebench_postcutoff_pass1': ('code', 128, 1),
    'ifbench_strict': ('if', 300, 1),
    'gpqa_diamond_avg4': ('science', 198, 4),
}
RUNS = {
    'm-pg-s42': ('formal_20260906', 'Joint PG'),
    's-pg-s42': ('formal_20260906', 'Math-only PG'),
    'm-tk64-dr-s42': ('student_top64_20260907', 'Joint Student64 · DR'),
    'm-tk64-dt-s42': ('student_top64_20260907', 'Joint Student64 · DT'),
    'm-tk64-gt-s42': ('student_top64_20260907', 'Joint Student64 · GT'),
    's-tk64-s42': ('student_top64_single_20260907_launch', 'Math-only Student64'),
    'm-itk64-dr-s42': ('student_top64_intersection_20260907', 'Joint Student64 → I64 · DR'),
    'm-itk64-dt-s42': ('student_top64_intersection_20260907', 'Joint Student64 → I64 · DT'),
    'm-itk64-gt-s42': ('student_top64_intersection_20260907', 'Joint Student64 → I64 · GT'),
    's-itk64-s42': ('student_top64_single_intersection_20260907', 'Math-only Student64 → I64'),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def read_events(path, issues):
    """Freeze the byte prefix read, excluding only a concurrently written final line."""
    payload = path.read_bytes()
    if payload and not payload.endswith(b'\n'):
        tail = payload.rsplit(b'\n', 1)[-1]
        try:
            json.loads(tail)
        except json.JSONDecodeError:
            payload = payload[:-len(tail)]
            issues.append({'source': str(path), 'reason': 'incomplete final line excluded'})
    return payload, [json.loads(line) for line in payload.splitlines() if line.strip()]


def merge_events(records, source):
    """Use the recorded namespace clock; keep line provenance for every scalar."""
    events = {}
    for line, record in enumerate(records, 1):
        metrics = record['metrics']
        clock = record['step_key']
        if clock not in metrics:
            raise ValueError(f'Missing clock {clock} in {source}:{line}')
        step = metrics[clock]
        entry = events.setdefault(step, {'step': step, 'clock': clock, 'metrics': {}, 'sources': {}})
        for key, value in metrics.items():
            # Shared workers can log disjoint fields at the same namespace step.
            # The original stream is archived; a repeated field uses its final value.
            entry['metrics'][key] = value
            entry['sources'][key] = {'file': source, 'line': line}
    return [events[s] for s in sorted(events)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=HERE.parents[1] / 'slime_opd_geometry/outputs/mopd_qwen3')
    ap.add_argument('--output', type=Path, default=HERE / 'paper_curves_20260908')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources, issues, runs, evaluations, raw_eval_events = [], [], {}, [], []

    def freeze(path, relative):
        payload, records = read_events(path, issues)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                compressed.write(payload)
        sources.append({'file': relative, 'source': str(path.resolve()), 'uncompressed_bytes': len(payload),
                        'sha256': hashlib.sha256(payload).hexdigest(), 'records': len(records)})
        return records

    for name, (group, label) in RUNS.items():
        root = args.root / group / name
        if not (root / 'metrics').exists():
            issues.append({'source': str(root), 'reason': 'run metrics not available locally'})
            continue
        metadata = json.loads((root / 'provenance/run_manifest.json').read_text())
        command = metadata['command']
        config = {command[i]: command[i+1] for i in range(len(command)-1)
                  if command[i] in {'--wandb-team', '--wandb-project'}}
        wandb_id_file = root / 'wandb_run_id.txt'
        wandb_id = wandb_id_file.read_text().strip() if wandb_id_file.exists() else None
        team, project = config.get('--wandb-team'), config.get('--wandb-project')
        transition_file = root / 'provenance/loss_transition.json'
        transition = json.loads(transition_file.read_text()) if transition_file.exists() else {}
        entry = {'id': name, 'label': label, 'source': str(root.resolve()),
                 'wandb_id': wandb_id,
                 'wandb_url': f'https://wandb.ai/{team}/{project}/runs/{wandb_id}' if all((team, project, wandb_id)) else None,
                 'loss': metadata['protocol_cli'].get('mopd_loss'),
                 'parent_run': Path(transition['parent_run']).name if transition else None,
                 'first_intersection_update': transition.get('first_intersection_update'),
                 'inherited_updates': transition.get('inherited_optimizer_updates', 0),
                 'complete_on_disk': (root / 'run_complete.json').exists(), 'events': {}}
        # Copy scalar logs and accounting, never credentials or model tensors.
        for namespace in ('train', 'rollout', 'mopd', 'eval'):
            path = root / 'metrics' / f'{namespace}.jsonl'
            if not path.exists():
                continue
            relative = f'raw/training/{name}/{namespace}.jsonl.gz'
            entry['events'][namespace] = merge_events(freeze(path, relative), relative)
        path = root / 'allocation.jsonl'
        allocations = freeze(path, f'raw/training/{name}/allocation.jsonl.gz') if path.exists() else []
        by_step = {}
        for row in allocations:
            by_step[row['optimizer_updates_after']] = row
        prompt_totals, token_total, costs = defaultdict(int), 0, []
        for step, row in sorted(by_step.items()):
            feedback = row.get('feedback', {})
            token_total += feedback.get('generated_tokens', 0)
            for task, count in row['counts'].items():
                prompt_totals[task] += count * row['step_global_batch_size']
            costs.append({'step': step, 'generated_tokens': token_total, 'prompts_by_domain': dict(prompt_totals)})
        entry['exposure'] = costs
        entry['allocation_has_contiguous_history'] = sorted(by_step) == list(range(1, max(by_step, default=0)+1))
        runs[name] = entry

    for index in sorted(args.root.glob('*/*/capability_eval/step_*/artifacts/index.jsonl')):
        root = index.parents[1]
        marker = root / 'run_complete.json'
        if not marker.exists():
            issues.append({'source': str(root), 'reason': 'evaluation has no completion marker'})
            continue
        subject = root.parents[1].name
        match = re.match(r'((?:m|s)-(?:pg|itk64|tk64)(?:-dr|-dt|-gt)?-s42)(?:-|$)', subject)
        family = match.group(1) if match else subject
        if family not in RUNS and family != 'initial_student' and not family.startswith('teacher_'):
            continue
        tag = str(root.relative_to(args.root)).replace('/', '__')
        marker_data = json.loads(marker.read_text())
        completed = marker_data.get('completed_at_utc', '')
        eval_file = root / 'metrics/eval.jsonl'
        if not eval_file.exists():
            issues.append({'source': str(root), 'reason': 'no raw evaluation scalar mirror'})
            continue
        raw_path = f'raw/evaluation/{tag}/eval.jsonl.gz'
        merged = merge_events(freeze(eval_file, raw_path), raw_path)
        _, indices = read_events(index, issues)
        step = int(root.name.split('_')[1])
        raw_eval_events.append({'attempt': tag, 'family': family, 'checkpoint_step': step, 'events': merged})
        dump(args.output / f'raw/evaluation/{tag}/run_complete.json', marker_data)
        dump(args.output / f'raw/evaluation/{tag}/artifact_index.json', indices)
        for record in indices:
            for dataset, info in record.get('datasets', {}).items():
                if dataset not in DATASETS:
                    continue
                domain, expected_prompts, repeats = DATASETS[dataset]
                artifact = root / info['run_relative_path']
                if not artifact.exists():
                    issues.append({'source': str(artifact), 'reason': 'missing evaluation response artifact'})
                    continue
                sha, rewards = hashlib.sha256(), []
                with artifact.open('rb') as stream:
                    for line in stream:
                        sha.update(line)
                        if not line.strip():
                            continue
                        response = json.loads(line)
                        rewards.append({k: response.get(k) for k in ['prompt_index', 'reward', 'num_updates', 'status']})
                counts = Counter(r['prompt_index'] for r in rewards)
                valid = bool(rewards) and all(isinstance(r['reward'], (int, float)) and math.isfinite(r['reward'])
                                               and 0 <= r['reward'] <= 1 for r in rewards)
                score = sum(r['reward'] for r in rewards)/len(rewards) if valid else None
                key = f'eval/capability/{dataset}'
                scalars = [e for e in merged if key in e['metrics']]
                logged = scalars[-1]['metrics'][key] if scalars else None
                checks = {'artifact_sha256': sha.hexdigest() == info['sha256'],
                          'counts': len(rewards) == expected_prompts * repeats and len(counts) == expected_prompts
                                    and set(counts.values()) == {repeats},
                          'checkpoint_step': all(r['num_updates'] == step for r in rewards),
                          'terminal_status': all(r['status'] in {'completed', 'truncated'} for r in rewards),
                          'legal_rewards': valid,
                          'raw_scalar_matches': score is not None and logged is not None and abs(score-logged) < 1e-10}
                compact = f'raw/evaluation/{tag}/{dataset}_rewards.json'
                dump(args.output / compact, {'source': str(artifact), 'sha256': sha.hexdigest(), 'responses': rewards})
                metrics = scalars[-1]['metrics'] if scalars else {}
                entry = {'family': family, 'subject': subject, 'attempt': tag, 'step': step, 'dataset': dataset,
                         'domain': domain, 'score': logged, 'recomputed_score': score,
                         'truncation_rate': metrics.get(key + '/truncation_rate'),
                         'response_length': metrics.get(f'eval/{dataset}/response_len/mean'),
                         'completed_at_utc': completed, 'verified': all(checks.values()), 'checks': checks,
                         'metric': key, 'source': scalars[-1]['sources'][key] if scalars else None,
                         'reward_evidence': compact, 'prompts': expected_prompts, 'samples_per_prompt': repeats}
                evaluations.append(entry)
                if not entry['verified']:
                    issues.append({'source': str(artifact), 'reason': 'evaluation verification failed', 'checks': checks})

    chosen = set()
    for row in sorted(evaluations, key=lambda r: (r['completed_at_utc'], r['attempt'])):
        key = (row['family'], row['step'], row['dataset'])
        row['selected'] = row['verified'] and key not in chosen
        if row['selected']:
            chosen.add(key)
        run = runs.get(row['family'])
        if run:
            costs = next((r for r in run['exposure'] if r['step'] == row['step']), None)
            row['domain_training_prompts'] = costs['prompts_by_domain'].get(row['domain'], 0) if costs else None
            row['generated_tokens'] = costs['generated_tokens'] if costs else None
            first = run['first_intersection_update']
            row['training_phase'] = ('student_topk' if row['step'] < first else 'mixed_student_topk_then_intersection') if first else run['loss']
        else:
            row['training_phase'] = 'reference'
    dataset = {'schema_version': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_kind': 'Local lossless scalar mirrors written immediately before wandb.log(metrics). No cloud history sampling.',
               'evaluation_selection': 'First complete, verified execution per family/checkpoint/dataset, chosen by completion time then path, never by score. All other attempts preserved.',
               'runs': runs, 'evaluations': evaluations, 'raw_evaluations': raw_eval_events,
               'sources': sources, 'issues': issues,
               'missing_metrics': ['Policy entropy is absent from captured training keys; sampled negative log probability is not substituted.'],
               'clock_notes': {'train/step': 'Recorded gradient-microbatch log index, not optimizer update.',
                               'mopd/update': 'Completed optimizer updates.',
                               'rollout/step': 'Recorded rollout index (policy before the corresponding update).',
                               'eval': 'Checkpoint update from artifact num_updates; eval-local step is not the training clock.'}}
    # Gzip avoids duplicating hundreds of megabytes of repeated provenance strings.
    with gzip.open(args.output / 'data.json.gz', 'wt') as stream:
        json.dump(dataset, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    write_csv(args.output / 'evaluation_attempts.csv', evaluations)
    write_csv(args.output / 'run_inventory.csv', [{k: v for k, v in row.items() if k not in {'events', 'exposure'}} for row in runs.values()])
    write_csv(args.output / 'source_manifest.csv', sources)
    dump(args.output / 'export_summary.json', {'runs': len(runs), 'scalar_files': len(sources),
          'evaluation_attempts': len(evaluations), 'verified_attempts': sum(r['verified'] for r in evaluations),
          'selected_eval_points': sum(r['selected'] for r in evaluations), 'issues': issues})
    print(json.dumps({'output': str(args.output), 'runs': len(runs), 'evaluations': len(evaluations),
                      'verified': sum(r['verified'] for r in evaluations), 'issues': len(issues)}, indent=2))


if __name__ == '__main__':
    main()
