"""Read-only inventory of visible completion markers and existing evidence sources."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'experiments/aligned_evidence_20260910'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    rg = shutil.which('rg')
    if rg:
        result = subprocess.run([rg, '--files', str(source/'outputs')], capture_output=True, text=True, check=True)
        files = result.stdout.splitlines()
    else:
        files = [str(p) for p in (source/'outputs').rglob('run_complete.json')]
    markers = []
    for name in sorted(files):
        path = Path(name)
        if path.name != 'run_complete.json':
            continue
        record = json.loads(path.read_text())
        markers.append({'path':str(path.relative_to(source)), 'sha256':digest(path),
                        'status':record.get('status'), 'step':record.get('final_num_updates'),
                        'completed_at_utc':record.get('completed_at_utc'),
                        'kind':'capability' if 'capability_eval' in path.parts else 'training_or_diagnostic'})
    sources = []
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    for record in manifest['sources']:
        path = source/record['source']
        if not path.is_file():
            sources.append({'source':record['source'], 'state':'unavailable_here'})
            continue
        actual = digest(path)
        sources.append({'source':record['source'], 'state':'matches' if actual==record['sha256'] else 'changed_since_freeze',
                        'current_sha256':actual,'frozen_sha256':record['sha256']})
    shares = {}
    for model in ['M-PG','M-I64-DR','M-I64-DT','M-I64-GT']:
        rows = defaultdict(dict)
        for line in (BUNDLE/f'raw/{model}_rollout.jsonl').read_text().splitlines():
            r = json.loads(line)['metrics']
            if 'rollout/step' in r:
                rows[int(r['rollout/step'])].update(r)
        shares[model] = {domain:100*sum(rows[i][f'mopd/task/{domain}/token_share'] for i in range(50))/50
                         for domain in ['math','code','if','science']}
    report = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'source_root':str(source),
              'scope':'Visible outputs completion markers; a marker is an attempt, not an independent experiment or seed. No live process status inferred.',
              'marker_counts_by_family':dict(Counter(r['path'].split('/')[1] for r in markers)),
              'markers':markers, 'bundle_source_checks':sources,
              'source_check_counts':dict(Counter(r['state'] for r in sources)),
              'first50_mean_token_share_percent':shares,
              'token_share_aggregation':'Arithmetic mean of per-rollout shares at indices 0..49; not pooled token counts.',
              'capability_suites':[{k:r[k] for k in ['model','step','verification','scores']} for r in json.loads((BUNDLE/'capability.json').read_text())]}
    target = ROOT/'experiments/completed_inventory_20260911.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'markers':len(markers),'families':report['marker_counts_by_family'],
                      'source_checks':report['source_check_counts'],'report':str(target)},indent=2))

if __name__=='__main__':
    main()
