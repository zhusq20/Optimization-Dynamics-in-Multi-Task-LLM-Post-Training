#!/usr/bin/env python3
"""Export the 27 figure specifications; experimental charts are saved separately."""
from pathlib import Path
from collections import Counter
import csv
import json
import re

HERE=Path(__file__).resolve().parent
DOC=HERE/'EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md'
OUT=HERE/'local_probe_results_20260908'
FIG=HERE.parent/'figures/local_probe_results_20260908'
TITLES=[
 ['Response lengths','Domain token shares','Response loss weights'],
 ['Length-gradient covariance','Gradient norm / direction','Held-out KL change'],
 ['Per-domain capability','Macro score vs tokens','Worst-domain change'],
 ['Cumulative BF16 changes','Update energy concentration','Layer-wise BF16 changes'],
 ['Teacher-pair overlap across conditions','Raw gradient vs BF16 alignment','Zero-gradient control'],
 ['Teacher-pair JS across checkpoints','JS vs routed support distance','Same-prefix control'],
 ['BF16 changed fraction across states','Gradient and writeback scale','Distance to full-vocab step'],
 ['Actual online BF16 steps','PG / TopK capability','Capability vs exposure / cost'],
 ['Paired training seeds','1024 / 4096 response cap','Diagnostic batch sensitivity']]
CLAIMS=['Token balancing: observed weights','Token balancing: local mechanism','Token balancing: online outcomes',
        'Update sparsity: cumulative changes','Teacher influence in one student','Teacher distribution distance',
        'Supervision density: local controls','Learning outcomes and actual updates','Robustness / optional ninth row']
PACKAGES=['completed local probes','completed local probes','existing evaluations','existing scans','completed local probes','completed local probes','completed local probes','existing trajectories','cancelled / out of scope']
DATA_KEYS=['local responses + weights','local covariance + gradient comparisons + heldout','existing evaluations: not refreshed',
           'existing cumulative scans: not refreshed','local teacher pairs + zero controls','local teacher JS + routed/common pairs',
           'local density branches + thresholds + pairs','existing evaluations / trajectories: not refreshed','cancelled or outside this campaign']

def main():
    rows=[]
    for line in DOC.read_text().splitlines():
        match=re.match(r'\| \*\*(F([1-9])([abc])) \[([EPNX])\]\*\*',line)
        if not match:continue
        cells=[c.strip() for c in line.split('|')[1:-1]]
        panel,row,col,status=match.groups();r=int(row)-1;c=ord(col)-ord('a')
        path=FIG/f'{panel}.pdf'
        rows.append({'panel_id':panel,'row':r+1,'column':col,'core_24':r<8,'status':status,
                     'title':TITLES[r][c],'claim':CLAIMS[r],'plot_and_required_data':cells[1],
                     'existing_evidence_and_next_experiment':cells[2],'work_packages':PACKAGES[r],
                     'snapshot_data_keys':DATA_KEYS[r],'publication_matrix_accepted':False,
                     'specification':DOC.name,'standalone_pdf':str(path.relative_to(HERE.parent)) if path.exists() else None,
                     'caption_file':'local_probe_results_20260908/FIGURE_CAPTIONS.md' if path.exists() else None})
    assert len(rows)==27 and len({r['panel_id'] for r in rows})==27
    OUT.mkdir(exist_ok=True)
    with (OUT/'figure_plan.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'figure_plan.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print('Exported 27 specifications; standalone experimental assets:',sum(r['standalone_pdf'] is not None for r in rows))

if __name__=='__main__':main()
