#!/usr/bin/env python3
"""Export the 27-row plan from the Chinese roadmap and draw its layout."""
from pathlib import Path
from collections import Counter
import csv
import json
import re
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DOC=HERE/'EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md'
OUT=HERE/'local_probe_results_20260908'
FIG=HERE.parent/'figures/local_probe_results_20260908'
TITLES=[
 ['Response lengths','Domain token shares','Response loss weights'],
 ['Length-gradient covariance','Gradient norm / direction','Held-out KL change'],
 ['Per-domain capability','Macro score vs tokens','Worst-domain change'],
 ['Cumulative BF16 changes','Update energy concentration','Layer-wise BF16 changes'],
 ['Teacher support overlap','Raw gradient vs BF16 alignment','Zero-gradient control'],
 ['Teacher JS matrix','JS vs routed support distance','Same-prefix control'],
 ['Local BF16 threshold curves','Gradient and writeback scale','Distance to full-vocab step'],
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
        pid,row,col,status=match.groups();r=int(row)-1;c=ord(col)-ord('a')
        rows.append({'panel_id':pid,'row':r+1,'column':col,'core_24':r<8,'status':status,
            'title':TITLES[r][c],'claim':CLAIMS[r],'plot_and_required_data':cells[1],
            'existing_evidence_and_next_experiment':cells[2], 'work_packages':PACKAGES[r],
            'snapshot_data_keys':DATA_KEYS[r], 'publication_matrix_accepted':False,
            'specification':'EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md'})
    assert len(rows)==27 and len({r['panel_id'] for r in rows})==27
    counts=Counter(r['status'] for r in rows[:24])
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    with (OUT/'figure_plan.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'figure_plan.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,axes=plt.subplots(9,3,figsize=(12,16.4))
    palette={'E':('#e6f2ee','#20735a'),'P':('#fff3df','#9b6218'),'N':('#eef0f4','#586374'),'X':('#f4e8e8','#976363')}
    label={'E':'EXPLORATORY DATA AVAILABLE','P':'PARTIAL DATA','N':'NOT FILLED BY THIS CAMPAIGN','X':'CANCELLED FROM CURRENT SCOPE'}
    for row,ax in zip(rows,axes.flat):
        ax.set_xticks([]);ax.set_yticks([]);r=row['row']-1;c=ord(row['column'])-ord('a')
        bg,fg=palette[row['status']];ax.set_facecolor(bg)
        for spine in ax.spines.values():spine.set_color('#d8dce1')
        ax.text(.06,.79,row['panel_id'],transform=ax.transAxes,fontweight='bold',fontsize=14,color=fg)
        ax.text(.06,.43,textwrap.fill(row['title'],27),transform=ax.transAxes,fontsize=11,va='center',color='#1d2835')
        ax.text(.06,.12,label[row['status']],transform=ax.transAxes,fontsize=7.5,color=fg)
        if c==0:ax.text(-.56,.5,textwrap.fill(CLAIMS[r],18),transform=ax.transAxes,ha='left',va='center',fontsize=9,color='#35404f')
    fig.suptitle('Evidence roadmap: 8 core rows x 3 panels + optional row 9',fontsize=16,y=.995,va='top')
    fig.text(.5,.956,'Local probes completed: 2026-09-08 20:45 UTC | Core: 15 exploratory, 5 partial, 4 not filled\nAvailability is not publication readiness. This is a plan, not experimental measurements.',ha='center',va='center',fontsize=9)
    fig.subplots_adjust(left=.19,right=.98,top=.918,bottom=.025,hspace=.21,wspace=.10)
    fig.savefig(FIG/'figure_roadmap_9x3.pdf',bbox_inches='tight')
    fig.savefig(FIG/'figure_roadmap_9x3.png',dpi=130,bbox_inches='tight');plt.close(fig)
    print('Exported 27 panel specifications; core status counts:',dict(Counter(r['status'] for r in rows[:24])))
if __name__=='__main__':main()
