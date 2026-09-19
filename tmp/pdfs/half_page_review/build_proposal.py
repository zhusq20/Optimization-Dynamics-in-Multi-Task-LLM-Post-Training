from pathlib import Path
import json
import re
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path.cwd()
out = root / 'output/half_page_proposal'
out.mkdir(parents=True, exist_ok=True)
table_path = root / 'sections/table_main_outcomes.tex'
grid_path = root / 'experiments/visual_mvp_20260918/qwen_all_checkpoints_table.tex'
active = '\n'.join(line for line in table_path.read_text().splitlines()
                   if not line.lstrip().startswith('%'))
initial_row = re.search(r'Initial student\s*&(.+?)\\\\', active, re.S).group(1)
initial = np.array([float(x) for x in re.findall(r'\d+\.\d+', initial_row)[:4]])
endpoint = {}
optimizer = 'Adam'
for row in active.split(r'\\'):
    if r'\textit{SGD}' in row:
        optimizer = 'SGD'
    match = re.search(r'\n((?:PG|I16|I64)-(?:DR|DT|GT))\s*\n?&', row)
    if match:
        stats = re.findall(r'\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$', row)
        assert len(stats) == 5
        endpoint[(match.group(1), optimizer)] = [float(x[0]) for x in stats[:4]]
assert len(endpoint) == 10
records = []
config = None
for line in grid_path.read_text().splitlines():
    cells = line.split('&')
    if len(cells) != 7:
        continue
    if re.match(r'^(PG|I16|I64)-', cells[0]):
        config = tuple(x.strip() for x in cells[0].split('/'))
    if config and re.search(r'\d+', cells[1]):
        step = int(re.search(r'\d+', cells[1]).group())
        scores = endpoint[config] if step == 500 else [float(re.search(r'\d+\.\d+', c).group()) for c in cells[2:6]]
        gain = np.array(scores) - initial
        records.append(dict(configuration=config[0], optimizer=config[1], update=step,
                            scores=scores, mean_gain=float(gain.mean()),
                            minimum_domain_gain=float(gain.min()),
                            worst_domain=['Math','Code','IF','Science'][int(gain.argmin())]))
assert len(records) == 37
intermediate = [r for r in records if r['update'] != 500]
hidden = [r for r in intermediate if r['mean_gain'] > 0 and r['minimum_domain_gain'] < 0]
assert len(intermediate) == 27 and len(hidden) == 10
assert all(r['minimum_domain_gain'] >= 0 for r in records if r['update'] == 500)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
    'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.6,'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios':[1.15,1]})
fig.subplots_adjust(left=.075, right=.99, top=.78, bottom=.20, wspace=.34)
ax=axes[0]
ax.axhspan(-4.7,0, xmin=1.1/7.6, color='#fff0e7',zorder=0)
ax.axhline(0,color='#777777',lw=.8)
ax.axvline(0,color='#777777',lw=.8)
colors={'PG / Adam':'#C45A27','I16/I64 / Adam':'#3279A8','PG / SGD':'#248B69'}
def family(r):
    return 'PG / SGD' if r['optimizer']=='SGD' else ('PG / Adam' if r['configuration'].startswith('PG') else 'I16/I64 / Adam')
for r in records:
    color=colors[family(r)]
    final=r['update']==500
    ax.scatter(r['mean_gain'],r['minimum_domain_gain'],s=26 if final else 23,
               facecolors=color if final else 'white', edgecolors=color,
               marker='s' if final else 'o',lw=.8,zorder=3,alpha=.90)
selected=next(r for r in records if (r['configuration'],r['optimizer'],r['update'])==('PG-DR','Adam',250))
ax.annotate('PG-DR, step 250',xy=(selected['mean_gain'],selected['minimum_domain_gain']),
            xytext=(3.4,-3.5),fontsize=7,arrowprops=dict(arrowstyle='-',lw=.6,color='#555555'))
ax.text(3.65,-.6,'Mean gain,\ndomain regression',ha='center',va='top',fontsize=7,color='#8D4724')
ax.set(xlim=(-1.1,6.5),ylim=(-4.7,4.5),xticks=[0,2,4,6],yticks=[-4,-2,0,2,4],
       xlabel='Mean score change (pp)',ylabel='Worst-domain change (pp)')
ax.set_title('(a) All 37 recorded checkpoints',loc='left',pad=9)
ax.grid(alpha=.16,zorder=0)
fig.legend(handles=[Line2D([0],[0],marker='o',lw=0,color=c,label=n,markersize=4) for n,c in colors.items()],
           loc='upper left',bbox_to_anchor=(.057,1.01),ncol=3,frameon=False,fontsize=7,
           handletextpad=.4,columnspacing=1.1)
fig.text(.089,.035,'Open circle: intermediate     Filled square: step 500',fontsize=7,color='#444444')

ax=axes[1]
rr=sorted([r for r in records if r['configuration']=='PG-DR' and r['optimizer']=='Adam'],key=lambda r:r['update'])
x=[0]+[r['update'] for r in rr]
mean=[0]+[r['mean_gain'] for r in rr]
science=[0]+[r['scores'][3]-initial[3] for r in rr]
ax.axhspan(-4.8,0,color='#fff0e7')
ax.axhline(0,color='#777777',lw=.8)
ax.plot(x,mean,'o-',color='#34495E',lw=1.4,ms=3.5,label='Four-domain mean')
ax.plot(x,science,'s--',color='#C45A27',lw=1.4,ms=3.5,label='Science (GPQA)')
ax.annotate('+2.6',xy=(250,mean[3]),xytext=(263,mean[3]+.45),fontsize=7,color='#34495E')
ax.annotate('-2.0',xy=(250,science[3]),xytext=(263,science[3]-.75),fontsize=7,color='#C45A27')
ax.set(xlim=(-15,530),ylim=(-4.8,5.9),xticks=[0,100,250,500],yticks=[-4,-2,0,2,4],
       xlabel='Optimizer update',ylabel='Score change from initialization (pp)')
ax.set_title('(b) PG-DR / Adam trajectory',loc='left',pad=9)
ax.legend(loc='upper left',fontsize=7,frameon=False,handlelength=1.6)
ax.grid(alpha=.16)
fig.text(.63,.035,'Recorded point estimates; no significance test',fontsize=7,color='#444444',ha='center')
fig.savefig(out/'reviewer_bridge.png',dpi=220)
fig.savefig(out/'reviewer_bridge.svg')
audit={'sources':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [table_path,grid_path]],
       'initial_scores':initial.tolist(),'records':records,'intermediate_count':len(intermediate),
       'positive_mean_negative_domain_count':len(hidden),'positive_mean_negative_domain_records':hidden,
       'notes':['Endpoints replaced by active Table 1 domain means; obsolete endpoint rows in the full-grid table are not used.',
                'Mean changes recomputed from displayed domain scores, so up to 0.01 pp rounding differences from displayed macro scores are possible.',
                'Intermediate points use the recorded grid summaries; complete matched per-seed grid outcomes are unavailable in this checkout.',
                'Counts are descriptive, correlated checkpoint observations; zero crossings are not significance tests.',
                'This figure does not establish a causal link between update geometry and capability outcomes.']}
(out/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'output':str(out),'records':len(records),'hidden_regressions':len(hidden),'all_endpoints_nonnegative':True}))
