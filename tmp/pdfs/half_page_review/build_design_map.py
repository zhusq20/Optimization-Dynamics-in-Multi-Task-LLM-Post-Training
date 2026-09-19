from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path('output/half_page_proposal')
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(7.2,2.7))
fig.subplots_adjust(left=.005,right=.995,top=.99,bottom=.015)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
panels=[(.02,'(a) Fixed-state diagnostic comparisons','#27658D','#EDF4F9',[
    ('Shared within a comparison','Saved student weights and cached prefixes\nSame teacher targets for normalization/loss controls'),
    ('Controlled changes','Loss averaging, vocabulary support, or teacher/input\nSaved/reset Adam moments; norm-matched SGD'),
    ('Measured outcomes','Gradient direction; FP32 step; BF16 writeback\nOne-step held-out KL in normalization controls')]),
    (.525,'(b) End-to-end capability comparisons','#326F5B','#EEF6F1',[
    ('Common starting point','Student initialization and training protocol\nFresh on-policy responses at every update'),
    ('Training configurations','PG / I16 / I64; DR / DT / GT\nAdam and SGD with their training learning rates'),
    ('Measured outcomes','Per-domain learning curves and final scores\nTable 1: update 500, ten joint-training configurations')])]
for x,title,color,fill,boxes in panels:
    width=.455
    ax.text(x,.965,title,ha='left',va='top',fontsize=9,fontweight='bold',color=color)
    for i,(head,body) in enumerate(boxes):
        y=.68-i*.265
        ax.add_patch(FancyBboxPatch((x,y),width,.205,boxstyle='round,pad=0.007,rounding_size=0.013',
                     linewidth=.8,edgecolor=color,facecolor=fill))
        ax.text(x+.012,y+.174,head,ha='left',va='top',fontsize=8,fontweight='bold',color=color)
        ax.text(x+.012,y+.113,body,ha='left',va='top',fontsize=7.3,linespacing=1.45,color='#20272E')
        if i<2:
            ax.annotate('',xy=(x+width/2,y-.043),xytext=(x+width/2,y-.011),
                        arrowprops=dict(arrowstyle='-|>',color=color,lw=.8,mutation_scale=9))
ax.text(.5,.045,'Local controls interpret update measurements; training runs measure capability outcomes.',
        ha='center',va='center',fontsize=7.5,color='#444444')
fig.savefig(out/'experimental_evidence_map.png',dpi=240)
fig.savefig(out/'experimental_evidence_map.svg')
print((out/'experimental_evidence_map.png').resolve())
