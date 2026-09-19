from pathlib import Path
import json
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures/table1_sync_20260918'
TMP = ROOT / 'tmp/pdfs/table1_figures'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / 'experiments/table1_sync_20260918/derived_data.json').read_text())
initial = np.array(data['initial'], dtype=float)
ns = np.array(data['uncertainty']['n'], dtype=float)
grid = {(r['configuration'], r['optimizer'], r['update']): np.array(r['scores'], dtype=float) for r in data['grid']}
single = {(r['configuration'], r['update']): np.array(r['scores'], dtype=float) for r in data['single']}
endpoint = {k: np.array(v['scores'], dtype=float) for k, v in data['endpoints'].items()}
teacher = []
for line in (ROOT / 'experiments/aligned_evidence_20260910/teacher_table.tex').read_text().splitlines():
    if re.match(r'^(Math|Code|If|Science) &', line):
        teacher.append(float(line.split('&')[2]))
teacher = np.array(teacher)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 8,
    'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'pdf.fonttype': 42, 'savefig.facecolor': 'white',
})
colors = {'DR': '#D55E00', 'DT': '#009E73', 'GT': '#CC79A7'}
markers = {'DR': 's', 'DT': '^', 'GT': 'D'}
steps = [0, 50, 100, 250, 500]


def sd(scores):
    a = np.asarray(scores)
    return np.sqrt(a * (100-a) / ns)


def mean_sd(scores):
    return np.sqrt(np.sum(sd(scores)**2, axis=-1)) / 4


def curve(rule):
    return np.array([initial] + [grid[(f'I64-{rule}', 'Adam', t)] for t in steps[1:]])


def format_axis(ax):
    ax.grid(axis='y', color='#D8DCE1', lw=.5)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 100, 250, 500])
    ax.set_xlim(-25, 530)
    ax.set_xlabel('Optimizer update', labelpad=2)


def source_patch(src, clip):
    """Keep the specified diagnostic rectangle, deleting all exterior content."""
    doc = fitz.open(src)
    p = doc[0]
    r = p.rect
    c = fitz.Rect(clip)
    for outside in [fitz.Rect(0, 0, r.width, c.y0), fitz.Rect(0, c.y1, r.width, r.height),
                    fitz.Rect(0, c.y0, c.x0, c.y1), fitz.Rect(c.x1, c.y0, r.width, c.y1)]:
        if not outside.is_empty:
            p.add_redact_annot(outside, fill=(1, 1, 1))
    p.apply_redactions(images=2, graphics=2, text=0)
    return doc


# Figure 1: retain diagnostic panels (a,b), regenerate capability curves from tables.
fig = plt.figure(figsize=(6.7, 4.5))
axes_pos = [[.744, .627, .24, .293], [.076, .145, .24, .293],
            [.410, .145, .24, .293], [.744, .145, .24, .293]]
names = ['MATH-500', 'LiveCodeBench v6', 'IFBench strict', 'GPQA (avg@4)']
limits = [(65, 81), (13, 26), (13, 34), (27, 40)]
for i, pos in enumerate(axes_pos):
    ax = fig.add_axes(pos)
    for rule in colors:
        arr = curve(rule)
        ax.errorbar(steps, arr[:, i], yerr=sd(arr)[:, i], color=colors[rule],
                    marker=markers[rule], ms=3.2, lw=1.1, elinewidth=.6, capsize=2,
                    label=rule, zorder=3)
    ax.axhline(initial[i], color='#737373', ls=':', lw=.9)
    ax.axhline(teacher[i], color='#504681', ls='--', lw=.9)
    ax.text(.99, teacher[i]+.18, f'T {teacher[i]:.1f}', color='#504681', fontsize=6.2,
            ha='right', va='bottom', transform=ax.get_yaxis_transform())
    ax.text(.99, initial[i]-.18, f'S {initial[i]:.1f}', color='#737373', fontsize=6.2,
            ha='right', va='top', transform=ax.get_yaxis_transform())
    ax.set_ylim(*limits[i])
    ax.set_ylabel('Score (%)', labelpad=2)
    ax.set_title(f'({chr(99+i)}) {names[i]}', loc='left', pad=5)
    format_axis(ax)
handles = [Line2D([0],[0], color=colors[r], marker=markers[r], lw=1, ms=3, label=r) for r in colors]
handles += [Line2D([0],[0], color='#737373', ls=':', label='Initial student'),
            Line2D([0],[0], color='#504681', ls='--', label='Domain teacher')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.56, .015), ncol=5, frameon=False,
           columnspacing=1.2, handlelength=2.3)
base = TMP / 'normalization_base.pdf'
fig.savefig(base)
plt.close(fig)
doc = fitz.open(base)
clip = fitz.Rect(0, 0, 327, 167)
patch = source_patch(ROOT / 'figures/nine_panel_20260914/section3_normalization.pdf', clip)
doc[0].show_pdf_page(clip, patch, 0, clip=clip)
doc.save(OUT / 'section3_normalization.pdf', garbage=4, deflate=True)

# Figure 4: preserve fixed-state diagnostics (a,b), rebuild all capability contrasts.
arrs = []
for name in ['S-I64', 'M-I64', 'M-I16']:
    columns = []
    for step in [100, 250, 500]:
        if name.startswith('S-'):
            delta = single[('S-I64', step)] - single[('S-PG', step)]
        else:
            cfg = 'I64-DR' if name == 'M-I64' else 'I16-DR'
            delta = grid[(cfg, 'Adam', step)] - grid[('PG-DR', 'Adam', step)]
        columns.append(delta)
    arrs.append(np.array(columns).T)
heat = np.vstack(arrs)
fig = plt.figure(figsize=(5.5, 1.8))
ax = fig.add_axes([.805, .178, .182, .746])
ax.imshow(heat, cmap='RdBu', vmin=-4, vmax=4, aspect='auto')
ax.set_xticks([0,1,2], ['100','250','500'])
ax.set_yticks(range(12), ['Math','Code','IF','GPQA']*3)
ax.tick_params(axis='both', length=0, pad=2, labelsize=6.2)
ax.set_xlabel('Update', labelpad=2, fontsize=7)
for spine in ax.spines.values(): spine.set_visible(False)
for r in range(12):
    for c in range(3):
        v = heat[r,c]
        label = f'{v:+.1f}'
        if label == '-0.0': label = '+0.0'
        ax.text(c, r, label, ha='center', va='center', fontsize=6.2,
                color='white' if abs(v)>2.7 else 'black')
for r in [3.5, 7.5]: ax.axhline(r, color='white', lw=1.5)
for y, label in zip([1.5, 5.5, 9.5], ['S-I64', 'M-I64', 'M-I16']):
    ax.text(-1.95, y, label, rotation=90, va='center', ha='center', fontsize=6.8, clip_on=False)
fig.text(.695, .95, '(c) Score change (pp)', fontsize=7.8, va='center')
base = TMP / 'supervision_base.pdf'
fig.savefig(base)
plt.close(fig)
doc = fitz.open(base)
src = ROOT / 'figures/layout_followup_20260918/section5_supervision.pdf'
clip = fitz.Rect(0, 0, 274, 116)
patch = source_patch(src, clip)
doc[0].show_pdf_page(clip, patch, 0, clip=clip)
clip = fitz.Rect(0, 116, 290, 129.6)
patch = source_patch(src, clip)
doc[0].show_pdf_page(clip, patch, 0, clip=clip)
doc.save(OUT / 'section5_supervision.pdf', garbage=4, deflate=True)

# Supporting aggregate curves: uncertainty applies to the four-domain mean only.
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.05))
fig.subplots_adjust(left=.09, right=.985, bottom=.25, top=.85, wspace=.34)
for rule in colors:
    arr = curve(rule)
    axes[0].errorbar(steps, arr.mean(axis=1), yerr=mean_sd(arr), color=colors[rule],
                     marker=markers[rule], ms=3.2, lw=1.1, elinewidth=.6, capsize=2)
    axes[1].plot(steps, (arr-initial).min(axis=1), color=colors[rule], marker=markers[rule], ms=3.2, lw=1.1)
axes[0].axhline(initial.mean(), color='#737373', ls=':', lw=.9)
axes[0].axhline(teacher.mean(), color='#504681', ls='--', lw=.9)
axes[0].text(515, teacher.mean()+.08, f'T {teacher.mean():.2f}', ha='right', va='bottom', color='#504681', fontsize=6.2)
axes[0].text(515, initial.mean()-.08, f'S {initial.mean():.2f}', ha='right', va='top', color='#737373', fontsize=6.2)
axes[0].set_ylim(31.8, 42.2)
axes[0].set_ylabel('Mean score (%)')
axes[0].set_title('(a) Four-domain mean', loc='left')
axes[1].axhline(0, color='#737373', ls=':', lw=.9)
axes[1].set_ylabel('Worst-domain change (pp)')
axes[1].set_title('(b) Retention from initialization', loc='left')
for ax in axes: format_axis(ax)
fig.legend(handles=handles[:3], loc='lower center', bbox_to_anchor=(.53, -.02), ncol=3, frameon=False)
fig.savefig(OUT / 'section3_capability_support.pdf')
plt.close(fig)

# Normalization contrast plot: mean difference +/- one estimated standard deviation.
fig, axes = plt.subplots(1, 4, figsize=(6.7, 1.75))
fig.subplots_adjust(left=.10, right=.98, bottom=.30, top=.80, wspace=.75)
base = endpoint['I64-DR / Adam']
for i, ax in enumerate(axes):
    for y, rule in enumerate(['GT','DT']):
        v = endpoint[f'I64-{rule} / Adam']
        delta = v[i]-base[i]
        stdev = math.hypot(sd(v)[i], sd(base)[i])
        ax.errorbar(delta, y, xerr=stdev, fmt=markers[rule], color=colors[rule],
                    ms=4, capsize=3, elinewidth=.8, lw=1)
    ax.axvline(0, color='#999999', ls=':', lw=.8)
    ax.set_yticks([0,1], ['GT - DR','DT - DR'])
    ax.set_ylim(1.65, -.65)
    ax.set_title(['Math','Code','IF','GPQA'][i], loc='center')
    ax.set_xlabel('Difference (pp)', fontsize=6.5)
    ax.grid(axis='x', color='#E0E0E0', lw=.4)
    ax.set_axisbelow(True)
fig.savefig(OUT / 'normalization_contrasts.pdf')
plt.close(fig)

for p in sorted(OUT.glob('*.pdf')):
    doc = fitz.open(p)
    doc[0].get_pixmap(matrix=fitz.Matrix(2.6, 2.6)).save(TMP / (p.stem+'.png'))
    print(p.relative_to(ROOT), doc[0].rect)
