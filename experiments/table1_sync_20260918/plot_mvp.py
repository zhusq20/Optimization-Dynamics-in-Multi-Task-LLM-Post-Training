from pathlib import Path
import json
import math
import re
import sys
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
# Read the active Table 1 directly; commented historical rows are excluded.
table = '\n'.join(line for line in (ROOT / 'sections/table_main_outcomes.tex').read_text().splitlines()
                  if not line.lstrip().startswith('%'))
initial_row = re.search(r'Initial student\s*&(.+?)\\\\', table, re.S).group(1)
initial = np.array([float(x) for x in re.findall(r'\d+\.\d+', initial_row)[:4]])
endpoint, endpoint_std = {}, {}
optimizer = 'Adam'
for row in table.split(r'\\'):
    if r'\textit{SGD}' in row:
        optimizer = 'SGD'
    match = re.search(r'\n((?:PG|I16|I64)-(?:DR|DT|GT))\s*\n?&', row)
    if match:
        stats = re.findall(r'\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$', row)
        assert len(stats) == 6, row
        key = (match.group(1), optimizer)
        endpoint[key] = np.array([float(v) for v, std in stats[:4]])
        endpoint_std[key] = np.array([float(std) for v, std in stats[:4]])
assert len(endpoint) == 10
# Intermediate checkpoints retain their reported point scores.
grid = {}
config = None
for line in (ROOT / 'experiments/visual_mvp_20260918/qwen_all_checkpoints_table.tex').read_text().splitlines():
    cells = line.split('&')
    if len(cells) != 7:
        continue
    if re.match(r'^(PG|I16|I64)-', cells[0]):
        config = tuple(x.strip() for x in cells[0].split('/'))
    if config and re.search(r'\d+', cells[1]):
        step = int(re.search(r'\d+', cells[1]).group())
        grid[(*config, step)] = (endpoint[config] if step == 500 else
            np.array([float(re.search(r'\d+\.\d+', x).group()) for x in cells[2:6]]))
single = {}
for line in (ROOT / 'experiments/appendix_mvp_20260914/capability_table.tex').read_text().splitlines():
    if line.startswith('S-'):
        cells = line.split('&')
        single[(cells[0].strip(), int(cells[1]))] = np.array([
            float(re.search(r'\d+\.\d+', x).group()) for x in cells[2:6]])

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
steps = [0, 100, 250, 500]
seed_sd = json.loads(
    (ROOT / 'experiments/table1_sync_20260918/figure1_seed_sd.json').read_text())


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
# Display-only horizontal offsets separate rules at the same checkpoint.
x_offsets = {'DR': -24, 'DT': 0, 'GT': 24}


def display_x(checkpoints, rule):
    checkpoints = np.asarray(checkpoints)
    # Keep the shared initial checkpoint compact; widen SD-bearing groups.
    return checkpoints + x_offsets[rule] * np.where(checkpoints < 100, .35, 1.)


for i, pos in enumerate(axes_pos):
    ax = fig.add_axes(pos)
    for rule in colors:
        arr = curve(rule)
        display_steps = display_x(steps, rule)
        ax.plot(display_steps, arr[:, i], color=colors[rule], lw=.85, alpha=.8, zorder=2)
        ax.plot(display_steps, arr[:, i], color=colors[rule], marker=markers[rule],
                ls='none', ms=4, markeredgecolor='white', markeredgewidth=.45,
                label=rule, zorder=5)
        # User-provided measured seed SD at 0/100/250.
        sd_steps = seed_sd['steps']
        ax.errorbar(display_x(sd_steps, rule), [arr[steps.index(t), i] for t in sd_steps],
                    yerr=np.array(seed_sd['sd'][rule])[:, i],
                    color=colors[rule], fmt='none', elinewidth=.9, capsize=1.3, capthick=.9, zorder=4)
        # Preserve the training-seed SD reported in Table 1 at update 500.
        ax.errorbar([500 + x_offsets[rule]], [arr[-1, i]],
                    yerr=[endpoint_std[(f'I64-{rule}', 'Adam')][i]],
                    color=colors[rule], fmt='none', elinewidth=.9, capsize=1.3, capthick=.9, zorder=4)
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
    ax.set_xlim(-25, 545)
handles = [Line2D([0],[0], color=colors[r], marker=markers[r], lw=.85, ms=4,
                  markeredgecolor='white', markeredgewidth=.45, label=r) for r in colors]
handles += [Line2D([0],[0], color='#737373', ls=':', label='Initial student'),
            Line2D([0],[0], color='#504681', ls='--', label='Domain teacher')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.56, .015), ncol=5, frameon=False,
           columnspacing=1.2, handlelength=2.3)
fig.text(.56, .003, '± SD over 3 training seeds at updates 100, 250, and 500', ha='center', va='bottom', fontsize=6.5)
base = TMP / 'normalization_base.pdf'
fig.savefig(base)
plt.close(fig)
doc = fitz.open(base)
clip = fitz.Rect(0, 0, 327, 167)
patch = source_patch(ROOT / 'figures/nine_panel_20260914/section3_normalization.pdf', clip)
doc[0].show_pdf_page(clip, patch, 0, clip=clip)
doc.save(OUT / 'section3_normalization.pdf', garbage=4, deflate=True)
if '--figure1-only' in sys.argv:
    doc[0].get_pixmap(matrix=fitz.Matrix(2.6, 2.6)).save(TMP / 'section3_normalization.png')
    print('Figure 1: measured seed SD at 0/100/250; Table 1 seed SD at 500.')
    sys.exit(0)

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


assert abs(heat[11, 2] - 0.04) < 1e-9
assert abs(heat[7, 2] + 1.23) < 1e-9
for name in ['section3_normalization', 'section5_supervision']:
    with fitz.open(OUT / (name + '.pdf')) as rendered:
        rendered[0].get_pixmap(matrix=fitz.Matrix(2.6, 2.6)).save(TMP / (name + '.png'))
print('Figure 4 GPQA @500: I16 = +0.04 pp; I64 = -1.23 pp.')
print('Figure 1: measured seed SD at 0/100/250; Table 1 seed SD at 500.')
