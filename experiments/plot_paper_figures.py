#!/usr/bin/env python3
"""Render question-led paper figures, preserving every plotted raw observation."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
DOMAINS = ['math', 'code', 'if', 'science']
DOMAIN_NAMES = ['Math', 'Code', 'Instruction following', 'Science']
DATASETS = ['math500_pass1', 'livecodebench_postcutoff_pass1', 'ifbench_strict', 'gpqa_diamond_avg4']
BENCHMARKS = ['MATH-500 pass@1', 'LiveCodeBench pass@1', 'IFBench strict', 'GPQA Diamond avg@4']
COLORS = ['#27699c', '#c05d28', '#268574', '#8a65a7']
METHODS = ['zero_gradient', 'sampled_pg', 'student_topk', 'topk_intersection', 'teacher_topk', 'full_vocab']
METHOD_NAMES = ['Zero gradient', 'PG', 'Student Top64', 'Top64 intersection', 'Teacher Top64', 'Full vocabulary']
METHOD_COLORS = ['#69747c', COLORS[0], '#8b9cab', COLORS[1], COLORS[3], COLORS[2]]
CLOCK_LABELS = {'train/step': 'Gradient-microbatch log index', 'mopd/update': 'Optimizer updates',
                'rollout/step': 'Rollout index', 'checkpoint_step': 'Checkpoint optimizer updates'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 180})


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def chart(height=4.6):
    fig, ax = plt.subplots(figsize=(7.6, height), layout='constrained')
    ax.grid(axis='y', color='#e4e8eb', lw=.7)
    ax.set_axisbelow(True)
    return fig, ax


def draw_raw(ax, events, key, label, color, scale=1, linewidth=.9):
    rows = [e for e in events if isinstance(e['metrics'].get(key), (int, float))
            and np.isfinite(e['metrics'][key])]
    if not rows:
        return []
    x = np.asarray([e['step'] for e in rows])
    y = np.asarray([e['metrics'][key] * scale for e in rows])
    # Do not bridge a missing log interval with an invented continuous segment.
    discontinuities = np.where(np.diff(x) > 1)[0] + 1
    for i, ix in enumerate(np.split(np.arange(len(x)), discontinuities)):
        ax.plot(x[ix], y[ix], color=color, lw=linewidth, label=label if i == 0 else None,
                marker='.' if len(ix) == 1 else None)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=HERE / 'paper_curves_20260908')
    ap.add_argument('--probes', type=Path, default=HERE / 'local_probe_results_20260908')
    ap.add_argument('--output', type=Path, default=HERE.parent / 'figures/paper_curves_20260908')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.data / 'data.json.gz', 'rt') as stream:
        data = json.load(stream)
    for source in data['sources']:
        with gzip.open(args.data / source['file'], 'rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != source['sha256']:
            raise ValueError('Raw scalar mirror changed: ' + source['file'])
    probe_manifest = json.loads((args.probes / 'data_manifest.json').read_text())
    for record in probe_manifest['files']:
        with (args.probes / record['path']).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256']:
            raise ValueError('Local probe evidence changed: ' + record['path'])
    probes = {j['id']: json.loads((args.probes / 'raw' / j['id'] / 'measurements.json').read_text())
              for j in probe_manifest['jobs']}
    tables = {}
    for name in ['normalization_gradients', 'responses', 'pairs', 'thresholds', 'branches', 'teacher_js']:
        with (args.probes / 'plot_data' / f'{name}.csv').open() as stream:
            tables[name] = list(csv.DictReader(stream))
    figures, manifest, plotted, layout_checks = {}, [], [], []

    def save(fig, name, question, reading, caption, kind='line', tier='appendix', sources=None):
        figures[name] = fig
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            xlo, xhi = sorted(ax.get_xlim())
            ylo, yhi = sorted(ax.get_ylim())
            labels = [ax.xaxis.label, ax.yaxis.label]
            labels.extend(label for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()) if xlo <= tick <= xhi)
            labels.extend(label for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()) if ylo <= tick <= yhi)
            if ax.get_legend():
                labels.extend(ax.get_legend().get_texts())
            for label in labels:
                if not label.get_text():
                    continue
                box = label.get_window_extent(renderer)
                within = box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1
                layout_checks.append({'figure': name, 'label': label.get_text(), 'inside_export': bool(within)})
                if not within:
                    raise ValueError(f'Clipped figure label in {name}: {label.get_text()}')
        for extension in ('pdf', 'png', 'svg'):
            fig.savefig(args.output / f'{name}.{extension}')
        plt.close(fig)
        manifest.append({'id': name, 'question': question, 'reading': reading, 'caption': caption,
                         'type': kind, 'tier': tier, 'sources': sources or [], 'axes': len(fig.axes)})

    def raw_points(name, run, rows, key, scale=1):
        plotted.extend({'figure': name, 'run': run, 'metric': key, 'x': e['step'],
                        'y': e['metrics'][key] * scale, 'raw_value': e['metrics'][key],
                        'scale': scale, 'clock': e['clock'], 'source': e['sources'][key]}
                       for e in rows)

    selected = [r for r in data['evaluations'] if r['selected']]
    reference = {(r['family'], r['dataset']): r for r in selected if r['training_phase'] == 'reference'}
    # First show the observed online outcome, with familiar absolute benchmark units.
    for domain, dataset, benchmark in zip(DOMAINS, DATASETS, BENCHMARKS):
        name = '01_eval_' + domain
        fig, ax = chart()
        initial = reference['initial_student', dataset]
        teacher = reference['teacher_' + domain, dataset]
        ax.axhline(100 * initial['score'], color='#888888', ls=':', lw=1.2, label='Initial student')
        ax.axhline(100 * teacher['score'], color='#333333', ls='--', lw=1.2, label='Domain teacher')
        for family, color, label in [('m-pg-s42', COLORS[0], 'Joint PG'), ('s-pg-s42', COLORS[1], 'Math-only PG')]:
            rows = sorted([r for r in selected if r['family'] == family and r['dataset'] == dataset], key=lambda r: r['step'])
            rows = [initial, *rows]
            ax.plot([r['step'] for r in rows], [100*r['score'] for r in rows], 'o-', color=color,
                    label=label, lw=2, markersize=5, markerfacecolor='white' if family.startswith('s-') else color)
            for r in rows:
                plotted.append({'figure': name, 'run': family, 'metric': r['metric'], 'x': r['step'],
                                'y': 100*r['score'], 'raw_value': r['score'], 'scale': 100,
                                'clock': 'checkpoint_step', 'source': r['source'], 'attempt': r['attempt']})
        ax.set(xlabel='Checkpoint optimizer updates', ylabel=benchmark + ' (%)', xlim=(-12, 515))
        ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1))
        save(fig, name, f'{benchmark}：联合训练的能力随训练如何变化？',
             '先看实心蓝线，再与初始学生及领域教师虚线比较；橙线是只训练数学后的同一测试集表现。',
             f'Raw {benchmark} evaluations at the recorded checkpoints; no smoothing or confidence bands. '
             'Blue: joint four-domain PG. Orange: math-only PG, also evaluated on other domains. '
             'Both start from the same measured initial checkpoint. Dotted and dashed lines are measured student and domain-teacher references. '
             'One training seed; the first complete verified evaluation attempt is selected by completion time, never by score. '
             'All attempts are retained in evaluation_attempts.csv and the raw explorer. The math-only series stops at its last available evaluation. '
             'Equal updates do not imply equal per-domain exposure; see the math sample-exposure view.', tier='main')

    # Per-domain sample exposure is available from the actual allocation records.
    fig, ax = chart()
    dataset = DATASETS[0]
    initial = reference['initial_student', dataset]
    ax.axhline(100 * reference['teacher_math', dataset]['score'], color='#333333', ls='--', lw=1.2, label='Math teacher')
    for family, color, label in [('m-pg-s42', COLORS[0], 'Joint PG'), ('s-pg-s42', COLORS[1], 'Math-only PG')]:
        rows = sorted([r for r in selected if r['family'] == family and r['dataset'] == dataset
                       and r.get('domain_training_prompts') is not None], key=lambda r: r['step'])
        ax.plot([0]+[r['domain_training_prompts']/1000 for r in rows], [100*initial['score']]+[100*r['score'] for r in rows],
                'o-', color=color, label=label, lw=2, markersize=5)
    ax.set(xlabel='Cumulative math training prompts (thousands)', ylabel='MATH-500 pass@1 (%)')
    ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1))
    save(fig, '02_eval_math_exposure', '数学能力差异能否由训练样本暴露解释？',
         '同样的评估点改用已消耗数学prompt数；横轴来自allocation记录。',
         'The same selected MATH-500 scores as 01_eval_math, against cumulative math prompts from deduplicated allocation records. '
         'One response per prompt in these runs. The joint run uses 16 math prompts per update and the math-only run uses 64. '
         'This plot compares sample exposure, not wall-time or GPU efficiency.')

    # Four signals on one stable run: domain colors have the same meaning throughout.
    joint = data['runs']['m-pg-s42']
    for name, namespace, metric, ylabel, question, reading, scale, tier in [
        ('03_train_domain_kl', 'mopd', 'teacher_loss', 'Sampled teacher–student log ratio (nats)',
         '学生接近各领域教师的速度是否一致？', '四条线是Joint PG各领域的原始teacher_loss；用它解释评估轨迹，不能代替能力分数。', 1, 'main'),
        ('04_train_token_share', 'rollout', 'token_share', 'Valid response token share (%)',
         '相同prompt份额是否带来相同token份额？', '虚线是每域25%的prompt分配；原始token份额随生成长度变化。', 100, 'main'),
        ('A01_response_length', 'rollout', 'mean_response_length', 'Mean response length (tokens)',
         'token份额变化是否伴随回答长度变化？', '展示实际在线生成的长度轨迹；与局部cap512/2048探针分别解释。', 1, 'appendix'),
        ('A02_truncation', 'rollout', 'truncation_rate', 'Truncated responses (%)',
         '训练回答是否越来越容易触及长度上限？', '每个点来自当次rollout，不做窗口平均。', 100, 'appendix'),
    ]:
        fig, ax = chart()
        for task, label, color in zip(DOMAINS, DOMAIN_NAMES, COLORS):
            key = f'mopd/task/{task}/{metric}'
            rows = draw_raw(ax, joint['events'][namespace], key, label, color, scale)
            raw_points(name, joint['id'], rows, key, scale)
        if metric == 'token_share':
            ax.axhline(25, color='#444444', ls='--', lw=1.1, label='Prompt share: 25% each')
        if metric == 'teacher_loss':
            values = [e['metrics'][f'mopd/task/{task}/teacher_loss'] for e in joint['events']['mopd'] for task in DOMAINS]
            if min(values) > 0:
                ax.set_yscale('log')
                ylabel = 'Sampled log ratio (nats; log scale)'
        clock = 'mopd/update' if namespace == 'mopd' else 'rollout/step'
        ax.set(xlabel=CLOCK_LABELS[clock], ylabel=ylabel)
        ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1))
        save(fig, name, question, reading,
             f'Joint PG, seed 42. All recorded {metric} observations, without smoothing, rebinning, or downsampling. '
             f'The horizontal clock is {clock}. Colors identify domains consistently across the training diagnostics. '
             'teacher_loss in this PG run estimates the sampled student-minus-teacher log-probability ratio, not an exact full-vocabulary KL. '
             'The four series describe one run, not four independent seeds.', tier=tier)

    for domain, benchmark in zip(DOMAINS, BENCHMARKS):
        name = 'A03_raw_reward_' + domain
        fig, ax = chart()
        key = f'rollout/reward/{domain}/mean'
        for family, color, label in [('m-pg-s42', COLORS[0], 'Joint PG'), ('s-pg-s42', COLORS[1], 'Math-only PG')]:
            rows = draw_raw(ax, data['runs'][family]['events']['rollout'], key, label, color)
            raw_points(name, family, rows, key)
        ax.set(xlabel='Rollout index', ylabel='Logged task reward (mean)')
        ax.legend(frameon=False)
        save(fig, name, f'{domain}：原始training reward怎样变化？',
             '这是训练数据上的原始verifier读数。需结合独立benchmark曲线判断泛化。',
             f'Raw {key}, with every recorded observation. This reward is observed on training prompts and has coefficient zero in the OPD loss. '
             'Verifier and infrastructure problems, if present in the original run, remain in the raw values; these curves are not rescored or substituted for benchmark evaluations.')

    for name, namespace, key, ylabel in [
        ('A04_raw_train_loss', 'train', 'train/loss', 'Logged PG objective'),
        ('A05_raw_sampled_logratio', 'train', 'train/sampled_reverse_kl_logratio', 'Sampled reverse-KL log ratio (nats)'),
        ('A06_aggregate_gradient', 'mopd', 'mopd/aggregate_grad_norm', 'Aggregate pre-clipping gradient norm'),
    ]:
        fig, ax = chart()
        for family, color, label in [('m-pg-s42', COLORS[0], 'Joint PG'), ('s-pg-s42', COLORS[1], 'Math-only PG')]:
            rows = draw_raw(ax, data['runs'][family]['events'][namespace], key, label, color, linewidth=.65)
            raw_points(name, family, rows, key)
        clock = 'train/step' if namespace == 'train' else 'mopd/update'
        ax.set(xlabel=CLOCK_LABELS[clock], ylabel=ylabel)
        ax.legend(frameon=False)
        save(fig, name, f'{key}：未经平滑的训练记录是什么样？',
             '保留所有原始点。train/step是梯度microbatch序号；完整梯度范数使用mopd/update记录。',
             f'Raw {key} against {clock}, without smoothing or filtering. A train/step increment is a logged gradient microbatch, '
             'not an optimizer update. The two PG runs use the same logged metric. PG surrogate loss and sampled KL have distinct meanings. '
             'Zero train/grad_norm values from accumulation slices are not used as the aggregate optimizer gradient norm.')

    # Preserve continuation identities. Never relabel the inherited Student64 section as I64.
    for domain, dataset, benchmark in zip(DOMAINS, DATASETS, BENCHMARKS):
        name = 'A07_eval_continuations_' + domain
        fig, ax = chart()
        for reduction, color in zip(['dr', 'dt', 'gt'], COLORS):
            family = f'm-itk64-{reduction}-s42'
            rows = sorted([r for r in selected if r['family'] == family and r['dataset'] == dataset], key=lambda r: r['step'])
            ax.plot([r['step'] for r in rows], [100*r['score'] for r in rows], 'o--', color=color,
                    label='Student64 → I64 · ' + reduction.upper(), lw=1.8)
            for r in rows:
                plotted.append({'figure': name, 'run': family, 'metric': r['metric'], 'x': r['step'],
                                'y': 100*r['score'], 'raw_value': r['score'], 'scale': 100,
                                'clock': 'checkpoint_step', 'source': r['source'], 'training_phase': r['training_phase']})
        ax.axhline(100*reference['initial_student', dataset]['score'], color='#888888', ls=':', label='Initial student')
        ax.axhline(100*reference['teacher_'+domain, dataset]['score'], color='#333333', ls='--', label='Domain teacher')
        ax.set(xlabel='Checkpoint optimizer updates', ylabel=benchmark+' (%)', xlim=(75, 275))
        ax.legend(frameon=False, fontsize=9, ncol=2, loc='lower left', bbox_to_anchor=(0, 1))
        save(fig, name, f'{benchmark}：已有归一化续训分支有什么评估证据？',
             '100步属于Student64父轨迹；250步若有数据则已切换为交集目标。数学/IF尚无250步完整评估时只画单点。',
             'Existing joint Student64-to-I64 continuations, with distinct domain-response (DR), domain-token (DT), and global-token (GT) reductions. '
             'I64 begins at updates 137/152/151 for DR/DT/GT. Their step-100 evaluations are inherited Student64 checkpoints, not I64-trained endpoints. '
             'Only completed verified evaluations are plotted; no extrapolation to unmeasured checkpoints. Dashed connectors cross a change of objective. '
             'These are mixed-history trajectories, not from-base controlled I64 ablations.')

    # Local mechanism figures are added below; their saved optimizer states are explicit.
    render_probes(args, probes, tables, chart, save, plotted)
    main_order = [r['id'] for r in manifest if r['tier'] == 'main']
    all_order = main_order + [r['id'] for r in manifest if r['tier'] != 'main']
    for filename, order in [('main_figures.pdf', main_order), ('all_figures.pdf', all_order)]:
        with PdfPages(args.output / filename) as pdf:
            for name in order:
                pdf.savefig(figures[name])
    for fig in figures.values():
        plt.close(fig)
    dump(args.data / 'figure_manifest.json', {'figures': manifest, 'main_order': main_order,
          'all_order': all_order, 'raw_policy': 'All logged points; no smoothing in exported static figures.',
          'counts': {kind: sum(r['type'] == kind for r in manifest) for kind in ['line', 'point', 'bar', 'heatmap']},
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    dump(args.data / 'plotted_values.json', plotted)
    dump(args.data / 'layout_checks.json', layout_checks)
    write_gallery(args, data, manifest, main_order)
    write_explorer(args, data)
    print(json.dumps({'figures': len(manifest), 'main_figures': len(main_order), 'output': str(args.output)}, indent=2))


def write_gallery(args, data, manifest, main_order):
    by_id = {r['id']: r for r in manifest}
    header = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOPD · 训练轨迹与机制图</title><style>
@font-face{font-family:MopdSans;src:url('fonts/MopdSans.woff') format('woff');font-display:swap}
body{font:16px/1.7 MopdSans,system-ui,sans-serif;color:#243541;background:#f5f7f8;margin:0}main{max-width:1060px;margin:auto;padding:36px 24px}
h1{font-size:32px;line-height:1.3}h2{font-size:22px}a{color:#246b9e}nav{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}
.lead{font-size:18px}.note{padding:18px 22px;background:#e9f1f6;border-left:4px solid #27699c}
article{background:white;border:1px solid #dde4e8;border-radius:8px;padding:24px;margin:24px 0}article img{width:100%;max-width:860px;display:block;margin:10px auto}
.tag{font-size:12px;letter-spacing:.08em;color:#667783}.caption{font-size:14px;color:#536673}summary{cursor:pointer;font-weight:600}code{font-size:13px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}
</style><main><div class="tag">MOPD · OBSERVED TRAINING AND LOCAL MECHANISMS</div>
<h1>先看能力轨迹，再解释训练为什么这样变化</h1>
<p class="lead">评估与原始训练曲线是主线。每张图只回答一个问题，坐标、参照和适用范围写清楚。</p>
<nav><a href="main_figures.pdf">正文候选 PDF</a><a href="raw_curves.html">W&amp;B 原始曲线浏览器</a><a href="all_figures.pdf">全部图 PDF</a><a href="../../experiments/paper_curves_20260908/DESIGN_zh.md">论文参考与图表配置</a><a href="../../experiments/paper_curves_20260908/FIGURE_CAPTIONS.md">逐图 caption</a></nav>'''
    body = [header]
    body.append('<p class="note">联合 PG 的 MATH-500 从 <b>55.6% → 66.4%</b>，IFBench strict 从 <b>17.3% → 14.7%</b>。先用领域评估看保留和退化，再检查 KL、长度与 token 份额。局部 BF16 探针用于机制解释。</p>')
    body.append('<p>每张图单独导出 PDF / PNG / SVG。所有静态训练曲线保留原始点、默认不平滑。评估点只来自完整核验产物；单个训练 seed 不绘制虚假的置信区间。</p>')

    def card(row):
        name = html.escape(row['id'])
        return (f'<article id="{name}"><div class="tag">{name} · {row["type"].upper()}</div>'
                f'<h2>{html.escape(row["question"])}</h2><p>{html.escape(row["reading"])}</p>'
                f'<a href="{name}.pdf"><img src="{name}.png" loading="lazy" alt="{html.escape(row["question"])}"></a>'
                f'<p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p>'
                f'<details><summary>图注、数据和解释范围</summary><p class="caption">{html.escape(row["caption"])}</p></details></article>')
    for name in main_order:
        body.append(card(by_id[name]))
    body.append('<details id="appendix"><summary>展开补充图：raw reward / loss、长度、截断、续训评估和局部对照</summary>')
    for row in manifest:
        if row['tier'] != 'main':
            body.append(card(row))
    body.append('</details><h2>数据来源与缺口</h2><p>原始scalar镜像由训练代码在同一次调用中先写JSONL，再提交给 <code>wandb.log(metrics)</code>。本页使用这些完整本地镜像，没有经W&amp;B网页抽样。浏览器链接可打开原run。</p>')
    body.append('<p>Policy entropy 未记录，未用负log-probability冒充；GPU成本比较尚未完成跨续训的统一核算。Student64→I64的切换点保留：DR 137、DT 152、GT 151、单任务101。100步评估属于父轨迹。</p>')
    body.append('<p>参考：<a href="https://arxiv.org/pdf/2608.19098v1">Open-MOPD Fig.1 / 3 / 4</a> 的能力与预算轨迹；<a href="https://arxiv.org/pdf/2606.30406">MOPD Fig.2 / 3</a> 的分域样本效率与训练诊断。这里展示本项目的真实测量。</p></main></html>')
    font = base64.b64encode((args.data / 'fonts/MopdSans.woff').read_bytes()).decode()
    (args.output / 'index.html').write_text('\n'.join(body).replace('fonts/MopdSans.woff', 'data:font/woff;base64,'+font))
    captions = ['# Figure captions', '', 'Static training curves retain all raw points. Local-probe envelopes are bank min–max, not training-seed confidence intervals.', '']
    for row in manifest:
        captions.extend([f'## {row["id"]}', '', row['question'], '', row['caption'], ''])
    (args.data / 'FIGURE_CAPTIONS.md').write_text('\n'.join(captions))


def write_explorer(args, data):
    # Curate useful scalar keys but retain every point for each selected key.
    # Full metric streams, including keys outside this view, remain in raw/*.jsonl.gz.
    training_keys = {
        'train/loss', 'train/sampled_reverse_kl_logratio', 'train/student_top64_surrogate',
        'train/student_top64_retained_mass', 'train/student_teacher_top64_intersection_surrogate',
        'train/student_teacher_top64_intersection_normalized_logratio',
        'train/student_teacher_top64_intersection_size', 'train/student_teacher_top64_intersection_retained_weight',
        'train/student_teacher_top64_intersection_empty_fraction', 'mopd/aggregate_grad_norm',
        'mopd/aggregate_grad_clipped', 'rollout/response_len/mean', 'rollout/truncated_ratio',
        'rollout/sandbox/infrastructure_errors', 'rollout/sandbox/execution_errors',
    }
    for task in DOMAINS:
        training_keys.add(f'rollout/reward/{task}/mean')
        for metric in ['teacher_loss', 'token_share', 'prompt_share', 'mean_response_length', 'truncation_rate']:
            training_keys.add(f'mopd/task/{task}/{metric}')
    series = []
    for name, run in data['runs'].items():
        for namespace, events in run['events'].items():
            keys = sorted({key for e in events for key in e['metrics']} & training_keys)
            for key in keys:
                rows = [e for e in events if isinstance(e['metrics'].get(key), (int, float)) and np.isfinite(e['metrics'][key])]
                if not rows:
                    continue
                series.append({'run': name, 'label': run['label'], 'namespace': namespace,
                               'key': key, 'clock': rows[0]['clock'],
                               'x': [e['step'] for e in rows], 'y': [e['metrics'][key] for e in rows],
                               'line': [e['sources'][key]['line'] for e in rows],
                               'source': rows[0]['sources'][key]['file'], 'url': run['wandb_url'],
                               'note': f'原始 {namespace} 事件；'+(f"交集目标从 update {run['first_intersection_update']} 开始。本run只含续训段；父轨迹 {run['parent_run']} 单独可选。" if run['parent_run'] else '独立run；无窗口平均。')})
    grouped = defaultdict(list)
    for row in data['evaluations']:
        if row['verified']:
            grouped[row['family'], row['dataset']].append(row)
    for (family, dataset), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (r['step'], r['completed_at_utc']))
        for field, key in [('score', 'eval/capability/'+dataset),
                           ('truncation_rate', 'eval/capability/'+dataset+'/truncation_rate'),
                           ('response_length', 'eval/'+dataset+'/response_len/mean')]:
            present = [r for r in rows if r.get(field) is not None]
            if not present:
                continue
            series.append({'run': family, 'label': data['runs'].get(family, {}).get('label', family),
                           'namespace': 'eval', 'key': key, 'clock': 'checkpoint_step',
                           'x': [r['step'] for r in present], 'y': [r[field] for r in present],
                           'line': [r['source']['line'] for r in present],
                           'sources': [r['source']['file'] for r in present],
                           'source': '', 'url': None,
                           'note': '所有完整核验的评估attempt；相同checkpoint重跑保留为独立点，不当作训练seed。eval本地step已按产物映射回checkpoint update。'})
    # Export a reusable long-form CSV alongside the frozen original streams.
    with (args.data / 'raw_curve_values.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'namespace', 'metric', 'clock', 'step', 'value', 'source', 'source_line'])
        for s in series:
            for i, (x, y) in enumerate(zip(s['x'], s['y'])):
                writer.writerow([s['run'], s['namespace'], s['key'], s['clock'], x, y,
                                 s.get('sources', [s['source']] * len(s['x']))[i], s['line'][i]])
    template = (Path(__file__).with_name('raw_curve_explorer.html')).read_text()
    font = base64.b64encode((args.data / 'fonts/MopdSans.woff').read_bytes()).decode()
    template = template.replace('fonts/MopdSans.woff', 'data:font/woff;base64,'+font)
    payload = json.dumps({'series': series, 'clocks': CLOCK_LABELS}, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (args.output / 'raw_curves.html').write_text(template.replace('__CURVE_DATA__', payload.replace('</', '<\\/')))
    dump(args.data / 'raw_curve_inventory.json', {'series': len(series), 'raw_points': sum(len(s['x']) for s in series),
         'training_runs': len(data['runs']), 'available_metrics': sorted({s['key'] for s in series}),
         'missing_metrics': data['missing_metrics'], 'default_smoothing': 0,
         'series_inventory': [{k: v for k, v in s.items() if k not in {'x', 'y', 'line', 'sources'}} | {'points': len(s['x'])} for s in series]})


def render_probes(args, probes, tables, chart, save, plotted):
    scope = ('Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, '
             'not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. ')

    def record(name, row, metric, value, **extra):
        plotted.append({'figure': name, 'job': row['job'], 'metric': metric, 'y': float(value),
                        'source': '../local_probe_results_20260908/' + row['source'],
                        **{k: row[k] for k in ['state', 'left', 'right', 'branch', 'reduction_pair', 'threshold'] if k in row}, **extra})

    jobs = ['normalization-m-pg250-draw42', 'normalization-m-pg250-draw43',
            'normalization-m-pg500-draw42', 'normalization-m-pg500-draw43',
            'normalization-m-pg500-draw42-long2048',
            'normalization-m-pg500-draw42-quota1234-uniform',
            'normalization-m-pg500-draw42-quota1234-prompt']
    labels = ['PG250 · bank 42', 'PG250 · bank 43', 'PG500 · bank 42', 'PG500 · bank 43',
              'PG500 · long responses', 'PG500 · unequal quota', 'PG500 · prompt prior']
    name = '05_normalization_direction'
    fig, ax = chart(5.1)
    for pair, offset, color, label in [('DR_DT', -.14, COLORS[0], 'Domain-token vs domain-response'),
                                       ('DR_GT', .14, COLORS[1], 'Global-token vs domain-response')]:
        values = []
        for job in jobs:
            row = next(r for r in tables['normalization_gradients'] if r['job'] == job and r['reduction_pair'] == pair)
            values.append(float(row['cosine']))
            record(name, row, 'raw_gradient_cosine', float(row['cosine']), pair=pair)
        ax.plot(values, np.arange(len(jobs))+offset, 'o', color=color, label=label, markersize=6)
    ax.set(yticks=np.arange(len(jobs)), yticklabels=labels, xlabel='Gradient cosine vs domain-response baseline', xlim=(.60, 1.025))
    ax.invert_yaxis()
    ax.axvline(1, color='#555555', ls='--', lw=1)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1), fontsize=9)
    save(fig, name, '仅改变平均规则，会改变梯度方向吗？',
         '每行固定同一个学生和response bank。越接近右侧1，方向越接近domain-response基准；点向左偏离表示方向改变。',
         scope + 'Each point is one recorded pre-clipping gradient cosine, replacing the previous seven-condition heatmap. '
         'The first four rows use cap512; the long-response row uses cap2048. The final two rows use the same unequal quota bank '
         'with uniform and prompt-proportional domain priors. The reference at one indicates identical directions. '
         'No ordering or interpolation is imposed on these categorical conditions.', kind='point', tier='main')

    pair_rows = [r for r in tables['pairs'] if r['kind'] == 'teachers' and float(r['threshold']) == 1e-5]
    conditions = [(250, 'sampled_pg'), (500, 'sampled_pg'), (250, 'topk_intersection'), (500, 'topk_intersection')]
    condition_labels = ['PG probe · step 250', 'PG probe · step 500', 'I64 probe · step 250', 'I64 probe · step 500']
    name = '06_teacher_input_control'
    fig, ax = chart()
    for mode, offset, color, label in [('routed', -.13, COLORS[0], 'Task-specific inputs'),
                                     ('common', .13, COLORS[1], 'Identical shared inputs')]:
        for i, (step, loss) in enumerate(conditions):
            rows = [r for r in pair_rows if r['state'] == f'm-pg{step}'
                    and r['left'].startswith(mode+'/') and r['right'].startswith(mode+'/')
                    and r['left'].endswith('/'+loss) and r['right'].endswith('/'+loss)]
            assert len(rows) == 12, (step, loss, mode, len(rows))
            values = [float(r['gradient_cosine']) for r in rows]
            mean = float(np.mean(values))
            ax.errorbar(mean, i+offset, xerr=[[mean-min(values)], [max(values)-mean]], fmt='o',
                        color=color, capsize=3, label=label if i == 0 else None, markersize=6)
            for r, v in zip(rows, values):
                record(name, r, 'raw_gradient_cosine', v, mode=mode, loss=loss, step=step)
    ax.set(yticks=range(4), yticklabels=condition_labels, xlabel='Cosine between teacher-specific raw gradients', xlim=(-.08, 1.045))
    ax.invert_yaxis()
    ax.axvline(0, color='#888888', ls=':', lw=.8)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1))
    save(fig, name, '教师方向差异来自教师本身，还是来自输入不同？',
         '蓝色是各教师使用各自任务输入，橙色是同一组输入。横向距离直接显示这一控制带来的变化。',
         scope + 'Means and full minimum–maximum ranges over six teacher pairs and two banks per condition. '
         'PG and I64 are local probe objectives on the indicated PG-trained states. Shared inputs change the observed gradient alignment substantially. '
         'Ranges show the observed dependent measurements, not confidence intervals.', kind='point', tier='main')

    name = 'A08_gradient_vs_writeback'
    fig, ax = chart()
    for metric, offset, color, label in [('gradient_cosine', -.13, COLORS[0], 'Raw gradient'),
                                        ('bf16_cosine', .13, COLORS[1], 'Saved-Adam BF16 writeback')]:
        for i, (step, loss) in enumerate(conditions):
            rows = [r for r in pair_rows if r['state'] == f'm-pg{step}' and r['left'].startswith('routed/')
                    and r['right'].startswith('routed/') and r['left'].endswith('/'+loss) and r['right'].endswith('/'+loss)]
            values = [float(r[metric]) for r in rows]
            mean = float(np.mean(values))
            ax.errorbar(mean, i+offset, xerr=[[mean-min(values)], [max(values)-mean]], fmt='o', color=color,
                        capsize=3, label=label if i == 0 else None)
            for r, v in zip(rows, values):
                record(name, r, metric, v)
    ax.set(yticks=range(4), yticklabels=condition_labels, xlabel='Cosine between teacher-specific directions', xlim=(-.08, 1.045))
    ax.invert_yaxis()
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1))
    save(fig, name, '原始梯度近正交，BF16写回也近正交吗？',
         '同一行比较同一组教师对的梯度与写回方向；保存的Adam历史参与局部写回。',
         scope+'Task-specific inputs. Each marker is the mean over the same six pairs and two banks; segments are observed min–max. '
         'Strong update alignment does not identify teacher-specific historical subnetworks.', kind='point')

    thresholds = [0, 1e-8, 1e-7, 1e-6, 1e-5]
    states = ['m-pg500', 'm-pg250', 's-pg250', 'm-i64dr250', 's-i64250']
    for state in states:
        name = '07_threshold_' + state
        fig, ax = chart()
        for method, label, color, marker in zip(METHODS, METHOD_NAMES, METHOD_COLORS, ['x', 'o', 'v', 's', '^', 'D']):
            if state == 'm-pg500' and method in {'student_topk', 'teacher_topk'}:
                continue
            means, low, high = [], [], []
            for threshold in thresholds:
                rows = [r for r in tables['thresholds'] if r['state'] == state and r['kind'] == 'density'
                        and r['branch'] == method and float(r['threshold']) == threshold]
                assert len(rows) == 4
                values = [float(r['changed_percent']) for r in rows]
                means.append(np.mean(values)); low.append(min(values)); high.append(max(values))
                for r, v in zip(rows, values):
                    record(name, r, 'bf16_changed_percent', v, threshold=threshold, loss=method)
            ax.plot(thresholds, means, marker=marker, markersize=5, markerfacecolor='none',
                    lw=1.6, color=color, label=label, ls='--' if method == 'full_vocab' else '-')
            ax.fill_between(thresholds, low, high, color=color, alpha=.08)
        ax.set_xscale('symlog', linthresh=1e-8)
        ax.set_xticks(thresholds, ['0', '$10^{-8}$', '$10^{-7}$', '$10^{-6}$', '$10^{-5}$'])
        ax.set(xlabel='Absolute BF16 change threshold', ylabel='Parameters exceeding threshold (%)', ylim=(0, None))
        ax.legend(frameon=False, ncol=2, fontsize=9)
        save(fig, name, f'{state}：监督目标的变化范围结论依赖阈值吗？',
             '横轴是有顺序的阈值，纵轴是超过阈值的参数百分比。曲线越接近，局部写回范围越相近；重合不表示方向相同。',
             scope+f'Student state {state}, four banks 42–45. Lines are means; faint envelopes are observed bank min–max, not confidence intervals. '
             'Threshold zero means exact nonzero BF16 changes. All thresholds were measured; none are interpolated observations. '
             'The main PG500 view focuses on PG, intersection, full vocabulary and zero-current-gradient; Student64 and Teacher64 remain in the source table '
             'and other state views. I64-labeled student states have mixed Student64-to-I64 histories. '
             'The zero-gradient control retains Adam state. Similar support sizes do not establish equal directions or capabilities.',
             tier='main' if state == 'm-pg500' else 'appendix')

    name = 'A09_objective_norms'
    fig, ax = chart()
    rows = [r for r in tables['branches'] if r['kind'] == 'density' and r['state'] == 'm-pg500']
    for metric, offset, color, label in [('gradient_l2', -.13, COLORS[0], 'Raw gradient / PG mean'),
                                        ('bf16_l2', .13, COLORS[1], 'BF16 writeback / PG mean')]:
        denominator = np.mean([float(r[metric]) for r in rows if r['branch'] == 'sampled_pg'])
        for i, method in enumerate(METHODS):
            observed = [r for r in rows if r['branch'] == method]
            values = [float(r[metric])/denominator for r in observed]
            mean = float(np.mean(values))
            ax.errorbar(mean, i+offset, xerr=[[mean-min(values)], [max(values)-mean]], fmt='o', color=color,
                        capsize=3, label=label if i == 0 else None)
            for r, v in zip(observed, values):
                record(name, r, metric+'_over_pg_mean', v, denominator=float(denominator), raw_value=float(r[metric]))
    ax.axvline(1, color='#555555', ls='--', lw=1)
    ax.set(yticks=range(6), yticklabels=METHOD_NAMES, xlabel='Norm relative to matched PG mean (PG = 1)')
    ax.invert_yaxis()
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1))
    save(fig, name, '监督目标改变了梯度尺度，是否也改变写回尺度？',
         '所有点固定在PG500状态。两种范数各除以自己的PG均值，所以可以读相对变化，不能把两种原始范数直接相除。',
         scope+'PG500, four banks. Ratios use the matched PG mean for each quantity separately; means are ratios of means. '
         'Segments show the per-bank range after division by that fixed denominator. Replaces the F7b heatmap.', kind='point')

    name = 'A10_teacher_overlap_matrix'
    rows = [r for r in pair_rows if r['state'] == 'm-pg500' and r['left'].startswith('common/')
            and r['right'].startswith('common/') and r['left'].endswith('/topk_intersection')
            and r['right'].endswith('/topk_intersection')]
    matrix = np.eye(4)
    for a in range(4):
        for b in range(a+1, 4):
            pair = [r for r in rows if {r['left'].split('/')[1], r['right'].split('/')[1]} == {DOMAINS[a], DOMAINS[b]}]
            matrix[a, b] = matrix[b, a] = np.mean([float(r['jaccard']) for r in pair])
            for r in pair:
                record(name, r, 'jaccard', float(r['jaccard']), teachers=[DOMAINS[a], DOMAINS[b]])
    fig, ax = chart(5.5)
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
    ax.grid(False)
    ax.set(xticks=range(4), xticklabels=['Math', 'Code', 'IF', 'Science'], yticks=range(4), yticklabels=['Math', 'Code', 'IF', 'Science'])
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f'{matrix[a,b]:.3f}', ha='center', va='center', color='white' if matrix[a,b]>.6 else '#222222')
    fig.colorbar(im, ax=ax, fraction=.05, label='BF16 support Jaccard (0–1)')
    save(fig, name, '同一输入下，四个教师的BF16写回集合有多重叠？',
         '这是唯一保留的热力图：两个轴都是教师，格子有明确的两两关系。',
         scope+'PG500 state, common inputs, I64 local objective, absolute threshold 1e-5. '
         'Each off-diagonal cell averages the two recorded banks. Diagonal entries equal one by identity. '
         'This selected controlled condition complements the full PG/I64 and step250/500 input comparisons, not a universal teacher-overlap result.', kind='heatmap')


if __name__ == '__main__':
    main()
