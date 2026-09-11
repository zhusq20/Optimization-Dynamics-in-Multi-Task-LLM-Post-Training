#!/usr/bin/env python3
"""Import verified local probes and render the F1/F2/F5/F6/F7 figure families."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
DOMAINS = ['math', 'code', 'if', 'science']
DOMAIN_NAMES = ['Math', 'Code', 'IF', 'Science']
METHODS = ['zero_gradient', 'sampled_pg', 'student_topk', 'topk_intersection', 'teacher_topk', 'full_vocab']
LABELS = ['Zero grad.', 'PG', 'Student64', 'Overlap64', 'Teacher64', 'Full vocab.']
COLORS = ['#777777', '#2769ad', '#7c91aa', '#21836b', '#8961a8', '#c56b32']
THRESHOLDS = ['0', '1e-08', '1e-07', '1e-06', '1e-05']
PAIRS = [(a, b) for i, a in enumerate(DOMAINS) for b in DOMAINS[i+1:]]
PAIR_COLORS = ['#2769ad', '#c56b32', '#21836b', '#8961a8', '#aa8528', '#687786']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 180})


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def import_probes(source, target):
    plan = json.loads((source / 'plan.json').read_text())
    records, files = [], []
    for stage in plan['stages']:
        for job in stage['jobs']:
            original = Path(job['output'])
            verification = json.loads((original / 'verified.json').read_text())
            destination = target / 'raw' / job['id']
            destination.mkdir(parents=True, exist_ok=True)
            # Large probability tensors are not required to reproduce these figures.
            names = [name for name in verification['artifacts'] if name != 'distributions.pt']
            for name in [*names, 'verified.json']:
                src = original / name
                actual = digest(src)
                if name != 'verified.json' and actual != verification['artifacts'][name]:
                    raise ValueError(f'Verified artifact changed: {src}')
                dst = destination / name
                shutil.copyfile(src, dst)
                files.append({'path': str(dst.relative_to(target)), 'sha256': actual, 'source': str(src)})
            records.append({key: job[key] for key in ['id', 'kind', 'state', 'snapshot_step']})
    for src in [source / 'analysis/RESULTS_zh.md', source / 'analysis/results_summary.json',
                source / 'validation/no_step_cast_audit.json']:
        dst = target / src.name
        shutil.copyfile(src, dst)
        files.append({'path': dst.name, 'sha256': digest(dst), 'source': str(src)})
    dump(target / 'data_manifest.json', {
        'campaign': source.name, 'completed_at_utc': json.loads((source / 'status.json').read_text())['completed_at_utc'],
        'jobs': records, 'files': files,
        'scope': '31 local probes; no new online training or capability evaluation',
        'reproduction': 'Figures require only this copied package, NumPy and Matplotlib; no models or GPUs.',
    })


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine_pair(data, a, b):
    return next(x for x in data['pairwise_bf16_writebacks'] if {x['left'], x['right']} == {a, b})


def teacher_pairs(data, mode, loss):
    return [x for x in data['pairwise_bf16_writebacks']
            if x['left'].startswith(mode + '/') and x['right'].startswith(mode + '/')
            and x['left'].endswith('/' + loss) and x['right'].endswith('/' + loss)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-from', type=Path)
    parser.add_argument('--data', type=Path, default=HERE / 'local_probe_results_20260908')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'figures/local_probe_results_20260908')
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.import_from:
        import_probes(args.import_from, args.data)
    manifest = json.loads((args.data / 'data_manifest.json').read_text())
    for record in manifest['files']:
        if digest(args.data / record['path']) != record['sha256']:
            raise ValueError('Imported evidence changed: ' + record['path'])
    jobs = manifest['jobs']
    data = {j['id']: json.loads((args.data / 'raw' / j['id'] / 'measurements.json').read_text()) for j in jobs}
    assert len(jobs) == 31
    density = [j for j in jobs if j['kind'] == 'density']
    normalization = sorted([j for j in jobs if j['kind'] == 'normalization'], key=lambda j: j['id'])
    teachers = [j for j in jobs if j['kind'] == 'teachers']
    tables = args.data / 'plot_data'
    tables.mkdir(exist_ok=True)
    branch_rows, pair_rows, threshold_rows, covariance_rows, response_rows, heldout_rows, js_rows = [], [], [], [], [], [], []
    normalization_gradient_rows = []
    for job in jobs:
        d = data[job['id']]
        base = {'job': job['id'], 'kind': job['kind'], 'state': job['state'],
                'source': f"raw/{job['id']}/measurements.json"}
        for name, branch in d['branches'].items():
            m = branch['metrics']
            branch_rows.append({**base, 'branch': name, 'gradient_l2': m['gradient_l2_before_clipping'],
                                'clip_coefficient': m['clip_coefficient'], 'master_update_l2': m['master_update']['l2'],
                                'bf16_l2': m['bf16_writeback']['l2'],
                                'bf16_nonzero_fraction': m['bf16_writeback']['fraction_above_0'],
                                'bf16_fraction_above_1e5': m['bf16_writeback']['fraction_above_1e-05']})
            for threshold in THRESHOLDS:
                threshold_rows.append({**base, 'branch': name, 'threshold': float(threshold),
                                       'changed_percent': 100 * m['bf16_writeback']['fraction_above_' + threshold]})
            for domain, value in branch.get('heldout_kl_decrease', {}).items():
                heldout_rows.append({**base, 'branch': name, 'domain': domain, 'kl_decrease': value})
        for pair in d.get('pairwise_bf16_writebacks', d.get('bf16_comparisons', [])):
            m = pair['metrics']
            for threshold, support in m['support'].items():
                row = {**base, 'left': pair['left'], 'right': pair['right'], 'threshold': float(threshold),
                       'gradient_cosine': pair.get('raw_gradient', {}).get('cosine'),
                       'bf16_cosine': m['cosine'], 'bf16_l2_difference': m['l2_difference'],
                       'left_count': support['left'], 'right_count': support['right'],
                       'intersection': support['intersection'], 'union': support['union'], 'jaccard': support['jaccard'],
                       'layer_random_ratio': support['layer_random_jaccard_ratio_of_expectations']}
                for rho in ['0.0001', '0.0005', '0.001']:
                    fixed = m.get('fixed_count_support', {}).get(rho, {})
                    row[f'fixed_{rho}_defined'] = fixed.get('defined')
                    row[f'fixed_{rho}_count'] = fixed.get('count_per_branch')
                    row[f'fixed_{rho}_jaccard'] = fixed.get('jaccard')
                pair_rows.append(row)
        for distance in d.get('teacher_distances', []):
            js_rows.append({**base, **distance})
        if job['kind'] == 'normalization':
            for pair, values in d['gradient_comparisons'].items():
                normalization_gradient_rows.append({**base, 'reduction_pair': pair, **values})
            root = args.data / 'raw' / job['id']
            banks = json.loads((root / 'banks.json').read_text())
            weights = json.loads((root / 'weights.json').read_text())
            for i, response in enumerate(banks['train']):
                response_rows.append({**base, 'response': i, 'domain': response['task'], 'length': len(response['response_ids']),
                                      'truncated': response['truncated'],
                                      **{rule: weights['response_coefficients'][rule][i] for rule in ['DR', 'DT', 'GT']}})
            for domain, value in d['decomposition'].items():
                covariance_rows.append({**base, 'domain': domain, 'mean_length': value['mean_length'],
                                        'covariance_correction_l2': value['covariance_l2'] / value['mean_length'],
                                        'relative_residual': value['relative_residual']})
    for name, rows in [('branches', branch_rows), ('pairs', pair_rows), ('thresholds', threshold_rows),
                       ('covariance', covariance_rows), ('responses', response_rows), ('heldout', heldout_rows), ('teacher_js', js_rows),
                       ('normalization_gradients', normalization_gradient_rows)]:
        write_csv(tables / f'{name}.csv', rows)

    figures, primary, mapping = [], [], []

    def new_figure(title, subtitle):
        fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.35))
        fig.subplots_adjust(left=.055, right=.975, top=.79, bottom=.22, wspace=.38)
        fig.suptitle(title, y=.985, fontsize=12)
        fig.text(.5, .91, subtitle, ha='center', fontsize=8)
        fig.text(.5, .025, 'Local HF probes with saved Adam state; BF16 proposed writebacks, not online steps. Diagnostic banks are not training seeds.', ha='center', fontsize=8)
        return fig, axes

    def save(fig, name, panels, used_jobs, is_primary=False):
        for suffix in ['pdf', 'png']:
            fig.savefig(args.output / f'{name}.{suffix}', bbox_inches='tight')
        figures.append(fig)
        if is_primary:
            primary.append(fig)
        mapping.append({'figure': name, 'panels': panels, 'jobs': used_jobs, 'primary': is_primary})

    def heatmap(ax, values, xs, ys, title, label, diverging=False, limits=None):
        values = np.asarray(values)
        lo, hi = limits if limits else ((-abs(values).max(), abs(values).max()) if diverging else (0, values.max()))
        im = ax.imshow(values, aspect='auto', cmap='RdBu' if diverging else 'YlGnBu', vmin=lo, vmax=hi)
        ax.set_xticks(range(len(xs)), xs)
        ax.set_yticks(range(len(ys)), ys)
        ax.set_title(title)
        for i, row in enumerate(values):
            for k, value in enumerate(row):
                ink = 'white' if (abs(value) > .68 * hi if diverging else value > .60 * hi) else '#202630'
                ax.text(k, i, f'{value:.3f}', ha='center', va='center', fontsize=7, color=ink)
        ax.figure.colorbar(im, ax=ax, shrink=.8, pad=.025, label=label)

    # F1: complete local batches and an explicit unequal-quota weight control.
    ordinary = [j for j in normalization if j['id'].endswith(('draw42', 'draw43'))]
    short = [j['id'].replace('normalization-m-pg', '').replace('-draw', '/') for j in ordinary]
    fig, axes = new_figure('F1 | Response lengths and loss weights', 'Four cap512 banks: PG250/500, draws42/43. Unequal-quota control: PG500/draw42, quotas 1/2/3/4, uniform domain prior.')
    for k, domain in enumerate(DOMAINS):
        for b, job in enumerate(ordinary):
            rows = [x for x in response_rows if x['job'] == job['id'] and x['domain'] == domain]
            for i, row in enumerate(rows):
                axes[0].scatter(k + (b - 1.5) * .12 + (i - .5) * .025, row['length'],
                                marker='x' if row['truncated'] else 'o', color=COLORS[k+1], s=25)
    axes[0].axhline(512, color='#777', ls=':', lw=1)
    axes[0].set(xticks=range(4), xticklabels=DOMAIN_NAMES, ylabel='Generated response tokens', title='F1a  Lengths (x = truncated)')
    for k, domain in enumerate(DOMAINS):
        vals = []
        for job in ordinary:
            rows = [x for x in response_rows if x['job'] == job['id']]
            vals.append(100 * sum(x['length'] for x in rows if x['domain'] == domain) / sum(x['length'] for x in rows))
        axes[1].plot(range(4), vals, 'o-', label=DOMAIN_NAMES[k], color=COLORS[k+1])
    axes[1].axhline(25, ls=':', color='#777', label='Prompt share')
    axes[1].set(xticks=range(4), xticklabels=short, ylabel='Domain token share (%)', title='F1b  Equal quotas, unequal token shares')
    axes[1].legend(ncol=2)
    quota = 'normalization-m-pg500-draw42-quota1234-uniform'
    rows = [x for x in response_rows if x['job'] == quota]
    for k, rule in enumerate(['DR', 'DT', 'GT']):
        axes[2].bar(np.arange(10)+(k-1)*.24, [100*x[rule] for x in rows], width=.24, label=rule, color=COLORS[k+1])
    axes[2].set(xticks=range(10), xticklabels=['M', 'C1', 'C2', 'I1', 'I2', 'I3', 'S1', 'S2', 'S3', 'S4'],
                ylabel='Response coefficient (%)', title='F1c  Unequal quotas: effective weights')
    axes[2].legend(ncol=3)
    save(fig, 'F1_lengths_and_weights', ['F1a', 'F1b', 'F1c'], [j['id'] for j in ordinary]+[quota], True)

    # F2 uses all seven banks; no selection by favorable KL outcomes.
    labels = ['250/42', '250/43', '500/42', '500/42 long', 'quota/prompt', 'quota/uniform', '500/43']
    fig, axes = new_figure('F2 | Normalization changes local gradients', 'All seven Overlap64-loss probes. KL panel is exploratory: one held-out response/domain; small changes and repeat-run numerical differences.')
    for k, domain in enumerate(DOMAINS):
        vals = [next(x['covariance_correction_l2'] for x in covariance_rows if x['job']==j['id'] and x['domain']==domain) for j in normalization]
        axes[0].scatter(np.arange(7)+(k-1.5)*.12, vals, color=COLORS[k+1], label=DOMAIN_NAMES[k], s=22)
    axes[0].set(xticks=range(7), xticklabels=labels, ylabel='||Cov(length, gradient)|| / mean length', title='F2a  Length-weighting correction')
    axes[0].tick_params(axis='x', rotation=55)
    axes[0].legend(ncol=2)
    cos = [[data[j['id']]['gradient_comparisons'][key]['cosine'] for key in ['DR_DT','DR_GT','DT_GT']] for j in normalization]
    heatmap(axes[1], cos, ['DR-DT','DR-GT','DT-GT'], labels, 'F2b  Raw-gradient direction', 'Cosine', limits=(0,1))
    kl = [[1000*data[j['id']]['branches'][key]['heldout_kl_decrease']['macro'] for key in ['zero_gradient','DR','DT','GT']] for j in normalization]
    heatmap(axes[2], kl, ['Zero','DR','DT','GT'], labels, 'F2c  Held-out macro KL decrease', 'Before - after (1e-3 nats)', True)
    save(fig, 'F2_normalization', ['F2a','F2b','F2c'], [j['id'] for j in normalization], True)

    for step in [250,500]:
        selected = [j for j in teachers if j['snapshot_step']==step]
        runs = [data[j['id']] for j in selected]
        for loss, loss_label in [('sampled_pg','PG'),('topk_intersection','Overlap64')]:
            fig, axes = new_figure(f'F5 | Teacher updates at M-PG/{step} | {loss_label}', 'Two diagnostic banks (42/43); six shared teacher pairs/bank. Support threshold = 1e-5. Error bars show observed bank min-max, not confidence intervals.')
            mat = np.eye(4)
            for i,a in enumerate(DOMAINS):
                for k,b in enumerate(DOMAINS):
                    if i==k: continue
                    mat[i,k] = np.mean([cosine_pair(d,f'routed/{a}/{loss}',f'routed/{b}/{loss}')['metrics']['support']['1e-05']['jaccard'] for d in runs])
            heatmap(axes[0],mat,DOMAIN_NAMES,DOMAIN_NAMES,'F5a  Routed support overlap','Jaccard',limits=(0,1))
            for mode,marker in [('routed','o'),('common','^')]:
                for pair_id,(a,b) in enumerate(PAIRS):
                    pairs = [cosine_pair(d,f'{mode}/{a}/{loss}',f'{mode}/{b}/{loss}') for d in runs]
                    axes[1].scatter([x['raw_gradient']['cosine'] for x in pairs],[x['metrics']['cosine'] for x in pairs],
                                    color=PAIR_COLORS[pair_id],marker=marker,s=30,alpha=.75)
                axes[1].scatter([],[],color='#444',marker=marker,label=mode)
            axes[1].set(xlabel='Raw-gradient cosine',ylabel='BF16 proposed-writeback cosine',title='F5b  Input control and direction',xlim=(-.1,1.04),ylim=(0,1.04))
            axes[1].legend()
            for offset,mode,color in [(-.15,'routed',COLORS[1]),(.15,'common',COLORS[5])]:
                for k,teacher in enumerate(DOMAINS):
                    ps=[cosine_pair(d,'zero_gradient',f'{mode}/{teacher}/{loss}') for d in runs]
                    vals=np.array([x['metrics']['support']['1e-05']['jaccard'] for x in ps])
                    random=np.mean([x['metrics']['support']['1e-05']['layer_random_jaccard_ratio_of_expectations'] for x in ps])
                    axes[2].errorbar(k+offset,vals.mean(),yerr=[[vals.mean()-vals.min()],[vals.max()-vals.mean()]],fmt='o',capsize=3,color=color,label=mode if k==0 else None)
                    axes[2].scatter(k+offset,random,color=color,marker='x',s=25)
            axes[2].scatter([],[],marker='x',color='#444',label='Layer random ratio')
            axes[2].set(xticks=range(4),xticklabels=DOMAIN_NAMES,yscale='log',ylim=(1e-4,1.6),ylabel='Jaccard with zero-gradient step',title='F5c  Shared optimizer-state control')
            axes[2].legend(loc='center left')
            save(fig,f'F5_teacher_overlap_pg{step}_{loss_label}', ['F5a','F5b','F5c'],[j['id'] for j in selected],step==500 and loss=='topk_intersection')

            fig,axes=new_figure(f'F6 | Teacher distribution distance at M-PG/{step} | {loss_label}', 'JS uses common four-domain prefixes. Each point is a bank/pair measurement; teacher pairs are dependent. No regression or significance claim.')
            distances=[{frozenset([x['left'],x['right']]):x['full_vocab_js'] for x in d['teacher_distances']} for d in runs]
            mat=np.zeros((4,4))
            for i,a in enumerate(DOMAINS):
                for k,b in enumerate(DOMAINS):
                    if i!=k:mat[i,k]=np.mean([dist[frozenset([a,b])] for dist in distances])
            heatmap(axes[0],mat,DOMAIN_NAMES,DOMAIN_NAMES,'F6a  Teacher JS on common prefixes','JS (nats)')
            for ax,mode,panel in [(axes[1],'routed','F6b'),(axes[2],'common','F6c')]:
                for index,(a,b) in enumerate(PAIRS):
                    xs=[dist[frozenset([a,b])] for dist in distances]
                    ys=[1-cosine_pair(d,f'{mode}/{a}/{loss}',f'{mode}/{b}/{loss}')['metrics']['support']['1e-05']['jaccard'] for d in runs]
                    ax.scatter(xs,ys,color=PAIR_COLORS[index],s=28,alpha=.8,label=f'{a[0].upper()}-{b[0].upper()}')
                ax.set(xlabel='Teacher JS on common prefixes (nats)',ylabel='1 - BF16 support Jaccard',title=f'{panel}  {mode.capitalize()} inputs',ylim=(0,.4))
                ax.set_xlim(0,max(mat.max(),max(max(v for key,v in dist.items() if 'student' not in key) for dist in distances))*1.12)
                ax.legend(ncol=3,loc='upper right')
            save(fig,f'F6_teacher_distance_pg{step}_{loss_label}', ['F6a','F6b','F6c'],[j['id'] for j in selected],step==500 and loss=='topk_intersection')

    states=['m-pg250','m-pg500','s-pg250','m-i64dr250','s-i64250']
    for state in states:
        selected=[j for j in density if j['state']==state]
        runs=[data[j['id']] for j in selected]
        history=' | mixed Student64 -> Overlap64 training history' if 'i64' in state else ''
        fig,axes=new_figure(f'F7 | Local supervision controls | {state}{history}', 'Four banks (42-45); cap256, up to four prefixes/response. Bands/bars are bank min-max. Different losses change weighting as well as support.')
        for branch,label,color in zip(METHODS,LABELS,COLORS):
            metrics=[d['branches'][branch]['metrics'] for d in runs]
            vals=np.array([[100*m['bf16_writeback']['fraction_above_'+t] for t in THRESHOLDS] for m in metrics])
            axes[0].plot(range(5),vals.mean(0),'o-',label=label,color=color,ms=3)
            axes[0].fill_between(range(5),vals.min(0),vals.max(0),color=color,alpha=.10)
            axes[1].scatter([m['gradient_l2_before_clipping'] for m in metrics],[m['bf16_writeback']['l2'] for m in metrics],color=color,label=label,s=26,alpha=.8)
        axes[0].set(xticks=range(5),xticklabels=['0','1e-8','1e-7','1e-6','1e-5'],ylabel='Changed parameters (%)',xlabel='Absolute BF16 change threshold',title='F7a  Writeback threshold curves',ylim=(0,None))
        axes[0].legend(ncol=2)
        maximum=max(d['branches'][branch]['metrics']['bf16_writeback']['l2'] for d in runs for branch in METHODS)
        axes[1].set(xscale='symlog',xlabel='Pre-clip gradient L2 (symlog)',ylabel='BF16 proposed-writeback L2',title='F7b  Gradient vs writeback magnitude',ylim=(0,maximum*1.15))
        axes[1].legend(ncol=2,loc='lower left')
        for i,(branch,label,color) in enumerate(zip(METHODS[:-1],LABELS[:-1],COLORS[:-1])):
            vals=np.array([cosine_pair(d,branch,'full_vocab')['metrics']['cosine'] for d in runs])
            axes[2].scatter(i+np.linspace(-.08,.08,len(vals)),vals,color=color,s=15,alpha=.55)
            axes[2].errorbar(i,vals.mean(),yerr=[[vals.mean()-vals.min()],[vals.max()-vals.mean()]],fmt='o',color=color,capsize=3)
        axes[2].set(xticks=range(5),xticklabels=LABELS[:-1],ylabel='BF16 cosine to full-vocabulary step',title='F7c  Agreement with full-vocab reference',ylim=(0,1.04))
        axes[2].tick_params(axis='x',rotation=25)
        save(fig,f'F7_density_{state}', ['F7a','F7b','F7c'],[j['id'] for j in selected],state=='m-pg500')

    with PdfPages(args.output/'main_local_probe_figures.pdf') as pdf:
        for fig in primary:pdf.savefig(fig,bbox_inches='tight')
    with PdfPages(args.output/'all_local_probe_figures.pdf') as pdf:
        for fig in figures:pdf.savefig(fig,bbox_inches='tight')
    for fig in figures:plt.close(fig)
    dump(args.data/'figure_manifest.json', {'figures':mapping,'primary_pages':len(primary),'all_pages':len(figures),
        'source_manifest_sha256':digest(args.data/'data_manifest.json'),'plot_script_sha256':digest(Path(__file__)),
        'missing_panels_not_filled':['F3a','F3b','F3c','F4a','F4b','F4c','F8a','F8b','F8c','F9a','F9b','F9c'],
        'selection':'Primary F5/F6 use PG500 + current Overlap64 loss; all checkpoint/loss variants are retained. No selection by effect size.',
        'table_rows':{'branches':len(branch_rows),'pairs_with_threshold':len(pair_rows),'thresholds':len(threshold_rows),
                      'responses':len(response_rows),'covariance':len(covariance_rows),'heldout':len(heldout_rows),'teacher_js':len(js_rows),
                      'normalization_gradients':len(normalization_gradient_rows)}})
    print(json.dumps({'verified_jobs':len(jobs),'figure_pages':len(figures),'primary_pages':len(primary),'output':str(args.output)}))


if __name__=='__main__':
    main()
