#!/usr/bin/env python3
"""Import verified probes and render one standalone chart per experimental panel."""
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
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18, 'axes.labelsize': 19, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                     'legend.fontsize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 240})


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

    from matplotlib.lines import Line2D
    figures, mapping, plotted_rows = [], [], []
    captions = {}

    def canvas(size=(6.6, 4.9)):
        fig, ax = plt.subplots(figsize=size)
        fig.subplots_adjust(left=.16, right=.96, bottom=.19, top=.96)
        return fig, ax

    def save(fig, panel, used_jobs, caption, encoding):
        # No titles, panel IDs or prose inside exported experimental images.
        assert not fig.texts
        assert all(not ax.get_title() for ax in fig.axes)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            boxes=[text.get_window_extent(renderer) for text in ax.texts]
            for i,box in enumerate(boxes):
                assert not any(box.overlaps(other) for other in boxes[i+1:]), f'Overlapping matrix labels in {panel}'
        for extension in ['pdf', 'png', 'svg']:
            fig.savefig(args.output / f'{panel}.{extension}', bbox_inches='tight', pad_inches=.06)
        figures.append(fig)
        captions[panel] = caption
        mapping.append({'figure': panel, 'panels': [panel], 'jobs': used_jobs,
                        'encoding': encoding, 'caption_file': 'FIGURE_CAPTIONS.md',
                        'standalone': True, 'main_axes': 1, 'title': None})

    def point(panel, job, measure, value, **coordinates):
        plotted_rows.append({'panel': panel, 'job': job, 'source': f'raw/{job}/measurements.json',
                             'measure': measure, 'value': float(value), 'coordinates': coordinates})

    def matrix(values, columns, rows, label, limits, fmt='.3f', size=(7.3, 5.0), diverging=False):
        fig, ax = canvas(size)
        fig.subplots_adjust(left=.23, right=.89, bottom=.23)
        values = np.asarray(values)
        lo, hi = limits
        im = ax.imshow(values, aspect='auto', cmap='RdBu' if diverging else 'YlGnBu', vmin=lo, vmax=hi)
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(rows)), rows)
        for i, row in enumerate(values):
            for k, value in enumerate(row):
                dark = abs(value)/hi > .68 if diverging else (value-lo)/(hi-lo) > .60
                ax.text(k, i, format(value, fmt), ha='center', va='center', fontsize=15,
                        color='white' if dark else '#15202b')
        cb = fig.colorbar(im, ax=ax, fraction=.055, pad=.035)
        cb.set_label(label, fontsize=17)
        cb.ax.tick_params(labelsize=14)
        return fig, ax

    ordinary = [j for j in normalization if j['id'].endswith(('draw42', 'draw43'))]
    bank_labels = [j['id'].replace('normalization-m-pg', '').replace('-draw', ' / ') for j in ordinary]
    fig, ax = canvas()
    for k, domain in enumerate(DOMAINS):
        for b, job in enumerate(ordinary):
            rows = [x for x in response_rows if x['job'] == job['id'] and x['domain'] == domain]
            for i, row in enumerate(rows):
                ax.scatter(k+(b-1.5)*.13+(i-.5)*.035, row['length'],
                           marker='x' if row['truncated'] else 'o', color=COLORS[k+1], s=55, linewidths=1.8)
                point('F1a', job['id'], 'response_length', row['length'], domain=domain,
                      response=row['response'], truncated=row['truncated'])
    ax.axhline(512, color='#888', linestyle=':', linewidth=1.3)
    ax.set(xticks=range(4), xticklabels=DOMAIN_NAMES, ylabel='Response length (tokens)', ylim=(0,550))
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#555',label='Complete'),
                       Line2D([],[],marker='x',ls='',color='#555',label='Truncated')], loc='lower left')
    save(fig, 'F1a', [j['id'] for j in ordinary],
         'Response lengths in four local diagnostic banks at M-PG checkpoints 250 and 500 (draws 42/43). Each bank contains two training responses per domain. Crosses denote truncation at the 512-token cap; these are capped diagnostic lengths, not an estimate of the natural online length distribution.',
         'color=domain; marker=completion; small horizontal offsets separate banks/responses')

    fig, ax = canvas()
    for k, domain in enumerate(DOMAINS):
        vals=[]
        for job in ordinary:
            rows=[x for x in response_rows if x['job']==job['id']]
            value=100*sum(x['length'] for x in rows if x['domain']==domain)/sum(x['length'] for x in rows)
            vals.append(value);point('F1b',job['id'],'domain_token_percent',value,domain=domain)
        ax.plot(range(4),vals,marker=['o','s','^','D'][k],linestyle='-',color=COLORS[k+1],
                label=DOMAIN_NAMES[k],ms=10 if k==0 else 6,mfc='none' if k==0 else COLORS[k+1],mew=1.5,lw=2)
    ax.axhline(25,color='#777',ls=':',lw=1.5,label='Prompt share')
    ax.set(xticks=range(4),xticklabels=bank_labels,xlabel='Checkpoint / bank',ylabel='Domain token share (%)')
    fig.subplots_adjust(bottom=.32)
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.27),fontsize=13)
    save(fig,'F1b',[j['id'] for j in ordinary],
         'Domain token shares in the same four cap512 banks as F1a. Each domain supplies 25% of prompts (dotted line). Token shares differ from prompt shares and must not be interpreted as the domain coefficients of response-normalized DR.',
         'color=domain; x=checkpoint/bank')

    quota='normalization-m-pg500-draw42-quota1234-uniform'
    rows=[x for x in response_rows if x['job']==quota]
    fig,ax=canvas((7.0,4.9))
    for k,rule in enumerate(['DR','DT','GT']):
        ax.bar(np.arange(10)+(k-1)*.25,[100*x[rule] for x in rows],width=.25,label=rule,color=COLORS[k+1])
        for row in rows:point('F1c',quota,'response_weight_percent',100*row[rule],rule=rule,response=row['response'],domain=row['domain'])
    ax.set(xticks=range(10),xticklabels=['M','C1','C2','I1','I2','I3','S1','S2','S3','S4'],ylabel='Response weight (%)')
    ax.tick_params(axis='x',labelsize=14);ax.legend(ncol=3)
    save(fig,'F1c',[quota],
         'Effective response coefficients for domain-response (DR), domain-token (DT), and global-token (GT) averaging on one shared PG500 bank. Domain quotas are 1/2/3/4 for math/code/IF/science, with a uniform domain prior. M, C, I, and S denote these domains. The matched prompt-share prior control is retained in the data package.',
         'bar color=reduction; x=response; domain letters defined in caption')

    norm_labels=['250 / 42','250 / 43','500 / 42','500 / 42 · long','Quota · prompt','Quota · uniform','500 / 43']
    fig,ax=canvas((7.0,5.6));fig.subplots_adjust(left=.33,bottom=.16)
    for k,domain in enumerate(DOMAINS):
        vals=[next(x['covariance_correction_l2'] for x in covariance_rows if x['job']==j['id'] and x['domain']==domain) for j in normalization]
        ax.scatter(vals,np.arange(7)+(k-1.5)*.13,color=COLORS[k+1],s=50,label=DOMAIN_NAMES[k])
        for j,value in zip(normalization,vals):point('F2a',j['id'],'covariance_correction_l2',value,domain=domain)
    ax.set(yticks=range(7),yticklabels=norm_labels,xlabel='Length–gradient correction (L2)');ax.invert_yaxis()
    ax.legend(ncol=2,loc='lower right',fontsize=13)
    save(fig,'F2a',[j['id'] for j in normalization],
         'Magnitude of the length-weighting correction, ||Cov(T,g)|| / mean(T), for each domain in all seven normalization probes. Zero corrections are retained. The long bank uses cap2048; other banks use cap512. Quota/prompt and quota/uniform reuse the same unequal-quota responses. The covariance identity has maximum relative residual about 1.8e-16; this checks a fixed-batch identity, not distributed reducer correctness.',
         'color=domain; rows=all seven probe conditions; no categorical connecting lines')
    vals=[]
    for j in normalization:
        row=[]
        for key in ['DR_DT','DR_GT','DT_GT']:
            value=data[j['id']]['gradient_comparisons'][key]['cosine'];row.append(value)
            point('F2b',j['id'],'raw_gradient_cosine',value,reduction_pair=key)
        vals.append(row)
    fig,ax=matrix(vals,['DR–DT','DR–GT','DT–GT'],norm_labels,'Gradient cosine',(0,1),size=(7.0,5.6))
    fig.subplots_adjust(left=.30,bottom=.12)
    save(fig,'F2b',[j['id'] for j in normalization],
         'Raw-gradient cosine for the three reduction pairs in all seven probes, with the same student, saved Adam state, and responses within each probe. The values describe pre-clipping gradients, not BF16 writebacks. The long-response condition gives DR–DT and DR–GT cosines of approximately 0.706 and 0.658.',
         'rows=probe; columns=reduction pair; every cell is a measured cosine')
    vals=[]
    for j in normalization:
        row=[]
        for rule in ['zero_gradient','DR','DT','GT']:
            value=1000*data[j['id']]['branches'][rule]['heldout_kl_decrease']['macro'];row.append(value)
            point('F2c',j['id'],'heldout_macro_kl_decrease_millinats',value,rule=rule)
        vals.append(row)
    limit=np.abs(vals).max()
    fig,ax=matrix(vals,['Zero','DR','DT','GT'],norm_labels,'KL decrease (10⁻³ nats)',(-limit,limit),size=(7.3,5.6),diverging=True)
    fig.subplots_adjust(left=.29,bottom=.12)
    save(fig,'F2c',[j['id'] for j in normalization],
         'Macro-average held-out full-vocabulary reverse-KL decrease (before minus after) for the zero-gradient Adam control and DR/DT/GT. All seven positive and negative outcomes are shown. Each domain has only one held-out response. Small cross-run numerical differences were observed even for the theoretically prior-invariant GT control on the reused quota bank, so this panel is exploratory and does not establish a ranking of downstream capability.',
         'rows=probe; columns=control/reduction; diverging scale symmetric about zero')

    teacher_runs={(step,draw):data[f'teachers-m-pg{step}-draw{draw}'] for step in [250,500] for draw in [42,43]}
    conditions=[(step,loss,mode) for step in [250,500] for loss in ['sampled_pg','topk_intersection'] for mode in ['routed','common']]
    cond_labels=[f"{step}\n{'PG' if loss=='sampled_pg' else 'I64'}\n{'R' if mode=='routed' else 'C'}" for step,loss,mode in conditions]
    teacher_ids=[j['id'] for j in teachers]
    pair_labels=[f'{DOMAIN_NAMES[DOMAINS.index(a)]}–{DOMAIN_NAMES[DOMAINS.index(b)]}' for a,b in PAIRS]
    values=[]
    for a,b in PAIRS:
        row=[]
        for step,loss,mode in conditions:
            observations=[]
            for draw in [42,43]:
                d=teacher_runs[step,draw];value=cosine_pair(d,f'{mode}/{a}/{loss}',f'{mode}/{b}/{loss}')['metrics']['support']['1e-05']['jaccard']
                observations.append(value);point('F5a',d['job'],'bf16_jaccard',value,step=step,loss=loss,mode=mode,left=a,right=b,bank=draw)
            row.append(np.mean(observations))
        values.append(row)
    fig,ax=matrix(values,cond_labels,pair_labels,'Jaccard',(0,1),fmt='.2f',size=(8.3,5.3))
    ax.tick_params(axis='x',labelsize=14)
    ax.axvline(3.5,color='white',lw=3)
    save(fig,'F5a',teacher_ids,
         'Teacher-pair overlap of BF16 proposed-writeback supports (absolute threshold 1e-5), jointly showing checkpoints 250/500, PG/I64 losses, and routed/common inputs. Each cell averages the two diagnostic banks 42/43 within its labeled condition; conditions are not pooled. R denotes each teacher’s routed domain; C denotes identical common four-domain prefixes and matched PG actions. Only the six distinct teacher pairs are shown, avoiding redundant diagonal and symmetric cells. Support counts, bank-level ranges and equal-count controls remain in the source tables.',
         'rows=six teacher pairs; columns=checkpoint×loss×input; cells=two-bank means')

    loss_colors={'sampled_pg':COLORS[1],'topk_intersection':COLORS[3]}
    step_markers={250:'o',500:'s'}
    fig,ax=canvas((6.8,5.1))
    for step,loss,mode in conditions:
        for draw in [42,43]:
            d=teacher_runs[step,draw]
            for pair in teacher_pairs(d,mode,loss):
                x=pair['raw_gradient']['cosine'];y=pair['metrics']['cosine']
                ax.scatter(x,y,s=53,marker=step_markers[step],edgecolors=loss_colors[loss],
                           facecolors=loss_colors[loss] if mode=='routed' else 'none',alpha=.72,linewidths=1.3)
                point('F5b',d['job'],'bf16_cosine',y,gradient_cosine=x,step=step,loss=loss,mode=mode,left=pair['left'],right=pair['right'])
    ax.set(xlabel='Raw-gradient cosine',ylabel='BF16-writeback cosine',xlim=(-.09,1.04),ylim=(.78,1.015))
    handles=[Line2D([],[],ls='',marker='o',color=loss_colors[k],label=v) for k,v in [('sampled_pg','PG'),('topk_intersection','I64')]]
    handles += [Line2D([],[],ls='',marker=step_markers[k],color='#555',label=f'Step {k}') for k in [250,500]]
    handles += [Line2D([],[],ls='',marker='o',color='#555',markerfacecolor=c,label=l) for c,l in [('#555','Routed'),('none','Common')]]
    ax.legend(handles=handles,ncol=2,loc='center',bbox_to_anchor=(.49,.38),fontsize=13)
    save(fig,'F5b',teacher_ids,
         'Raw-gradient versus BF16 proposed-writeback alignment for every teacher pair, bank, checkpoint and loss. Color identifies PG/I64, circle/square identifies step250/500, and filled/open markers identify routed/common inputs. The 96 plotted comparisons share teachers and contexts and are not independent replicates. Matched inputs yield highly aligned gradients, whereas routed gradients are nearly orthogonal despite substantial BF16 alignment.',
         'color=loss; shape=checkpoint; fill=input condition; 96 measured points')

    fig,ax=canvas((8.1,5.5));fig.subplots_adjust(bottom=.24,left=.14)
    for i,(step,loss,mode) in enumerate(conditions):
        for k,teacher in enumerate(DOMAINS):
            for b,draw in enumerate([42,43]):
                d=teacher_runs[step,draw];m=cosine_pair(d,'zero_gradient',f'{mode}/{teacher}/{loss}')['metrics']['support']['1e-05']
                x=i+(k-1.5)*.13+(b-.5)*.038
                ax.scatter(x,m['jaccard'],color=COLORS[k+1],s=34,alpha=.75)
                ax.scatter(x,m['layer_random_jaccard_ratio_of_expectations'],color=COLORS[k+1],s=35,marker='x',alpha=.8)
                point('F5c',d['job'],'zero_control_jaccard',m['jaccard'],step=step,loss=loss,mode=mode,teacher=teacher,
                      random_ratio=m['layer_random_jaccard_ratio_of_expectations'])
    ax.set(xticks=range(8),xticklabels=cond_labels,yscale='log',ylim=(1e-4,1.6),ylabel='Zero-gradient overlap')
    ax.tick_params(axis='x',labelsize=14)
    handles=[Line2D([],[],ls='',marker='o',color=COLORS[k+1],label=l) for k,l in enumerate(DOMAIN_NAMES)]
    handles += [Line2D([],[],ls='',marker=m,color='#555',label=l) for m,l in [('o','Observed'),('x','Layer random')]]
    ax.legend(handles=handles,ncol=3,loc='center',fontsize=13)
    save(fig,'F5c',teacher_ids,
         'Overlap with the zero-gradient Adam control for each teacher, checkpoint, loss and input condition. R/C denote routed/common as in F5a. Dots are individual-bank Jaccards; crosses are the layer-matched ratio of expected intersection to expected union, not an empirical expected Jaccard. The logarithmic axis exposes the separation from the random reference. Shared optimizer history must be considered when interpreting BF16 support overlap.',
         'x=checkpoint×loss×input; color=teacher; dot=observed bank; cross=layer-random ratio')

    distances={key:{frozenset([v['left'],v['right']]):v['full_vocab_js'] for v in d['teacher_distances']} for key,d in teacher_runs.items()}
    vals=[]
    for a,b in PAIRS:
        row=[]
        for step in [250,500]:
            obs=[]
            for draw in [42,43]:
                value=distances[step,draw][frozenset([a,b])];obs.append(value)
                point('F6a',teacher_runs[step,draw]['job'],'teacher_js',value,step=step,bank=draw,left=a,right=b)
            row.append(np.mean(obs))
        vals.append(row)
    fig,ax=matrix(vals,['250','500'],pair_labels,'JS (nats)',(0,np.max(vals)),fmt='.4f',size=(5.7,5.1))
    fig.subplots_adjust(left=.34,bottom=.16);ax.set_xlabel('Student checkpoint')
    save(fig,'F6a',teacher_ids,
         'Full-vocabulary Jensen–Shannon divergence between each distinct teacher pair, averaged over two common-prefix banks at each student checkpoint. JS is computed before applying either local loss and is therefore shown once rather than duplicated for PG and I64. Teachers are fixed; checkpoint-specific differences reflect different generated prefix banks, not changing teacher weights.',
         'rows=unique teacher pair; columns=checkpoint; cells=two-bank JS mean; no duplicate loss dimension')
    xmax=max(v for table in distances.values() for pair,v in table.items() if 'student' not in pair)*1.08
    for panel,mode in [('F6b','routed'),('F6c','common')]:
        fig,ax=canvas((6.8,5.1))
        for step in [250,500]:
            for loss in loss_colors:
                for draw in [42,43]:
                    d=teacher_runs[step,draw]
                    for a,b in PAIRS:
                        x=distances[step,draw][frozenset([a,b])]
                        y=1-cosine_pair(d,f'{mode}/{a}/{loss}',f'{mode}/{b}/{loss}')['metrics']['support']['1e-05']['jaccard']
                        ax.scatter(x,y,color=loss_colors[loss],marker=step_markers[step],s=56,alpha=.7,edgecolors='white',linewidths=.35)
                        point(panel,d['job'],'bf16_support_distance',y,teacher_js=x,step=step,loss=loss,mode=mode,left=a,right=b)
        ax.set(xlabel='Teacher JS (nats)',ylabel='1 − BF16 Jaccard',xlim=(0,xmax),ylim=(0,.4))
        ax.set_xticks([0,.01,.02,.03],['0.00','0.01','0.02','0.03'])
        handles=[Line2D([],[],ls='',marker='o',color=loss_colors[k],label=v) for k,v in [('sampled_pg','PG'),('topk_intersection','I64')]]
        handles += [Line2D([],[],ls='',marker=step_markers[k],color='#555',label=f'Step {k}') for k in [250,500]]
        ax.legend(handles=handles,ncol=2,loc='lower right' if mode=='routed' else 'upper right',fontsize=13)
        save(fig,panel,teacher_ids,
             f'Teacher JS on common prefixes versus BF16 support distance under {mode} inputs, combining both checkpoints and both losses while preserving their labels. Each point represents one teacher pair in one bank. F6b and F6c use identical axis limits for a matched routed/common comparison. Pairwise observations are dependent; no regression, significance test, or causal relationship is asserted.',
             'color=PG/I64; shape=250/500; 48 observations; axes shared with the other input condition')

    states=['m-pg250','m-pg500','s-pg250','m-i64dr250','s-i64250']
    state_labels=['M-PG / 250','M-PG / 500','S-PG / 250','M-I64* / 250','S-I64* / 250']
    short_methods=['Zero','PG','ST64','I64','T64','Full']
    state_runs={s:[data[j['id']] for j in density if j['state']==s] for s in states}
    vals=[]
    for state in states:
        row=[]
        for method in METHODS:
            obs=[]
            for d in state_runs[state]:
                value=100*d['branches'][method]['metrics']['bf16_writeback']['fraction_above_1e-05'];obs.append(value)
                point('F7a',d['job'],'bf16_changed_percent',value,state=state,loss=method,threshold=1e-5)
            row.append(np.mean(obs))
        vals.append(row)
    # Scaling the displayed unit preserves four-decimal percentage precision
    # without cramming six-character strings into adjacent cells.
    fig,ax=matrix(np.asarray(vals)*100,short_methods,state_labels,'Changed parameters (10⁻² %)',(0,np.max(vals)*100),fmt='.2f',size=(7.6,5.0))
    fig.subplots_adjust(left=.28,bottom=.13)
    save(fig,'F7a',[j['id'] for j in density],
         'Percentage of parameters whose local BF16 proposed writeback exceeds the prespecified absolute threshold 1e-5, jointly showing all five student states and six objectives/controls. Displayed units are 0.01 percentage points (a cell value of 3.52 represents 0.0352%). Cells average four diagnostic banks (42–45) within each condition. Zero is the zero-gradient Adam control; ST64 selects student Top64, I64 their intersection with teacher Top64, T64 selects teacher Top64, and Full is the full-vocabulary local reference. Asterisks mark states with mixed Student64→I64 training history. Other thresholds and bank ranges remain in the plotting tables; the full threshold sweep is not overlaid as 30 unreadable curves.',
         'rows=five labeled states; columns=six objectives; cells=four-bank mean at fixed threshold1e-5')

    fig,ax=canvas((7.3,6.3));fig.subplots_adjust(left=.16,right=.97,bottom=.34,top=.97)
    state_markers=['o','s','^','D','P']
    for state,marker in zip(states,state_markers):
        for method,color in zip(METHODS,COLORS):
            for d in state_runs[state]:
                m=d['branches'][method]['metrics'];x=m['gradient_l2_before_clipping'];y=m['bf16_writeback']['l2']
                ax.scatter(x,y,marker=marker,color=color,s=47,alpha=.68,linewidths=.4,edgecolors='white')
                point('F7b',d['job'],'bf16_l2',y,gradient_l2=x,state=state,loss=method)
    ax.set(xscale='symlog',xlabel='Pre-clip gradient L2',ylabel='BF16-writeback L2',ylim=(.023,.05))
    method_handles=[Line2D([],[],ls='',marker='o',color=c,label=l) for c,l in zip(COLORS,short_methods)]
    state_handles=[Line2D([],[],ls='',marker=m,color='#555',label=l.replace(' / ',' ')) for m,l in zip(state_markers,state_labels)]
    fig.legend(handles=method_handles,ncol=6,loc='lower center',bbox_to_anchor=(.54,.16),fontsize=13,handletextpad=.2,columnspacing=.8)
    fig.legend(handles=state_handles,ncol=3,loc='lower center',bbox_to_anchor=(.54,.015),fontsize=13,handletextpad=.25,columnspacing=1.)
    save(fig,'F7b',[j['id'] for j in density],
         'Pre-clipping raw-gradient L2 versus BF16 proposed-writeback L2 for all five student states, six objectives and four banks (120 observations, with identical zero-control coordinates overlapping). Color denotes objective and marker denotes student state. The gradient axis uses a symmetric logarithmic scale to retain the zero control. The writeback axis is linear and explicitly zoomed to the observed range. The two norms are distinct quantities; similar writeback norms do not establish equal directions or learning effects. Asterisks denote mixed training history.',
         'color=objective; shape=student state; raw bank points; symlog x, linear zoomed y')

    vals=[]
    for state in states:
        row=[]
        for method in METHODS[:-1]:
            obs=[]
            for d in state_runs[state]:
                value=cosine_pair(d,method,'full_vocab')['metrics']['cosine'];obs.append(value)
                point('F7c',d['job'],'bf16_cosine_to_full',value,state=state,loss=method)
            row.append(np.mean(obs))
        vals.append(row)
    fig,ax=matrix(vals,short_methods[:-1],state_labels,'Cosine to full-vocab step',(0,1),fmt='.3f',size=(7.3,5.0))
    fig.subplots_adjust(left=.29,bottom=.13)
    save(fig,'F7c',[j['id'] for j in density],
         'Cosine similarity between each local BF16 proposed writeback and the full-vocabulary reference from the same student, Adam state and prefix bank. Cells average four banks per state/objective. All five states are shown together; the trivial full-versus-full column is omitted. Asterisks retain mixed-history provenance. The full-vocabulary branch is a local reference, not an online-trained capability baseline.',
         'rows=five states; columns=five non-reference objectives; cells=four-bank mean cosine')

    # This convenience PDF still contains exactly one independently exported chart per page.
    with PdfPages(args.output/'all_panels.pdf') as pdf:
        for fig in figures:pdf.savefig(fig,bbox_inches='tight',pad_inches=.06)
    for fig in figures:plt.close(fig)
    dump(args.data/'panel_values.json',plotted_rows)
    lines=['# 实验图 Caption 草稿','',
           '每个文件只含一张独立图。图内仅保留轴、刻度、矩阵数值和必要图例；标题、实验条件与解释放在以下 caption。',
           '', '所有新图均为保存 Adam 状态上的 HF 局部模拟 BF16 写回，不是真实在线单步或能力评测。不同条件保持分开，诊断 bank 不是训练 seed。', '']
    for panel in captions:lines += [f'## {panel}', '', captions[panel], '']
    (args.data/'FIGURE_CAPTIONS.md').write_text('\n'.join(lines)+'\n')
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>实验图</title>',
          '<style>body{font:18px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#172433}section{margin:36px 0 64px}img{max-width:100%;height:auto;border:1px solid #ddd}a{color:#2463a6}nav{line-height:2.1}</style>',
          '<h1>独立实验图</h1><nav>'+ ' · '.join(f'<a href="#{p}">{p}</a>' for p in captions)+'</nav>']
    for panel in captions:html.append(f'<section id="{panel}"><p>{panel} · <a href="{panel}.pdf">PDF</a> · <a href="{panel}.png">PNG</a> · <a href="{panel}.svg">SVG</a></p><a href="{panel}.pdf"><img src="{panel}.png" alt="{panel}"></a></section>')
    html.append('</html>');(args.output/'index.html').write_text('\n'.join(html))
    dump(args.data/'figure_manifest.json',{'figures':mapping,'layout':'one chart per file; no triptychs',
        'standalone_panels':len(figures),'review_pdf_pages':len(figures),'source_manifest_sha256':digest(args.data/'data_manifest.json'),
        'plot_script_sha256':digest(Path(__file__)),'aggregation':'checkpoint/loss/input conditions remain distinct; only banks within cells are averaged',
        'font_points':{'base':18,'axes':19,'ticks':16,'matrix_values':15,'compact_legends':13},
        'title_count':0,'explanatory_figure_text_count':0,'panel_values_sha256':digest(args.data/'panel_values.json'),
        'table_rows':{'branches':len(branch_rows),'pairs_with_threshold':len(pair_rows),'thresholds':len(threshold_rows),
                      'responses':len(response_rows),'covariance':len(covariance_rows),'heldout':len(heldout_rows),
                      'teacher_js':len(js_rows),'normalization_gradients':len(normalization_gradient_rows)}})
    print(json.dumps({'verified_jobs':len(jobs),'standalone_panels':len(figures),'output':str(args.output)}))


if __name__=='__main__':
    main()
