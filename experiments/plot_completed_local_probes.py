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
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import html

    BLUE, ORANGE, INK = '#246a9a', '#c46624', '#263541'
    sequential = LinearSegmentedColormap.from_list('paper_blue', ['#f6f9fc', '#accbdc', '#246a9a'])
    diverging = LinearSegmentedColormap.from_list('paper_balance', ['#b75b22', '#fffdf8', '#246a9a'])
    figures, mapping, plotted_rows, summaries, layout_checks = [], [], [], [], []
    captions, questions, readings, tiers = {}, {}, {}, {}

    def point(panel, job, measure, value, **coordinates):
        filename={'response_length':'banks.json','token_minus_prompt_share_pp':'banks.json',
                  'response_weight_percent':'weights.json'}.get(measure,'measurements.json')
        sources=[f'raw/{job}/{filename}']
        if measure=='response_weight_percent':sources.append(f'raw/{job}/banks.json')
        if measure=='l2_over_pg_mean':
            sources=sorted(set(sources+[f"raw/{j['id']}/measurements.json" for j in density if j['state']==coordinates['state']]))
        plotted_rows.append({'panel': panel, 'job': job, 'source': f'raw/{job}/{filename}','source_files':sources,
                             'measure': measure, 'value': float(value), 'coordinates': coordinates})

    def summary(panel, row, column, observations, statistic='mean', **extra):
        a=np.asarray(observations,dtype=float)
        result={'panel':panel,'row':row,'column':column,'statistic':statistic,
                'value':float(np.mean(a)),'minimum':float(np.min(a)),'maximum':float(np.max(a)),
                'observations':len(a),**extra}
        summaries.append(result)
        return result['value']

    def matrix(values, columns, rows, label, limits, fmt='.2f', size=(8.2,5.4),
               center=None, separators=(), group_labels=None, group_size=2, fontsize=16):
        # Labels and numbers carry identities; colors encode only the measured value.
        fig,ax=plt.subplots(figsize=size)
        left=.31 if group_labels else .29
        fig.subplots_adjust(left=left,right=.98,bottom=.30,top=.98)
        values=np.asarray(values,dtype=float)
        lo,hi=limits
        norm=TwoSlopeNorm(vmin=lo,vcenter=center,vmax=hi) if center is not None else plt.Normalize(lo,hi)
        cmap=diverging if center is not None else sequential
        im=ax.imshow(values,aspect='auto',norm=norm,cmap=cmap,interpolation='nearest')
        ax.set_xticks(range(len(columns)),columns,fontsize=16)
        ax.set_yticks(range(len(rows)),rows,fontsize=16)
        ax.tick_params(axis='both',length=0,pad=8)
        for spine in ax.spines.values():spine.set_visible(False)
        for i,row in enumerate(values):
            for k,value in enumerate(row):
                rgba=cmap(norm(value))
                rgb=np.asarray(rgba[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                luminance=float(np.dot(linear,[.2126,.7152,.0722]))
                black_contrast=(luminance+.05)/.05;white_contrast=1.05/(luminance+.05)
                text_color='black' if black_contrast>=white_contrast else 'white'
                assert max(black_contrast,white_contrast)>=4.5
                ax.text(k,i,format(value,fmt),ha='center',va='center',fontsize=fontsize,
                        color=text_color)
        for y in separators:ax.axhline(y,color='white',lw=3)
        if group_labels:
            for i,name in enumerate(group_labels):
                ax.text(-.19,i*group_size+(group_size-1)/2,name,transform=ax.get_yaxis_transform(),
                        ha='right',va='center',fontsize=16,color=INK)
            ax.tick_params(axis='y',labelsize=15,pad=5)
        cbax=fig.add_axes([left+.08,.105,.98-left-.16,.028])
        cb=fig.colorbar(im,cax=cbax,orientation='horizontal')
        if center==1:cb.set_ticks([0,.5,1,1.5,2])
        cb.outline.set_visible(False);cb.ax.tick_params(labelsize=14,length=3,pad=3)
        cb.set_label(label,fontsize=17,labelpad=6)
        return fig,ax

    def compare(rows, series, xlabel, limits, log=False, size=(8.8,5.7), separators=()):
        # Each endpoint has a dedicated vertical lane. Ranges never obscure another series.
        fig,ax=plt.subplots(figsize=size)
        fig.subplots_adjust(left=.30,right=.98,bottom=.25,top=.98)
        for i in range(len(rows)):
            ax.axhspan(i-.44,i+.44,color='#f3f6f8' if i%2==0 else 'white',zorder=0)
            av,bv=[np.mean(s['values'][i]) for s in series]
            ax.plot([av,bv],[i-.13,i+.13],color='#b7c2cb',lw=1.2,zorder=1)
        ax._comparison_centers=[]
        for k,s in enumerate(series):
            for i,values in enumerate(s['values']):
                mid=float(np.mean(values));low=float(np.min(values));high=float(np.max(values))
                ax.errorbar(mid,i+(-.13 if k==0 else .13),xerr=[[mid-low],[high-mid]],
                            fmt=['o','s'][k],ms=8,color=[BLUE,ORANGE][k],elinewidth=1.6,
                            capsize=3,mec='white',mew=.7,zorder=3)
                ax._comparison_centers.append((mid,i+(-.13 if k==0 else .13)))
        ax.set_yticks(range(len(rows)),rows,fontsize=16)
        ax.set_ylim(len(rows)-.55,-.55);ax.set_xlim(*limits)
        ax.set_xlabel(xlabel,fontsize=19,labelpad=9)
        ax.tick_params(axis='y',length=0,pad=10)
        ax.spines['left'].set_visible(False)
        if log:ax.set_xscale('log')
        else:ax.axvline(0,color='#bdc6cd',lw=1,zorder=0)
        for y in separators:ax.axhline(y,color='#bec8d0',lw=1)
        handles=[Line2D([],[],marker=['o','s'][k],color=[BLUE,ORANGE][k],lw=1.6,
                         markersize=8,label=s['name']) for k,s in enumerate(series)]
        ax.legend(handles=handles,ncol=2,frameon=False,fontsize=15,loc='upper center',
                  bbox_to_anchor=(.5,-.16),handlelength=1.7,columnspacing=1.3)
        return fig,ax

    def save(fig,panel,used_jobs,question,reading,caption,encoding,tier='appendix'):
        assert not fig.texts and all(not ax.get_title() for ax in fig.axes)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            boxes=[t.get_window_extent(renderer) for t in ax.texts]
            for i,box in enumerate(boxes):
                assert not any(box.overlaps(other) for other in boxes[i+1:]), f'Overlapping numbers/labels: {panel}'
            centers=ax.transData.transform(getattr(ax,'_comparison_centers',[])) if hasattr(ax,'_comparison_centers') else []
            for i,center in enumerate(centers):
                assert all(np.linalg.norm(center-other)>8*fig.dpi/72 for other in centers[i+1:]), f'Overlapping comparison markers: {panel}'
        layout_checks.append({'panel':panel,'matrix_labels_overlap':False,'comparison_markers_overlap':False,
                              'axes_titles':0,'figure_prose':0,'matrix_contrast_minimum':4.5})
        for ext in ['pdf','png','svg']:fig.savefig(args.output/f'{panel}.{ext}',bbox_inches='tight',pad_inches=.08)
        figures.append((panel,fig));captions[panel]=caption;questions[panel]=question;readings[panel]=reading;tiers[panel]=tier
        mapping.append({'figure':panel,'panels':[panel],'jobs':used_jobs,'question':question,
                        'encoding':encoding,'suggested_placement':tier,'caption_file':'FIGURE_CAPTIONS.md',
                        'standalone':True,'main_axes':1,'title':None})

    ordinary=[j for j in normalization if j['id'].endswith(('draw42','draw43'))]
    bank_labels=[f"Step {j['snapshot_step']}\nBank {j['id'].rsplit('draw',1)[1]}" for j in ordinary]
    values=[];response_labels=[]
    for domain,name in zip(DOMAINS,DOMAIN_NAMES):
        for ordinal in range(2):
            row=[];response_labels.append(f'{name} {ordinal+1}')
            for j in ordinary:
                response=[r for r in response_rows if r['job']==j['id'] and r['domain']==domain][ordinal]
                row.append(response['length'])
                point('F1a',j['id'],'response_length',response['length'],domain=domain,
                      response=response['response'],truncated=response['truncated'])
            values.append(row)
    fig,ax=matrix(values,bank_labels,response_labels,'Response length (tokens)',(0,512),fmt='.0f',
                  size=(8.1,6.0),separators=[1.5,3.5,5.5])
    save(fig,'F1a',[j['id'] for j in ordinary],
         '诊断批次中的长度差异有多大，512-token上限影响了哪些观测？',
         'Math与Science全部触及512-token上限；不能用这批长度推断自然回答分布。',
         'Response lengths in four fixed diagnostic banks (columns: student checkpoint / bank draw). Rows enumerate the two responses per domain within each bank; response indices do not identify matched prompts across banks. Every 512-token response is truncated at the cap; shorter responses completed. These capped local banks are not an estimate of the natural online response-length distribution.',
         'one response per labeled cell; no jitter or overlaid points')

    values=[]
    for domain in DOMAINS:
        row=[]
        for j in ordinary:
            rr=[r for r in response_rows if r['job']==j['id']]
            token=100*sum(r['length'] for r in rr if r['domain']==domain)/sum(r['length'] for r in rr)
            value=token-25;row.append(value)
            point('F1b',j['id'],'token_minus_prompt_share_pp',value,domain=domain,token_percent=token,prompt_percent=25)
        values.append(row)
    fig,ax=matrix(values,bank_labels,DOMAIN_NAMES,'Token share − prompt share\n(percentage points)',
                  (-10,10),fmt='+.1f',center=0,size=(8.0,4.8))
    save(fig,'F1b',[j['id'] for j in ordinary],
         '相同prompt配额下，各任务的token份额偏离了多少？',
         '每格直接显示相对25% prompt份额的偏差，正负号即可区分增权和减权。',
         'Token share minus prompt share, in percentage points, for the banks in F1a. Each domain contributes 25% of prompts; positive values indicate a larger share of generated tokens and negative values a smaller share. This is the domain weighting induced by global-token averaging on these batches, not the domain coefficient under domain-response averaging. Rows remain separate even when numerical values coincide.',
         'rows=domain; columns=checkpoint/bank; signed deviation from the explicit 25% reference')

    quota='normalization-m-pg500-draw42-quota1234-uniform'
    rr=[r for r in response_rows if r['job']==quota];values=[];labels=[];counts={d:0 for d in DOMAINS}
    for r in rr:
        counts[r['domain']]+=1;labels.append(f"{DOMAIN_NAMES[DOMAINS.index(r['domain'])]} {counts[r['domain']]}")
        values.append([100*r[k] for k in ['DR','DT','GT']])
        for k in ['DR','DT','GT']:point('F1c',quota,'response_weight_percent',100*r[k],rule=k,response=r['response'],domain=r['domain'])
    fig,ax=matrix(values,['Domain\nresponse (DR)','Domain\ntoken (DT)','Global\ntoken (GT)'],labels,
                  'Effective response weight (%)',(0,25),fmt='.2f',size=(8.4,6.7),separators=[.5,2.5,5.5])
    save(fig,'F1c',[quota],
         'DR、DT、GT究竟怎样改变同一批回答的训练权重？',
         '沿列看域内回答是否等权，沿行看同一回答在三种规则下如何变化。',
         'Effective coefficients multiplying each response-mean loss on the same PG500 bank. Math/code/instruction-following/science supply 1/2/3/4 responses and the intended domain prior is uniform. Domain-response (DR) gives equal weight to responses within each domain; domain-token (DT) changes their relative weights with length while preserving the domain total; global-token (GT) also changes domain totals. All columns sum to 100%. The matched prompt-share-prior control is retained in the source data.',
         'one response per row; three explicitly named reductions; numerical weights with common scale',tier='main')

    norm_ids=[j['id'] for j in ordinary]+['normalization-m-pg500-draw42-long2048',
             'normalization-m-pg500-draw42-quota1234-uniform','normalization-m-pg500-draw42-quota1234-prompt']
    norm_labels=['250 · bank 42','250 · bank 43','500 · bank 42','500 · bank 43',
                 '500 · cap 2048','500 · quota / uniform','500 · quota / prompt']
    values=[]
    for job in norm_ids:
        row=[]
        for domain in DOMAINS:
            value=next(r['covariance_correction_l2'] for r in covariance_rows if r['job']==job and r['domain']==domain)
            row.append(value);point('F2a',job,'covariance_correction_l2',value,domain=domain)
        values.append(row)
    fig,ax=matrix(values,DOMAIN_NAMES,norm_labels,'Length-weighting correction (L2)',(0,np.max(values)),
                  fmt='.2f',size=(8.8,5.9),separators=[3.5,4.5])
    save(fig,'F2a',norm_ids,
         '长度加权造成的梯度修正主要出现在哪些任务？',
         '同一尺度的数值矩阵让零值、IF的大修正和不等配额对照都可以直接查读。',
         'Magnitude of the length-weighting correction, ||Cov(T,g)|| / mean(T), by domain and diagnostic condition. The first four rows use balanced cap512 banks, the fifth cap2048, and the last two share quota1/2/3/4 responses under uniform/prompt-share domain priors. Both quota banks and the long bank use draw42. Displayed 0.00 values are rounded; full precision is retained. The fixed-batch identity has maximum relative residual 1.8e-16 and does not itself establish capability improvement or distributed reducer correctness.',
         'rows=seven conditions; columns=domains; one correction per cell')

    values=[]
    for job in norm_ids:
        row=[]
        for pair in ['DR_DT','DR_GT','DT_GT']:
            v=data[job]['gradient_comparisons'][pair]['cosine'];row.append(v)
            point('F2b',job,'raw_gradient_cosine',v,reduction_pair=pair)
        values.append(row)
    fig,ax=matrix(values,['DR vs DT','DR vs GT','DT vs GT'],norm_labels,'Gradient cosine',(0,1),
                  fmt='.3f',size=(8.6,5.8),separators=[3.5,4.5])
    save(fig,'F2b',norm_ids,
         '只改平均方式，梯度方向是否也会改变？',
         '余弦1表示方向相同；长回答批次DR–DT为0.706、DR–GT为0.658。',
         'Cosine similarity of pre-clipping gradients under domain-response (DR), domain-token (DT), and global-token (GT) averaging. Within each row, the student, saved optimizer state and responses are identical; cosine one means equal direction. Row conditions are as in F2a. Direction differences depend on the batch: for example, the long-response row gives 0.706 and 0.658 for DR–DT and DR–GT. These local directions do not rank online capability.',
         'all seven conditions in common order; directly labeled reduction pairs; fixed [0,1] scale',tier='main')

    values=[]
    for job in norm_ids:
        control=data[job]['branches']['zero_gradient']['heldout_kl_decrease']['macro'];row=[]
        for rule in ['DR','DT','GT']:
            observed=data[job]['branches'][rule]['heldout_kl_decrease']['macro']
            v=1000*(observed-control);row.append(v)
            point('F2c',job,'extra_kl_decrease_vs_zero_millinats',v,rule=rule,rule_kl_decrease=observed,zero_kl_decrease=control)
        values.append(row)
    bound=np.ceil(np.max(np.abs(values))*2)/2
    fig,ax=matrix(values,['DR','DT','GT'],norm_labels,'Extra KL decrease vs zero-gradient\n(millinats)',
                  (-bound,bound),fmt='+.2f',center=0,size=(8.8,5.8),separators=[3.5,4.5])
    save(fig,'F2c',norm_ids,
         '局部KL改善是否超过仅沿用Adam历史的zero-gradient对照？',
         '正值才表示超过zero对照；正负混杂，不能据此排名归一化规则。',
         'Additional macro held-out KL decrease relative to the zero-gradient Adam control: 1000 × (KL decrease under the reduction − KL decrease under zero current gradient), in millinats. Positive cells outperform the control on this diagnostic bank; negative cells do not. Each domain has only one held-out response. The matched quota reruns also exhibit small unexplained GT numerical differences, so these exploratory measurements cannot support a rule ranking, significance claim or capability claim. Unadjusted values, including the control, remain in heldout.csv.',
         'signed paired difference against same-job optimizer control; all seven rows; raw values retained')

    teacher_runs={(j['snapshot_step'],int(j['id'].rsplit('draw',1)[1])):data[j['id']] for j in teachers}
    teacher_ids=[j['id'] for j in teachers]
    conditions=[(250,'sampled_pg'),(250,'topk_intersection'),(500,'sampled_pg'),(500,'topk_intersection')]
    condition_labels=['PG · step 250','I64 · step 250','PG · step 500','I64 · step 500']
    condition_columns=['PG\n250','I64\n250','PG\n500','I64\n500']
    modes=['routed','common'];mode_names={'routed':'Task inputs','common':'Shared inputs'}

    def pair_observations(panel,step,loss,mode,measure):
        obs=[]
        for draw in [42,43]:
            d=teacher_runs[step,draw]
            for a,b in PAIRS:
                p=cosine_pair(d,f'{mode}/{a}/{loss}',f'{mode}/{b}/{loss}')
                value=p['raw_gradient']['cosine'] if measure=='raw_gradient_cosine' else p['metrics']['cosine']
                obs.append(value);point(panel,d['job'],measure,value,step=step,loss=loss,mode=mode,left=a,right=b,bank=draw)
        return obs

    series=[]
    for mode in modes:
        values=[]
        for label,(step,loss) in zip(condition_labels,conditions):
            obs=pair_observations('F5a',step,loss,mode,'raw_gradient_cosine');values.append(obs)
            summary('F5a',label,mode,obs)
        series.append({'name':'Task-specific inputs' if mode=='routed' else 'Shared inputs','values':values})
    fig,ax=compare(condition_labels,series,'Teacher-pair gradient cosine',(-.08,1.04),size=(8.7,4.7),separators=[1.5])
    ax.set_xticks([0,.25,.5,.75,1]);ax.axvline(1,color='#bdc6cd',lw=1,zorder=0)
    save(fig,'F5a',teacher_ids,
         '教师梯度的近正交，是否仍存在于相同输入上？',
         '任务输入下的均值靠近0；共享输入下靠近1。输入条件是解释教师差异的必要控制。',
         'Teacher-pair raw-gradient alignment under task-specific versus shared inputs. PG and I64 are shown at both student checkpoints. Each marker is the mean of all six teacher pairs in two diagnostic banks within the labeled condition; horizontal segments show the full observed minimum–maximum over those 12 dependent pair/bank values, not a confidence interval. Small vertical offsets separate the two input conditions. Routed inputs mix teacher and task differences; the shared-input control uses the same prefix bank and PG actions across teachers. Near-orthogonal routed gradients therefore do not establish disjoint teacher-specific subnetworks.',
         'four checkpoint/loss rows; two input conditions in separate lanes; mean and full observed range',tier='main')

    labels=[];series=[{'name':'Raw gradient','values':[]},{'name':'BF16 update','values':[]}]
    for mode in modes:
        for label,(step,loss) in zip(condition_labels,conditions):
            row=f'{label}\n{mode_names[mode]}';labels.append(row)
            for s,measure in zip(series,['raw_gradient_cosine','bf16_cosine']):
                obs=pair_observations('F5b',step,loss,mode,measure);s['values'].append(obs)
                summary('F5b',row,measure,obs)
    fig,ax=compare(labels,series,'Teacher-pair cosine',(-.08,1.04),size=(8.8,7.4),separators=[3.5])
    ax.set_xticks([0,.25,.5,.75,1]);ax.axvline(1,color='#bdc6cd',lw=1,zorder=0)
    save(fig,'F5b',teacher_ids,
         '原始梯度近正交，是否意味着实际写回方向也近正交？',
         '在任务输入下，梯度余弦接近0，而BF16写回余弦仍较高；共享输入对照保留在下半部分。',
         'Raw-gradient and BF16 proposed-update cosine on the same teacher pairs, with input conditions in separate rows. Markers and minimum–maximum segments summarize all six pairs across two banks per condition as in F5a; overlapping numerical ranges occupy separate vertical lanes. All local updates start from the same saved Adam state within a checkpoint. Stronger alignment after optimizer processing does not attribute historical updates to individual teachers; zero-current-gradient controls are shown in F5c.',
         'eight explicit condition rows; gradient and update means/ranges in separate lanes; no cloud of 96 points',tier='main')

    labels=[];series=[{'name':'Observed','values':[]},{'name':'Layer-random reference','values':[]}]
    for mode in modes:
        for label,(step,loss) in zip(condition_labels,conditions):
            row=f'{label}\n{mode_names[mode]}';labels.append(row);observed=[];random=[]
            for draw in [42,43]:
                d=teacher_runs[step,draw]
                for teacher in DOMAINS:
                    s=cosine_pair(d,'zero_gradient',f'{mode}/{teacher}/{loss}')['metrics']['support']['1e-05']
                    observed.append(s['jaccard']);random.append(s['layer_random_jaccard_ratio_of_expectations'])
                    point('F5c',d['job'],'zero_gradient_jaccard',s['jaccard'],step=step,loss=loss,mode=mode,
                          teacher=teacher,layer_random_ratio=s['layer_random_jaccard_ratio_of_expectations'])
            for s,obs in zip(series,[observed,random]):s['values'].append(obs);summary('F5c',row,s['name'],obs)
    fig,ax=compare(labels,series,'Jaccard with zero-gradient update',(1e-4,1.4),log=True,size=(8.8,7.4),separators=[3.5])
    ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1],['0.0001','0.001','0.01','0.1','1'])
    save(fig,'F5c',teacher_ids,
         '不同教师的写回是否大量包含zero-gradient也会移动的参数？',
         '观测重叠与逐层匹配随机参照相隔多个数量级；只给两种参照颜色，教师差异用范围保留。',
         'Support overlap with the zero-current-gradient Adam control at absolute BF16 threshold 1e-5. Each point is the mean over four teachers and two banks within one labeled condition, with the full minimum–maximum range. The layer-random reference is the ratio of expected intersection to expected union under matched per-layer counts, not the expected Jaccard itself. The horizontal axis is logarithmic. A separate all-coordinate audit at PG250/500 found BF16(saved master) identical to the saved model before any Adam step, excluding pre-existing master/model disagreement at those two checkpoints.',
         'eight condition rows; observed and random mean/range; logarithmic axis with explicit numeric ticks')

    distances={(step,draw):{frozenset([r['left'],r['right']]):r['full_vocab_js'] for r in d['teacher_distances']}
               for (step,draw),d in teacher_runs.items()}
    ordered_pairs=sorted(PAIRS,key=lambda pair:np.mean([v[frozenset(pair)] for v in distances.values()]),reverse=True)
    pair_labels=[f'{DOMAIN_NAMES[DOMAINS.index(a)]}–{DOMAIN_NAMES[DOMAINS.index(b)]}' for a,b in ordered_pairs]
    values=[]
    for pair_label,(a,b) in zip(pair_labels,ordered_pairs):
        row=[]
        for step in [250,500]:
            obs=[]
            for draw in [42,43]:
                v=distances[step,draw][frozenset([a,b])];obs.append(v)
                point('F6a',teacher_runs[step,draw]['job'],'teacher_js',v,step=step,bank=draw,left=a,right=b)
            row.append(summary('F6a',pair_label,str(step),obs))
        values.append(row)
    fig,ax=matrix(values,['Step 250','Step 500'],pair_labels,'Teacher distribution distance (JS, nats)',
                  (0,.04),fmt='.4f',size=(7.7,5.6))
    save(fig,'F6a',teacher_ids,
         '哪对教师的输出分布更不同？',
         '教师对按平均JS从大到小排序；F6b/c保持完全相同的行顺序以核对更新差异。',
         'Full-vocabulary Jensen–Shannon divergence between teachers on shared student prefixes. Cells average two banks within a student checkpoint. Rows are ordered by mean JS across the four banks solely to define a common display order for F6a–c; checkpoint values remain separate. JS is measured before applying the local objective and is not duplicated by PG/I64. The teachers are fixed: checkpoint-specific differences reflect changed student prefix banks.',
         'six distinct pairs in a common order; two checkpoint columns; no redundant loss dimension')

    for panel,mode in [('F6b','routed'),('F6c','common')]:
        values=[]
        for pair_label,(a,b) in zip(pair_labels,ordered_pairs):
            row=[]
            for label,(step,loss) in zip(condition_labels,conditions):
                obs=[]
                for draw in [42,43]:
                    d=teacher_runs[step,draw]
                    v=1-cosine_pair(d,f'{mode}/{a}/{loss}',f'{mode}/{b}/{loss}')['metrics']['support']['1e-05']['jaccard']
                    obs.append(v);point(panel,d['job'],'bf16_support_distance',v,step=step,bank=draw,loss=loss,
                                        mode=mode,left=a,right=b,teacher_js=distances[step,draw][frozenset([a,b])])
                row.append(summary(panel,pair_label,label,obs))
            values.append(row)
        label='Update-set distance (1 − Jaccard)\nTask-specific inputs' if mode=='routed' else 'Update-set distance (1 − Jaccard)\nShared inputs'
        fig,ax=matrix(values,condition_columns,pair_labels,label,(0,.4),fmt='.3f',size=(8.4,5.6))
        save(fig,panel,teacher_ids,
             'JS较大的教师对是否也有更不同的更新位置？' if mode=='routed' else '共享输入后，教师对的更新位置差异如何变化？',
             '沿用F6a的教师对顺序；逐列查读PG/I64及250/500，避免用重叠散点暗示相关性。' if mode=='routed' else '与F6b使用相同的行序、列序和0–0.4色标，便于直接比较输入控制。',
             f'BF16 update-set distance under {"task-specific" if mode=="routed" else "shared"} inputs, at absolute threshold 1e-5. A value of zero means identical selected parameter sets. Each cell averages two banks for one teacher pair, checkpoint and loss. Teacher-pair order is identical to F6a, and F6b/c share the same numerical scale and column order. All PG/I64 and 250/500 conditions are retained. The small teacher pool and shared pairs support a descriptive comparison only; no regression, significance test or universal monotone JS–update relationship is claimed.',
             'six named teacher pairs × four named checkpoint/loss conditions; fixed matched color scale')

    states=['m-pg250','m-pg500','s-pg250','m-i64dr250','s-i64250']
    state_labels=['Joint PG · 250','Joint PG · 500','Single PG · 250','Joint I64* · 250','Single I64* · 250']
    method_labels=['Zero\ngradient','PG','Student\nTop64','Top64\nintersection','Teacher\nTop64','Full\nvocab.']
    state_runs={s:[data[j['id']] for j in density if j['state']==s] for s in states}
    values=[]
    for state,label in zip(states,state_labels):
        row=[]
        for method,name in zip(METHODS,method_labels):
            obs=[]
            for d in state_runs[state]:
                v=100*d['branches'][method]['metrics']['bf16_writeback']['fraction_above_1e-05'];obs.append(v)
                point('F7a',d['job'],'bf16_changed_percent',v,state=state,loss=method,threshold=1e-5)
            row.append(summary('F7a',label,name.replace('\n',' '),obs))
        values.append(row)
    fig,ax=matrix(values,method_labels,state_labels,'Parameters changed by more than 10⁻⁵ (%)',(0,.09),
                  fmt='.4f',size=(9.9,5.3),fontsize=15)
    save(fig,'F7a',[j['id'] for j in density],
         '在同一学生状态上，更密的监督是否明显扩大参数写回范围？',
         '每行固定学生与Adam状态，横向比较六个目标；图中直接使用百分比，无需再换算10⁻²%。',
         'Percentage of parameters with absolute local BF16 proposed change greater than 1e-5. Each row fixes the student and saved Adam state; each cell averages four diagnostic banks (42–45). Read across a row to compare objectives, rather than attributing differences between independently trained states to the local loss. Zero gradient retains Adam history; Student Top64, intersection with teacher Top64 (I64), teacher-selected Top64 and full vocabulary are distinct local objectives. Asterisks denote mixed Student64-to-I64 training history, not all-I64 training. Other thresholds and bank ranges are available in the data tables.',
         'five student states × six local objectives; direct percent units; matched within-row comparison',tier='main')

    alternatives=[m for m in METHODS if m!='sampled_pg']
    alternative_labels=[n for m,n in zip(METHODS,method_labels) if m!='sampled_pg']
    values=[];row_labels=[]
    for state,label in zip(states,state_labels):
        for metric,name in [('gradient_l2_before_clipping','Gradient'),('bf16_l2','Update')]:
            row=[];row_labels.append(name)
            def read_metric(d,method):
                m=d['branches'][method]['metrics']
                return m['bf16_writeback']['l2'] if metric=='bf16_l2' else m[metric]
            denominator=float(np.mean([read_metric(d,'sampled_pg') for d in state_runs[state]]))
            for method,col in zip(alternatives,alternative_labels):
                obs=[]
                for d in state_runs[state]:
                    absolute=read_metric(d,method);v=absolute/denominator;obs.append(v)
                    point('F7b',d['job'],'l2_over_pg_mean',v,state=state,loss=method,metric=metric,
                          absolute_l2=absolute,pg_state_mean=denominator)
                row.append(summary('F7b',f'{label} / {name}',col.replace('\n',' '),obs,
                                   statistic='ratio_of_four_bank_means',pg_reference=denominator))
            values.append(row)
    fig,ax=matrix(values,alternative_labels,row_labels,'L2 norm / matched PG mean (PG = 1)',(0,2),
                  center=1,fmt='.3f',size=(9.6,7.2),group_labels=state_labels,separators=[1.5,3.5,5.5,7.5],fontsize=16)
    save(fig,'F7b',[j['id'] for j in density],
         '目标改变了梯度尺度，是否也按同等比例改变写回尺度？',
         '每个状态的Gradient/Update两行共用PG=1参照：梯度可大幅缩小，写回范数仍接近1。',
         'Pre-clipping raw-gradient and BF16 proposed-update L2 norms, each divided by the corresponding PG norm at the same student checkpoint. Values are ratios of four-bank means, not means of per-bank ratios and not gradient-to-update ratios. The PG column would be exactly one and is omitted; both rows use the same reference level and color scale. The zero-gradient column still shows nonzero writebacks from saved Adam history. Similar update norms do not imply identical update directions or learning outcomes; F7c reports angular differences. Asterisks identify mixed training history as in F7a.',
         'two adjacent metric rows per student; five objectives; ratio of bank means to metric-specific PG reference; no scatter',tier='main')

    values=[]
    for state,label in zip(states,state_labels):
        row=[]
        for method,name in zip(METHODS[:-1],method_labels[:-1]):
            obs=[]
            for d in state_runs[state]:
                cosine=cosine_pair(d,method,'full_vocab')['metrics']['cosine']
                angle=float(np.degrees(np.arccos(np.clip(cosine,-1,1))));obs.append(angle)
                point('F7c',d['job'],'bf16_angle_to_full_degrees',angle,state=state,loss=method,cosine=cosine)
            row.append(summary('F7c',label,name.replace('\n',' '),obs))
        values.append(row)
    fig,ax=matrix(values,method_labels[:-1],state_labels,'Angle to full-vocabulary update (degrees)',(0,40),
                  fmt='.1f',size=(9.0,5.3))
    save(fig,'F7c',[j['id'] for j in density],
         '写回范数相近时，方向是否仍有差异？',
         '0°表示与同bank的full-vocabulary写回同向；角度直接显示余弦接近1时仍存在的差别。',
         'Angle between each local BF16 proposed update and the full-vocabulary update from the same student, Adam state and bank. Each cell is the mean of four per-bank angles, computed as arccos(cosine) in degrees; it is not the angle of a mean update or arccos of a mean cosine. Zero degrees means aligned directions. Full-versus-full is zero by construction and omitted. Full vocabulary is a local reference, not a claim of an optimal direction or a completed online capability baseline. Asterisks retain mixed-history provenance.',
         'five student states × five objectives; directly readable angle; common [0,40] degree scale')

    main_order=['F1c','F2b','F5a','F5b','F7a','F7b']
    all_order=main_order+[p for p,_ in figures if p not in main_order]
    figure_by_id=dict(figures)
    for filename,order in [('all_panels.pdf',all_order),('main_panels.pdf',main_order)]:
        with PdfPages(args.output/filename) as pdf:
            for panel in order:pdf.savefig(figure_by_id[panel],bbox_inches='tight',pad_inches=.08)
    for _,fig in figures:plt.close(fig)
    dump(args.data/'panel_values.json',plotted_rows);dump(args.data/'panel_summaries.json',summaries)
    dump(args.data/'layout_checks.json',layout_checks)
    caption_lines=['# 实验图 Caption 草稿','',
        '下列说明用于论文caption；不写入图片。正文候选顺序：'+ ' → '.join(main_order)+'。其他图作为补充诊断。', '',
        '**共同范围**：全部为固定学生与保存Adam状态下的HF局部梯度/模拟BF16写回，不是实际线上训练步或能力评测。', '',
        'I64是学生与教师Top64交集；权重仍按原学生Top64归一化，交集上不再归一化。诊断bank不是训练seed；所有范围均为描述性观测范围，不是置信区间。', '']
    for panel in all_order:
        caption_lines += [f'## {panel}', '',f'审稿人问题：{questions[panel]}', '',captions[panel], '']
    (args.data/'FIGURE_CAPTIONS.md').write_text('\n'.join(caption_lines)+'\n')
    review_lines=['# 作图审稿检查与阅读顺序','',
        '本轮重画以“一个明确问题、一个可读比较”为验收标准。旧版主要问题是把多种条件压在同一散点云中，并让读者用多重图例猜测比较对象。', '',
        '## 正文与补充材料','',
        '建议正文使用六张独立图，每项论点两张：F1c/F2b（平均规则→梯度方向），F5a/F5b（输入控制→梯度与写回），F7a/F7b（局部目标→范围与尺度）。', '',
        'F1a/F1b是受cap约束的样本描述；F2a验证局部修正；F2c数值不稳定、仅可作探索诊断；F5c检查Adam历史；F6a–c逐教师对核对分布与位置；F7c避免从范数相近推出方向等价。', '',
        'F6不再暗示JS与更新位置有稳定回归关系。所有教师对保留在同序矩阵；并未只挑有利条件。', '',
        '## 逐图问题与图形选择','',
        '| 图 | 必须能回答的问题 | 当前读法 | 建议位置 |','|---|---|---|---|']
    for panel in all_order:review_lines.append(f'| {panel} | {questions[panel]} | {readings[panel]} | {tiers[panel]} |')
    review_lines += ['', '## 保留与转换约定','',
        '- F5a/b在每个checkpoint/loss/input条件内汇总六个教师对×两个bank，F5c汇总四个教师×两个bank。点为算术均值、线为全部观测min–max；不构造CI，不当作独立重复。',
        '- F6/F7矩阵仅在固定条件内平均bank，不跨checkpoint或loss。F6行序使用四批JS均值仅作排序。',
        '- F1b减去已知25% prompt参照；F2c减去同job的zero-gradient KL下降；F7b分别将两个范数的四bank均值除以同状态PG的对应范数均值；F7c先逐bank将cosine变成角度再平均。',
        '- 每个原始值与转换参数都记录在panel_values.json；矩阵或区间汇总另见panel_summaries.json；189个源文件保持原SHA256。',
        '- 图内只保留必要轴标签、数值和最多两系列图例；不用颜色+点形+空心叠加编码多个条件，不用图内标题或说明段落。',
        '- 英文caption草稿给出样本、定义、统计单位及限制。浏览页的问题/解读在图片外，便于讨论；导出的PDF/PNG/SVG保持纯图。', '']
    (args.data/'REVIEWER_FIGURE_AUDIT_zh.md').write_text('\n'.join(review_lines)+'\n')
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>实验图：问题与证据</title><style>body{font:18px/1.65 system-ui;max-width:1080px;margin:32px auto;padding:0 24px;color:#20313e}h1{font-size:28px}h2{font-size:23px}section{margin:40px 0 70px;border-top:1px solid #dce2e7;padding-top:18px}img{width:100%;height:auto}a{color:#246a9a}nav{line-height:2.4}.caption{font-size:16px}summary{cursor:pointer}p{max-width:950px}</style>',
          '<h1>实验图：按论文问题阅读</h1><p>正文候选按三项论点排列；每个文件一张图，说明全部在图外。</p>',
          '<p><a href="main_panels.pdf">六张正文候选（一图一页）</a> · <a href="all_panels.pdf">全部15张（一图一页）</a></p>',
          '<nav>'+ ' · '.join(f'<a href="#{p}">{p}</a>' for p in all_order)+'</nav>']
    for panel in all_order:
        page += [f'<section id="{panel}"><h2>{panel} · {html.escape(questions[panel])}</h2>',
                 f'<p>{html.escape(readings[panel])}</p><p>{"正文候选" if tiers[panel]=="main" else "补充诊断"} · <a href="{panel}.pdf">PDF</a> · <a href="{panel}.png">PNG</a> · <a href="{panel}.svg">SVG</a></p>',
                 f'<a href="{panel}.pdf"><img loading="lazy" src="{panel}.png" alt="{html.escape(questions[panel])}"></a>',
                 f'<details><summary>论文 caption</summary><p class="caption">{html.escape(captions[panel])}</p></details></section>']
    page.append('</html>');(args.output/'index.html').write_text('\n'.join(page))
    dump(args.data/'figure_manifest.json',{'figures':mapping,'layout':'one chart per file; no scatter clouds or triptychs',
        'standalone_panels':len(figures),'review_pdf_pages':len(figures),'main_pdf_pages':len(main_order),
        'main_reading_order':main_order,'all_reading_order':all_order,
        'source_manifest_sha256':digest(args.data/'data_manifest.json'),'plot_script_sha256':digest(Path(__file__)),
        'aggregation':'conditions remain separate; F5 mean/min–max over named pair/teacher observations, F6/F7 bank means; transformations documented',
        'font_points':{'base':18,'axes':19,'ticks':16,'matrix_values':15,'legends':15},
        'title_count':0,'explanatory_figure_text_count':0,'scatter_cloud_count':0,'max_comparison_series':2,
        'panel_values_sha256':digest(args.data/'panel_values.json'),'panel_summaries_sha256':digest(args.data/'panel_summaries.json'),
        'table_rows':{'branches':len(branch_rows),'pairs_with_threshold':len(pair_rows),'thresholds':len(threshold_rows),
                      'responses':len(response_rows),'covariance':len(covariance_rows),'heldout':len(heldout_rows),
                      'teacher_js':len(js_rows),'normalization_gradients':len(normalization_gradient_rows)}})
    print(json.dumps({'verified_jobs':len(jobs),'standalone_panels':len(figures),'main_panels':len(main_order),'output':str(args.output)}))


if __name__=='__main__':
    main()
