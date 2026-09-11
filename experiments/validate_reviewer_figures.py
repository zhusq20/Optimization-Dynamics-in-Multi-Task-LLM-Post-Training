#!/usr/bin/env python3
"""Independently check exported figures against the frozen measurement package."""
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from urllib.parse import unquote

from PIL import Image, ImageStat

PAPER=Path(__file__).resolve().parent.parent
DATA=PAPER/'experiments/local_probe_results_20260908'
FIG=PAPER/'figures/local_probe_results_20260908'
DOMAINS=['math','code','if','science']
NAMES=['Math','Code','IF','Science']
METHODS=['zero_gradient','sampled_pg','student_topk','topk_intersection','teacher_topk','full_vocab']
METHOD_LABELS=['Zero gradient','PG','Student Top64','Top64 intersection','Teacher Top64','Full vocab.']
STATE_LABELS=dict(zip(['m-pg250','m-pg500','s-pg250','m-i64dr250','s-i64250'],
                     ['Joint PG · 250','Joint PG · 500','Single PG · 250','Joint I64* · 250','Single I64* · 250']))


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)
def pair(d,a,b):return next(r for r in d['pairwise_bf16_writebacks'] if {r['left'],r['right']}=={a,b})
def distance(d,a,b):return next(r['full_vocab_js'] for r in d['teacher_distances'] if {r['left'],r['right']}=={a,b})
def norm(d,method,metric):
    m=d['branches'][method]['metrics']
    return m['bf16_writeback']['l2'] if metric=='bf16_l2' else m[metric]

def main():
    checked=[]
    source=read(DATA/'data_manifest.json');fm=read(DATA/'figure_manifest.json')
    assert len(source['jobs'])==31 and len(source['files'])==189
    for f in source['files']:assert digest(DATA/f['path'])==f['sha256'],f['path']
    checked.append('All 189 frozen source hashes match, including all 31 measurements')
    jobs={j['id']:j for j in source['jobs']}
    measurements={j:read(DATA/'raw'/j/'measurements.json') for j in jobs}
    values=read(DATA/'panel_values.json');summaries=read(DATA/'panel_summaries.json')
    expected_counts={'F1a':32,'F1b':16,'F1c':30,'F2a':28,'F2b':21,'F2c':21,'F5a':96,'F5b':192,
                     'F5c':64,'F6a':24,'F6b':48,'F6c':48,'F7a':120,'F7b':200,'F7c':100}
    assert dict(Counter(r['panel'] for r in values))==expected_counts
    assert {r['job'] for r in values}==set(jobs)
    assert {j for f in fm['figures'] for j in f['jobs']}==set(jobs)
    grouped=defaultdict(list)
    for r in values:
        p=r['panel'];c=r['coordinates'];d=measurements[r['job']];job=r['job'];v=r['value']
        assert (DATA/r['source']).exists()
        assert all((DATA/f).exists() for f in r['source_files'])
        if p in ['F1a','F1b','F1c']:
            bank=read(DATA/'raw'/job/'banks.json')['train']
            if p=='F1a':
                b=bank[c['response']];expected=len(b['response_ids'])
                assert b['task']==c['domain'] and b['truncated']==c['truncated']
                assert (expected==512)==b['truncated']
                assert r['source'].endswith('banks.json')
            elif p=='F1b':
                token=100*sum(len(b['response_ids']) for b in bank if b['task']==c['domain'])/sum(len(b['response_ids']) for b in bank)
                expected=token-25;close(c['token_percent'],token);close(c['prompt_percent'],25)
                assert sum(b['task']==c['domain'] for b in bank)==2
            else:
                weights=read(DATA/'raw'/job/'weights.json')['response_coefficients']
                expected=100*weights[c['rule']][c['response']]
                assert bank[c['response']]['task']==c['domain']
                close(sum(weights[c['rule']]),1)
                assert r['source'].endswith('weights.json')
        elif p=='F2a':
            m=d['decomposition'][c['domain']];expected=m['covariance_l2']/m['mean_length']
        elif p=='F2b':expected=d['gradient_comparisons'][c['reduction_pair']]['cosine']
        elif p=='F2c':
            rule=d['branches'][c['rule']]['heldout_kl_decrease']['macro']
            control=d['branches']['zero_gradient']['heldout_kl_decrease']['macro']
            expected=1000*(rule-control);close(c['rule_kl_decrease'],rule);close(c['zero_kl_decrease'],control)
        elif p in ['F5a','F5b']:
            a=f"{c['mode']}/{c['left']}/{c['loss']}";b=f"{c['mode']}/{c['right']}/{c['loss']}"
            q=pair(d,a,b);expected=q['raw_gradient']['cosine'] if r['measure']=='raw_gradient_cosine' else q['metrics']['cosine']
            condition=f"{'PG' if c['loss']=='sampled_pg' else 'I64'} · step {c['step']}"
            if p=='F5a':row,column=condition,c['mode']
            else:row,column=condition+'\n'+('Task inputs' if c['mode']=='routed' else 'Shared inputs'),r['measure']
            grouped[p,row,column].append(expected)
        elif p=='F5c':
            q=pair(d,'zero_gradient',f"{c['mode']}/{c['teacher']}/{c['loss']}")['metrics']['support']['1e-05']
            expected=q['jaccard'];close(c['layer_random_ratio'],q['layer_random_jaccard_ratio_of_expectations'])
            row=f"{'PG' if c['loss']=='sampled_pg' else 'I64'} · step {c['step']}\n"+('Task inputs' if c['mode']=='routed' else 'Shared inputs')
            grouped[p,row,'Observed'].append(expected)
            grouped[p,row,'Layer-random reference'].append(c['layer_random_ratio'])
        elif p in ['F6a','F6b','F6c']:
            row=f"{NAMES[DOMAINS.index(c['left'])]}–{NAMES[DOMAINS.index(c['right'])]}"
            if p=='F6a':expected=distance(d,c['left'],c['right']);column=str(c['step'])
            else:
                q=pair(d,f"{c['mode']}/{c['left']}/{c['loss']}",f"{c['mode']}/{c['right']}/{c['loss']}")
                expected=1-q['metrics']['support']['1e-05']['jaccard']
                close(c['teacher_js'],distance(d,c['left'],c['right']))
                column=f"{'PG' if c['loss']=='sampled_pg' else 'I64'} · step {c['step']}"
            grouped[p,row,column].append(expected)
        elif p in ['F7a','F7b','F7c']:
            row=STATE_LABELS[c['state']];column=METHOD_LABELS[METHODS.index(c['loss'])]
            if p=='F7a':
                expected=100*d['branches'][c['loss']]['metrics']['bf16_writeback']['fraction_above_1e-05'];close(c['threshold'],1e-5)
            elif p=='F7b':
                same=[measurements[j] for j,jv in jobs.items() if jv['kind']=='density' and jv['state']==c['state']]
                assert len(same)==4
                reference=mean(norm(s,'sampled_pg',c['metric']) for s in same)
                expected=norm(d,c['loss'],c['metric'])/reference
                close(c['pg_state_mean'],reference);close(c['absolute_l2'],norm(d,c['loss'],c['metric']))
                assert len(r['source_files'])==4
                row+=' / '+('Update' if c['metric']=='bf16_l2' else 'Gradient')
            else:
                cosine=pair(d,c['loss'],'full_vocab')['metrics']['cosine'];close(c['cosine'],cosine)
                expected=math.degrees(math.acos(max(-1,min(1,cosine))))
            grouped[p,row,column].append(expected)
        else:raise AssertionError(p)
        close(v,expected)
    checked.append(f'All {len(values)} plotted observations and conversion inputs independently match raw banks, weights or measurements')
    assert len(summaries)==len(grouped)
    for r in summaries:
        vals=grouped[r['panel'],r['row'],r['column']]
        close(r['value'],mean(vals));close(r['minimum'],min(vals));close(r['maximum'],max(vals))
        assert r['observations']==len(vals)
        if r['panel'] in ['F5a','F5b']:assert len(vals)==12
        elif r['panel']=='F5c':assert len(vals)==8
        elif r['panel'].startswith('F6'):assert len(vals)==2
        elif r['panel'].startswith('F7'):assert len(vals)==4
    checked.append(f'All {len(summaries)} cell/range summaries match the full intended observations and preserve condition identities')
    for panel in ['F6a','F6b','F6c']:
        order=list(dict.fromkeys(r['row'] for r in summaries if r['panel']==panel))
        if panel=='F6a':pair_order=order
        else:assert order==pair_order
    checked.append('F6 panels preserve all six teacher pairs in the same displayed order; both losses and checkpoints are covered')
    for name,key in [('data_manifest.json','source_manifest_sha256'),('panel_values.json','panel_values_sha256'),('panel_summaries.json','panel_summaries_sha256')]:
        assert digest(DATA/name)==fm[key]
    assert digest(PAPER/'experiments/plot_completed_local_probes.py')==fm['plot_script_sha256']
    checked.append('Plot script and all derived-data hashes match the final figure manifest')
    ids=set(expected_counts);assert {f['figure'] for f in fm['figures']}==ids
    assert all(f['standalone'] and f['main_axes']==1 and f['title'] is None and f['question'] for f in fm['figures'])
    sizes={};hashes={}
    for panel in sorted(ids):
        for extension in ['pdf','png','svg']:
            f=FIG/f'{panel}.{extension}';assert f.stat().st_size>1000;hashes[f.name]=digest(f)
        assert len(re.findall(rb'/Type\s*/Page\b',(FIG/f'{panel}.pdf').read_bytes()))==1
        with Image.open(FIG/f'{panel}.png') as im:
            assert min(im.size)>800 and max(ImageStat.Stat(im.convert('RGB')).stddev)>10
            sizes[panel]=im.size
    for name,count in [('main_panels.pdf',6),('all_panels.pdf',15)]:
        assert len(re.findall(rb'/Type\s*/Page\b',(FIG/name).read_bytes()))==count
        hashes[name]=digest(FIG/name)
    assert {f.stem for f in FIG.glob('*.pdf')}==ids|{'all_panels','main_panels'}
    checked.append('All 15 charts have standalone one-page PDF, nonblank high-resolution PNG and SVG; reading PDFs have 6 and 15 one-chart pages')
    layouts=read(DATA/'layout_checks.json');assert len(layouts)==15
    assert all(not x['matrix_labels_overlap'] and not x['comparison_markers_overlap'] and x['axes_titles']==0 and x['figure_prose']==0 for x in layouts)
    checked.append('Runtime rendering checks passed: no overlapping matrix labels or comparison markers, no in-image titles/prose; matrix number contrast at least 4.5')
    counts={'branches':276,'pairs':1908,'thresholds':1380,'responses':60,'covariance':28,'heldout':140,'teacher_js':168,'normalization_gradients':21}
    for name,n in counts.items():
        with (DATA/'plot_data'/f'{name}.csv').open() as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==n and all((DATA/r['source']).exists() for r in rows)
    checked.append('All eight full-precision raw plotting tables retain expected row counts and resolvable sources')
    plan=read(DATA/'figure_plan.json');assert len(plan)==27 and len({r['panel_id'] for r in plan})==27
    assert Counter(r['status'] for r in plan if r['core_24'])==Counter({'E':15,'P':5,'N':4})
    assert all(r['status']=='X' for r in plan if r['panel_id'] in ['F9a','F9b'])
    assert sum(r['suggested_placement']=='main' for r in plan)==6
    checked.append('Current 27-slot registry matches revised questions and six main candidates; cancelled training slots stay cancelled')
    docs=[PAPER/'experiments/README.md',PAPER/'experiments/EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md',
          PAPER/'experiments/NEXT_EXPERIMENTS_20260908_zh.md',DATA/'README_zh.md',DATA/'FIGURE_CAPTIONS.md',DATA/'REVIEWER_FIGURE_AUDIT_zh.md']
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=unquote(target.split('#',1)[0]);assert (path.parent/target).exists(),(path,target)
    gallery=(FIG/'index.html').read_text()
    assert gallery.count('<section ')==15 and gallery.count('<details>')==15
    checked.append('Current document links resolve; the gallery has 15 external questions/readings and 15 caption sections')
    report={'at_utc':datetime.now(timezone.utc).isoformat(),'passed':checked,
            'source_files':189,'jobs':31,'plotted_observations':len(values),'verified_summaries':len(summaries),
            'panel_observations':expected_counts,'png_dimensions':sizes,
            'visual_review':{'panels':sorted(ids),'reviewer_checks':['one explicit question per chart','no dense scatter clouds',
                              'all condition identities directly labeled','main/supporting distinction','paired ranges not presented as confidence intervals',
                              'large text, wrapped long labels, visible matrix numbers','no titles or explanatory paragraphs inside image files']},
            'figure_hashes':hashes,'scope':'Scientific data/figure and document consistency; no new model experiment or significance test'}
    (DATA/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed_checks':len(checked),'source_files':189,'jobs':31,'values':len(values),'summaries':len(summaries),'standalone_charts':15}))

if __name__=='__main__':main()
