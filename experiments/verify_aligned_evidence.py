"""Audit the portable evidence and the compiled manuscript without source GPUs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

import matplotlib
import numpy as np
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments/aligned_evidence_20260910'
FIG=ROOT/'figures/aligned_evidence_20260910'
checks=[]


def check(name,condition,details=None):
    if not condition:raise AssertionError((name,details))
    checks.append({'check':name,'passed':True,'details':details})


def read(name):return json.loads((DATA/name).read_text())


def main():
    manifest=read('manifest.json');frozen=0
    for r in manifest['sources']:
        if r.get('note'):continue
        check('frozen_source_hash',hashlib.sha256((DATA/r['frozen']).read_bytes()).hexdigest()==r['sha256'],r['frozen'])
        frozen+=1
    cap=read('capability.json')
    expected={('Initial',0),('M-PG',50),('M-PG',100),('M-I64-DR',50),('M-I64-DR',100),
              ('M-I64-DT',50),('M-I64-GT',50),('M-I64-GT',100),
              ('S-PG',100),('S-PG',250),('S-PG',500),('S-I64',100),('S-I64',250)}
    check('complete_capability_grid',len(cap)==13 and {(r['model'],r['step']) for r in cap}==expected)
    check('total_evaluated_responses',sum(sum(r['response_counts'].values()) for r in cap)==22360)
    rows=[json.loads(l) for l in (DATA/'prompt_scores.jsonl').read_text().splitlines()]
    check('local_response_count',len(rows)==15480)
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['step'],r['domain'])].append(r)
    for r in cap:
        if r['verification']!='raw_artifacts':continue
        for d,score in r['scores'].items():
            rr=groups[(r['model'],r['step'],d)]
            check('response_mean_matches_score',abs(np.mean([x['reward'] for x in rr])-score)<1e-12,[r['model'],r['step'],d])
            check('unique_prompt_sample',len({(x['prompt_index'],x['sample']) for x in rr})==len(rr))
    audit=read('gpqa_score_audit.json')
    check('gpqa_rescore',audit=={'scorer':'final-answer-v2','responses':7128,'disagreements':[]})
    teachers=read('teacher_references.json')
    check('teacher_reference_domains',len(teachers)==4 and all('base' not in r['teacher'].lower() for r in teachers))
    check('science_teacher_rescore',next(r for r in teachers if r['teacher']=='teacher_science')['current_scorer_disagreements']==0)
    endpoint=next(r for r in cap if r['model']=='S-PG' and r['step']==500)
    check('endpoint_raw_evidence',endpoint['verification']=='raw_artifacts')
    conversion=read('raw/S-PG_500_export_verified.json')
    check('endpoint_model_conversion',conversion['step']==500 and conversion['tensor_keys']==310
          and conversion['exact_original_key_set'] and conversion['all_finite']
          and conversion['serialized_tensors_equal_converted_native'])
    single=read('raw/single_protocol.json');shared=read('raw/protocol.json')
    check('single_protocol_alignment',all(single[k]==shared[k] for k in
          ['initialization','student','teachers','prompt_format','response_semantics','evaluation','datasets']))
    comp=read('paired_comparisons.json')
    check('paired_comparison_count',len(comp)==40)
    for r in comp:
        left=groups[(r['left'],r['left_step'],r['domain'])];right=groups[(r['right'],r['right_step'],r['domain'])]
        identity=lambda rr:{(x['prompt_index'],x['identity']) for x in rr}
        check('paired_question_identity',identity(left)==identity(right))
        check('paired_effect_equals_score_difference',abs(100*(np.mean([x['reward'] for x in right])-np.mean([x['reward'] for x in left]))-r['delta_pp'])<1e-10)
    measured=[json.loads(l) for l in (DATA/'raw/measurements.jsonl').read_text().splitlines()]
    count=Counter(r['kind'] for r in measured)
    check('mechanism_record_counts',len(measured)==433 and count['crossed_pair']==192 and count['teacher_pair']==48 and count['teacher_js']==12,dict(count))
    check('numerical_controls',read('raw/numerical_validation.json')['status']=='passed')
    summary=read('raw/results_summary.json')
    geo=read('raw/online_context.json')['geometry']
    at=lambda q:next(r['metrics'] for r in geo if r['model']=='M-PG' and r['step']==100 and r['quantity']==q)
    check('headline_fp32_activity',round(100*(1-at('delta_fp32')['sparsity_at_0']),1)==94.0)
    check('headline_bf16_activity',round(100*(1-at('delta_bf16')['sparsity_at_0']),2)==3.41)
    check('unique_coordinate_count',at('delta_fp32')['parameters']==1720574976)
    check('local_probability_coverage',len(summary['coverage'])==8 and min(r['student_mass_in_teacher_top64'] for r in summary['coverage'])>.99918)
    clocks={}
    for m,n in [('M-PG',128),('M-I64-DR',100),('M-I64-DT',100),('M-I64-GT',115)]:
        rec=defaultdict(dict)
        for line in (DATA/f'raw/{m}_rollout.jsonl').read_text().splitlines():
            r=json.loads(line)['metrics']
            if 'rollout/step' in r:rec[int(r['rollout/step'])].update(r)
        check('rollout_clock_coverage',set(rec)==set(range(n)),[m,len(rec)])
        for r in rec.values():
            shares=[r[f'mopd/task/{d}/token_share'] for d in ['math','code','if','science']]
            check('token_share_denominator',abs(sum(shares)-1)<1e-10 and all(0<=x<=.8 for x in shares))
        clocks[m]=len(rec)
    gt=defaultdict(dict)
    for line in (DATA/'raw/M-I64-GT_rollout.jsonl').read_text().splitlines():
        r=json.loads(line)['metrics']
        if 'rollout/step' in r:gt[int(r['rollout/step'])].update(r)
    check('body_first50_token_shares',
          [round(100*np.mean([gt[i][f'mopd/task/{d}/token_share'] for i in range(50)]),2)
           for d in ['math','if']]==[44.05,14.98])
    fig_manifest=read('figure_manifest.json')
    check('plot_script_hash',hashlib.sha256((ROOT/'experiments/plot_aligned_evidence.py').read_bytes()).hexdigest()==fig_manifest['script_sha256'])
    check('eleven_figures',len(fig_manifest['figures'])==11)
    figure_details=[]
    required_labels={'normalization50':('Difference (pp)',4),'cumulative_geometry':('Nonzero (%)',2),
                     'supervision_density':('Nonzero (%)',2)}
    for r in fig_manifest['figures']:
        for src in r['sources']:check('figure_source_exists',(DATA/src).is_file(),src)
        for ext in ['pdf','png','svg']:check('figure_format_exists',(FIG/(r['id']+'.'+ext)).is_file(),[r['id'],ext])
        doc=pymupdf.open(FIG/(r['id']+'.pdf'));page=doc[0];text=page.get_text()
        check('figure_has_content',len(doc)==1 and len(text)>70 and len(page.get_drawings())>12,r['id'])
        if r['id'] in required_labels:
            label,n=required_labels[r['id']];check('figure_axis_labels_fit',text.count(label)==n,[r['id'],label,text.count(label)])
        figure_details.append({'figure':r['id'],'width_pt':page.rect.width,'height_pt':page.rect.height,'text_characters':len(text)})
    root_tex=(ROOT/'iclr2027_conference.tex').read_text()
    section_names=re.findall(r'\\input\{(sections/[^}]+)\}',root_tex)
    tex='\n'.join((ROOT/(name+'.tex')).read_text() for name in section_names)
    labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\ref\{([^}]+)\}',tex)
    check('all_tex_references_defined',set(refs)<=set(labels),sorted(set(refs)-set(labels)))
    check('unique_tex_labels',len(labels)==len(set(labels)))
    figure_paths=re.findall(r'\\includegraphics\[[^]]+\]\{([^}]+)\}',tex)
    check('only_aligned_figures_in_manuscript',len(figure_paths)==11 and all(p.startswith('figures/aligned_evidence_20260910/') for p in figure_paths))
    check('figure_manifest_matches_manuscript',
          {Path(p).stem for p in figure_paths}=={r['id'] for r in fig_manifest['figures']})
    check('no_todo_placeholders',not re.search(r'\\missing|TODO|to be measured|remain to be measured',tex))
    check('obsolete_numeric_claims_removed',not any(x in tex for x in ['5.90','9.92','55.6','97.87','GPAS']))
    log=(ROOT/'iclr2027_conference.log').read_text()
    check('latex_references_and_layout',not re.search(r'undefined|multiply defined|Overfull|LaTeX Error',log))
    pdf=pymupdf.open(ROOT/'iclr2027_conference.pdf');body='\n'.join(p.get_text() for p in pdf)
    check('compiled_figures_present',all(f'Figure {i}:' in body for i in range(1,12)))
    check('compiled_tables_present',all(t in body for t in [f'Table {i}:' for i in range(1,8)]))
    environment={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'pymupdf':pymupdf.__version__}
    (DATA/'plotting_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    report={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':len(checks),
            'frozen_sources':frozen,'capability_suites':len(cap),'capability_responses':22360,'local_response_records':len(rows),
            'gpqa_rescored':7128,'teacher_gpqa_rescored':792,'paired_contrasts':len(comp),'mechanism_records':len(measured),
            'rollout_clocks':clocks,'figures':figure_details,'compiled_pdf_pages':len(pdf),
            'pdf_sha256':hashlib.sha256((ROOT/'iclr2027_conference.pdf').read_bytes()).hexdigest(),
            'environment':environment,'checks':checks}
    (DATA/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','figures']},indent=2))


if __name__=='__main__':main()
