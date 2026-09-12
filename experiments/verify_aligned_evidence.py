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
    expected=set()
    for source in manifest['sources']:
        match=re.fullmatch(r'raw/(Initial|M-PG|M-I64-DR|M-I64-DT|M-I64-GT|S-PG|S-I64)_(\d+)_complete.json',source['frozen'])
        if match:
            marker=read(source['frozen'])
            check('capability_completion_marker',marker['status']=='complete' and marker['final_num_updates']==int(match[2]),source['frozen'])
            expected.add((match[1],int(match[2])))
    remote=read('raw/single_capability.json')['runs']
    expected|={(r['model'],r['step']) for r in remote}
    check('complete_capability_grid',len(cap)==len(expected) and {(r['model'],r['step']) for r in cap}==expected)
    counts={'Math':500,'Code':128,'IF':300,'GPQA':792}
    check('complete_suite_response_counts',all(r['response_counts']==counts for r in cap))
    total_responses=sum(sum(r['response_counts'].values()) for r in cap)
    check('total_evaluated_responses',total_responses==len(expected)*sum(counts.values()),total_responses)
    rows=[json.loads(l) for l in (DATA/'prompt_scores.jsonl').read_text().splitlines()]
    local=[r for r in cap if r['verification']=='raw_artifacts']
    check('local_response_count',len(rows)==sum(sum(r['response_counts'].values()) for r in local),len(rows))
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['step'],r['domain'])].append(r)
    for r in cap:
        if r['verification']!='raw_artifacts':continue
        for d,score in r['scores'].items():
            rr=groups[(r['model'],r['step'],d)]
            check('response_mean_matches_score',abs(np.mean([x['reward'] for x in rr])-score)<1e-12,[r['model'],r['step'],d])
            check('unique_prompt_sample',len({(x['prompt_index'],x['sample']) for x in rr})==len(rr))
    audit=read('gpqa_score_audit.json')
    check('gpqa_rescore',audit=={'scorer':'final-answer-v2','responses':sum(r['domain']=='GPQA' for r in rows),'disagreements':[]})
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
    raw_ids={(r['model'],r['step']) for r in local}
    expected_pairs={(('Initial',0),identity) for identity in raw_ids if identity!=('Initial',0)}
    expected_pairs|={(('M-I64-DR',50),(m,50)) for m in ['M-I64-DT','M-I64-GT']}
    for left,right in [('S-PG','S-I64'),('M-PG','M-I64-DR')]:
        common={step for model,step in raw_ids if model==left}&{step for model,step in raw_ids if model==right}
        expected_pairs|={((left,step),(right,step)) for step in common}
    expected_contrasts={(left,right,domain) for left,right in expected_pairs for domain in counts}
    check('paired_comparison_coverage',len(comp)==len(expected_contrasts) and
          {((r['left'],r['left_step']),(r['right'],r['right_step']),r['domain']) for r in comp}==expected_contrasts,len(comp))
    for r in comp:
        left=groups[(r['left'],r['left_step'],r['domain'])];right=groups[(r['right'],r['right_step'],r['domain'])]
        identity=lambda rr:{(x['prompt_index'],x['identity']) for x in rr}
        check('paired_question_identity',identity(left)==identity(right))
        check('paired_effect_equals_score_difference',abs(100*(np.mean([x['reward'] for x in right])-np.mean([x['reward'] for x in left]))-r['delta_pp'])<1e-10)
    endpoint_rows={}
    for setting,left,right in [('Single','S-PG','S-I64'),('Joint','M-PG','M-I64-DR')]:
        comparisons=[r for r in comp if r['left']==left and r['right']==right and r['left_step']==r['right_step']]
        endpoint=max(r['left_step'] for r in comparisons)
        for row in comparisons:
            if row['left_step']==endpoint:endpoint_rows[setting,endpoint,row['domain']]=row
    endpoint_seen=[]
    for line in (DATA/'endpoint_paired_table.tex').read_text().splitlines():
        if not line.startswith(('Single &','Joint &')):continue
        setting,step,domain,effect,interval=[x.strip() for x in line.rstrip('\\').split('&')]
        identity=(setting,int(step),domain);endpoint_seen.append(identity);row=endpoint_rows[identity]
        limits=[float(x) for x in interval.strip('[] ').split(',')]
        check('endpoint_paired_table_values_match_full_precision_source',
              [float(effect)]+limits==[float(f'{row[key]:.2f}') for key in ['delta_pp','lo_pp','hi_pp']],identity)
    check('endpoint_paired_table_complete',len(endpoint_seen)==len(endpoint_rows) and set(endpoint_seen)==set(endpoint_rows))
    fixed=read('normalization_fixed_batch_20260912.json')['records']
    seeds=sorted({r['bank_seed'] for r in fixed})
    fixed_rows={(r['step'],seeds.index(r['bank_seed'])+1):r for r in fixed}
    check('fixed_batch_comparison_grid_complete',set(fixed_rows)=={(step,bank) for step in [100,250] for bank in [1,2,3]})
    for row in fixed:
        raw=read(row['source'])
        check('fixed_batch_export_matches_frozen_measurement',raw['status']=='complete' and
              raw['snapshot_step']==row['step'] and raw['gradient_comparisons']==row['gradient_comparisons'],row['source'])
    fixed_seen=[]
    for line in (DATA/'normalization_fixed_batch_table.tex').read_text().splitlines():
        if not re.match(r'^\d+\s*&',line):continue
        fields=[x.strip() for x in line.rstrip('\\').split('&')]
        identity=(int(fields[0]),int(fields[1]));fixed_seen.append(identity);row=fixed_rows[identity]
        check('fixed_batch_table_values_match_full_precision_source',
              [float(x) for x in fields[2:]]==[float(f"{row['gradient_comparisons'][key]['cosine']:.4f}") for key in ['DR_DT','DR_GT','DT_GT']],identity)
    check('fixed_batch_table_complete',len(fixed_seen)==len(fixed_rows) and set(fixed_seen)==set(fixed_rows))
    measured=[json.loads(l) for l in (DATA/'raw/measurements.jsonl').read_text().splitlines()]
    count=Counter(r['kind'] for r in measured)
    check('mechanism_record_counts',len(measured)==433 and count['crossed_pair']==192 and count['teacher_pair']==48 and count['teacher_js']==12,dict(count))
    check('numerical_controls',read('raw/numerical_validation.json')['status']=='passed')
    summary=read('raw/results_summary.json')
    geo=read('raw/online_context.json')['geometry']
    extra_geo=read('online_geometry_20260912.json')
    for source in extra_geo['sources']:
        model=Path(source['frozen']).name.split('_paper_')[0]
        raw=[json.loads(line) for line in (DATA/source['frozen']).read_text().splitlines()]
        for row in extra_geo['geometry']:
            if row['model']==model:
                check('additional_geometry_matches_frozen_log',
                      {k:v for k,v in row.items() if k!='model'} in raw,[model,row['step'],row['quantity']])
    geo+=extra_geo['geometry']
    geometry_keys=[(r['model'],r['step'],r['quantity']) for r in geo]
    check('geometry_coordinates_not_duplicated',len(geometry_keys)==len(set(geometry_keys)))
    density=next(r for r in read('completed_local_followup_20260912.json')['experiments']
                 if r['experiment']=='density-mpg100-long-bank1042')
    density_raw=[json.loads(line) for line in (DATA/density['source']).read_text().splitlines()]
    check('density_export_matches_frozen_optimizer_records',density['optimizer']==[r for r in density_raw if r['kind']=='optimizer'
          and r['mode']=='history' and r['loss'] in ['sampled_pg','topk_intersection','teacher_topk','full_vocab']
          and r['quantity'] in ['master','bf16']])
    density_grid=[r for r in density['optimizer'] if r['mode']=='history' and r['loss'] in
                  ['sampled_pg','topk_intersection','teacher_topk','full_vocab'] and r['quantity'] in ['master','bf16']]
    expected_density={(bank,loss,quantity) for bank in range(4) for loss in
                      ['sampled_pg','topk_intersection','teacher_topk','full_vocab'] for quantity in ['master','bf16']}
    check('complete_four_bank_density_comparison',len(density_grid)==32 and
          {(r['bank'],r['loss'],r['quantity']) for r in density_grid}==expected_density)
    check('density_run_completed',read(str(Path(density['source']).parent/'run_complete.json'))['status']=='complete')
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
    figures={r['id']:r for r in fig_manifest['figures']}
    for name,model_groups in [('normalization_capability',[['M-I64-DR','M-I64-DT','M-I64-GT']]),
                              ('capability',[['S-PG','S-I64'],['M-PG','M-I64-DR']])]:
        figure=figures[name]
        for model_group in model_groups:
            common=set.intersection(*({r['step'] for r in cap if r['model']==m} for m in model_group))
            check('capability_figure_has_only_complete_matched_checkpoints',
                  all(figure['evaluated_steps'][m]==sorted(common) for m in model_group),[name,model_group,sorted(common)])
    common_geometry=set.intersection(*({r['step'] for r in geo if r['model']==m and r['quantity']==q}
                                      for m in ['M-PG','M-I64-DR'] for q in ['delta_fp32','delta_bf16']))
    check('geometry_figure_has_complete_matched_checkpoints',
          all(figures['cumulative_geometry']['evaluated_steps'][m]==sorted(common_geometry) for m in ['M-PG','M-I64-DR'])
          and figures['cumulative_geometry']['endpoint_step']==max(common_geometry))
    check('density_figure_uses_complete_four_bank_experiment',figures['supervision_density']['experiment']==density['experiment']
          and figures['supervision_density']['banks']==list(range(4)))
    check('token_share_figure_uses_common_interval',figures['all_token_shares']['rollout_indices']==list(range(50)))
    paired_table_ids={('Initial',0)}
    density_table_ids=set()
    for model_group in [['M-I64-DR','M-I64-DT','M-I64-GT'],['S-PG','S-I64'],['M-PG','M-I64-DR']]:
        common=set.intersection(*({r['step'] for r in cap if r['model']==m} for m in model_group))
        identities={(model,step) for model in model_group for step in common}
        paired_table_ids|=identities
        if len(model_group)==2:density_table_ids|=identities
    for name,expected_ids in [('capability_table.tex',paired_table_ids),('density_capability_table.tex',density_table_ids)]:
        actual=[]
        for line in (DATA/name).read_text().splitlines():
            fields=line.split(' & ')
            if fields[0] in {r['model'] for r in cap}:actual.append((fields[0],int(fields[1])))
        check('capability_table_has_only_complete_matched_comparisons',len(actual)==len(expected_ids) and set(actual)==expected_ids,name)
    figure_details=[]
    required_labels={'normalization50':('Difference (pp)',4),'cumulative_geometry':('Nonzero (%)',2),
                     'supervision_density':('Nonzero (%)',2)}
    for r in fig_manifest['figures']:
        check('three_column_figure_grid',r['columns']==3 and r['rows']==(r['panels']+2)//3,r['id'])
        check('figure_labels_fit_canvas_and_do_not_overlap',r['text_within_canvas'] and not r['overlapping_tick_labels'],r['id'])
        bounds=r['panel_bounds']
        check('panels_fit_three_column_width',len(bounds)==r['panels'] and all(0<w<=1/3 for x,y,w,h in bounds),r['id'])
        check('panel_axes_do_not_overlap',all(x+w<=a+1e-4 or a+c<=x+1e-4 or y+h<=b+1e-4 or b+d<=y+1e-4
              for i,(x,y,w,h) in enumerate(bounds) for a,b,c,d in bounds[i+1:]),r['id'])
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
    notes=[]
    for start in re.finditer(r'\\completionnote\{',tex):
        cursor=start.end()-1;arguments=[]
        for argument in range(2):
            while cursor<len(tex) and tex[cursor].isspace():cursor+=1
            check('completion_note_has_two_arguments',cursor<len(tex) and tex[cursor]=='{')
            position=cursor+1;level=1
            for end in range(position,len(tex)):
                if tex[end]=='{' and tex[end-1]!='\\':level+=1
                if tex[end]=='}' and tex[end-1]!='\\':level-=1
                if level==0:break
            check('completion_note_braces_balanced',level==0)
            arguments.append(tex[position:end]);cursor=end+1
        notes.append('\n'.join(arguments))
    check('incomplete_experiments_have_annotations',len(notes)>=3,len(notes))
    check('completion_notes_do_not_embed_partial_figures_or_tables',all(not re.search(r'\\includegraphics|\\input|\\begin\{(?:tabular|table|figure)',note) for note in notes))
    check('obsolete_numeric_claims_removed',not any(x in tex for x in ['5.90','9.92','55.6','97.87','GPAS']))
    log=(ROOT/'iclr2027_conference.log').read_text()
    check('latex_references_and_layout',not re.search(r'undefined|multiply defined|Overfull|LaTeX Error',log))
    pdf=pymupdf.open(ROOT/'iclr2027_conference.pdf');body='\n'.join(p.get_text() for p in pdf)
    figure_count=len(re.findall(r'\\begin\{figure\*?\}',tex))
    table_count=len(re.findall(r'\\begin\{table\*?\}',tex))
    check('compiled_figures_present',all(f'Figure {i}:' in body for i in range(1,figure_count+1)))
    check('compiled_tables_present',all(f'Table {i}:' in body for i in range(1,table_count+1)))
    environment={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'pymupdf':pymupdf.__version__}
    (DATA/'plotting_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    report={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':len(checks),
            'frozen_sources':frozen,'capability_suites':len(cap),'capability_responses':total_responses,'local_response_records':len(rows),
            'gpqa_rescored':audit['responses'],'teacher_gpqa_rescored':792,'paired_contrasts':len(comp),'mechanism_records':len(measured),
            'rollout_clocks':clocks,'figures':figure_details,'compiled_pdf_pages':len(pdf),
            'pdf_sha256':hashlib.sha256((ROOT/'iclr2027_conference.pdf').read_bytes()).hexdigest(),
            'environment':environment,'checks':checks}
    (DATA/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','figures']},indent=2))


if __name__=='__main__':main()
