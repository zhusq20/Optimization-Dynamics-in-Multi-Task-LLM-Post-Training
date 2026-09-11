#!/usr/bin/env python3
"""Audit plotted values against the frozen source records and local probe formulas."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DATA = HERE / 'paper_curves_20260908'
FIGURES = HERE.parent / 'figures/paper_curves_20260908'


def main():
    with gzip.open(DATA / 'data.json.gz', 'rt') as stream:
        data = json.load(stream)
    caches = {}
    def original(source, line):
        if source not in caches:
            with gzip.open(DATA / source, 'rt') as stream:
                caches[source] = [json.loads(s) for s in stream if s.strip()]
        return caches[source][line-1]

    checked = 0
    with (DATA / 'raw_curve_values.csv').open() as stream:
        for row in csv.DictReader(stream):
            event = original(row['source'], int(row['source_line']))
            assert str(event['metrics'][row['metric']]) == row['value'], row
            if row['clock'] != 'checkpoint_step':
                assert float(event['metrics'][row['clock']]) == float(row['step']), row
            checked += 1
    plotted = json.loads((DATA / 'plotted_values.json').read_text())
    local, static_raw = 0, 0
    for row in plotted:
        if isinstance(row['source'], dict):
            src = row['source']
            event = original(src['file'], src['line'])
            observed = event['metrics'][row['metric']]
            assert observed == row['raw_value'], row
            assert math.isclose(observed * row['scale'], row['y'], rel_tol=1e-12, abs_tol=1e-12), row
            if row['clock'] != 'checkpoint_step':
                assert event['metrics'][row['clock']] == row['x'], row
            static_raw += 1
            continue
        source = DATA / row['source']
        evidence = json.loads(source.read_text())
        metric = row['metric']
        if metric == 'raw_gradient_cosine' and 'reduction_pair' in row:
            observed = evidence['gradient_comparisons'][row['reduction_pair']]['cosine']
        elif metric in {'raw_gradient_cosine', 'gradient_cosine', 'bf16_cosine', 'jaccard'}:
            pair = next(p for p in evidence['pairwise_bf16_writebacks'] if p['left'] == row['left'] and p['right'] == row['right'])
            if metric in {'raw_gradient_cosine', 'gradient_cosine'}:
                observed = pair['raw_gradient']['cosine']
            elif metric == 'bf16_cosine':
                observed = pair['metrics']['cosine']
            else:
                observed = pair['metrics']['support'][str(float(row['threshold']))]['jaccard']
        elif metric == 'bf16_changed_percent':
            threshold = '0' if float(row['threshold']) == 0 else str(float(row['threshold']))
            observed = 100 * evidence['branches'][row['branch']]['metrics']['bf16_writeback']['fraction_above_'+threshold]
        elif metric.endswith('_over_pg_mean'):
            branch = evidence['branches'][row['branch']]['metrics']
            absolute = branch['gradient_l2_before_clipping'] if metric.startswith('gradient') else branch['bf16_writeback']['l2']
            assert absolute == row['raw_value'], row
            observed = absolute / row['denominator']
        else:
            raise ValueError('Unvalidated transform: '+metric)
        assert math.isclose(observed, row['y'], rel_tol=1e-12, abs_tol=1e-12), row
        local += 1
    # Check that evaluation attempts are selected by chronology, not maximum score.
    first = {}
    for row in sorted(data['evaluations'], key=lambda r: (r['completed_at_utc'], r['attempt'])):
        assert row['verified'] and all(row['checks'].values())
        key = (row['family'], row['step'], row['dataset'])
        assert row['selected'] == (key not in first)
        first.setdefault(key, row)
        run = data['runs'].get(row['family'])
        if run and run['first_intersection_update'] and row['step'] < run['first_intersection_update']:
            assert row['training_phase'] == 'student_topk'
    manifest = json.loads((DATA / 'figure_manifest.json').read_text())
    for row in manifest['figures']:
        for suffix in ['pdf', 'png', 'svg']:
            assert (FIGURES / f"{row['id']}.{suffix}").stat().st_size > 100
    assert len(re.findall(rb'/Type\s*/Page\b', (FIGURES / 'main_figures.pdf').read_bytes())) == 9
    assert len(re.findall(rb'/Type\s*/Page\b', (FIGURES / 'all_figures.pdf').read_bytes())) == 30
    assert manifest['counts'] == {'line': 25, 'point': 4, 'bar': 0, 'heatmap': 1}
    assert sum(r['type'] == 'line' for r in manifest['figures'] if r['tier'] == 'main') == 7
    assert manifest['script_sha256'] == hashlib.sha256((HERE / 'plot_paper_figures.py').read_bytes()).hexdigest()
    layout = json.loads((DATA / 'layout_checks.json').read_text())
    assert all(r['inside_export'] for r in layout)
    report = {'status': 'passed', 'raw_curve_values_checked': checked,
              'static_raw_values_checked': static_raw, 'local_probe_values_checked': local,
              'evaluation_attempts_verified': len(data['evaluations']), 'selected_eval_points': len(first),
              'exported_figures': 30, 'main_pdf_pages': 9, 'all_pdf_pages': 30,
              'labels_inside_export_checked': len(layout), 'method_transitions_preserved': True,
              'browser_validation': json.loads((DATA / 'browser_validation.json').read_text())
                                    if (DATA / 'browser_validation.json').exists() else 'not run by this validator'}
    (DATA / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
