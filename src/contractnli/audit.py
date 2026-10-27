"""Read-only paired evaluation of saved predictions, including legacy runs."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .data import LABELS
from .metrics import classification, paired_metrics
from .prompts import parse_first_line, parse_label
from .storage import digest, write_json


def audit_predictions(path, metrics_path=None, *, bootstrap_samples=20000, seed=20261027, overlap_tolerance=.02):
    import hashlib
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Install analysis dependencies: pip install -e ".[analysis]"') from exc

    if bootstrap_samples < 100 or not 0 <= overlap_tolerance < 1:
        raise ValueError("Use at least 100 bootstrap samples and a valid overlap tolerance")
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or len({r['key'] for r in rows}) != len(rows):
        raise ValueError("Empty predictions or duplicate keys")
    methods = sorted({r['condition'] for r in rows})
    if set(methods) - {'original', 'naive', 'contextualized', 'longllmlingua'} or 'original' not in methods:
        raise ValueError("Audit requires original and known comparison conditions")
    methods = ['original'] + [m for m in methods if m != 'original']
    grouped = {m: sorted((r for r in rows if r['condition'] == m), key=lambda r: r['example_id']) for m in methods}
    base = grouped['original']
    if len({r['example_id'] for r in base}) != len(base):
        raise ValueError("Duplicate example IDs")
    for m, rs in grouped.items():
        if [r['example_id'] for r in rs] != [r['example_id'] for r in base]:
            raise ValueError("Conditions contain different example IDs")
        for a, r in zip(base, rs):
            if any(a[k] != r[k] for k in ('gold', 'document_id', 'hypothesis_id', 'hypothesis', 'original_tokens')):
                raise ValueError("Paired example metadata disagree")
            if r['key'] != f"{m}/{r['example_id']}" or r['gold'] not in LABELS:
                raise ValueError("Invalid key or gold label")
            if r['prediction'] != parse_label(r['generation']['text']):
                raise ValueError("Stored prediction disagrees with strict parser")
            if r['original_tokens'] <= 0 or r['context_tokens'] <= 0 or not math.isfinite(r['generation']['seconds']) or r['generation']['seconds'] < 0:
                raise ValueError("Invalid token counts or latency")
    docs = sorted({r['document_id'] for r in base})
    doc_index = {d: i for i, d in enumerate(docs)}
    weights = np.random.default_rng(seed).multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=bootstrap_samples)
    report = {'schema_version': 1, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'documents': len(docs), 'pairs_per_condition': len(base),
              'bootstrap': {'unit': 'paired document clusters including all hypotheses', 'samples': bootstrap_samples,
                            'seed': seed, 'interval': 'percentile 95%, exploratory, unadjusted'},
              'analyses': {}, 'format_diagnostics': {}, 'hypotheses': [], 'length_overlap_diagnostic': [],
              'notes': ['Original files and timings are never modified.',
                        'First-line parsing is a secondary sensitivity analysis, not a replacement for the primary protocol.',
                        'Length-overlap selection is gold-blind but changes the evaluated sample; it is not a full length-matched experiment.',
                        'Document bootstrap does not measure variability across model seeds or hypotheses.',
                        'This test split has already been inspected; subsequent adjustments are exploratory. Use dev for further tuning.']}
    source_metrics = json.loads(Path(metrics_path).read_text(encoding='utf-8-sig')) if metrics_path else None
    if source_metrics:
        report['source_metrics_sha256'] = hashlib.sha256(Path(metrics_path).read_bytes()).hexdigest()
        if set(source_metrics['metrics']) != set(methods):
            raise ValueError("Source metrics conditions disagree with predictions")
    for mode, parser in [('strict', parse_label), ('first_line_sensitivity', parse_first_line)]:
        evaluated = {m: [{**r, 'prediction': parser(r['generation']['text'])} for r in rs] for m, rs in grouped.items()}
        stats, bootstrap = {}, {}
        for m, rs in evaluated.items():
            stats[m] = classification(rs)
            stats[m]['correct_n'] = sum(r['prediction'] == r['gold'] for r in rs)
            if mode == 'strict' and source_metrics:
                old = source_metrics['metrics'][m]
                if any(not math.isclose(stats[m][k], old[k], rel_tol=0, abs_tol=1e-12) for k in ('accuracy', 'macro_f1')):
                    raise ValueError(f"Source metrics do not reproduce for {m}")
                if any(stats[m][k] != old[k] for k in ('n', 'invalid_outputs', 'confusion')):
                    raise ValueError(f"Source metrics counts do not reproduce for {m}")
            if m != 'original':
                stats[m]['vs_original'] = paired_metrics(evaluated['original'], rs)
            cm = np.zeros((len(docs), 3, 4), dtype=np.int64)
            for r in rs:
                cm[doc_index[r['document_id']], LABELS.index(r['gold']), LABELS.index(r['prediction']) if r['prediction'] in LABELS else 3] += 1
            sampled = (weights @ cm.reshape(len(docs), -1)).reshape(-1, 3, 4)
            tp = np.diagonal(sampled[:, :, :3], axis1=1, axis2=2)
            denominator = sampled.sum(2) + sampled[:, :, :3].sum(1)
            f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
            bootstrap[m] = {'accuracy': tp.sum(1) / sampled.sum((1, 2)), 'macro_f1': f1.mean(1)}
        contrasts = {}
        for i, a in enumerate(methods):
            for b in methods[:i]:
                contrasts[f'{a}_minus_{b}'] = {k: {'difference': stats[a][k] - stats[b][k],
                    'ci95': np.quantile(bootstrap[a][k] - bootstrap[b][k], [.025, .975]).tolist()}
                    for k in ('accuracy', 'macro_f1')}
        report['analyses'][mode] = {'conditions': stats, 'contrasts': contrasts}
    for m, rs in grouped.items():
        bad = [r for r in rs if r['prediction'] is None]
        report['format_diagnostics'][m] = {'invalid_n': len(bad), 'invalid_rate': len(bad) / len(rs),
            'finish_reasons': dict(Counter(r['generation']['finish_reason'] for r in bad)),
            'first_line_recoverable_n': sum(parse_first_line(r['generation']['text']) is not None for r in bad),
            'first_line_correct_n': sum(parse_first_line(r['generation']['text']) == r['gold'] for r in bad),
            'examples': [{'example_id': r['example_id'], 'gold': r['gold'], 'raw_output': r['generation']['text']} for r in bad[:5]]}
    for h in sorted({r['hypothesis_id'] for r in base}):
        subset = {m: [r for r in rs if r['hypothesis_id'] == h] for m, rs in grouped.items()}
        report['hypotheses'].append({'id': h, 'hypothesis': subset['original'][0]['hypothesis'],
            'gold_counts': dict(Counter(r['gold'] for r in subset['original'])),
            'strict': {m: classification(rs) for m, rs in subset.items()},
            'first_line_sensitivity': {m: classification([{**r, 'prediction': parse_first_line(r['generation']['text'])} for r in rs])
                                       for m, rs in subset.items()}})
    compressed = [m for m in methods if m != 'original']
    for i, a in enumerate(compressed):
        for b in compressed[:i]:
            idx = [j for j, (ra, rb) in enumerate(zip(grouped[a], grouped[b]))
                   if abs(ra['context_tokens'] / ra['original_tokens'] - rb['context_tokens'] / rb['original_tokens']) <= overlap_tolerance]
            report['length_overlap_diagnostic'].append({'a': a, 'b': b, 'ratio_tolerance': overlap_tolerance,
                'n': len(idx), 'coverage': len(idx) / len(base),
                'documents': len({base[j]['document_id'] for j in idx}),
                'gold_counts': dict(Counter(base[j]['gold'] for j in idx)),
                'strict_metrics': {m: classification([grouped[m][j] for j in idx]) for m in (a, b)} if idx else None,
                'warning': 'Selected subset only. Does not control content, selection bias, or all documents.'})
    report['audit_fingerprint'] = digest(report)
    return report


def save_audit(predictions_path, metrics_path, output_dir, **kwargs):
    directory = Path(output_dir)
    source_dirs = {Path(p).resolve().parent for p in (predictions_path, metrics_path) if p}
    if directory.resolve() in source_dirs:
        raise ValueError("Choose a separate audit output directory; source run is read-only")
    result = audit_predictions(predictions_path, metrics_path, **kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'audit.json', result)
    import csv
    with (directory / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['evaluation', 'condition', 'accuracy', 'macro_f1', 'invalid_outputs', 'correct_n'])
        for mode, analysis in result['analyses'].items():
            for m, stat in analysis['conditions'].items():
                writer.writerow([mode, m, *[stat[k] for k in ('accuracy', 'macro_f1', 'invalid_outputs', 'correct_n')]])
    return result
