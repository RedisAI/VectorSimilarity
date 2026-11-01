"""Summarize paired end-to-end timings without hiding slower groups or recall loss."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(root):
    groups = defaultdict(lambda: defaultdict(list))
    total = 0
    for path in sorted((root / 'results').glob('*/latencies.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ['history', 'membership', 'ratio', 'syntax', 'ef'])
            groups[key][row['build'], row['policy']].append(row)
            total += 1
    summaries = []
    for key, data in sorted(groups.items()):
        result = dict(zip(['history', 'membership', 'ratio', 'syntax', 'ef'], key))
        result['strategies'] = {}
        for (build, policy), rows in data.items():
            latencies = [r['ms'] for r in rows]
            recalls = [r['recall'] for r in rows]
            result['strategies'][build + '_' + policy] = {
                'samples': len(rows), 'p50_ms': float(np.median(latencies)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'mean_recall': float(np.mean(recalls)), 'min_recall': min(recalls),
                'mean_tie_adjusted_recall': float(np.mean([r['tie_adjusted_recall'] for r in rows])),
                'errors': sum(r['error'] is not None for r in rows),
                'incomplete': sum(r['count'] != 200 for r in rows),
                'over_500ms': sum(r['ms'] >= 500 for r in rows)}
        baseline = {(r['qi'], r['repeat']): r for r in data['baseline', 'AUTO']}
        candidate = {(r['qi'], r['repeat']): r for r in data['candidate', 'AUTO']}
        if baseline.keys() != candidate.keys():
            result['incomplete_pairing'] = True
        else:
            result['paired_median_speedup'] = float(np.median(
                [baseline[k]['ms'] / candidate[k]['ms'] for k in baseline]))
        b = result['strategies']['baseline_AUTO']
        c = result['strategies']['candidate_AUTO']
        result['median_slowdown'] = c['p50_ms'] / b['p50_ms']
        result['p95_slowdown'] = c['p95_ms'] / b['p95_ms']
        summaries.append(result)
    report = {'measurements': total, 'groups': len(summaries), 'results': summaries,
              'median_regressions_over_10pct': [r for r in summaries if r['median_slowdown'] > 1.1]}
    fresh = root / 'results/fresh/controls.jsonl'
    holes = root / 'results/holes/controls.jsonl'
    if fresh.exists() and holes.exists():
        def load(path):
            return {(tuple(r['case']), r['qi'], r['ef']): r
                    for r in map(json.loads, path.read_text().splitlines())}
        a, b = load(fresh), load(holes)
        report['history_controls'] = {
            'cases_fresh': len(a), 'cases_holes': len(b),
            'all_batch_keys_and_scores_equal': a.keys() == b.keys() and all(
                a[k]['batches'] == b[k]['batches'] for k in a),
            'all_bf_keys_and_scores_equal': a.keys() == b.keys() and all(
                a[k]['truth'] == b[k]['truth'] for k in a)}
    (root / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ['results', 'median_regressions_over_10pct']}, indent=2))
    print('Groups with >10% median regression:', len(report['median_regressions_over_10pct']))
    for r in summaries:
        if r['ratio'] == 26 and r['syntax'] == 'negative':
            print(json.dumps(r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
