# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Retain per-query regressions hidden by workload-level latency percentiles."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from audit_results import read


def summarize(root):
    groups = defaultdict(lambda: defaultdict(list))
    for directory in sorted((root / 'results').iterdir()):
        for row in map(json.loads, read(directory / 'latencies.jsonl').splitlines()):
            if row['policy'] == 'AUTO':
                key = tuple(row[k] for k in
                            ['history', 'membership', 'ratio', 'syntax', 'ef', 'original_qi'])
                groups[key][row['build']].append(row)
    rows = []
    for key, strategies in sorted(groups.items()):
        assert all(len(strategies[name]) == 3 for name in ['baseline', 'candidate'])
        b, c = [statistics.median(row['ms'] for row in strategies[name])
                for name in ['baseline', 'candidate']]
        rows.append({
            **dict(zip(['history', 'membership', 'ratio', 'syntax', 'ef', 'query_id'], key)),
            'baseline_median_ms': b, 'candidate_median_ms': c, 'slowdown': c / b,
            'baseline_recall': strategies['baseline'][0]['recall'],
            'candidate_recall': strategies['candidate'][0]['recall'],
        })
    report = {'query_workloads': len(rows), 'histories': {}, 'rows': rows}
    for history in sorted({r['history'] for r in rows}):
        selected = [r for r in rows if r['history'] == history]
        report['histories'][history] = {
            'query_workloads': len(selected),
            'over_10pct_slower': sum(r['slowdown'] > 1.1 for r in selected),
            'over_2x_slower': sum(r['slowdown'] > 2 for r in selected),
            'worst_queries': sorted(selected, key=lambda r: r['slowdown'], reverse=True)[:10],
        }
    (root / 'query-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
