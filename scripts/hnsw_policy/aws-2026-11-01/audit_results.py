# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Audit completeness and exact-reference consistency of a completed AWS run."""
import argparse
import gzip
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

RATIOS = [1, 5, 10, 17.5, 26, 50, 95, 100]
STRATEGIES = [('baseline', 'AUTO'), ('candidate', 'AUTO'),
              ('baseline', 'ADHOC_BF'), ('baseline', 'BATCHES')]


def read(path):
    if path.exists():
        return path.read_text()
    with gzip.open(str(path) + '.gz', 'rt') as stream:
        return stream.read()


def digest(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def pairs(items):
    return dict(zip(items[::2], items[1::2]))


def audit(root):
    report = {'histories': {}}
    controls_by_history = {}
    for history in ['fresh', 'holes']:
        directory = root / 'results' / history
        controls = {}
        for row in map(json.loads, read(directory / 'controls.jsonl').splitlines()):
            key = (tuple(row['case']), row['qi'], row['ef'])
            assert key not in controls
            assert len(row['truth']) == len(row['batches']) == 200
            controls[key] = row
        assert len(controls) == 1920
        controls_by_history[history] = controls
        seen = set()
        auto_hashes = Counter()
        for row in map(json.loads, read(directory / 'latencies.jsonl').splitlines()):
            key = tuple(row[k] for k in
                        ['membership', 'ratio', 'syntax', 'ef', 'qi', 'repeat', 'build', 'policy'])
            assert key not in seen, key
            seen.add(key)
            assert row['history'] == history
            assert row['error'] is None and row['count'] == row['reported_count'] == 200
            assert row['ms'] > 0 and 0 <= row['recall'] <= 1
            case = (row['membership'], RATIOS.index(row['ratio']), row['syntax'])
            control = controls[case, row['qi'], row['ef']]
            if row['policy'] in ['ADHOC_BF', 'BATCHES']:
                expected = control['truth' if row['policy'] == 'ADHOC_BF' else 'batches']
                assert row['ids_sha256'] == digest(expected)
            else:
                auto_hashes[tuple(key[:5]) + (row['build'], row['ids_sha256'])] += 1
        expected = {
            (*case, *strategy)
            for case in itertools.product(['random', 'projection'], RATIOS,
                                          ['positive', 'negative'], [10, 400, 800],
                                          range(20), range(3))
            for strategy in STRATEGIES
        }
        assert seen == expected, (len(seen), len(expected))
        assert all(count == 3 for count in auto_hashes.values()), 'AUTO results changed across repeats'
        states = {}
        for phase in ['loaded', 'before', 'after']:
            state = json.loads(read(directory / (phase + '.json')))
            states[phase] = {}
            for build, shards in state.items():
                assert len(shards) == 2
                states[phase][build] = []
                for i, shard in enumerate(shards):
                    vectors = pairs(shard['vectors'])
                    backend = pairs(vectors['BACKEND_INDEX'])
                    count = [73187, 73186][i]
                    assert vectors['INDEX_SIZE'] == vectors['INDEX_LABEL_COUNT'] == count
                    assert pairs(vectors['FRONTEND_INDEX'])['INDEX_SIZE'] == 0
                    assert vectors['BACKGROUND_INDEXING'] == backend['NUMBER_OF_MARKED_DELETED'] == 0
                    assert backend['INDEX_SIZE'] == backend['INDEX_LABEL_COUNT'] == count
                    states[phase][build].append(shard['max_doc_id'])
            assert states[phase]['baseline'] == states[phase]['candidate']
        assert states['loaded'] == states['before'] == states['after']
        report['histories'][history] = {
            'timings': len(seen), 'controls': len(controls), 'max_doc_ids': states['after']['baseline'],
            'forced_timing_ids_match_controls': True, 'auto_ids_stable_across_repeats': True,
            'complete_pairing': True, 'errors': 0, 'incomplete': 0, 'pending_vector_deletions': 0}
    a, b = controls_by_history['fresh'], controls_by_history['holes']
    assert a == b, 'Fresh and hole histories have different exact or forced-batch results'
    report['exact_and_batch_keys_scores_equal_across_histories'] = True
    report['limitations'] = ('Cross-build control equality is asserted by the measurement harness before '
                             'timing. This audit checks the retained common controls and all timed IDs; '
                             'it does not reconstruct scores absent from the latency rows.')
    (root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
