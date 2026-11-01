"""Paired end-to-end policy measurements on two dedicated, two-shard clusters."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import redis
from redis.cluster import key_slot

RATIOS = [1, 5, 10, 17.5, 26, 50, 95, 100]
INDEX = 'product_v4'
K = 200
DIM = 768


def serial(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, dict):
        return {serial(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def pairs(value):
    value = serial(value)
    return dict(zip(value[::2], value[1::2]))


def token(membership, ri):
    return f'{membership[0]}{ri}'


def predicate(membership, ri, syntax):
    t = token(membership, ri)
    if syntax == 'positive':
        return f'@channels:{{{t}keep}}'
    return ' '.join(f'-@channels:{{{t}{part}}}' for part in 'abc')


def expression(case, ef, policy, exact=False):
    pred = predicate(*case)
    if exact:
        pred = f'@channels:{{all}} ({pred})'
    option = '' if policy == 'AUTO' else f'HYBRID_POLICY {policy}'
    search_width = '' if policy == 'ADHOC_BF' else f'EF_RUNTIME {ef}'
    return f'({pred})=>[KNN {K} @embedding $BLOB {search_width} {option} AS d]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['load', 'measure'])
    parser.add_argument('--ports', nargs=4, type=int, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--history', choices=['fresh', 'holes'], required=True)
    parser.add_argument('--n', type=int, default=146373)
    parser.add_argument('--queries', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--efs', type=int, nargs='+', default=[10, 400, 800])
    args = parser.parse_args()
    output = args.root / 'results' / args.history
    output.mkdir(parents=True, exist_ok=True)
    clusters = {
        name: [redis.Redis(port=port, protocol=2, socket_timeout=60) for port in ports]
        for name, ports in zip(['baseline', 'candidate'], [args.ports[:2], args.ports[2:]])
    }
    shard_tags = [next(str(i) for i in range(1000) if key_slot(str(i).encode()) // 8192 == shard)
                  for shard in range(2)]
    counts = [(args.n + 1) // 2, args.n // 2]
    targets = [1152500, 1058100] if args.n == 146373 else [15 * n for n in counts]
    vectors = np.memmap(args.root / 'vectors.f32', mode='r', dtype='<f4', shape=(args.n, DIM))

    def key(i):
        return f'product:{{{shard_tags[i % 2]}}}:{i}'

    def state():
        result = {}
        for name, nodes in clusters.items():
            result[name] = [{'port': node.connection_pool.connection_kwargs['port'],
                             'server': node.info('server'),
                             'memory': node.info('memory'),
                             'config': node.config_get('search-*workers'),
                             'modules': node.execute_command('MODULE', 'LIST'),
                             'index': node.execute_command('FT.INFO', INDEX),
                             'max_doc_id': node.execute_command('_FT.DEBUG', 'GET_MAX_DOC_ID', INDEX),
                             'vectors': node.execute_command('_FT.DEBUG', 'VECSIM_INFO', INDEX, 'embedding')}
                            for node in nodes]
        return serial(result)

    def wait_index(nodes):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            ok = True
            for shard, node in enumerate(nodes):
                info = pairs(node.execute_command('_FT.DEBUG', 'VECSIM_INFO', INDEX, 'embedding'))
                ok &= info['INDEX_LABEL_COUNT'] == counts[shard]
                ok &= pairs(info['FRONTEND_INDEX'])['INDEX_SIZE'] == 0
                ok &= not info['BACKGROUND_INDEXING']
            if ok:
                return
            time.sleep(.2)
        raise RuntimeError('Indexing did not settle')

    # Membership is fixed before any measurements and identical in both builds.
    ranks = {}
    for shard, n in enumerate(counts):
        random_order = np.random.default_rng(58301 + shard).permutation(n)
        projection_order = np.argsort(vectors[shard::2, 0], kind='stable')
        for membership, order in [('random', random_order), ('projection', projection_order)]:
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            ranks[shard, membership] = rank

    if args.action == 'load':
        for name, nodes in clusters.items():
            if any(node.dbsize() for node in nodes):
                raise RuntimeError('Refusing to load into a nonempty cluster')
            for node in nodes:
                for option in ['search-min-operation-workers', 'search-min-maintenance-workers', 'search-workers']:
                    node.config_set(option, 0)
                node.config_set('search-timeout', 20000)
                node.config_set('search-on-timeout', 'fail')
            nodes[0].execute_command('FT.CREATE', 'probe', 'ON', 'JSON', 'PREFIX', 1, 'probe:',
                                    'SCHEMA', '$.channels[*]', 'AS', 'channels', 'TAG',
                                    '$.embedding', 'AS', 'embedding', 'VECTOR', 'HNSW', 6,
                                    'TYPE', 'FLOAT64', 'DIM', DIM, 'DISTANCE_METRIC', 'COSINE')
            for shard, node in enumerate(nodes):
                node.execute_command('JSON.SET', f'probe:{{{shard_tags[shard]}}}:1', '$',
                                     json.dumps({'channels': ['all', 'r0keep', 'r7keep'],
                                                 'embedding': [1.0]*DIM}))
            for case in [('random', 0, 'positive'), ('random', 0, 'negative'),
                         ('random', 7, 'positive'), ('random', 7, 'negative')]:
                for ef in args.efs:
                    for policy in ['AUTO', 'BATCHES', 'ADHOC_BF']:
                        result = nodes[0].execute_command(
                            'FT.SEARCH', 'probe', expression(case, ef, policy, policy == 'ADHOC_BF'),
                            'RETURN', 1, 'd', 'SORTBY', 'd', 'LIMIT', 0, K, 'DIALECT', 2,
                            'PARAMS', 2, 'BLOB', np.ones(DIM, dtype='<f8').tobytes())
                        assert result[0] == 2, result
            nodes[0].execute_command('FT.DROPINDEX', 'probe', 'DD')
            assert all(node.dbsize() == 0 for node in nodes)
            print(f'{name}: query-option preflight passed', flush=True)
            nodes[0].execute_command('FT.CREATE', INDEX, 'ON', 'JSON', 'PREFIX', 1, 'product:',
                                    'SCHEMA', '$.channels[*]', 'AS', 'channels', 'TAG',
                                    '$.embedding', 'AS', 'embedding', 'VECTOR', 'HNSW', 12,
                                    'TYPE', 'FLOAT64', 'DIM', DIM, 'DISTANCE_METRIC', 'COSINE',
                                    'M', 16, 'EF_CONSTRUCTION', 100, 'EF_RUNTIME', 10)
            pipes = [node.pipeline(transaction=False) for node in nodes]
            for i in range(args.n):
                shard, local_i = i % 2, i // 2
                pipe = pipes[shard]
                if args.history == 'holes':
                    # Interleave deleted non-vector documents without changing the live HNSW graph.
                    holes = targets[shard] - counts[shard]
                    num_holes = ((local_i + 1) * holes // counts[shard]
                                 - local_i * holes // counts[shard])
                    dummy = f'product:{{{shard_tags[shard]}}}:hole'
                    for _ in range(num_holes):
                        pipe.execute_command('JSON.SET', dummy, '$', '{"channels":["hole"]}')
                        pipe.delete(dummy)
                tags = ['all']
                for membership in ['random', 'projection']:
                    rank = ranks[shard, membership][local_i]
                    for ri, ratio in enumerate(RATIOS):
                        passing = rank < int(counts[shard] * ratio / 100)
                        suffix = 'keep' if passing else 'abc'[local_i % 3]
                        tags.append(token(membership, ri) + suffix)
                doc = {'channels': tags, 'embedding': vectors[i].astype(float).tolist()}
                pipe.execute_command('JSON.SET', key(i), '$', json.dumps(doc, separators=(',', ':')))
                if i % 200 == 199:
                    for pipe in pipes:
                        pipe.execute()
                if i % 10000 == 9999:
                    print(f'{args.history} {name}: loaded {i+1}/{args.n}', flush=True)
            for pipe in pipes:
                pipe.execute()
            wait_index(nodes)
            for shard, node in enumerate(nodes):
                observed = node.execute_command('_FT.DEBUG', 'GET_MAX_DOC_ID', INDEX)
                expected = targets[shard] if args.history == 'holes' else counts[shard]
                assert observed == expected, (observed, expected)
                assert node.dbsize() == counts[shard]
            print(f'{args.history} {name}: graph complete', flush=True)
        (output / 'loaded.json').write_text(json.dumps(state(), indent=2))
        return

    for nodes in clusters.values():
        wait_index(nodes)
        for node in nodes:
            node.config_set('search-workers', 2)
    (output / 'before.json').write_text(json.dumps(state(), indent=2))
    blobs = np.fromfile(args.root / 'queries.f32', dtype='<f4').reshape(-1, DIM)[:args.queries].astype('<f8')
    cases = [(m, ri, syntax) for m in ['random', 'projection'] for ri in range(len(RATIOS))
             for syntax in ['positive', 'negative']]

    def query(build, case, qi, ef, policy, profile=False, exact=False):
        prefix = ['FT.PROFILE', INDEX, 'SEARCH', 'QUERY'] if profile else ['FT.SEARCH', INDEX]
        return clusters[build][0].execute_command(
            *prefix, expression(case, ef, policy, exact), 'RETURN', 1, 'd', 'SORTBY', 'd', 'LIMIT', 0, K,
            'TIMEOUT', 20000, 'DIALECT', 2, 'PARAMS', 2, 'BLOB', blobs[qi].tobytes())

    def hits(reply):
        assert len(reply) % 2 == 1, reply[:5]
        return {key.decode(): float(pairs(values)['d']) for key, values in zip(reply[1::2], reply[2::2])}

    # Compare complete membership, not just counts, for the equivalent filter expressions.
    membership_evidence = []
    for m in ['random', 'projection']:
        for ri, ratio in enumerate(RATIOS):
            expected = sum(int(n * ratio / 100) for n in counts)
            ids = []
            for build in clusters:
                for syntax in ['positive', 'negative']:
                    response = clusters[build][0].execute_command(
                        'FT.SEARCH', INDEX, predicate(m, ri, syntax), 'NOCONTENT',
                        'LIMIT', 0, args.n, 'TIMEOUT', 20000, 'DIALECT', 2)
                    assert response[0] == expected and len(response)-1 == expected
                    ids.append(set(response[1:]))
            assert all(part == ids[0] for part in ids)
            membership_evidence.append({'membership': m, 'ratio': ratio, 'count': expected,
                                        'ids_sha256': hashlib.sha256(b'\n'.join(sorted(ids[0]))).hexdigest()})
    (output / 'membership.json').write_text(json.dumps(membership_evidence, indent=2))
    truth = {}
    controls = {}
    with (output / 'controls.jsonl').open('w') as out:
        for ci, case in enumerate(cases):
            for qi in range(args.queries):
                a = hits(query('baseline', case, qi, 10, 'ADHOC_BF', exact=True))
                b = hits(query('candidate', case, qi, 10, 'ADHOC_BF', exact=True))
                assert a == b, ('BF differs across builds', case, qi)
                assert len(a) == K
                truth[case, qi] = a
                for ef in args.efs:
                    a = hits(query('baseline', case, qi, ef, 'BATCHES'))
                    b = hits(query('candidate', case, qi, ef, 'BATCHES'))
                    assert a == b, ('BATCHES differs across builds', case, qi, ef)
                    controls[case, qi, ef] = a
                    out.write(json.dumps({'case': case, 'qi': qi, 'ef': ef, 'truth': truth[case, qi],
                                          'batches': a}) + '\n')
            out.flush()
            print(f'{args.history}: control equality passed {ci+1}/{len(cases)}', flush=True)
    profiles = []
    strategies = [('baseline', 'AUTO'), ('candidate', 'AUTO'),
                  ('baseline', 'ADHOC_BF'), ('baseline', 'BATCHES')]
    for case in cases:
        for ef in args.efs:
            for build, policy in strategies:
                profiles.append({'case': case, 'ef': ef, 'build': build, 'policy': policy,
                                 'profile': serial(query(build, case, 0, ef, policy, profile=True,
                                                         exact=policy == 'ADHOC_BF'))})
    (output / 'profiles.json').write_text(json.dumps(profiles))
    jobs = [(case, qi, ef, repeat) for repeat in range(args.repeats)
            for case in cases for qi in range(args.queries) for ef in args.efs]
    random.Random(58302).shuffle(jobs)
    with (output / 'latencies.jsonl').open('w') as out:
        for ji, (case, qi, ef, repeat) in enumerate(jobs):
            shift = ji % len(strategies)
            for build, policy in strategies[shift:] + strategies[:shift]:
                start = time.perf_counter_ns()
                error = None
                result = [0]
                try:
                    result = query(build, case, qi, ef, policy, exact=policy == 'ADHOC_BF')
                except redis.ResponseError as exc:
                    error = str(exc)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                found = hits(result)
                exact_ids = truth[case, qi]
                recall = len(found.keys() & exact_ids.keys()) / K
                tie_recall = sum(score <= max(exact_ids.values()) for score in found.values()) / K
                row = {'history': args.history, 'membership': case[0], 'ratio': RATIOS[case[1]],
                       'syntax': case[2], 'qi': qi, 'original_qi': qi+2000, 'ef': ef,
                       'repeat': repeat, 'build': build, 'policy': policy, 'ms': elapsed,
                       'count': len(found), 'reported_count': result[0], 'recall': recall,
                       'tie_adjusted_recall': tie_recall,
                       'error': error, 'ids_sha256': hashlib.sha256('\n'.join(sorted(found)).encode()).hexdigest()}
                out.write(json.dumps(row)+'\n')
            if ji % 100 == 99:
                out.flush()
                print(f'{args.history}: measured {ji+1}/{len(jobs)} query groups', flush=True)
    (output / 'after.json').write_text(json.dumps(state(), indent=2))
    print(f'{args.history}: MEASURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
