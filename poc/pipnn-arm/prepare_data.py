"""Prepare real, disjoint FP32 L2 inputs and exact ground truth on the ARM host."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path('/home/ubuntu/pipnn-poc-20261114/data')
ROOT.mkdir(parents=True, exist_ok=True)

def read_bin(path, dtype='<f4'):
    n, d = np.fromfile(path, dtype='<u4', count=2)
    return np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(int(n), int(d)))

def write_bin(path, array):
    with path.open('wb') as f:
        np.array(array.shape, dtype='<u4').tofile(f)
        array.tofile(f)

def prepare(name, base, queries, normalize=False):
    start = time.monotonic()
    # Compare exact raw-vector content, not only label identities.
    query_hashes = {hashlib.sha256(q.tobytes()).digest() for q in queries}
    overlaps = {hashlib.sha256(x.tobytes()).digest() for x in base} & query_hashes
    queries = np.array([q for q in queries if hashlib.sha256(q.tobytes()).digest() not in overlaps], dtype='<f4')[:1000]
    if len(queries) < 100:
        raise RuntimeError(f'{name}: insufficient disjoint queries: {len(queries)}')
    base = np.array(base, dtype='<f4')
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError('nonfinite inputs')
    if normalize:
        for x in (base, queries):
            norms = np.linalg.norm(x.astype(np.float64), axis=1)
            if (norms == 0).any():
                raise ValueError('zero vector')
            x /= norms[:, None]
    normalized_overlaps = set()
    if normalize:
        hashes = {hashlib.sha256(q.tobytes()).digest() for q in queries}
        normalized_overlaps = {hashlib.sha256(x.tobytes()).digest() for x in base} & hashes
        queries = np.array([q for q in queries if hashlib.sha256(q.tobytes()).digest() not in normalized_overlaps], dtype='<f4')
    if len(base) < 11:
        raise ValueError('at least 11 base vectors required for recall@10 boundary checks')
    k = min(100, len(base))
    # Float64 ground truth; recompute shortlisted distances by direct subtraction.
    b64 = base.astype(np.float64)
    norms = np.einsum('ij,ij->i', b64, b64)
    ids, distances = [], []
    for i in range(0, len(queries), 64):
        q = queries[i:i+64].astype(np.float64)
        scores = np.einsum('ij,ij->i', q, q)[:, None] + norms[None, :] - 2 * (q @ b64.T)
        chosen = np.argpartition(scores, k-1, axis=1)[:, :k]
        chosen = np.take_along_axis(chosen, np.argsort(np.take_along_axis(scores, chosen, axis=1), axis=1), axis=1)
        exact = np.array([np.sum((b64[c] - row)**2, axis=1) for row, c in zip(q, chosen)])
        order = np.argsort(exact, axis=1, kind='stable')
        chosen = np.take_along_axis(chosen, order, axis=1)
        exact = np.take_along_axis(exact, order, axis=1)
        ids.append(chosen.astype('<u4'))
        distances.append(exact)
    ids, distances = np.concatenate(ids), np.concatenate(distances)
    # Exclude trivial near-identical queries and ambiguous top-10 boundaries;
    # otherwise multiple correct label sets can make ID recall misleading.
    near = distances[:, 0] < 1e-12
    tied = np.abs(distances[:, 10] - distances[:, 9]) < 1e-10
    keep = ~(near | tied)
    queries, ids, distances = queries[keep], ids[keep], distances[keep].astype('<f4')
    if len(queries) < 100:
        raise ValueError('insufficient unambiguous held-out queries')
    out = ROOT / name
    out.mkdir(exist_ok=True)
    write_bin(out/'base.fbin', base)
    write_bin(out/'query.fbin', queries)
    write_bin(out/'truth.ibin', ids)
    # ParlayANN ground truth contains IDs followed by distances, after one header.
    with (out/'truth-with-dist.bin').open('wb') as f:
        np.array(ids.shape, dtype='<u4').tofile(f)
        ids.tofile(f)
        distances.tofile(f)
    report = dict(name=name, n=len(base), dim=base.shape[1], queries=len(queries),
                  metric='L2 on FP32 unit-normalized vectors' if normalize else 'L2',
                  excluded_overlap_hashes=len(overlaps), normalized_overlap_hashes=len(normalized_overlaps),
                  excluded_near_identical=int(near.sum()), excluded_tied_top10=int(tied.sum()), truth='exhaustive float64 norm/dot scan with direct-distance shortlist reranking',
                  preparation_seconds=time.monotonic()-start,
                  base_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
                  query_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
                  thread_env={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
    (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    source = Path('/home/ubuntu/mod21066/data')
    prepare('deep100k', read_bin(source/'deep.base.100K.fbin'), read_bin(source/'deep.query.public.10K.fbin'))
    source = Path('/home/ubuntu/sq8-e2e/vdb/datasets/dbpedia-openai-975K-1536-angular-100neighbors/dbpedia_openai_975K')
    base = np.load(source/'vectors.npy', mmap_mode='r')[:100000]
    queries = []
    with (source/'tests.jsonl').open() as f:
        for i, line in enumerate(f):
            if i == 2000:
                break
            record = json.loads(line)
            if record['conditions']:
                raise ValueError('filtered query in unfiltered benchmark')
            queries.append(record['query'])
    prepare('dbpedia100k', base, np.asarray(queries, dtype='<f4'), normalize=True)
