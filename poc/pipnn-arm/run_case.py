"""Run one isolated serial benchmark case on dorer-arm-big, saving raw evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('dataset', choices=['deep100k', 'dbpedia100k'])
p.add_argument('engine', choices=['hnsw', 'pipnn'])
p.add_argument('--repeat', type=int, default=1)
p.add_argument('--m', type=int, default=32)
p.add_argument('--efc', type=int, default=200)
p.add_argument('--alpha', type=float, default=1.2)
p.add_argument('--replicas', type=int, default=1)
a = p.parse_args()
root = Path('/home/ubuntu/pipnn-poc-20261114')
os.chdir(root)
data = root/'data'/a.dataset
out = root/'results'
out.mkdir(exist_ok=True)
env = dict(os.environ, PARLAY_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
beams = [10, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 800, 1000]
tag = f'{a.dataset}-{a.engine}-' + (f'm{a.m}-c{a.efc}' if a.engine == 'hnsw' else f'a{a.alpha}-trees{a.replicas}') + f'-r{a.repeat}'

def run(cmd, label):
    names = subprocess.check_output(['ps', '-eo', 'comm='], text=True).splitlines()
    busy = [x.strip() for x in names if x.strip() in ('vecsim_bench', 'pipnn_query', 'cc1plus', 'cc1', 'ninja', 'make') or x.strip().startswith(('neighbors-', 'bm_'))]
    if busy:
        raise RuntimeError('Competing build/benchmark processes: ' + repr(busy))
    cmd = [str(x) for x in cmd]
    record = {'command': cmd, 'start_unix': time.time(), 'env': {k: env[k] for k in ['PARLAY_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
    log = out/f'{tag}.{label}.log'
    if log.exists():
        raise RuntimeError(f'refusing to overwrite {log}')
    with log.open('w') as f, (out/f'{tag}.{label}.time').open('w') as err:
        process = subprocess.run(['/usr/bin/time', '-v', *cmd], env=env, stdout=f, stderr=err)
    record.update(end_unix=time.time(), returncode=process.returncode)
    (out/f'{tag}.{label}.json').write_text(json.dumps(record, indent=2)+'\n')
    if process.returncode:
        raise RuntimeError(f'{tag}: {label} failed ({process.returncode})')
    print(f'{tag} {label} complete in {record["end_unix"]-record["start_unix"]:.2f}s', flush=True)

if a.engine == 'hnsw':
    run([root/'vecsim_bench', data/'base.fbin', data/'query.fbin', data/'truth.ibin', a.m, a.efc, a.repeat, ','.join(map(str, beams))], 'run')
else:
    binary = root/'upstream/build/algorithms/PipNN'
    graph = out/f'{tag}.graph'
    run([binary/'neighbors-pipnn_FLOAT_T_EUCLIDEAN', '-base_path', data/'base.fbin', '-graph_outfile', graph,
         '-num_clusters', a.replicas, '-fanout', 30, '-fanout_scheme', '10,3', '-cluster_size', 1024,
         '-mst_deg', 2, '-prune', '-prune_all', '-prune_degree', 64, '-table_size', 160,
         '-top_level_leaders', 1000, '-fraction_leaders', 0.005, '-hash_bits', 12, '-leaf_method', 'EigenKNN',
         '-quantize_bits', 0, '-quantize_mode', 0, '-alpha', a.alpha], 'build')
    run([binary/'pipnn_query', graph, data/'base.fbin', data/'query.fbin', data/'truth-with-dist.bin', 10, *beams], 'query')
