"""Summarize raw completed runs; only measured recall points qualify."""
import json
import argparse
from pathlib import Path
import re
from statistics import median

parser = argparse.ArgumentParser()
parser.add_argument('--evidence', type=Path, default=Path(__file__).parent/'evidence')
root = parser.parse_args().evidence/'results'

def kv(line):
    return {k:float(v) for k,v in re.findall(r'(\w+)=([-+0-9.eE]+)(?=\s|$)',line)}

def wall_seconds(path):
    text = path.read_text()
    v = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)', text)[1]
    total = 0
    for part in v.split(':'):
        total = total*60 + float(part)
    return total

runs=[]
for file in sorted(root.glob('*.log')):
    name=file.name
    timing=file.with_suffix('.time')
    if not timing.exists() and name.endswith('.query.log'):
        timing=root/(name.removesuffix('.query.log')+'.time')
    if not timing.exists() or 'Exit status: 0' not in timing.read_text():
        continue
    if '-hnsw-' in name:
        lines=file.read_text().splitlines()
        header=next((kv(x) for x in lines if x.startswith('n=')),None)
        points=[kv(x) for x in lines if x.startswith('ef=')]
        if not header or not points: continue
        m=re.search(r'^(.*)-hnsw-m(\d+)-c(\d+)-r(\d+)',name)
        runs.append(dict(dataset=m[1], engine='hnsw', config=f'M{m[2]}-efC{m[3]}', repeat=int(m[4]),
                         build_s=header['ready_from_fbin_seconds'], rss_bytes=header['resident_index_process_bytes'],
                         peak_bytes=header['build_peak_rss_bytes'],
                         points=[dict(effort=x['ef'], recall=x['recall10'], p95_us=x['p95_us'], p50_us=x['p50_us']) for x in points], file=name))
    elif '-pipnn-' in name and name.endswith('.query.log'):
        lines=file.read_text().splitlines()
        mem=next((kv(x) for x in lines if x.startswith('resident_index')),None)
        points=[kv(x) for x in lines if x.startswith('beam=')]
        if not mem or not points: continue
        m=re.search(r'^(.*)-pipnn-a([0-9.]+)(?:-trees(\d+))?-r(\d+)',name)
        stem=name.removesuffix('.query.log')
        timefile=root/(stem+'.build.time')
        if not timefile.exists(): timefile=root/(stem+'.time')
        peak=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',timefile.read_text())[1])*1024
        runs.append(dict(dataset=m[1], engine='pipnn', config=f'alpha{m[2]}-trees{m[3] or 1}', repeat=int(m[4]),
                         build_s=wall_seconds(timefile), rss_bytes=mem['resident_index_process_bytes'], peak_bytes=peak,
                         points=[dict(effort=x['beam'], recall=x['recall'], p95_us=x['p95_us'], p50_us=x['p50_us']) for x in points], file=name))

groups={}
for run in runs:
    groups.setdefault((run['dataset'],run['engine'],run['config']),[]).append(run)
summary=[]
for (dataset,engine,config), group in sorted(groups.items()):
    row=dict(dataset=dataset,engine=engine,config=config,repeats=len(group),
             build_s=median(x['build_s'] for x in group), build_range_s=[min(x['build_s'] for x in group),max(x['build_s'] for x in group)],
             rss_mb=median(x['rss_bytes'] for x in group)/1e6, peak_mb=max(x['peak_bytes'] for x in group)/1e6)
    efforts=set.union(*[{p['effort'] for p in x['points']} for x in group])
    for target in (.95,.99):
        feasible=[]
        for effort in efforts:
            observations=[p for x in group for p in x['points'] if p['effort']==effort]
            if len(observations)==len(group) and min(p['recall'] for p in observations)>=target:
                feasible.append(dict(effort=effort, observations=len(observations), recall_min=min(p['recall'] for p in observations),
                                     p95_us=median(p['p95_us'] for p in observations),
                                     p95_range_us=[min(p['p95_us'] for p in observations),max(p['p95_us'] for p in observations)]))
        row[f'at{int(target*100)}']=min(feasible,key=lambda x:x['p95_us']) if feasible else None
    summary.append(row)
output=dict(runs=runs,summary=summary,notes=['PiPNN build time is entire builder process including graph serialization; HNSW ready time includes fbin loading and build but no serialization.', 'Only effort settings meeting recall in every observed repeat at that setting qualify; observation counts are reported, with no interpolation.', 'RSS is loaded index process residency, not allocator-owned index bytes. Peak is build process memory, including input buffers.', 'Two repeats describe an observed range; they are not a confidence interval.'])
(root.parent/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
for row in summary: print(json.dumps(row))
