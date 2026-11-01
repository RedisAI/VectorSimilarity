"""Summarize direct native policy execution. Excludes no failed or slow cases."""
import csv, io, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
root=Path(__file__).parent
run=sys.argv[1]
raw=(root/f'{run}-results.csv').read_text()
rows=list(csv.DictReader(io.StringIO(raw[raw.index('n,dim,m,efc,bytes,seed,filter,ratio,k,query,estimate_kind,ef,repeat,'): ])))
fields=['n','dim','m','efc','bytes','seed','filter','ratio','k','estimate_kind','ef','policy']
groups=defaultdict(list)
for r in rows: groups[tuple(r[x] for x in fields)].append(r)
def percentile(xs,p):
    xs=sorted(xs)
    return xs[max(0,math.ceil(len(xs)*p)-1)]
summary=[]
for key,rs in groups.items():
    out=dict(zip(fields,key))
    bf=[float(r['adhoc_us']) for r in rs]
    auto=[float(r['batches_us']) for r in rs]
    out.update(samples=len(rs),bf_p50_us=statistics.median(bf),bf_p95_us=percentile(bf,.95),policy_p50_us=statistics.median(auto),policy_p95_us=percentile(auto,.95),mean_recall=statistics.mean(float(r['recall']) for r in rs),min_recall=min(float(r['recall']) for r in rs),switches=sum(int(r['switched']) for r in rs),initial_adhoc=sum(int(r['batch_count'])==0 for r in rs),paired_speedup=statistics.median(float(r['adhoc_us'])/float(r['batches_us']) for r in rs))
    summary.append(out)
(root/f'{run}-summary.json').write_text(json.dumps(dict(scope='Local native direct policy execution, same graph; excludes RediSearch filter-iterator and network costs. Progress is an external prototype, compiled is the current source helper.',measurements=len(rows),groups=summary),indent=2)+'\n')
print('Measurements',len(rows),'groups',len(summary))
print('Customer-like K200, 26%:')
for r in summary:
    if r['k']=='200' and r['ratio']=='0.26': print(json.dumps(r))
print('Progress policy worst ten accurate-estimate regressions vs BF:')
for r in sorted([r for r in summary if r['policy']=='progress' and r['estimate_kind']=='actual'],key=lambda r:r['paired_speedup'])[:10]:print(json.dumps(r))
