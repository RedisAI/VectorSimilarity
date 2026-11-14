#!/usr/bin/env python3
import pathlib, struct, sys
import numpy as np
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
base = rng.normal(size=(200, 16)).astype('<f4')
queries = rng.normal(size=(12, 16)).astype('<f4')
dist = ((queries[:, None, :].astype(np.float64) - base[None, :, :]) ** 2).sum(axis=2)
truth = np.argsort(dist, axis=1)[:, :10].astype('<u4')
for name, a in [('base.fbin', base), ('query.fbin', queries), ('truth.ibin', truth)]:
    with (out / name).open('wb') as f:
        f.write(struct.pack('<II', *a.shape)); f.write(a.tobytes())
