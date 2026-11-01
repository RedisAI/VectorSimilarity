#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/mod18264/source
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('../narrowed/manifest.json').read_text())
for name, expected in manifest['source_sha256'].items():
    actual = hashlib.file_digest(Path(name).open('rb'), 'sha256').hexdigest()
    assert actual == expected, (name, actual, expected)
print('All compiled source files match the local candidate')
PY
git diff --binary --full-index -- src/redisearch_rs/vector_score_source/src/lib.rs > ../narrowed/artifacts/root.patch
git -C deps/VectorSimilarity diff --binary --full-index -- src/VecSim/algorithms/hnsw/hnsw.h src/VecSim/algorithms/hnsw/hnsw_hybrid_policy.h > ../narrowed/artifacts/vecsim.patch
cmp ../narrowed/root.patch ../narrowed/artifacts/root.patch
cmp ../narrowed/vecsim.patch ../narrowed/artifacts/vecsim.patch
cd ../narrowed
ln -s ../../current/artifacts/baseline.so artifacts/baseline.so
ln -s ../../current/artifacts/rejson.so artifacts/rejson.so
ln -s ../current/vectors.f32 vectors.f32
sha256sum artifacts/*.so > artifacts/SHA256SUMS
sha256sum queries.f32 vectors.f32 > artifacts/data-SHA256SUMS
bash run_current.sh 2>&1 | tee run-narrowed.log | tail -50
