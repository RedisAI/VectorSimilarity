#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/mod18264/tools:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=8
cd /home/ubuntu/mod18264
mkdir -p narrowed/artifacts narrowed/logs
test -z "$(git -C source status --porcelain)"
test "$(git -C source rev-parse HEAD)" = 50b3cb6d667494982941dc09ae4fac65c2287174
test "$(git -C source/deps/VectorSimilarity rev-parse HEAD)" = 2d27f83f0cdb23f060983c5ca7eb74139fb05a93
git -C source apply --check ../narrowed/root.patch
git -C source/deps/VectorSimilarity apply --check ../../../narrowed/vecsim.patch
git -C source apply ../narrowed/root.patch
git -C source/deps/VectorSimilarity apply ../../../narrowed/vecsim.patch
cd source
if command -v sccache >/dev/null && sccache --show-stats; then
  export RUSTC_WRAPPER=sccache SCCACHE_CACHE_SIZE=30G SCCACHE_BASEDIRS=/home/ubuntu/mod18264
fi
taskset -c 0-7 ./build.sh 2>&1 | tee ../narrowed/logs/build-candidate.log | tail -35
cp bin/linux-x64-release/search-community/redisearch.so ../narrowed/artifacts/candidate.so
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
echo BUILD_NARROWED_COMPLETE
