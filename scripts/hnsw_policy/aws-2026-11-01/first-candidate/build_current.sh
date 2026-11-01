#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/mod18264/tools:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=8
cd /home/ubuntu/mod18264
mkdir -p current/artifacts current/logs current/json-source
if command -v sccache >/dev/null && sccache --show-stats; then
  export SCCACHE_CACHE_SIZE=30G SCCACHE_BASEDIRS=/home/ubuntu/mod18264 RUSTC_WRAPPER=sccache
fi
cd source
test -z "$(git status --porcelain)"
git fetch origin omer-mod18264-hnsw-policy
for revision in baseline:1fbe5191c925c710945a7ed526293d552e8e346a candidate:50b3cb6d667494982941dc09ae4fac65c2287174; do
  name=${revision%%:*}
  commit=${revision#*:}
  git checkout --detach "$commit"
  git submodule update --init --recursive
  taskset -c 0-7 ./build.sh 2>&1 | tee "../current/logs/build-$name.log" | tail -35
  cp bin/linux-x64-release/search-community/redisearch.so "../current/artifacts/$name.so"
  git rev-parse HEAD > "../current/artifacts/$name-commit.txt"
  git submodule status --recursive > "../current/artifacts/$name-submodules.txt"
  find bin/linux-x64-release -name CMakeCache.txt -exec cp {} "../current/artifacts/$name-CMakeCache.txt" \;
done
cd ../current/json-source
tar -xf ../redisjson-current.tar
taskset -c 0-7 cargo +1.94.0 build --release --locked -p redis_json 2>&1 | tee ../logs/build-json.log | tail -35
cp target/release/librejson.so ../artifacts/rejson.so
cd ..
sha256sum artifacts/*.so > artifacts/SHA256SUMS
/home/ubuntu/mod-18374/Redis-main-runtime/src/redis-server --version > artifacts/redis-version.txt
sha256sum /home/ubuntu/mod-18374/Redis-main-runtime/src/redis-server > artifacts/redis-sha256.txt
ldd /home/ubuntu/mod-18374/Redis-main-runtime/src/redis-server > artifacts/redis-libraries.txt
gcc --version > artifacts/gcc-version.txt
rustc --version > artifacts/rust-version.txt
echo BUILD_CURRENT_COMPLETE
