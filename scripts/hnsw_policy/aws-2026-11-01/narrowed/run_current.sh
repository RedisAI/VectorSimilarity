#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/mod18264
export MOD18264_REPO_ROOT=/home/ubuntu/mod18264/source
export REDIS_SERVER=/home/ubuntu/mod-18374/Redis-main-runtime/src/redis-server
export REDIS_CLI=/home/ubuntu/mod-18374/Redis-main-runtime/src/redis-cli
export RSV_ROOT=/home/ubuntu/mod18264/narrowed/rsv
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
names=()
cleanup() {
  for name in "${names[@]}"; do ./rsv.sh --name "$name" stop; done
}
on_exit() {
  local rc=$?
  if (( rc == 0 )); then cleanup; else
    echo "Run failed ($rc); preserving owned instances for diagnosis: ${names[*]}"
  fi
}
trap on_exit EXIT
for history in fresh holes; do
  ports=()
  for build in baseline candidate; do
    name="narrowed-$history-$build"
    names+=("$name")
    ./rsv.sh --name "$name" cluster-start --shards 2 \
      --module "$PWD/narrowed/artifacts/$build.so" --json-path "$PWD/narrowed/artifacts/rejson.so" \
      --modargs 'WORKERS 0 MIN_OPERATION_WORKERS 0 DEFAULT_DIALECT 2'
    mapfile -t nodeports < <(./rsv.sh --name "$name" ports)
    ports+=("${nodeports[@]}")
  done
  printf '%s\n' "${ports[@]}" > "narrowed/$history-ports.txt"
  args=(--ports "${ports[@]}" --root "$PWD/narrowed" --history "$history")
  venv/bin/python narrowed/compare_current.py load "${args[@]}" 2>&1 | tee "narrowed/logs/load-$history.log" | tail -15
  venv/bin/python narrowed/compare_current.py measure "${args[@]}" 2>&1 | tee "narrowed/logs/measure-$history.log" | tail -15
  cleanup
  names=()
done
echo AWS_COMPARISON_COMPLETE
