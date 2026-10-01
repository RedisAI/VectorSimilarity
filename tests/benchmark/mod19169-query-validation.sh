#!/usr/bin/env bash
set -euo pipefail
export ROOT="$GITHUB_WORKSPACE"
cd "$ROOT"
mkdir -p recall-results

git rev-parse HEAD > recall-results/source.txt
sha256sum tests/benchmark/data/*fp16*.hnsw* \
    tests/benchmark/data/*fp16-test_vectors.raw > recall-results/inputs.sha256
lscpu > recall-results/cpu.txt

for mode in baseline candidate; do
    value=0
    if [[ "$mode" == candidate ]]; then value=1; fi
    build_dir="$RUNNER_TEMP/mod19169-$mode"
    cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_FLAGS="-DMOD19169_SQ8_QUERY=$value -DMOD19169_RECALL_VALIDATION=1"
    cmake --build "$build_dir" --target bm_basics_single_fp16 bm_basics_multi_fp16 \
        --parallel "$(nproc)"
    sha256sum "$build_dir/benchmark/bm_basics_single_fp16" \
        "$build_dir/benchmark/bm_basics_multi_fp16" >> recall-results/binaries.sha256
done

query_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
echo "$query_cpu" > recall-results/query-cpu.txt

for dataset in single multi; do
    export MOD19169_GT_CACHE="$RUNNER_TEMP/mod19169-$dataset-reference.bin"
    test ! -e "$MOD19169_GT_CACHE"
    # Bracket the candidate with the identical baseline binary to expose time drift.
    for run in baseline-a baseline-b candidate-a candidate-b baseline-c; do
        mode="${run%-*}"
        binary="$RUNNER_TEMP/mod19169-$mode/benchmark/bm_basics_${dataset}_fp16"
        taskset -c "$query_cpu" "$binary" --benchmark_filter='TopK_HNSW_SQ8' --benchmark_repetitions=3 \
            --benchmark_out_format=json \
            --benchmark_out="recall-results/${dataset}-${run}_results.json"
        python3 - "recall-results/${dataset}-${run}_results.json" <<'CHECK_RESULTS'
import json, math, sys
with open(sys.argv[1]) as result_file:
    result = json.load(result_file)
rows = result['benchmarks']
assert not any(row.get('error_occurred') for row in rows), rows
samples = [row for row in rows if row.get('run_type', 'iteration') == 'iteration']
assert len(samples) == 9, len(samples)
for row in samples:
    assert row['iterations'] == 1000, row
    assert 0 <= row['BF_boundary_tied_queries'] <= 1000, row
    assert 0 <= row['Recall_vs_FP16_BF_IDs'] <= 1, row
    assert math.isfinite(row['real_time']) and math.isfinite(row['cpu_time']), row
CHECK_RESULTS
    done
    sha256sum "$MOD19169_GT_CACHE" >> recall-results/reference.sha256
done
sha256sum --check recall-results/inputs.sha256
