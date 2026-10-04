#!/usr/bin/env bash
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${RUNNER_TEMP:?}"
: "${ARCH:?}"
export ROOT="$GITHUB_WORKSPACE"
cd "$ROOT"
results_dir="$ROOT/three-query-modes-results"
mkdir -p "$results_dir"

set_case() {
    dataset="$1"
    if [[ "$dataset" == single ]]; then
        prefix=dbpedia-cosine-dim768
    else
        prefix=fashion_images_multi_value-cosine-dim512
    fi
    source_path="$ROOT/tests/benchmark/data/${prefix}-M64-efc512-fp16.hnsw_v3"
    queries_path="$ROOT/tests/benchmark/data/${prefix}-fp16-test_vectors.raw"
    saved_graph="$ROOT/tests/benchmark/data/${prefix}-M64-efc512-fp16-sq8.hnsw_v5"
    case_id="fp16-${dataset}-full"
    cache_path="$RUNNER_TEMP/${case_id}-three-modes-reference.bin"
    common_args=(--dtype fp16 --dataset "$dataset" --corpus full
        --source "$source_path" --queries "$queries_path" --cache "$cache_path")
}

case "${1:-}" in
    build)
        git rev-parse HEAD > "$results_dir/source.txt"
        lscpu > "$results_dir/cpu.txt"
        python3 -c 'import os; print(min(os.sched_getaffinity(0)))' > "$results_dir/query-cpu.txt"
        for dataset in single multi; do
            set_case "$dataset"
            sha256sum "$source_path" "$queries_path" "$saved_graph" >> "$results_dir/inputs.sha256"
        done
        for mode in baseline widen quantized; do
            case "$mode" in baseline) value=0 ;; widen) value=2 ;; quantized) value=1 ;; esac
            build_dir="$RUNNER_TEMP/mod19169-three-$mode"
            cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMOD19169_VALIDATION=ON \
                -DCMAKE_CXX_FLAGS="-DMOD19169_SQ8_QUERY=$value"
            cmake --build "$build_dir" --target bm_mod19169_validation test_sq8_query_modes --parallel "$(nproc)"
            sha256sum "$build_dir/benchmark/bm_mod19169_validation" "$build_dir/unit_tests/test_sq8_query_modes" >> "$results_dir/binaries.sha256"
            cp "$build_dir/CMakeCache.txt" "$results_dir/cmake-$mode.txt"
        done
        ;;
    check)
        for mode in baseline widen quantized; do
            "$RUNNER_TEMP/mod19169-three-$mode/unit_tests/test_sq8_query_modes" \
                --gtest_output="xml:$results_dir/unit-$mode.xml" \
                > "$results_dir/unit-$mode.log" 2>&1
        done
        ;;
    query)
        query_cpu=$(cat "$results_dir/query-cpu.txt")
        for dataset in single multi; do
            set_case "$dataset"
            test ! -e "$cache_path"
            "$RUNNER_TEMP/mod19169-three-baseline/benchmark/bm_mod19169_validation" \
                --stage reference "${common_args[@]}" --graph "$saved_graph" \
                --output "$results_dir/${case_id}-reference.json" \
                > "$results_dir/${case_id}-reference.log" 2>&1
            sha256sum "$cache_path" >> "$results_dir/references.sha256"
            for run in baseline-a baseline-b widen-a widen-b quantized-a quantized-b baseline-c; do
                mode="${run%-*}"
                output_path="$results_dir/${case_id}-graph-saved-query-${run}_results.json"
                taskset -c "$query_cpu" "$RUNNER_TEMP/mod19169-three-$mode/benchmark/bm_mod19169_validation" \
                    --stage query "${common_args[@]}" --graph "$saved_graph" \
                    --output "$results_dir/${case_id}-graph-saved-query-${run}-provenance.json" \
                    --benchmark_out_format=json --benchmark_out="$output_path" \
                    > "$results_dir/${case_id}-graph-saved-query-${run}.log" 2>&1
                python3 - "$output_path" <<'CHECK_RESULTS'
import json
import math
import sys
from pathlib import Path
path = Path(sys.argv[1])
result = json.loads(path.read_text())
report = json.loads(path.with_name(path.name.replace("_results.json", "-provenance.json")).read_text())
assert report["dtype"] == "fp16" and report["corpus"] == "full"
assert report["query_count"] == 1000 and report["k"] == 100
assert not any(row.get("error_occurred") for row in result["benchmarks"])
samples = [row for row in result["benchmarks"] if row.get("run_type", "iteration") == "iteration"]
assert len(samples) == 9
for row in samples:
    assert row["iterations"] == 1000 and row["time_unit"] == "ms"
    assert row["query_mode"] == report["query_mode"]
    assert row["label"] == "reference=FP32_BF_over_original_typed_values; query includes preprocessing"
    assert 0 <= row["FP32_BF_boundary_tied_queries"] <= 1000
    assert math.isfinite(row["Recall_vs_FP32_BF"]) and 0 <= row["Recall_vs_FP32_BF"] <= 1
    assert math.isfinite(row["real_time"]) and row["real_time"] > 0
    assert math.isfinite(row["cpu_time"]) and row["cpu_time"] > 0
CHECK_RESULTS
            done
        done
        sha256sum --check "$results_dir/inputs.sha256"
        ;;
    *)
        echo "usage: $0 {build|check|query}" >&2
        exit 2
        ;;
esac
