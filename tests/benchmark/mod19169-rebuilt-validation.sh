#!/usr/bin/env bash
set -euo pipefail
export ROOT="${GITHUB_WORKSPACE:?}"
: "${RUNNER_TEMP:?}"
: "${ARCH:?}"
cd "$ROOT"
results_dir="$ROOT/rebuilt-validation-results"
mkdir -p "$results_dir"

set_case() {
    dtype="$1"
    dataset="$2"
    corpus="$3"
    suffix=""
    if [[ "$dtype" == fp16 ]]; then suffix="-fp16"; fi
    if [[ "$dataset" == single ]]; then
        prefix="dbpedia-cosine-dim768"
    else
        prefix="fashion_images_multi_value-cosine-dim512"
    fi
    source_path="$ROOT/tests/benchmark/data/${prefix}-M64-efc512${suffix}.hnsw_v3"
    queries_path="$ROOT/tests/benchmark/data/${prefix}${suffix}-test_vectors.raw"
    saved_graph="$ROOT/tests/benchmark/data/${prefix}-M64-efc512${suffix}-sq8.hnsw_v5"
    case_id="${dtype}-${dataset}-${corpus}"
    cache_path="$RUNNER_TEMP/${case_id}-reference.bin"
    common_args=(--dtype "$dtype" --dataset "$dataset" --corpus "$corpus"
        --source "$source_path" --queries "$queries_path" --cache "$cache_path")
}

run_reference() {
    test ! -e "$cache_path"
    "$RUNNER_TEMP/mod19169-rebuilt-baseline/benchmark/bm_mod19169_validation" \
        --stage reference "${common_args[@]}" --graph "$saved_graph" \
        --output "$results_dir/${case_id}-reference.json" \
        > "$results_dir/${case_id}-reference.log" 2>&1
    sha256sum "$cache_path" >> "$results_dir/references.sha256"
}

query_graph() {
    graph_mode="$1"
    graph_path="$2"
    query_cpu=$(cat "$results_dir/query-cpu.txt")
    for run in baseline-a baseline-b candidate-a candidate-b baseline-c; do
        mode="${run%-*}"
        binary="$RUNNER_TEMP/mod19169-rebuilt-$mode/benchmark/bm_mod19169_validation"
        output_path="$results_dir/${case_id}-graph-${graph_mode}-query-${run}_results.json"
        taskset -c "$query_cpu" "$binary" --stage query "${common_args[@]}" \
            --graph "$graph_path" --output "$results_dir/${case_id}-graph-${graph_mode}-query-${run}-provenance.json" \
            --benchmark_out_format=json --benchmark_out="$output_path" \
            > "$results_dir/${case_id}-graph-${graph_mode}-query-${run}.log" 2>&1
        python3 - "$output_path" <<'CHECK_RESULTS'
import json
import math
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    result = json.load(f)
rows = result["benchmarks"]
assert not any(row.get("error_occurred") for row in rows), rows
samples = [r for r in rows if r.get("run_type", "iteration") == "iteration"]
assert len(samples) == 9, len(samples)
for r in samples:
    assert r["iterations"] == 1000, r
    assert r["time_unit"] == "ms", r
    assert r["label"] == "reference=FP32_BF_over_original_typed_values; query includes preprocessing", r
    assert 0 <= r["FP32_BF_boundary_tied_queries"] <= 1000, r
    assert math.isfinite(r["Recall_vs_FP32_BF"]) and 0 <= r["Recall_vs_FP32_BF"] <= 1, r
    assert math.isfinite(r["real_time"]) and r["real_time"] > 0, r
    assert math.isfinite(r["cpu_time"]) and r["cpu_time"] > 0, r
CHECK_RESULTS
    done
}

case "${1:-}" in
    build)
        git rev-parse HEAD > "$results_dir/source.txt"
        lscpu > "$results_dir/cpu.txt"
        python3 -c 'import os; print(min(os.sched_getaffinity(0)))' > "$results_dir/query-cpu.txt"
        for dtype in fp16 fp32; do
            for dataset in single multi; do
                set_case "$dtype" "$dataset" full
                sha256sum "$source_path" "$queries_path" "$saved_graph" >> "$results_dir/inputs.sha256"
            done
        done
        for mode in baseline candidate; do
            value=0
            if [[ "$mode" == candidate ]]; then value=1; fi
            build_dir="$RUNNER_TEMP/mod19169-rebuilt-$mode"
            cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMOD19169_VALIDATION=ON \
                -DCMAKE_CXX_FLAGS="-DMOD19169_SQ8_QUERY=$value"
            cmake --build "$build_dir" --target bm_mod19169_validation --parallel "$(nproc)"
            sha256sum "$build_dir/benchmark/bm_mod19169_validation" >> "$results_dir/binaries.sha256"
        done
        ;;
    rebuilt)
        for dtype in fp16 fp32; do
            for dataset in single multi; do
                set_case "$dtype" "$dataset" subset
                run_reference
                for graph_mode in baseline candidate; do
                    graph_path="$RUNNER_TEMP/${case_id}-${graph_mode}.hnsw"
                    binary="$RUNNER_TEMP/mod19169-rebuilt-$graph_mode/benchmark/bm_mod19169_validation"
                    /usr/bin/time -v -o "$results_dir/${case_id}-build-${graph_mode}-time.txt" \
                        "$binary" --stage build "${common_args[@]}" --graph "$graph_path" \
                        --output "$results_dir/${case_id}-build-${graph_mode}.json" \
                        > "$results_dir/${case_id}-build-${graph_mode}.log" 2>&1
                    sha256sum "$graph_path" >> "$results_dir/graphs.sha256"
                    query_graph "$graph_mode" "$graph_path"
                done
            done
        done
        sha256sum --check "$results_dir/inputs.sha256"
        ;;
    saved)
        for dataset in single multi; do
            set_case fp32 "$dataset" full
            run_reference
            query_graph saved "$saved_graph"
        done
        sha256sum --check "$results_dir/inputs.sha256"
        ;;
    *)
        echo "usage: $0 {build|rebuilt|saved}" >&2
        exit 2
        ;;
esac
