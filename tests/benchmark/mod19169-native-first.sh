#!/usr/bin/env bash
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${RUNNER_TEMP:?}"
: "${ARCH:?}"
[[ "$ARCH" == x86_64 ]] || { echo "Two-sum experiment requires the approved Intel runner" >&2; exit 2; }
export ROOT="$GITHUB_WORKSPACE"
cd "$ROOT"
results_dir="$ROOT/native-first-results"
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
    cache_path="$RUNNER_TEMP/${case_id}-native-first-reference.bin"
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
        for mode in baseline candidate; do
            case "$mode" in baseline) value=0 ;; candidate) value=1 ;; esac
            build_dir="$RUNNER_TEMP/mod19169-native-first-$mode"
            cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMOD19169_NATIVE_VALIDATION=ON \
                -DCMAKE_CXX_FLAGS="-DMOD19169_NATIVE_FP16=1 -DMOD19169_TWO_SUMS=$value"
            cmake --build "$build_dir" --target bm_mod19169_native_first --parallel "$(nproc)"
            sha256sum "$build_dir/benchmark/bm_mod19169_native_first" >> "$results_dir/binaries.sha256"
            cp "$build_dir/CMakeCache.txt" "$results_dir/cmake-$mode.txt"
        done
        ;;
    check)
        build_dir="$RUNNER_TEMP/mod19169-native-first-candidate"
        cmake --build "$build_dir" --target test_spaces test_components test_hnsw_sq8 --parallel "$(nproc)"
        if [[ "$ARCH" == x86_64 ]]; then
            python3 - <<'CHECK_CPU'
from pathlib import Path
assert "avx512_fp16" in Path("/proc/cpuinfo").read_text(), "Native FP16 must execute on this Intel runner"
CHECK_CPU
        fi
        for target in test_spaces test_components test_hnsw_sq8; do
            "$build_dir/unit_tests/$target" --gtest_output="xml:$results_dir/unit-$target.xml" \
                > "$results_dir/unit-$target.log" 2>&1
        done
        ctest --test-dir "$build_dir" -R "^tier_linkage$" --output-on-failure --no-tests=error \
            > "$results_dir/tier-linkage.log" 2>&1
        if [[ "$ARCH" == x86_64 ]]; then
            python3 - "$results_dir/unit-test_spaces.xml" <<'CHECK_NATIVE'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = root.findall('.//testsuite[@name="SQ8FP16NativeIPTest"]/testcase')
assert len(cases) == 7, "All seven native FP16 tests must execute"
assert all(case.get("status") == "run" and case.find("skipped") is None and case.find("failure") is None for case in cases)
print("Native FP16 executed cases:", len(cases))
CHECK_NATIVE
        fi
        ;;
    query)
        query_cpu=$(cat "$results_dir/query-cpu.txt")
        for dataset in single multi; do
            set_case "$dataset"
            test ! -e "$cache_path"
            "$RUNNER_TEMP/mod19169-native-first-baseline/benchmark/bm_mod19169_native_first" \
                --stage reference "${common_args[@]}" --graph "$saved_graph" \
                --output "$results_dir/${case_id}-reference.json" \
                > "$results_dir/${case_id}-reference.log" 2>&1
            sha256sum "$cache_path" >> "$results_dir/references.sha256"
            cp "$cache_path" "$results_dir/${case_id}-reference.bin"
            for run in baseline-a baseline-b candidate-a candidate-b baseline-c; do
                mode="${run%-*}"
                output_path="$results_dir/${case_id}-graph-saved-query-${run}_results.json"
                taskset -c "$query_cpu" "$RUNNER_TEMP/mod19169-native-first-$mode/benchmark/bm_mod19169_native_first" \
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
