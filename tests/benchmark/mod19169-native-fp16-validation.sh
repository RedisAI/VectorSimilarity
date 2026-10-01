#!/usr/bin/env bash
set -euo pipefail

export ROOT="${GITHUB_WORKSPACE:?}"
: "${RUNNER_TEMP:?}"
: "${ARCH:?}"
cd "$ROOT"
results_dir="$ROOT/native-fp16-results"
mkdir -p "$results_dir"

case "${1:-}" in
    build)
        git rev-parse HEAD > "$results_dir/source.txt"
        sha256sum tests/benchmark/data/*fp16*.hnsw* \
            tests/benchmark/data/*fp16-test_vectors.raw > "$results_dir/inputs.sha256"
        lscpu > "$results_dir/cpu.txt"
        python3 -c 'import os; print(min(os.sched_getaffinity(0)))' > "$results_dir/query-cpu.txt"

        for mode in baseline candidate; do
            value=0
            if [[ "$mode" == candidate ]]; then value=1; fi
            build_dir="$RUNNER_TEMP/mod19169-native-fp16-$mode"
            cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                -DCMAKE_CXX_FLAGS="-DMOD19169_NATIVE_FP16=$value -DMOD19169_RECALL_VALIDATION=1"
            targets=(bm_basics_single_fp16 bm_basics_multi_fp16)
            if [[ "$ARCH" == x86_64 && "$mode" == candidate ]]; then
                targets+=(test_spaces test_hnsw_sq8 test_components)
            fi
            cmake --build "$build_dir" --target "${targets[@]}" --parallel "$(nproc)"
            sha256sum "$build_dir/benchmark/bm_basics_single_fp16" \
                "$build_dir/benchmark/bm_basics_multi_fp16" >> "$results_dir/binaries.sha256"
            if [[ "$ARCH" == x86_64 && "$mode" == candidate ]]; then
                sha256sum "$build_dir/unit_tests/test_spaces" \
                    "$build_dir/unit_tests/test_hnsw_sq8" \
                    "$build_dir/unit_tests/test_components" >> "$results_dir/binaries.sha256"
            fi
        done
        ;;
    unit)
        if [[ "$ARCH" != x86_64 ]]; then
            exit 0
        fi
        if ! grep -Eqm1 '(^|[[:space:]])avx512_fp16([[:space:]]|$)' /proc/cpuinfo; then
            echo "Intel validation runner does not expose AVX512FP16" >&2
            exit 1
        fi
        build_dir="$RUNNER_TEMP/mod19169-native-fp16-candidate"
        failed=0
        for target in test_spaces test_hnsw_sq8 test_components; do
            if ! "$build_dir/unit_tests/$target" \
                --gtest_output="xml:$results_dir/unit-$target.xml" \
                > "$results_dir/unit-$target.log" 2>&1; then
                failed=1
            fi
            if [[ ! -s "$results_dir/unit-$target.xml" ]]; then
                failed=1
            fi
        done
        python3 - "$results_dir/unit-test_spaces.xml" <<'CHECK_NATIVE_TESTS'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = root.findall('.//testsuite[@name="SQ8FP16NativeIPTest"]/testcase')
assert cases, "SQ8FP16NativeIPTest did not execute"
for case in cases:
    assert case.get("status") == "run", ET.tostring(case, encoding="unicode")
    assert case.get("result") == "completed", ET.tostring(case, encoding="unicode")
    assert case.find("skipped") is None, ET.tostring(case, encoding="unicode")
CHECK_NATIVE_TESTS
        exit "$failed"
        ;;
    query)
        query_cpu=$(cat "$results_dir/query-cpu.txt")
        for dataset in single multi; do
            export MOD19169_GT_CACHE="$RUNNER_TEMP/mod19169-fp32bf-over-fp16-$dataset.bin"
            test ! -e "$MOD19169_GT_CACHE"
            for run in baseline-a baseline-b candidate-a candidate-b baseline-c; do
                mode="${run%-*}"
                binary="$RUNNER_TEMP/mod19169-native-fp16-$mode/benchmark/bm_basics_${dataset}_fp16"
                taskset -c "$query_cpu" "$binary" \
                    --benchmark_filter='TopK_HNSW_SQ8' --benchmark_repetitions=3 \
                    --benchmark_out_format=json \
                    --benchmark_out="$results_dir/${dataset}-${run}_results.json"
                python3 - "$results_dir/${dataset}-${run}_results.json" <<'CHECK_RESULTS'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    result = json.load(result_file)
rows = result["benchmarks"]
assert not any(row.get("error_occurred") for row in rows), rows
samples = [row for row in rows if row.get("run_type", "iteration") == "iteration"]
assert len(samples) == 9, len(samples)
for row in samples:
    assert row["iterations"] == 1000, row
    assert row.get("label") == "reference=FP32_BF_over_FP16_values", row
    assert 0 <= row["FP32_BF_boundary_tied_queries"] <= 1000, row
    recall = row["Recall_vs_FP32_BF_over_FP16_values"]
    assert math.isfinite(recall) and 0 <= recall <= 1, row
    assert math.isfinite(row["real_time"]) and row["real_time"] > 0, row
    assert math.isfinite(row["cpu_time"]) and row["cpu_time"] > 0, row
CHECK_RESULTS
            done
            sha256sum "$MOD19169_GT_CACHE" >> "$results_dir/reference.sha256"
        done
        sha256sum --check "$results_dir/inputs.sha256"
        ;;
    *)
        echo "usage: $0 {build|unit|query}" >&2
        exit 2
        ;;
esac
