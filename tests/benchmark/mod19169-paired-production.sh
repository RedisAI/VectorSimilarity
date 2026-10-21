#!/usr/bin/env bash
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${RUNNER_TEMP:?}"
: "${PRODUCTION_SOURCE:?}"
: "${ARCH:?}"
[[ "$ARCH" == x86_64 ]]
harness="$GITHUB_WORKSPACE/harness"
source_root="$GITHUB_WORKSPACE/source"
results_dir="$GITHUB_WORKSPACE/paired-production-results"
export ROOT="$source_root"
mkdir -p "$results_dir"

set_case() {
    dataset="$1"
    if [[ "$dataset" == single ]]; then
        prefix=dbpedia-cosine-dim768
    else
        prefix=fashion_images_multi_value-cosine-dim512
    fi
    data="$harness/tests/benchmark/data"
    source_path="$data/${prefix}-M64-efc512-fp16.hnsw_v3"
    queries_path="$data/${prefix}-fp16-test_vectors.raw"
    saved_graph="$data/${prefix}-M64-efc512-fp16-sq8.hnsw_v5"
    case_id="fp16-${dataset}-full"
    cache_path="$RUNNER_TEMP/${case_id}-paired-production-reference.bin"
    common_args=(--dtype fp16 --dataset "$dataset" --corpus full
        --source "$source_path" --queries "$queries_path" --cache "$cache_path")
}

case "${1:-}" in
build)
    [[ "$(git -C "$source_root" rev-parse HEAD)" == "$PRODUCTION_SOURCE" ]]
    git -C "$source_root" status --porcelain > "$results_dir/native-source-status.txt"
    test ! -s "$results_dir/native-source-status.txt"
    printf '%s\n' "$PRODUCTION_SOURCE" > "$results_dir/production-source.txt"
    git -C "$source_root" rev-parse 'HEAD^{tree}' > "$results_dir/production-tree.txt"
    git -C "$harness" rev-parse HEAD > "$results_dir/harness-source.txt"
    lscpu > "$results_dir/cpu.txt"
    ps -eo pid,pcpu,args --sort=-pcpu | sed -n '1,20p' > "$results_dir/processes.txt"
    python3 - <<'CPU' > "$results_dir/query-cpu.txt"
import os
from pathlib import Path
assert 'avx512_fp16' in Path('/proc/cpuinfo').read_text(), 'AVX-512 FP16 must execute'
print(min(os.sched_getaffinity(0)))
CPU
    for dataset in single multi; do
        set_case "$dataset"
        sha256sum "$source_path" "$queries_path" "$saved_graph" >> "$results_dir/inputs.sha256"
    done
    baseline="$RUNNER_TEMP/mod19169-paired-mixed-source"
    git -C "$source_root" worktree add --detach "$baseline" "$PRODUCTION_SOURCE"
    git -C "$baseline" submodule update --init --recursive --depth 1
    python3 - "$baseline/src/VecSim/spaces/IP_space.cpp" <<'CONTROL'
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = p.read_text()
old = 'if (dim >= spaces::SQ8_FP16_NATIVE_MIN_DIM && dim <= spaces::FP16_MAX_UNIT_IP_SIMD_DIM &&'
assert s.count(old) == 2, 'Only the IP and cosine native dispatch branches may change'
p.write_text(s.replace(old, old.replace('if (', 'if (false && ', 1)))
CONTROL
    git -C "$baseline" diff -- src/VecSim/spaces/IP_space.cpp > "$results_dir/mixed-control.patch"
    [[ "$(git -C "$baseline" diff --name-only)" == src/VecSim/spaces/IP_space.cpp ]]
    for mode in mixed native; do
        if [[ "$mode" == mixed ]]; then tree="$baseline"; value=0; else tree="$source_root"; value=1; fi
        build_dir="$RUNNER_TEMP/mod19169-paired-$mode"
        cmake -S "$harness/tests/benchmark/paired_production" -B "$build_dir" \
            -DSOURCE_ROOT="$tree" -DPAIRED_NATIVE_MODE="$value" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            > "$results_dir/configure-$mode.log" 2>&1
        cmake --build "$build_dir" --target bm_mod19169_paired --parallel "$(nproc)" \
            > "$results_dir/build-$mode.log" 2>&1
        cp "$build_dir/CMakeCache.txt" "$results_dir/CMakeCache-$mode.txt"
        sha256sum "$build_dir/bm_mod19169_paired" >> "$results_dir/binaries.sha256"
        sha256sum "$tree/src/VecSim/spaces/IP_space.cpp" >> "$results_dir/dispatch-sources.sha256"
    done
    git -C "$source_root" status --porcelain > "$results_dir/native-source-status-after.txt"
    test ! -s "$results_dir/native-source-status-after.txt"
    ;;
query)
    query_cpu=$(cat "$results_dir/query-cpu.txt")
    for dataset in single multi; do
        set_case "$dataset"
        test ! -e "$cache_path"
        taskset -c "$query_cpu" "$RUNNER_TEMP/mod19169-paired-mixed/bm_mod19169_paired" \
            --stage reference "${common_args[@]}" --graph "$saved_graph" \
            --output "$results_dir/${case_id}-reference.json" \
            > "$results_dir/${case_id}-reference.log" 2>&1
        sha256sum "$cache_path" >> "$results_dir/references.sha256"
        cp "$cache_path" "$results_dir/${case_id}-reference.bin"
        for run in mixed-a mixed-b native-a mixed-c native-b native-c mixed-d; do
            mode="${run%-*}"
            output_path="$results_dir/${case_id}-graph-saved-query-${run}_results.json"
            taskset -c "$query_cpu" "$RUNNER_TEMP/mod19169-paired-$mode/bm_mod19169_paired" \
                --stage query "${common_args[@]}" --graph "$saved_graph" \
                --output "$results_dir/${case_id}-graph-saved-query-${run}-provenance.json" \
                --benchmark_out_format=json --benchmark_out="$output_path" \
                > "$results_dir/${case_id}-graph-saved-query-${run}.log" 2>&1
            python3 - "$output_path" "$mode" <<'CHECK'
import json
import math
import sys
from pathlib import Path
path = Path(sys.argv[1])
result = json.loads(path.read_text())
report = json.loads(path.with_name(path.name.replace('_results.json', '-provenance.json')).read_text())
mode = 1 if sys.argv[2] == 'native' else 0
assert report['dtype'] == 'fp16' and report['corpus'] == 'full'
assert report['query_count'] == 1000 and report['k'] == 100 and report['query_mode'] == mode
assert not any(row.get('error_occurred') for row in result['benchmarks'])
samples = [row for row in result['benchmarks'] if row.get('run_type', 'iteration') == 'iteration']
assert len(samples) == 9
for row in samples:
    assert row['iterations'] == 1000 and row['time_unit'] == 'ms' and row['query_mode'] == mode
    assert row['label'] == 'reference=FP32_BF_over_original_typed_values; query includes preprocessing'
    assert math.isfinite(row['Recall_vs_FP32_BF']) and 0 <= row['Recall_vs_FP32_BF'] <= 1
    assert math.isfinite(row['real_time']) and row['real_time'] > 0
    assert math.isfinite(row['cpu_time']) and row['cpu_time'] > 0
CHECK
        done
    done
    sha256sum --check "$results_dir/inputs.sha256"
    sha256sum --check "$results_dir/references.sha256"
    python3 "$harness/tests/benchmark/mod19169_paired_summary.py" "$results_dir"
    echo PAIRED_SUMMARY_JSON_BEGIN
    cat "$results_dir/summary.json"
    echo PAIRED_SUMMARY_JSON_END
    ;;
*) echo "usage: $0 {build|query}" >&2; exit 2 ;;
esac
