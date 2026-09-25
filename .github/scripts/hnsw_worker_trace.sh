#!/usr/bin/env bash
set -euo pipefail

export ROOT="$PWD"
mkdir -p worker-trace-results
git rev-parse HEAD > worker-trace-results/revision.txt
git submodule status --recursive > worker-trace-results/submodules.txt
cmake -S . -B build-worker-trace -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g" \
    2>&1 | tee worker-trace-results/configure.log | tail -40
cmake --build build-worker-trace --target test_hnsw test_hnsw_sq8 --parallel 2 \
    2>&1 | tee worker-trace-results/build.log | tail -40
build-worker-trace/unit_tests/test_hnsw --gtest_filter='DISABLED_HNSWWorkerTrace.*' \
    --gtest_also_run_disabled_tests \
    --gtest_output=xml:worker-trace-results/results.xml \
    2>&1 | tee worker-trace-results/test.log

# Broaden validation only after the exact FIFO reproduction is green.
build-worker-trace/unit_tests/test_hnsw \
    --gtest_output=xml:worker-trace-results/hnsw-suite.xml \
    2>&1 | tee worker-trace-results/hnsw-suite.log | tail -60
build-worker-trace/unit_tests/test_hnsw_sq8 \
    --gtest_output=xml:worker-trace-results/sq8-suite.xml \
    2>&1 | tee worker-trace-results/sq8-suite.log | tail -60

# Exercise the schedule-sensitive insert/delete failure beyond the ordinary suite run.
build-worker-trace/unit_tests/test_hnsw \
    --gtest_filter='HNSWTieredIndexTest/*.alternateInsertDeleteAsync' --gtest_repeat=20 \
    --gtest_output=xml:worker-trace-results/insert-delete-repeat.xml \
    2>&1 | tee worker-trace-results/insert-delete-repeat.log | tail -60
