#!/usr/bin/env bash
set -euo pipefail

# Keep the test source identical in both builds; only the production header changes.
baseline=ec835be8c1ea758b91e94c459bde483cb46690f5
header=src/VecSim/algorithms/hnsw/hnsw.h
export ROOT="$PWD"
mkdir -p repair-results
git rev-parse HEAD > repair-results/proposed-revision.txt
git rev-parse "$baseline" > repair-results/baseline-revision.txt
trap 'git restore --source=HEAD -- "$header"' EXIT

git restore --source="$baseline" -- "$header"
cmake -S . -B build-repair -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g" -DUSE_SVS=OFF \
    2>&1 | tee repair-results/configure.log | tail -40
cmake --build build-repair --target test_hnsw --parallel 2 \
    2>&1 | tee repair-results/baseline-build.log | tail -40

set +e
build-repair/unit_tests/test_hnsw --gtest_filter='HNSWRepairChains.*' \
    --gtest_output=xml:repair-results/baseline.xml \
    2>&1 | tee repair-results/baseline.log
baseline_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$baseline_status" > repair-results/baseline-exit.txt
if [[ "$baseline_status" != 1 ]]; then
    echo "Expected regression assertion failures on the baseline, got exit $baseline_status"
    exit 1
fi

git restore --source=HEAD -- "$header"
cmake --build build-repair --target test_hnsw test_hnsw_sq8 --parallel 2 \
    2>&1 | tee repair-results/proposed-build.log | tail -40
set +e
build-repair/unit_tests/test_hnsw --gtest_filter='HNSWRepairChains.*' \
    --gtest_output=xml:repair-results/proposed.xml \
    2>&1 | tee repair-results/proposed.log
proposed_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$proposed_status" > repair-results/proposed-exit.txt

python3 - <<'PY'
import json
import xml.etree.ElementTree as ET
from pathlib import Path

results = {}
for variant in ("baseline", "proposed"):
    root = ET.parse(f"repair-results/{variant}.xml").getroot()
    cases = {}
    for case in root.iter("testcase"):
        assert case.get("status") == "run", ET.tostring(case, encoding="unicode")
        cases[case.get("name")] = {
            "passed": not list(case.iter("failure")),
            "properties": {p.get("name"): p.get("value") for p in case.iter("property")},
        }
    assert cases, f"No regression tests executed for {variant}"
    results[variant] = cases
assert results["baseline"].keys() == results["proposed"].keys()
Path("repair-results/comparison.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
for name, initial_count in (
    ("AdjacentReplacementsRemainFullyReachable", 40),
    ("BatchedFiveThousandVectorReplacementsRecoverAfterIdleAndGC", 5000),
    ("PendingFiveThousandVectorReplacementJobsPreserveReachability", 5000),
):
    assert name in results["baseline"], f"Missing required reproduction: {name}"
    assert not results["baseline"][name]["passed"], f"Baseline did not reproduce {name}"
    for variant in results:
        properties = results[variant][name]["properties"]
        assert int(properties["initial_count"]) == initial_count, (variant, name, properties)
        assert "final_count_before_gc" in properties and "final_count_after_gc" in properties
        if name.startswith("PendingFiveThousand"):
            assert int(properties["pending_callback_count"]) == 286, (variant, properties)
PY

if [[ "$proposed_status" != 0 ]]; then
    exit "$proposed_status"
fi

# The shared HNSW repair code also serves ordinary and SQ8 tiered indexes.
build-repair/unit_tests/test_hnsw --gtest_filter='-HNSWRepairChains.*' \
    --gtest_output=xml:repair-results/hnsw-suite.xml \
    2>&1 | tee repair-results/hnsw-suite.log | tail -60
build-repair/unit_tests/test_hnsw_sq8 --gtest_output=xml:repair-results/sq8-suite.xml \
    2>&1 | tee repair-results/sq8-suite.log | tail -60
