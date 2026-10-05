#!/usr/bin/env bash
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${RUNNER_TEMP:?}"
: "${ARCH:?}"
: "${PRODUCTION_SOURCE:?}"
[[ "$PRODUCTION_SOURCE" =~ ^[0-9a-f]{40}$ ]]
[[ "$ARCH" == x86_64 || "$ARCH" == arm64 ]]
cd "$GITHUB_WORKSPACE/source"
results_dir="$GITHUB_WORKSPACE/final-native-check-results"
build_dir="$RUNNER_TEMP/mod19169-final-production"
mkdir -p "$results_dir"
git rev-parse HEAD > "$results_dir/source.txt"
[[ "$(cat "$results_dir/source.txt")" == "$PRODUCTION_SOURCE" ]]
git rev-parse 'HEAD^{tree}' > "$results_dir/source-tree.txt"
git status --porcelain > "$results_dir/source-status.txt"
test ! -s "$results_dir/source-status.txt"
lscpu > "$results_dir/cpu.txt"
printf '%s\n' "$ARCH" > "$results_dir/architecture.txt"
python3 - "$ARCH" <<'CPU'
import platform
import sys
from pathlib import Path
expected = 'x86_64' if sys.argv[1] == 'x86_64' else 'aarch64'
assert platform.machine() == expected
if expected == 'x86_64':
    assert 'avx512_fp16' in Path('/proc/cpuinfo').read_text(), 'Native FP16 must execute'
CPU
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo > "$results_dir/configure.log" 2>&1
cp "$build_dir/CMakeCache.txt" "$results_dir/CMakeCache.txt"
cmake --build "$build_dir" --target test_spaces test_components test_hnsw_sq8 --parallel "$(nproc)" \
    > "$results_dir/build.log" 2>&1
for target in test_spaces test_components test_hnsw_sq8; do
    sha256sum "$build_dir/unit_tests/$target" >> "$results_dir/binaries.sha256"
    "$build_dir/unit_tests/$target" --gtest_output="xml:$results_dir/unit-$target.xml" \
        > "$results_dir/unit-$target.log" 2>&1
done
ctest --test-dir "$build_dir" -R '^tier_linkage$' --output-on-failure --no-tests=error \
    > "$results_dir/tier-linkage.log" 2>&1
python3 - "$results_dir" "$ARCH" <<'VERIFY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
folder, arch = Path(sys.argv[1]), sys.argv[2]
summary = {'source': (folder / 'source.txt').read_text().strip(), 'architecture': arch, 'targets': {}}
for target in ('test_spaces', 'test_components', 'test_hnsw_sq8'):
    root = ET.parse(folder / f'unit-{target}.xml').getroot()
    cases = root.findall('.//testcase')
    assert cases and all(case.find('failure') is None for case in cases)
    summary['targets'][target] = {'total': len(cases), 'passed': sum(case.get('status') == 'run' and case.find('skipped') is None for case in cases), 'skipped': sum(case.find('skipped') is not None for case in cases), 'disabled': sum(case.get('status') != 'run' for case in cases)}
    assert summary['targets'][target]['disabled'] == 0
    if target == 'test_spaces':
        native = root.findall('.//testsuite[@name="SQ8FP16NativeIPTest"]/testcase')
        assert len(native) == (7 if arch == 'x86_64' else 0)
        assert all(case.get('status') == 'run' and case.find('skipped') is None and case.find('failure') is None for case in native)
        summary['native_executed'] = len(native)
cache = (folder / 'CMakeCache.txt').read_text()
assert 'MOD19169' not in cache, 'Production build must not contain experiment switches'
assert '100% tests passed' in (folder / 'tier-linkage.log').read_text()
(folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
VERIFY
