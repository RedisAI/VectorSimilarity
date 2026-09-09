#!/usr/bin/env python3
# Copyright (c) 2006-Present, Redis Ltd. All rights reserved.
# Licensed under RSALv2, SSPLv1, or AGPLv3.

"""Branch-local SQ8 experiment: build and measure only inside GitHub Actions."""

import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import tempfile


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("This experiment may only build and benchmark in GitHub Actions")
    root = Path.cwd()
    config = json.loads((root / ".github/sq8-comparison.json").read_text())
    results = root / "sq8-comparison-results"
    results.mkdir()

    def run(args, name, cwd=root):
        args = [str(arg) for arg in args]
        print("Running:", " ".join(args), flush=True)
        with (results / (name + ".log")).open("w") as log:
            subprocess.run(args, cwd=cwd, env={**os.environ, "ROOT": str(cwd)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    baseline_sha = config["baseline"]
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_sha):
        raise ValueError("Baseline must be an exact commit SHA")
    candidate_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(["git", "fetch", "--depth=1", "origin", baseline_sha], "fetch-baseline")
    experiment = Path(tempfile.mkdtemp(prefix="sq8-paired-", dir=os.environ["RUNNER_TEMP"]))
    baseline = experiment / "baseline"
    run(["git", "clone", "--shared", "--no-checkout", root, baseline], "clone-baseline")
    run(["git", "checkout", "--detach", baseline_sha], "checkout-baseline", cwd=baseline)
    run(["git", "submodule", "update", "--init", "--recursive"], "baseline-submodules", cwd=baseline)

    # Both libraries receive the candidate's exact benchmark workload. AVX512F's new
    # direct chooser registration is unrelated to the measured dispatcher workload,
    # and cannot be compiled against the old library, which lacks that symbol.
    bench_path = Path("tests/benchmark/spaces_benchmarks/bm_spaces_sq8_fp32.cpp")
    bench_source = (root / bench_path).read_text()
    if config["experiment"] == "avx512f-dispatch":
        bench_source, count = re.subn(
            r"#ifdef OPT_AVX512F\nbool avx512f_supported = opt.avx512f;\n"
            r"INITIALIZE_BENCHMARKS_SET_L2\(.*?\);\n#endif\n",
            "", bench_source, count=1, flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError("Expected exactly one candidate-only AVX512F registration")
    (baseline / bench_path).write_text(bench_source)
    run(["git", "diff", "--", bench_path], "baseline-benchmark-overlay", cwd=baseline)

    build_dirs = {}
    for name, source in (("baseline", baseline), ("candidate", root)):
        build = experiment / (name + "-build")
        build_dirs[name] = build
        run(["cmake", "-S", source, "-B", build, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DUSE_SVS=ON",
             "-DVECSIM_BUILD_TESTS=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
            name + "-configure")
        targets = ["bm_spaces_sq8_fp32", "test_spaces"]
        if name == "candidate":
            targets.extend(["test_components", "test_hnsw_sq8"])
        run(["cmake", "--build", build, "--parallel", min(4, os.cpu_count()),
             "--target", *targets], name + "-build")
        (results / (name + "-compile-commands.json")).write_text(
            (build / "compile_commands.json").read_text())
        for target in targets[1:]:
            run([build / "unit_tests" / target,
                 "--gtest_output=xml:" + str(results / (name + "-" + target + ".xml"))],
                name + "-" + target, cwd=source)

    affinity = sorted(os.sched_getaffinity(0))
    cpu = affinity[-1]
    metadata = {
        **config, "candidate": candidate_sha, "cpu": cpu, "affinity": affinity,
        "rounds": 9, "min_time_seconds": 0.15, "build_type": "Release", "use_svs": True,
        "compiler": subprocess.check_output(["c++", "--version"], text=True),
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "run_url": "https://github.com/" + os.environ["GITHUB_REPOSITORY"] + "/actions/runs/"
        + os.environ["GITHUB_RUN_ID"],
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2))
    samples = {name: {} for name in build_dirs}
    expected_names = None
    # Warm up both binaries, then alternate AB / BA rounds to reduce time drift.
    # Compilation and correctness tests finish before any timed measurements start.
    for round_number in range(-1, metadata["rounds"]):
        order = ["baseline", "candidate"] if round_number % 2 == 0 else ["candidate", "baseline"]
        for name in order:
            output = results / f"{name}-round-{round_number}.json"
            run(["taskset", "--cpu-list", cpu,
                 build_dirs[name] / "benchmark/bm_spaces_sq8_fp32",
                 "--benchmark_filter=" + config["filter"],
                 "--benchmark_min_time=0.15", "--benchmark_repetitions=1",
                 "--benchmark_out=" + str(output), "--benchmark_out_format=json"],
                f"{name}-round-{round_number}")
            rows = json.loads(output.read_text())["benchmarks"]
            if not rows or any(row.get("error_occurred") for row in rows):
                raise RuntimeError("Benchmark selection is empty or contains unsupported/error cases")
            names = {row["name"] for row in rows}
            if expected_names is None:
                expected_names = names
            if names != expected_names:
                raise RuntimeError("Baseline and candidate must execute the same benchmark cases")
            for row in rows:
                if row["time_unit"] != "ns" or row["cpu_time"] <= 0:
                    raise ValueError("Expected positive nanosecond timings")
                if round_number >= 0:
                    samples[name].setdefault(row["name"], []).append(row["cpu_time"])

    summary = []
    rng = random.Random(1028)
    for name in sorted(expected_names):
        base = samples["baseline"][name]
        candidate = samples["candidate"][name]
        ratios = [new / old for old, new in zip(base, candidate)]
        boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
        low, high = boot[249], boot[9749]
        verdict = "improved" if high < 0.98 else "regressed" if low > 1.02 else "inconclusive/no clear change"
        summary.append({
            "name": name, "baseline_ns": statistics.median(base),
            "candidate_ns": statistics.median(candidate),
            "latency_change_pct": 100 * (statistics.median(ratios) - 1),
            "paired_bootstrap_95pct": [100 * (low - 1), 100 * (high - 1)],
            "baseline_cv_pct": 100 * statistics.stdev(base) / statistics.mean(base),
            "candidate_cv_pct": 100 * statistics.stdev(candidate) / statistics.mean(candidate),
            "verdict": verdict,
        })
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# SQ8 paired CI comparison", "", f"Baseline: `{baseline_sha}`",
             f"Candidate: `{candidate_sha}`", "",
             "Nine alternating paired rounds on one pinned CPU; negative change is faster.",
             "95% bootstrap intervals describe this runner only; improvement requires the full interval below -2%.",
             "", "| Case | Baseline ns | Candidate ns | Latency change | 95% interval | Verdict |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in summary:
        low, high = row["paired_bootstrap_95pct"]
        lines.append(f"| {row['name']} | {row['baseline_ns']:.2f} | {row['candidate_ns']:.2f} | "
                     f"{row['latency_change_pct']:+.1f}% | [{low:+.1f}%, {high:+.1f}%] | {row['verdict']} |")
    lines.extend(["", "SVS enabled in both focused builds. This is a hot-cache microbenchmark, not end-to-end query latency.",
                  "The no-VNNI dispatch experiment masks a feature on this runner; it does not measure older CPUs."])
    report = "\n".join(lines) + "\n"
    (results / "summary.md").write_text(report)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as step_summary:
        step_summary.write(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
