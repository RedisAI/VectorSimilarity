#!/usr/bin/env python3
# Copyright (c) 2006-Present, Redis Ltd. All rights reserved.
# Licensed under RSALv2, SSPLv1, or AGPLv3.
"""Build, verify and benchmark an isolated exact-SQ8 candidate only in GitHub Actions."""

import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("CI-only experiment")
    root = Path.cwd()
    results = root / "sq8-exact-results"
    results.mkdir()
    scratch = Path(tempfile.mkdtemp(prefix="sq8-exact-", dir=os.environ["RUNNER_TEMP"]))
    baseline = scratch / "baseline"
    baseline_sha = "838a30de4940f0794f84213e38c1b802d3f1db7c"
    cpu_headers = root / ".repro-deps/cpu_features/include"
    probe = root / ".github/scripts/sq8-exact-probe.cpp"
    check = root / ".github/scripts/sq8-exact-check.py"

    def run(args, label, cwd=root, extra_env=None):
        args = list(map(str, args))
        print(label + ": " + " ".join(args), flush=True)
        log_path = results / (label + ".log")
        with log_path.open("w") as output:
            process = subprocess.run(args, cwd=cwd, env={**os.environ, "ROOT": str(cwd), **(extra_env or {})},
                                     stdout=output, stderr=subprocess.STDOUT)
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-35:]), flush=True)
        process.check_returncode()

    run(["git", "fetch", "--depth=1", "origin", baseline_sha], "fetch-baseline")
    run(["git", "clone", "--shared", "--no-checkout", root, baseline], "clone-baseline")
    run(["git", "fetch", "--depth=1", "https://github.com/" + os.environ["GITHUB_REPOSITORY"], baseline_sha],
        "fetch-isolated-baseline", baseline)
    run(["git", "checkout", "--detach", baseline_sha], "checkout-baseline", baseline)
    metadata = {"baseline": baseline_sha, "candidate": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "architecture": platform.machine(), "compiler": subprocess.check_output(["g++", "--version"], text=True),
                "cpu": subprocess.check_output(["lscpu"], text=True),
                "run": "https://github.com/" + os.environ["GITHUB_REPOSITORY"] + "/actions/runs/" + os.environ["GITHUB_RUN_ID"]}
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def compile_probe(source, output, candidate, compiler="g++", sanitizer=False, build=None):
        flags = [compiler, "-std=c++20", "-O3", "-fno-fast-math", "-ffp-contract=off",
                 "-I" + str(source / "src"), "-I" + str(cpu_headers)]
        if candidate:
            flags.append("-DSQ8_EXACT_CANDIDATE")
        if sanitizer:
            flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-DSQ8_SANITIZER_MAIN"]
        else:
            flags += ["-shared", "-fPIC"]
        sources = [probe, source / "src/VecSim/memory/vecsim_malloc.cpp", source / "src/VecSim/memory/vecsim_base.cpp"]
        if build is None:
            flags.append("-DSQ8_STANDALONE")
            sources += [source / "src/VecSim/spaces/L2/L2.cpp", source / "src/VecSim/spaces/IP/IP.cpp"]
        else:
            sources += ["-Wl,--start-group"]
            for name in ("libVectorSimilaritySpaces.a", "libVectorSimilaritySpaces_no_optimization.a", "libcpu_features.a"):
                matches = list(build.rglob(name))
                assert len(matches) == 1, (name, matches)
                sources.append(matches[0])
            sources += ["-Wl,--end-group", "-lpthread", "-ldl"]
        run(flags + sources + ["-o", output], output.stem + "-compile")

    quick_base, quick_candidate = results / "baseline-quick.so", results / "candidate-quick.so"
    compile_probe(baseline, quick_base, False)
    compile_probe(root, quick_candidate, True)
    run(["python3", check, "quick", quick_candidate, quick_base, results], "quick-oracle")
    clang_candidate = results / "candidate-clang.so"
    compile_probe(root, clang_candidate, True, compiler="clang++")
    run(["python3", check, "clang", clang_candidate, quick_base, results], "clang-oracle")
    sanitized = results / "candidate-sanitized"
    compile_probe(root, sanitized, True, sanitizer=True)
    run([sanitized], "sanitizers", extra_env={"ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1", "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"})

    run(["git", "submodule", "update", "--init", "--recursive"], "baseline-submodules", baseline)
    full = {}
    for label, source in (("baseline", baseline), ("candidate", root)):
        build = scratch / (label + "-build")
        run(["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DUSE_SVS=ON", "-DVECSIM_BUILD_TESTS=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], label + "-configure", source)
        targets = ["test_spaces"]
        if label == "candidate":
            targets += ["test_components", "test_hnsw_sq8"]
        run(["cmake", "--build", build, "--parallel", "2", "--target", *targets], label + "-build", source)
        (results / (label + "-compile-commands.json")).write_text((build / "compile_commands.json").read_text())
        for target in targets:
            run([build / "unit_tests" / target, "--gtest_output=xml:" + str(results / (label + "-" + target + ".xml"))],
                label + "-" + target, source)
        run(["ctest", "--test-dir", build, "-R", "^tier_linkage$", "--output-on-failure"], label + "-tier-linkage", source)
        full[label] = results / (label + "-full.so")
        compile_probe(source, full[label], label == "candidate", build=build)
    run(["python3", check, "full", full["candidate"], full["baseline"], results], "full-oracle")
    run(["python3", check, "benchmark", full["candidate"], full["baseline"], results], "benchmark")


if __name__ == "__main__":
    main()
