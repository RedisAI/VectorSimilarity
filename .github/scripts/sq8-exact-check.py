#!/usr/bin/env python3
# Copyright (c) 2006-Present, Redis Ltd. All rights reserved.
# Licensed under RSALv2, SSPLv1, or AGPLv3.
"""Independent integer oracle and paired hot-cache benchmark; invoked only by CI."""

import ctypes as C
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import sys


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(raw):
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def units(raw):
    """FP32 as an integer in units of the smallest FP32 subnormal."""
    exponent = (raw >> 23) & 255
    mantissa = raw & 0x7fffff
    assert exponent != 255
    integer = mantissa if exponent == 0 else ((1 << 23) | mantissa) << (exponent - 1)
    return -integer if raw >> 31 else integer


def oracle(storage, query):
    dim = len(query)
    min_bits, delta_bits = struct.unpack_from("<II", storage, dim)
    m, d = units(min_bits), units(delta_bits)
    exact = sum((m + d * code - units(y)) ** 2 for code, y in zip(storage[:dim], query))
    # Use a monotonic binary search over all positive FP32 encodings, independently of
    # the production leading-bit/guard/sticky rounding implementation.
    max_bits = 0x7f7fffff
    maximum = units(max_bits) << 149
    previous = units(max_bits - 1) << 149
    if exact >= maximum + (maximum - previous) // 2:
        return 0x7f800000
    low, high = 0, max_bits
    while low < high:
        middle = (low + high + 1) // 2
        if (units(middle) << 149) <= exact:
            low = middle
        else:
            high = middle - 1
    if low == max_bits:
        return low
    below, above = exact - (units(low) << 149), (units(low + 1) << 149) - exact
    return low + int(above < below or (above == below and low & 1))


def library(path):
    lib = C.CDLL(str(path))
    lib.sq8_distance.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    lib.sq8_distance.restype = C.c_uint32
    lib.sq8_certified.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    lib.sq8_quantize.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p]
    lib.sq8_query.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p]
    lib.sq8_benchmark.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_size_t]
    lib.sq8_benchmark.restype = C.c_double
    lib.sq8_environment.argtypes = [C.c_int, C.c_int]
    return lib


def blob(codes, minimum, delta):
    return bytes(codes) + struct.pack("<IIII", minimum, delta, 0, 0)


def buffers(storage, query):
    return C.create_string_buffer(storage), (C.c_uint32 * (len(query) + 2))(*query, 0, 0)


def production_pair(lib, stored, query):
    dim = len(stored)
    original = (C.c_float * dim)(*stored)
    storage = C.create_string_buffer(dim + 16)
    lib.sq8_quantize(original, dim, storage)
    original_query = (C.c_float * dim)(*query)
    processed_query = (C.c_uint32 * (dim + 2))()
    lib.sq8_query(original_query, dim, processed_query)
    return bytes(storage), list(processed_query)[:dim], storage, processed_query


def verify(candidate, baseline, results, prefix):
    rng = random.Random(1028)
    fixtures = []
    known = []
    for stored, query in (
        ([1, 33554432, 66846720], [1, 33554432, 66846720]),
        ([1, 33554432, 66846720], [1, 33554436, 66846720]),
        ([1, 2**60, 255 * 2**53], [1, 2**60, 255 * 2**53]),
    ):
        storage, q, s_buffer, q_buffer = production_pair(candidate, stored, query)
        expected = oracle(storage, q)
        record = {"stored": stored, "query": query, "expected": value(expected)}
        record["candidate"] = value(candidate.sq8_distance(s_buffer, q_buffer, len(q), 1))
        # The baseline gets its own production preprocessed blobs, including query sums.
        _, bq, bs, bquery = production_pair(baseline, stored, query)
        record["baseline"] = value(baseline.sq8_distance(bs, bquery, len(bq), 1))
        known.append(record)
        fixtures.append((storage, q))
    for query in ([1, 2**-12], [1, 2**-12, 2**-24], [1] + [2**-12] * 3,
                  [2**-75], [value(bits(2**-75) + 1)], [2**-74], [2**-63], [2**64]):
        fixtures.append((blob([0] * len(query), 0, bits(1)), list(map(bits, query))))
    # Exact overflow midpoint and its adjacent lower value.
    max_distance_terms = []
    for exponent in range(104, 128):
        max_distance_terms.extend([2**(exponent // 2)] * (2 if exponent % 2 else 1))
    for query in (max_distance_terms, max_distance_terms + [2**51, 2**51]):
        fixtures.append((blob([0] * len(query), 0, bits(1)), list(map(bits, query))))
    for _ in range(6000):
        dim = rng.choice([1, 2, 3, 8, 9, 16, 31, 32, 65, 128])
        finite = lambda: rng.getrandbits(32) & 0xfeffffff
        fixtures.append((blob([rng.randrange(256) for _ in range(dim)], finite(), finite()),
                         [finite() for _ in range(dim)]))
    for _ in range(3000):
        dim = rng.choice([1, 3, 8, 15, 128, 1536])
        offset = rng.choice([0, 1000, 100000, -100000])
        stored = [offset + rng.uniform(-1, 1) for _ in range(dim)]
        query = [offset + rng.uniform(-1, 1) for _ in range(dim)]
        storage, q, _, _ = production_pair(candidate, stored, query)
        fixtures.append((storage, q))
    certified = 0
    for index, (storage, query) in enumerate(fixtures):
        expected = oracle(storage, query)
        sb, qb = buffers(storage, query)
        for mode in (0, 1, 2):
            actual = candidate.sq8_distance(sb, qb, len(query), mode)
            if actual != expected:
                failure = {"index": index, "mode": mode, "expected": expected, "actual": actual,
                           "storage": storage.hex(), "query": query}
                (results / (prefix + "-failure.json")).write_text(json.dumps(failure))
                raise AssertionError(failure)
        certified += candidate.sq8_certified(sb, qb, len(query))
    # Expect the same bitwise result under all rounding modes and flush-to-zero controls.
    environmental = fixtures[:15] + fixtures[15:165]
    expected_environment = [oracle(s, q) for s, q in environmental]
    try:
        for mode in range(4):
            for flush in range(2):
                candidate.sq8_environment(mode, flush)
                for (storage, query), expected in zip(environmental, expected_environment):
                    sb, qb = buffers(storage, query)
                    assert candidate.sq8_distance(sb, qb, len(query), 1) == expected
    finally:
        candidate.sq8_environment(0, 0)
    report = {"cases": len(fixtures), "comparisons": 3 * len(fixtures),
              "certified_cases": certified, "environmental_comparisons": 8 * len(environmental),
              "known_reproductions": known}
    (results / (prefix + "-oracle.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def benchmark(candidate, baseline, results):
    rng = random.Random(1028)
    rows = []
    cpu = sorted(os.sched_getaffinity(0))[-1]
    os.sched_setaffinity(0, {cpu})
    for dim in (8, 16, 128, 384, 768, 1536):
        for shape in ("ordinary", "large_offset", "wide_range"):
            offset = 100000 if shape == "large_offset" else 0
            stored = [offset + rng.uniform(-1, 1) for _ in range(dim)]
            query = [offset + rng.uniform(-1, 1) for _ in range(dim)]
            if shape == "wide_range":
                stored = [1.0] * dim
                stored[1], stored[-1] = 33554432.0, 66846720.0
                query = list(stored)
                query[1] += 4
            _, _, s, q = production_pair(candidate, stored, query)
            _, _, bs, bq = production_pair(baseline, stored, query)
            variants = {"baseline_scalar": (baseline, bs, bq, 0),
                        "baseline_dispatch": (baseline, bs, bq, 1),
                        "candidate_dispatch": (candidate, s, q, 1),
                        "candidate_exact_only": (candidate, s, q, 2)}
            samples = {name: [] for name in variants}
            iterations = {}
            for name, (lib, vs, vq, mode) in variants.items():
                duration = lib.sq8_benchmark(vs, vq, dim, mode, 10000)
                iterations[name] = max(100, min(50000000, int(5e7 / max(duration, 0.01))))
            for round_number in range(8):
                order = list(variants) if round_number % 2 == 0 else list(reversed(variants))
                for name in order:
                    lib, vs, vq, mode = variants[name]
                    ns = lib.sq8_benchmark(vs, vq, dim, mode, iterations[name])
                    if round_number:
                        samples[name].append(ns)
            paired = [c / b for c, b in zip(samples["candidate_dispatch"], samples["baseline_dispatch"])]
            boot = sorted(statistics.median(rng.choices(paired, k=len(paired))) for _ in range(2000))
            row = {"dim": dim, "shape": shape, "cpu": cpu,
                   "certified": bool(candidate.sq8_certified(s, q, dim)),
                   "median_ns": {k: statistics.median(v) for k, v in samples.items()},
                   "slowdown": statistics.median(paired), "paired_95pct": [boot[49], boot[1949]],
                   "samples_ns": samples}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples_ns"}), flush=True)
    (results / "benchmark.json").write_text(json.dumps(rows, indent=2))
    lines = ["# SQ8 exact L2 comparison", "", "Seven alternating paired rounds on a pinned CPU.",
             "Thread CPU time, hot-cache kernel only; not end-to-end search latency.", "",
             "| Shape | Dim | Baseline SIMD ns | Candidate ns | Slowdown | Certified |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        med = row["median_ns"]
        lines.append(f"| {row['shape']} | {row['dim']} | {med['baseline_dispatch']:.2f} | "
                     f"{med['candidate_dispatch']:.2f} | {row['slowdown']:.2f}x | {row['certified']} |")
    report = "\n".join(lines) + "\n"
    (results / "summary.md").write_text(report)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
        output.write(report)


if __name__ == "__main__":
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Builds, tests and benchmarks are CI-only")
    command, candidate_path, baseline_path, results_path = sys.argv[1:]
    candidate, baseline = library(candidate_path), library(baseline_path)
    results = Path(results_path)
    if command == "benchmark":
        benchmark(candidate, baseline, results)
    else:
        verify(candidate, baseline, results, command)
