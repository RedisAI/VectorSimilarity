#!/usr/bin/env python3
# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Validate frozen artifacts and drive the native VecSim filtered-search executable."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

import filtered_bench as bench


GRAPH_SCHEMA_VERSION = 1
DEFAULT_POLICIES = ["NATIVE_SCAN", "NATIVE_BATCHES"]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def checkout_context():
    """The working checkout is context, not proof of which source built a binary."""
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL).strip())
        tracked_diff = subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD", "--binary"], stderr=subprocess.DEVNULL)
        return {"path": str(root), "revision": revision, "dirty": dirty,
                "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
                "diff_note": "tracked diff only; untracked files contribute to dirty status but not this hash"}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(root), "revision": None, "dirty": None}


def binary_description(binary):
    completed = subprocess.run([str(binary), "describe"], capture_output=True, text=True,
                               check=True, timeout=30)
    description = json.loads(completed.stdout)
    policies = description.get("policies") if isinstance(description, dict) else None
    if not isinstance(policies, list) or not policies or any(
            not isinstance(label, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", label)
            for label in policies) or len(set(policies)) != len(policies):
        raise ValueError("native describe must report distinct policy labels")
    return description


def corpus_identity(bundle):
    return {key: bundle.manifest[key] for key in
            ("dtype", "metric", "dimensions", "document_count")} | {
                "vectors_sha256": bundle.manifest["files"]["vectors.f32"]}


def validate_graph(directory, bundle):
    directory = Path(directory).resolve()
    graph = directory / "graph.hnsw"
    metadata = bench.read_json(directory / "graph.json")
    if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != GRAPH_SCHEMA_VERSION:
        raise ValueError("unsupported native graph sidecar schema")
    if metadata.get("graph_file") != "graph.hnsw" or graph.is_symlink():
        raise ValueError("graph sidecar must name a regular graph.hnsw file")
    if metadata.get("corpus") != corpus_identity(bundle):
        raise ValueError("graph corpus does not match the bundle vectors, metric, or shape")
    construction = metadata.get("construction")
    if not isinstance(construction, dict):
        raise ValueError("graph sidecar lacks construction parameters")
    bench.require_int(construction.get("m"), "graph m", 2)
    bench.require_int(construction.get("ef_construction"), "graph ef_construction", 1)
    if construction.get("insertion_order") != "ascending_external_id":
        raise ValueError("unsupported graph insertion order")
    if metadata.get("graph_sha256") != bench.sha256(graph):
        raise ValueError("native graph checksum mismatch")
    return graph, metadata


def write_native_inputs(bundle, output):
    offsets = {}
    position = 0
    with (output / "filters.u64").open("xb") as target:
        for name, ids in bundle.filters.items():
            offsets[name] = (position, len(ids))
            np.asarray(ids, dtype="<u8").tofile(target)
            position += len(ids)
    with (output / "cases.tsv").open("x", encoding="ascii") as target:
        for index, case in enumerate(bundle.cases):
            offset, count = offsets[case["filter_id"]]
            target.write(f"{index}\t{case['query_id']}\t{case['k']}\t{offset}\t{count}\n")


def convert_results(bundle, raw_path, policies, repeats):
    rows, seen = [], set()
    for row in bench.read_jsonl(raw_path):
        if not isinstance(row, dict):
            raise ValueError("native result must be an object")
        index = bench.require_int(row.get("case_index"), "native case_index")
        repeat = bench.require_int(row.get("repeat"), "native repeat")
        if index >= len(bundle.cases) or repeat >= repeats:
            raise ValueError("native case/repeat is out of range")
        policy = row.get("requested_policy")
        if policy not in policies or row.get("effective_policy") != policy:
            raise ValueError("native result policy does not match its registered adapter")
        key = (index, repeat, policy)
        if key in seen:
            raise ValueError("duplicate native result")
        seen.add(key)
        if row.get("status") not in ("ok", "error", "timeout"):
            raise ValueError("invalid native result status")
        latency = row.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("native latency must be finite and nonnegative")
        case = bundle.cases[index]
        ids, distances = row.get("ids"), row.get("distances")
        if not isinstance(ids, list) or not isinstance(distances, list) or len(ids) != len(distances) or len(ids) > case["k"]:
            raise ValueError("native IDs/distances have invalid lengths")
        eligible = set(bundle.filters[case["filter_id"]])
        if any(type(doc_id) is not int or doc_id not in eligible for doc_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("native IDs must be distinct eligible document IDs")
        if any(isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance)
               for distance in distances):
            raise ValueError("native distances must be finite")
        counters = row.get("counters")
        if not isinstance(counters, dict) or any(value is not None and
                (type(value) is not int or value < 0) for value in counters.values()):
            raise ValueError("native counters must be nonnegative integers or null")
        converted = {key: row[key] for key in ("repeat", "requested_policy", "effective_policy",
                                               "status", "ids", "distances", "latency_ms", "counters")}
        if "error" in row:
            if not isinstance(row["error"], str):
                raise ValueError("native error must be a string")
            converted["error"] = row["error"]
        converted.update(bundle_id=bundle.bundle_id, case_id=case["case_id"])
        rows.append(converted)
    if len(seen) != len(bundle.cases) * len(policies) * repeats:
        raise ValueError("native results do not cover the complete case/policy/repeat matrix")
    return rows


def invoke_native(command, output, timeout):
    started = time.monotonic()
    with (output / "native.stdout").open("x", encoding="utf-8") as stdout, \
            (output / "native.stderr").open("x", encoding="utf-8") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=True)
    elapsed = time.monotonic() - started
    metadata = bench.read_json(output / "native.stdout")
    if not isinstance(metadata, dict):
        raise ValueError("native executable must return a JSON metadata object")
    return metadata, elapsed


def execute(args):
    bundle_root = Path(args.bundle).resolve()
    bundle = bench.load_bundle(bundle_root)
    binary = Path(args.binary).resolve(strict=True)
    binary_hash = bench.sha256(binary)
    description = binary_description(binary)
    output = Path(args.output).resolve()
    graph, graph_metadata = None, None
    if args.command == "run":
        if len(set(args.policies)) != len(args.policies) or any(
                label not in description["policies"] for label in args.policies):
            raise ValueError(f"choose distinct policies registered by this binary: {description['policies']}")
        graph, graph_metadata = validate_graph(args.graph, bundle)
    elif not 2 <= args.m <= 32767:
        raise ValueError("HNSW m must be between 2 and 32767")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1, "command": args.command, "status": "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id, "bundle": str(bundle_root),
        "binary": str(binary), "binary_sha256": binary_hash,
        "binary_description": description,
        "binary_source_note": "source_revision/source_dirty reported by the binary are configure-time metadata; binary_sha256 identifies the executable",
        "wrapper_checkout": checkout_context(),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "process_timeout_seconds": args.process_timeout_seconds,
    }
    command = [str(binary), args.command, "--dimensions", str(bundle.manifest["dimensions"]),
               "--documents", str(bundle.manifest["document_count"]),
               "--metric", bundle.manifest["metric"]]
    result_code = 0
    try:
        if args.command == "build":
            graph = output / "graph.hnsw"
            construction = {"m": args.m, "ef_construction": args.ef_construction,
                            "insertion_order": "ascending_external_id"}
            metadata["construction"] = construction
            command += ["--vectors", str(bundle_root / "vectors.f32"), "--graph", str(graph),
                        "--m", str(args.m), "--ef-construction", str(args.ef_construction)]
        else:
            write_native_inputs(bundle, output)
            metadata.update(graph_sha256=graph_metadata["graph_sha256"], graph=str(graph),
                            graph_metadata=graph_metadata,
                            policies=args.policies, repeats=args.repeats, ef_runtime=args.ef_runtime,
                            batch_size=args.batch_size, timeout_ms=args.timeout_ms,
                            input_checksums={name: bench.sha256(output / name)
                                             for name in ("filters.u64", "cases.tsv")})
            command += ["--graph", str(graph), "--queries", str(bundle_root / "queries.f32"),
                        "--query-count", str(bundle.manifest["query_count"]),
                        "--filters", str(output / "filters.u64"), "--cases", str(output / "cases.tsv"),
                        "--output", str(output / "native-results.jsonl"),
                        "--policies", ",".join(args.policies), "--repeats", str(args.repeats),
                        "--ef-runtime", str(args.ef_runtime), "--batch-size", str(args.batch_size),
                        "--timeout-ms", str(args.timeout_ms)]
        metadata["native_command"] = command
        native, elapsed = invoke_native(command, output, args.process_timeout_seconds)
        metadata.update(native=native, subprocess_wall_seconds=elapsed)
        if bench.sha256(binary) != binary_hash:
            raise ValueError("native binary changed during execution")
        if args.command == "build":
            if bench.sha256(bundle_root / "vectors.f32") != corpus_identity(bundle)["vectors_sha256"]:
                raise ValueError("bundle vectors changed during graph construction")
            graph_metadata = {
                "schema_version": GRAPH_SCHEMA_VERSION, "graph_file": "graph.hnsw",
                "graph_sha256": bench.sha256(graph), "creator_bundle_id": bundle.bundle_id,
                "corpus": corpus_identity(bundle), "construction": construction,
                "creator_binary_sha256": binary_hash, "native_build": native,
                "trusted_input_only": True,
            }
            bench.write_json(output / "graph.json", graph_metadata)
            metadata["graph_sha256"] = graph_metadata["graph_sha256"]
        else:
            if bench.sha256(graph) != graph_metadata["graph_sha256"]:
                raise ValueError("native graph changed during search")
            if bench.sha256(bundle_root / "queries.f32") != bundle.manifest["files"]["queries.f32"]:
                raise ValueError("bundle queries changed during search")
            rows = convert_results(bundle, output / "native-results.jsonl", args.policies, args.repeats)
            with (output / "results.jsonl").open("x", encoding="utf-8") as target:
                for row in rows:
                    target.write(json.dumps(row, allow_nan=False) + "\n")
            cases = {case["case_id"]: case for case in bundle.cases}
            incomplete = sum(len(row["ids"]) != min(cases[row["case_id"]]["k"],
                len(bundle.filters[cases[row["case_id"]]["filter_id"]])) for row in rows)
            failures = sum(row["status"] != "ok" for row in rows)
            metadata.update(result_count=len(rows), failure_count=failures, incomplete_count=incomplete,
                            results_sha256=bench.sha256(output / "results.jsonl"))
            if failures or incomplete:
                result_code = 1
        metadata["status"] = "completed" if result_code == 0 else "completed_with_failures"
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        bench.write_json(output / ("build.json" if args.command == "build" else "run.json"), metadata)
    return result_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "run"):
        command = subcommands.add_parser(name)
        command.add_argument("--bundle", required=True)
        command.add_argument("--binary", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--process-timeout-seconds", type=positive_int, default=3600)
        if name == "build":
            command.add_argument("--m", type=positive_int, default=16)
            command.add_argument("--ef-construction", type=positive_int, default=200)
        else:
            command.add_argument("--graph", required=True)
            command.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
            command.add_argument("--repeats", type=positive_int, default=3)
            command.add_argument("--ef-runtime", type=positive_int, default=200)
            command.add_argument("--batch-size", type=positive_int, default=100)
            command.add_argument("--timeout-ms", type=positive_int, default=10000)
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"native benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
