#!/usr/bin/env python3
# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Frozen datasets and an engine-independent oracle for filtered vector search."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np


SCHEMA_VERSION = 1
FILES = ("vectors.f32", "queries.f32", "filters.json", "cases.json")
TIE_ATOL = 1e-7
TIE_RTOL = 1e-7


@dataclass
class Bundle:
    manifest: dict
    vectors: np.ndarray
    queries: np.ndarray
    filters: dict
    cases: list
    bundle_id: str


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")


def read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=unique_object)


def require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_vectors(vectors, metric):
    for start in range(0, len(vectors), 4096):
        block = vectors[start:start + 4096]
        if not np.isfinite(block).all():
            raise ValueError("vectors must be finite")
        if metric == "cosine" and np.any(np.linalg.norm(block.astype(np.float64), axis=1) == 0):
            raise ValueError("cosine vectors must have nonzero norm")


def load_bundle(path):
    root = Path(path)
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema_version")
    dimensions = require_int(manifest.get("dimensions"), "dimensions", 1)
    documents = require_int(manifest.get("document_count"), "document_count", 1)
    query_count = require_int(manifest.get("query_count"), "query_count", 1)
    metric = manifest.get("metric")
    if metric not in ("l2", "cosine") or manifest.get("dtype") != "<f4":
        raise ValueError("metric must be l2/cosine and dtype must be <f4")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(FILES):
        raise ValueError("manifest must name exactly the four standard bundle files")
    for name in FILES:
        if (root / name).is_symlink():
            raise ValueError(f"bundle files must not be symlinks: {name}")
        if sha256(root / name) != manifest["files"][name]:
            raise ValueError(f"checksum mismatch: {name}")
    arrays = []
    for name, rows in (("vectors.f32", documents), ("queries.f32", query_count)):
        if (root / name).stat().st_size != rows * dimensions * 4:
            raise ValueError(f"wrong byte length: {name}")
        array = np.memmap(root / name, dtype="<f4", mode="r", shape=(rows, dimensions))
        validate_vectors(array, metric)
        arrays.append(array)
    filters = read_json(root / "filters.json")
    if not isinstance(filters, dict) or not filters:
        raise ValueError("filters must be a nonempty object")
    for name, ids in filters.items():
        if not name or not isinstance(ids, list):
            raise ValueError("filters require nonempty names and lists of IDs")
        for doc_id in ids:
            if require_int(doc_id, "filter document ID") >= documents:
                raise ValueError("filter document ID out of range")
        if ids != sorted(set(ids)):
            raise ValueError("filter IDs must be sorted and unique")
    cases = read_json(root / "cases.json")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "query_id", "filter_id", "k", "cohort"}:
            raise ValueError("unexpected case fields")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case IDs must be nonempty and unique")
        seen.add(case_id)
        if require_int(case["query_id"], "query_id") >= query_count:
            raise ValueError("query_id out of range")
        if not isinstance(case["filter_id"], str) or case["filter_id"] not in filters:
            raise ValueError("invalid case filter")
        if not isinstance(case["cohort"], str) or not case["cohort"] or case["cohort"] == "ALL":
            raise ValueError("invalid case filter/cohort")
        require_int(case["k"], "k", 1)
    return Bundle(manifest, arrays[0], arrays[1], filters, cases, sha256(root / "manifest.json"))


def distances_to(vectors, query, metric, block_size=4096):
    """Calculate each distance once, in float64 over the original FP32 values."""
    query = np.asarray(query, dtype=np.float64)
    distances = np.empty(len(vectors), dtype=np.float64)
    query_norm = np.linalg.norm(query)
    for start in range(0, len(vectors), block_size):
        block = np.asarray(vectors[start:start + block_size], dtype=np.float64)
        if metric == "l2":
            delta = block - query
            values = np.einsum("ij,ij->i", delta, delta)
        else:
            values = 1.0 - (block @ query) / (np.linalg.norm(block, axis=1) * query_norm)
            values = np.clip(values, 0.0, 2.0)
        distances[start:start + len(block)] = values
    return distances


def create_cases(vectors, queries, metric, seed):
    count = len(vectors)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(count)
    filters, cases = {}, []

    def add_filter(name, ids):
        filters[name] = sorted(int(x) for x in ids)

    def add_case(query_id, filter_id, k, cohort):
        cases.append({"case_id": f"q{query_id}:{filter_id}:k{k}", "query_id": query_id,
                      "filter_id": filter_id, "k": k, "cohort": cohort})

    for fraction in (1.0, 0.1, 0.01, 0.001, 0.0001):
        eligible = max(1, int(count * fraction))
        name = f"independent_{fraction:g}"
        add_filter(name, permutation[:eligible])
        for query_id in range(len(queries)):
            add_case(query_id, name, 10, f"independent/{fraction:g}/k10")
            if eligible >= 200 and fraction in (0.1, 0.01):
                add_case(query_id, name, 100, f"independent/{fraction:g}/k100")

    # These are static indexed fields. Each anchor is evaluated against its own frozen filters.
    for query_id in range(min(8, len(queries))):
        distances = distances_to(vectors, queries[query_id], metric)
        ranked = np.lexsort((np.arange(count), distances))
        for fraction in (0.1, 0.01, 0.001):
            eligible = max(1, int(count * fraction))
            for direction, ids in (("near", ranked[:eligible]), ("far", ranked[-eligible:])):
                name = f"anchor_{query_id}_{direction}_{fraction:g}"
                add_filter(name, ids)
                add_case(query_id, name, 10, f"anchor_{direction}/{fraction:g}/k10")
    for eligible in sorted({0, 1, min(9, count), min(10, count), min(11, count)}):
        name = f"semantic_{eligible}"
        add_filter(name, permutation[:eligible])
        add_case(0, name, 10, f"semantic/{eligible}/k10")
    return filters, cases


def write_bundle(output, vectors, queries, metric, seed, provenance):
    if metric not in ("l2", "cosine"):
        raise ValueError("metric must be l2 or cosine")
    if vectors.ndim != 2 or queries.ndim != 2 or vectors.shape[1] != queries.shape[1]:
        raise ValueError("document and query matrices must have the same dimensions")
    if not len(vectors) or not len(queries) or vectors.shape[1] == 0:
        raise ValueError("empty vector matrices are not supported")
    validate_vectors(vectors, metric)
    validate_vectors(queries, metric)
    filters, cases = create_cases(vectors, queries, metric, seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    np.asarray(vectors, dtype="<f4").tofile(root / "vectors.f32")
    np.asarray(queries, dtype="<f4").tofile(root / "queries.f32")
    write_json(root / "filters.json", filters)
    write_json(root / "cases.json", cases)
    manifest = {"schema_version": SCHEMA_VERSION, "dtype": "<f4", "metric": metric,
                "dimensions": vectors.shape[1], "document_count": len(vectors),
                "query_count": len(queries), "seed": seed,
                "id_mapping": "zero-based vector row", "provenance": provenance,
                "files": {name: sha256(root / name) for name in FILES}}
    write_json(root / "manifest.json", manifest)
    return load_bundle(root)


def read_matrix(path, file_format, count, dimensions):
    require_int(count, "source count", 1)
    require_int(dimensions, "dimensions", 1)
    offset = 0
    if file_format == "fbin":
        with Path(path).open("rb") as source:
            header = source.read(8)
        if len(header) != 8 or struct.unpack("<II", header) != (count, dimensions):
            raise ValueError("fbin header does not match the expected count and dimensions")
        offset = 8
    if Path(path).stat().st_size != offset + count * dimensions * 4:
        raise ValueError("input file has wrong byte length for little-endian FP32")
    return np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, dimensions))


def oracle_rows(bundle):
    cases_by_query = {}
    for case in bundle.cases:
        cases_by_query.setdefault(case["query_id"], []).append(case)
    for query_id, cases in cases_by_query.items():
        distances = distances_to(bundle.vectors, bundle.queries[query_id], bundle.manifest["metric"])
        local_count = min(1000, len(distances))
        local_ids = set(np.argpartition(distances, local_count - 1)[:local_count].tolist())
        for case in cases:
            eligible = np.asarray(bundle.filters[case["filter_id"]], dtype=np.int64)
            target = min(case["k"], len(eligible))
            if target:
                selected_distances = distances[eligible]
                boundary = float(np.partition(selected_distances, target - 1)[target - 1])
                tolerance = TIE_ATOL + TIE_RTOL * abs(boundary)
                strict_ids = eligible[selected_distances < boundary - tolerance]
                boundary_ids = eligible[np.abs(selected_distances - boundary) <= tolerance]
                candidates = eligible[selected_distances <= boundary]
                ordered = np.lexsort((candidates, distances[candidates]))[:target]
                top_ids = candidates[ordered]
            else:
                boundary, tolerance = None, None
                strict_ids = boundary_ids = top_ids = np.array([], dtype=np.int64)
            yield {"bundle_id": bundle.bundle_id, "case_id": case["case_id"],
                   "eligible_count": len(eligible), "target_count": target,
                   "ids": top_ids.tolist(), "distances": distances[top_ids].tolist(),
                   "strict_ids": strict_ids.tolist(), "boundary_ids": boundary_ids.tolist(),
                   "boundary_distance": boundary, "tie_tolerance": tolerance,
                   "local_matching_rate": len(local_ids.intersection(eligible.tolist())) / local_count}


def write_oracle(bundle, path):
    path = Path(path)
    metadata = path.with_suffix(path.suffix + ".meta.json")
    if path.exists() or metadata.exists():
        raise ValueError("oracle output already exists")
    with path.open("x", encoding="utf-8") as target:
        for row in oracle_rows(bundle):
            target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_json(metadata, {"schema_version": SCHEMA_VERSION, "bundle_id": bundle.bundle_id,
                          "sha256": sha256(path), "arithmetic": "float64 over original FP32",
                          "tie_atol": TIE_ATOL, "tie_rtol": TIE_RTOL})


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at line {line_number}")
            yield value


def load_oracle(bundle, path):
    metadata = read_json(Path(path).with_suffix(Path(path).suffix + ".meta.json"))
    if metadata.get("bundle_id") != bundle.bundle_id or metadata.get("sha256") != sha256(path):
        raise ValueError("oracle belongs to a different bundle or its checksum changed")
    rows = list(read_jsonl(path))
    truth = {row["case_id"]: row for row in rows}
    if len(truth) != len(rows) or set(truth) != {case["case_id"] for case in bundle.cases}:
        raise ValueError("oracle must contain exactly one record per case")
    if any(row.get("bundle_id") != bundle.bundle_id for row in rows):
        raise ValueError("oracle record bundle mismatch")
    return truth


def score_results(bundle, records, truth, policies, repeats, target_recall=0.95):
    require_int(repeats, "repeats", 1)
    if not math.isfinite(target_recall) or not 0 <= target_recall <= 1:
        raise ValueError("target_recall must be between 0 and 1")
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("declare a nonempty unique list of expected policies")
    cases = {case["case_id"]: case for case in bundle.cases}
    indexed = {}
    for record in records:
        if record.get("bundle_id") != bundle.bundle_id:
            raise ValueError("result bundle_id does not match this bundle")
        policy, repeat, case_id = (record.get(key) for key in ("requested_policy", "repeat", "case_id"))
        require_int(repeat, "result repeat")
        if policy not in policies or repeat >= repeats or case_id not in cases:
            raise ValueError("result refers to an unexpected policy, repeat, or case")
        key = (policy, repeat, case_id)
        if key in indexed:
            raise ValueError(f"duplicate result record: {key}")
        indexed[key] = record
    scored = []
    for policy in policies:
        for repeat in range(repeats):
            for case_id, case in cases.items():
                oracle = truth[case_id]
                target = oracle["target_count"]
                record = indexed.get((policy, repeat, case_id))
                status, issue, latency, recall = "missing", None, None, (0.0 if target else None)
                complete = False
                if record is not None:
                    status = record.get("status")
                    ids, scores = record.get("ids"), record.get("distances")
                    latency = record.get("latency_ms")
                    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
                        issue = "latency_ms must be finite and nonnegative"
                        latency = None
                    elif status not in ("ok", "error", "timeout"):
                        issue = "unknown status"
                    elif not isinstance(ids, list) or not isinstance(scores, list) or len(ids) != len(scores):
                        issue = "ids/distances must be same-length lists"
                    elif any(type(doc_id) is not int for doc_id in ids) or len(set(ids)) != len(ids):
                        issue = "document IDs must be integers without duplicates"
                    elif not set(ids).issubset(bundle.filters[case["filter_id"]]):
                        issue = "returned an ineligible or out-of-range document"
                    elif len(ids) > target:
                        issue = "returned more than min(k, eligible_count) documents"
                    elif any(type(score) not in (int, float) or not math.isfinite(score) for score in scores):
                        issue = "distances must be finite numbers"
                    if issue:
                        status = "invalid"
                    elif status == "ok":
                        returned = set(ids)
                        strict = set(oracle["strict_ids"])
                        boundary = set(oracle["boundary_ids"])
                        credit = len(returned & strict) + min(target - len(strict), len(returned & boundary))
                        recall = credit / target if target else None
                        complete = len(ids) == target
                scored.append({"case_id": case_id, "cohort": case["cohort"], "requested_policy": policy,
                               "repeat": repeat, "status": status, "recall": recall,
                               "complete": complete, "latency_ms": latency, "validation_error": issue})
    summaries = []
    for policy in policies:
        for repeat in range(repeats):
            cohorts = sorted({case["cohort"] for case in bundle.cases})
            for cohort in ["ALL", *cohorts]:
                rows = [row for row in scored if row["requested_policy"] == policy and
                        row["repeat"] == repeat and (cohort == "ALL" or row["cohort"] == cohort)]
                recalls = [row["recall"] for row in rows if row["recall"] is not None]
                latencies = [row["latency_ms"] for row in rows if row["status"] == "ok"]
                passed = all(row["status"] == "ok" and row["complete"] and
                             (row["recall"] is None or row["recall"] >= target_recall) for row in rows)
                summaries.append({"requested_policy": policy, "repeat": repeat, "cohort": cohort,
                                  "cases": len(rows), "failures": sum(row["status"] != "ok" for row in rows),
                                  "incomplete": sum(not row["complete"] for row in rows),
                                  "mean_recall": float(np.mean(recalls)) if recalls else None,
                                  "min_recall": min(recalls) if recalls else None,
                                  "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
                                  "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                                  "all_queries_reach_target": passed})
    return {"schema_version": SCHEMA_VERSION, "bundle_id": bundle.bundle_id,
            "target_recall": target_recall,
            "expected_policies": policies, "expected_repeats": repeats,
            "summaries": summaries, "queries": scored}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="make a deterministic synthetic bundle")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--documents", type=int, default=10000)
    prepare.add_argument("--dimensions", type=int, default=128)
    prepare.add_argument("--queries", type=int, default=32)
    prepare.add_argument("--seed", type=int, default=20221)
    prepare.add_argument("--metric", choices=("l2", "cosine"), default="cosine")
    ingest = commands.add_parser("import", help="import original little-endian FP32 arrays")
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--vectors", required=True)
    ingest.add_argument("--queries", required=True)
    ingest.add_argument("--format", choices=("raw", "fbin"), required=True)
    ingest.add_argument("--documents", type=int, required=True)
    ingest.add_argument("--query-count", type=int, required=True)
    ingest.add_argument("--dimensions", type=int, required=True)
    ingest.add_argument("--limit-documents", type=int)
    ingest.add_argument("--limit-queries", type=int)
    ingest.add_argument("--query-start", type=int, default=0, help="first source query row, for held-out splits")
    ingest.add_argument("--metric", choices=("l2", "cosine"), required=True)
    ingest.add_argument("--seed", type=int, default=20221)
    ingest.add_argument("--source", required=True, help="source URL or immutable dataset identifier")
    oracle = commands.add_parser("oracle", help="write untimed exhaustive filtered ground truth")
    oracle.add_argument("bundle")
    oracle.add_argument("--output")
    score = commands.add_parser("score", help="validate and score every declared run/case")
    score.add_argument("bundle")
    score.add_argument("results")
    score.add_argument("--oracle")
    score.add_argument("--policies", nargs="+", required=True)
    score.add_argument("--repeats", type=int, required=True, help="repeat IDs must be 0..N-1")
    score.add_argument("--output", required=True)
    score.add_argument("--target-recall", type=float, default=0.95)
    score.add_argument("--fail-below-target", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        for field in ("documents", "dimensions", "queries"):
            require_int(getattr(args, field), field, 1)
        rng = np.random.default_rng(args.seed)
        vectors = rng.standard_normal((args.documents, args.dimensions), dtype=np.float32)
        queries = rng.standard_normal((args.queries, args.dimensions), dtype=np.float32)
        bundle = write_bundle(args.output, vectors, queries, args.metric, args.seed,
                              {"source": "synthetic independent Gaussian FP32", "numpy_version": np.__version__})
        print(bundle.bundle_id)
    elif args.command == "import":
        vectors = read_matrix(args.vectors, args.format, args.documents, args.dimensions)
        queries = read_matrix(args.queries, args.format, args.query_count, args.dimensions)
        if Path(args.vectors).resolve() == Path(args.queries).resolve():
            raise ValueError("base and query arrays must be distinct source files")
        for limit, count in ((args.limit_documents, args.documents), (args.limit_queries, args.query_count)):
            if limit is not None and (require_int(limit, "limit", 1) > count):
                raise ValueError("prefix limit exceeds source count")
        require_int(args.query_start, "query_start")
        query_stop = args.query_count if args.limit_queries is None else args.query_start + args.limit_queries
        if args.query_start >= args.query_count or query_stop > args.query_count:
            raise ValueError("query subset exceeds source count")
        bundle = write_bundle(args.output, vectors[:args.limit_documents], queries[args.query_start:query_stop],
                              args.metric, args.seed, {"source": args.source, "format": args.format,
                              "base_sha256": sha256(args.vectors), "query_sha256": sha256(args.queries),
                              "source_document_count": args.documents, "source_query_count": args.query_count,
                              "document_subset": "fixed source prefix", "query_start": args.query_start,
                              "query_stop": query_stop, "numpy_version": np.__version__})
        print(bundle.bundle_id)
    elif args.command == "oracle":
        bundle = load_bundle(args.bundle)
        write_oracle(bundle, args.output or Path(args.bundle) / "ground_truth.jsonl")
    else:
        bundle = load_bundle(args.bundle)
        truth = load_oracle(bundle, args.oracle or Path(args.bundle) / "ground_truth.jsonl")
        report = score_results(bundle, read_jsonl(args.results), truth, args.policies, args.repeats,
                               args.target_recall)
        write_json(args.output, report)
        if any(row["failures"] or row["incomplete"] for row in report["summaries"]):
            return 1
        if args.fail_below_target and any(not row["all_queries_reach_target"] for row in report["summaries"]):
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
