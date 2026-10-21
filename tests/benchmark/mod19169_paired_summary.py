#!/usr/bin/env python3
"""Validate and summarize the seven-process FP16 saved-graph comparison.

Usage: python3 mod19169_paired_summary.py RESULTS_DIRECTORY
Latency is the median of three repeated mean-latency samples per process,
then the median across all four mixed or three native processes (ms/query).
All four mixed processes also form the A/A control.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

RUNS = ("mixed-a", "mixed-b", "native-a", "mixed-c", "native-b", "native-c", "mixed-d")
EFS = (100, 200, 400)
MIXED = ("mixed-a", "mixed-b", "mixed-c", "mixed-d")
NATIVE = ("native-a", "native-b", "native-c")
IDENTITY_FIELDS = (
    "source", "queries", "graph", "source_vectors", "source_labels", "selected_vectors",
    "selected_labels", "selection_hash", "selected_data_hash", "query_input_hash",
    "ground_truth_checksum", "boundary_tied_queries", "stored_byte_checksum",
    "graph_file_checksum", "serialized_graph_bytes", "M", "ef_construction", "normalization",
    "training_vectors", "mean_hash", "seed", "graph_builder_mode",
)
HASH_FIELDS = ("selection_hash", "selected_data_hash", "query_input_hash",
               "ground_truth_checksum", "stored_byte_checksum", "graph_file_checksum")
LABEL = "reference=FP32_BF_over_original_typed_values; query includes preprocessing"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, context, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{context}: expected finite number")
    require(not positive or value > 0, f"{context}: expected positive number")
    return value


def integer(value, context, low, high):
    require(type(value) is int and low <= value <= high,
            f"{context}: expected integer in [{low}, {high}]")
    return value


def read_json(path, allow_nonfinite=False):
    def reject_constant(value):
        raise ValueError(f"{path.name}: nonfinite JSON constant {value}")
    with path.open() as stream:
        # Google Benchmark emits NaN for aggregate CV counters whose mean is
        # zero. These aggregate rows are discarded; raw metrics are checked
        # individually before use. Provenance always uses strict parsing.
        value = json.load(stream, **({} if allow_nonfinite else {"parse_constant": reject_constant}))
    require(isinstance(value, dict), f"{path.name}: expected JSON object")
    return value


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_run(directory, dataset, run):
    prefix = f"fp16-{dataset}-full-graph-saved-query-{run}"
    path = directory / f"{prefix}_results.json"
    provenance_path = directory / f"{prefix}-provenance.json"
    report = read_json(provenance_path)
    result = read_json(path, allow_nonfinite=True)
    context = prefix
    mode = 0 if run.startswith("mixed-") else 1
    mode_name = "mixedFP32" if mode == 0 else "finalnative"
    expected = {"stage": "query", "dtype": "fp16", "dataset": dataset, "corpus": "full",
                "query_mode": mode, "dimension": 768 if dataset == "single" else 512,
                "query_count": 1000, "k": 100}
    for field, value in expected.items():
        require(report.get(field) == value and type(report.get(field)) is type(value),
                f"{context}: invalid {field}, expected {value!r}")
    if "query_mode_name" in report:
        require(report["query_mode_name"] == mode_name,
                f"{context}: invalid query_mode_name, expected {mode_name!r}")
    for field in IDENTITY_FIELDS:
        require(field in report, f"{context}: missing identity field {field}")
    for field in HASH_FIELDS:
        require(isinstance(report[field], str) and report[field].isdigit(),
                f"{context}: invalid {field}")
    count = 1000000 if dataset == "single" else 1111025
    labels = 1000000 if dataset == "single" else 44441
    for field, value in (("source_vectors", count), ("selected_vectors", count),
                         ("source_labels", labels), ("selected_labels", labels)):
        require(type(report[field]) is int and report[field] == value,
                f"{context}: {field} does not describe the complete corpus")
    integer(report["boundary_tied_queries"], context + ": boundary_tied_queries", 0, 1000)
    integer(report["serialized_graph_bytes"], context + ": serialized_graph_bytes", 1, 2**64 - 1)
    hits = report.get("per_query_hits")
    require(isinstance(hits, dict) and set(hits) == {f"ef{ef}" for ef in EFS},
            f"{context}: expected per_query_hits for ef100, ef200, ef400")
    for ef in EFS:
        values = hits[f"ef{ef}"]
        require(isinstance(values, list) and len(values) == 1000,
                f"{context}: ef{ef} must have 1000 per-query hits")
        for q, value in enumerate(values):
            integer(value, f"{context}: ef{ef} query {q} hits", 0, 100)
    rows = result.get("benchmarks")
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
            f"{context}: missing benchmark rows")
    require(not any(row.get("error_occurred") or row.get("error_message") for row in rows),
            f"{context}: benchmark reported an error")
    samples = [row for row in rows if row.get("run_type", "iteration") == "iteration"]
    require(len(samples) == 9, f"{context}: expected exactly 9 raw timing samples")
    grouped = {ef: [] for ef in EFS}
    pattern = re.compile(rf"MOD19169/{mode_name}/fp16/{dataset}/full/ef(100|200|400)(?:/|$)")
    for row in samples:
        match = pattern.match(row.get("run_name", row.get("name", "")))
        require(match is not None, f"{context}: unexpected benchmark name")
        ef = int(match.group(1))
        require(type(row.get("iterations")) is int and row["iterations"] == 1000,
                f"{context}: expected 1000 iterations")
        require(row.get("time_unit") == "ms", f"{context}: expected milliseconds")
        require(number(row.get("query_mode"), context + ": query_mode") == mode,
                f"{context}: wrong benchmark query_mode")
        if "query_mode_name" in row:
            require(row["query_mode_name"] == mode_name,
                    f"{context}: wrong benchmark query_mode_name")
        require(row.get("label") == LABEL, f"{context}: unexpected timing label")
        require(number(row.get("FP32_BF_boundary_tied_queries"), context + ": boundary ties")
                == report["boundary_tied_queries"],
                f"{context}: boundary-tie counter differs from provenance")
        recall = number(row.get("Recall_vs_FP32_BF"), context + ": recall")
        require(0 <= recall <= 1 and math.isclose(
            recall, sum(hits[f"ef{ef}"]) / 100000, rel_tol=0, abs_tol=1e-12),
            f"{context}: ef{ef} recall counter differs from per-query hits")
        for field in ("real_time", "cpu_time"):
            number(row.get(field), context + ": " + field, positive=True)
        grouped[ef].append(row)
    for ef, values in grouped.items():
        require(len(values) == 3, f"{context}: ef{ef} requires three raw samples")
        repetition_indices = [row.get("repetition_index") for row in values]
        if any(index is not None for index in repetition_indices):
            require(sorted(repetition_indices, key=str) == [0, 1, 2],
                    f"{context}: ef{ef} repetition indices must be 0, 1, 2")
    return {"identity": {key: report[key] for key in IDENTITY_FIELDS}, "hits": hits,
            "ef": {str(ef): {
                "real_time_ms_samples": [row["real_time"] for row in grouped[ef]],
                "cpu_time_ms_samples": [row["cpu_time"] for row in grouped[ef]],
                "median_real_time_ms": statistics.median(row["real_time"] for row in grouped[ef]),
                "median_cpu_time_ms": statistics.median(row["cpu_time"] for row in grouped[ef]),
                "recall_fraction": sum(hits[f"ef{ef}"]) / 100000,
            } for ef in EFS},
            "sources": [{"path": item.name, "sha256": digest(item)}
                        for item in (path, provenance_path)]}


def summarize(directory):
    summary = {"schema_version": 1, "latency_unit": "ms/query", "latency_clock": "real_time",
               "aggregation": "median across all four mixed or three native process medians; each process median is over three repeated mean-latency samples",
               "primary_mixed_runs": MIXED, "primary_native_runs": NATIVE,
               "baseline_aa_runs": [run for run in RUNS if run.startswith("mixed-")],
               "datasets": {}, "sources": []}
    lines = ["FP16 paired saved-graph comparison (1000 queries, k=100, full corpus)",
             "Latency: ms/query, real_time; median of process medians of 3 repeated mean-latency samples.",
             "Primary comparison: all 4 mixed processes vs all 3 native processes.",
             "A/A spread: (max - min) / median of all 4 mixed process medians.",
             "Deltas are native minus mixed; negative latency means faster.", ""]
    for dataset in ("single", "multi"):
        runs = {run: load_run(directory, dataset, run) for run in RUNS}
        identity = runs[RUNS[0]]["identity"]
        for run, value in runs.items():
            require(value["identity"] == identity,
                    f"{dataset}/{run}: data/query/graph/ground-truth identity differs")
        for mode_runs in ([run for run in RUNS if run.startswith("mixed-")], NATIVE):
            reference = runs[mode_runs[0]]["hits"]
            for run in mode_runs:
                require(runs[run]["hits"] == reference,
                        f"{dataset}/{run}: per-query hits changed within the same mode")
        ef_summary = {}
        lines.append(dataset + ":")
        for ef in EFS:
            key = str(ef)
            mixed = statistics.median(runs[run]["ef"][key]["median_real_time_ms"] for run in MIXED)
            native = statistics.median(runs[run]["ef"][key]["median_real_time_ms"] for run in NATIVE)
            aa = [runs[run]["ef"][key]["median_real_time_ms"]
                  for run in RUNS if run.startswith("mixed-")]
            spread = (max(aa) - min(aa)) / statistics.median(aa)
            mh = runs[MIXED[0]]["hits"][f"ef{ef}"]
            nh = runs[NATIVE[0]]["hits"][f"ef{ef}"]
            mr, nr = sum(mh) / 100000, sum(nh) / 100000
            gains, losses = sum(n > m for m, n in zip(mh, nh)), sum(n < m for m, n in zip(mh, nh))
            ef_summary[key] = {"mixed_latency_ms": mixed, "native_latency_ms": native,
                               "mixed_recall_fraction": mr, "native_recall_fraction": nr,
                               "native_minus_mixed_recall_percentage_points": (nr - mr) * 100,
                               "native_minus_mixed_latency_percent": (native / mixed - 1) * 100,
                               "baseline_aa_latency_spread_fraction": spread,
                               "per_query_native_gain_count": gains,
                               "per_query_native_loss_count": losses,
                               "per_query_equal_count": 1000 - gains - losses,
                               "per_query_native_minus_mixed_hits": [n - m for m, n in zip(mh, nh)]}
            lines.append(f"  ef{ef}: latency mixed={mixed:.9g}, native={native:.9g} ms/query "
                         f"({(native / mixed - 1) * 100:+.4f}%); recall mixed={mr:.6f}, "
                         f"native={nr:.6f} ({(nr - mr) * 100:+.4f} pp); "
                         f"A/A spread={spread * 100:.4f}%; query gains/losses/equal="
                         f"{gains}/{losses}/{1000-gains-losses}")
        summary["datasets"][dataset] = {"identity": identity, "ef": ef_summary,
                                          "runs": {run: value["ef"] for run, value in runs.items()}}
        summary["sources"].extend(source for value in runs.values() for source in value["sources"])
    evidence = sorted({path for path in directory.iterdir() if path.is_file() and (
        path.name == "production-source.txt" or path.suffix in (".sha256", ".sha256sum")
        or "hash" in path.name.lower())})
    for path in evidence:
        source = {"path": path.name, "sha256": digest(path)}
        if path.stat().st_size <= 1024 * 1024:
            try:
                source["text"] = path.read_text()
            except UnicodeDecodeError:
                pass
        summary["sources"].append(source)
    lines.extend(["", "Input evidence (SHA256):"])
    lines.extend(f"  {source['path']}: {source['sha256']}" for source in summary["sources"])
    return summary, "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        directory = Path(sys.argv[1])
        summary, readable = summarize(directory)
        # Write only after every input has passed validation.
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (directory / "summary.txt").write_text(readable)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Paired summary failed: {error}", file=sys.stderr)
        return 1
    print(readable, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
