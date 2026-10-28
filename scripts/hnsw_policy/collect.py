# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Collect raw paired timings and retain independent index shapes for validation."""

import argparse
import csv
import hashlib
import io
import json
import platform
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workloads", type=Path, default=Path(__file__).with_name("workloads.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    workloads = json.loads(args.workloads.read_text())
    metadata = {
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown"),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256((repo / "tests/benchmark/bm_hnsw_policy.cpp").read_bytes()).hexdigest(),
        "workloads": workloads,
    }
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("output directory belongs to a different build or workload set")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    keys = ["n", "dim", "m", "k", "subset", "bytes", "seed", "clustered"]
    with (args.output / "measurements.csv").open("w") as summary, \
            (args.output / "samples.csv").open("w") as samples:
        writer = csv.DictWriter(summary, lineterminator="\n", fieldnames=keys + ["split", "adhoc_us", "batches_us", "recall", "old_adhoc", "batch_count"])
        writer.writeheader()
        sample_writer = csv.DictWriter(samples, lineterminator="\n", fieldnames=keys + ["split", "query", "repeat", "adhoc_us", "batches_us", "recall", "old_adhoc", "batch_count"])
        sample_writer.writeheader()
        for split, configurations in workloads.items():
            for config in configurations:
                name = "-".join(map(str, config))
                print(f"Measuring {split}: {name}", flush=True)
                raw_path = args.output / f"{split}-{name}.csv"
                if raw_path.exists():
                    raw = raw_path.read_text()
                else:
                    partial = raw_path.with_suffix(".partial.csv")
                    with partial.open("w") as stdout, (args.output / f"{split}-{name}.log").open("w") as stderr:
                        result = subprocess.run([str(args.executable), *map(str, config)],
                                                stdout=stdout, stderr=stderr)
                    result.check_returncode()
                    partial.replace(raw_path)
                    raw = raw_path.read_text()
                grouped = defaultdict(list)
                # Test builds log index growth before emitting the CSV header.
                raw = raw[raw.index("n,dim,m,k,subset,"):]
                for row in csv.DictReader(io.StringIO(raw)):
                    grouped[tuple(row[key] for key in keys)].append(row)
                if len(grouped) != 54 or any(len(rows) != 9 for rows in grouped.values()):
                    raise ValueError(f"incomplete benchmark output for {name}")
                for key, rows in grouped.items():
                    sample_writer.writerows(dict(row, split=split) for row in rows)
                    output = dict(zip(keys, key), split=split)
                    for metric in ["adhoc_us", "batches_us", "recall"]:
                        output[metric] = statistics.median(float(row[metric]) for row in rows)
                    output["old_adhoc"] = rows[0]["old_adhoc"]
                    output["batch_count"] = min(int(row["batch_count"]) for row in rows)
                    writer.writerow(output)
                summary.flush()
                samples.flush()


if __name__ == "__main__":
    main()
