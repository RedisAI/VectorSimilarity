# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Summarize only completed frozen workloads, without excluding regressions."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "frozen-validation.json").read_text())
report = {
    "scope": "Native, same graph, paired with exact BF over preselected IDs; excludes Search filter traversal and network costs.",
    "queries_per_shape": manifest["queries_per_shape"],
    "query_offset": manifest["query_offset"],
    "repeats": 3,
    "regression_threshold": "Median of paired BF/policy latency ratios below 1/1.1 (>10% slowdown).",
    "shapes": [],
    "pending": [],
}
for shape in manifest["shapes"]:
    name = shape["name"]
    if not (root / f"{name}.complete").exists():
        report["pending"].append(name)
        continue
    summary = json.loads((root / f"{name}-summary.json").read_text())
    assert summary["measurements"] == 13440, name
    groups = [r for r in summary["groups"] if r["policy"] == "compiled"]
    result = dict(shape, measurements=summary["measurements"], estimates={})
    for estimate in ["actual", "full"]:
        rows = [r for r in groups if r["estimate_kind"] == estimate]
        result["estimates"][estimate] = {
            "groups": len(rows),
            "median_regressions_over_10pct": sum(r["paired_speedup"] < 1 / 1.1 for r in rows),
            "worst_paired_slowdown": max(1 / r["paired_speedup"] for r in rows),
            "worst_mean_recall": min(r["mean_recall"] for r in rows),
            "worst_query_recall": min(r["min_recall"] for r in rows),
            "customer_like_k200_26pct": [r for r in rows if r["k"] == "200" and r["ratio"] == "0.26"],
            "regressions": sorted(
                [r for r in rows if r["paired_speedup"] < 1 / 1.1],
                key=lambda r: r["paired_speedup"],
            ),
        }
    report["shapes"].append(result)
(root / "reserved-summary.json").write_text(json.dumps(report, indent=2) + "\n")
for shape in report["shapes"]:
    print(shape["name"], {
        e: {k: r[k] for k in ["groups", "median_regressions_over_10pct", "worst_paired_slowdown"]}
        for e, r in shape["estimates"].items()
    })
print("Pending:", report["pending"])
