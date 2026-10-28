# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Retrain the affected HNSW policy leaf from paired latency measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

FEATURES = ("k", "n", "dim", "m", "ratio")
BATCHES_MARGIN = 1.2
LICENSE = """/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
"""


def read_measurements(path):
    """Reject incomplete/invalid input rather than silently changing the training population."""
    with path.open() as source:
        rows = list(csv.DictReader(source))
    samples = []
    for row in rows:
        integers = {key: int(row[key]) for key in ("n", "dim", "m", "k", "subset", "bytes", "batch_count")}
        if any(integers[key] <= 0 for key in ("n", "dim", "m", "k", "subset")):
            raise ValueError("index, query and subset sizes must be positive")
        if integers["subset"] > integers["n"] or row["split"] not in ("train", "validation"):
            raise ValueError("invalid subset or dataset split")
        if integers["bytes"] not in (4, 8) or not 0 <= integers["batch_count"] <= min(integers["k"], integers["subset"]):
            raise ValueError("invalid element size or result count")
        a, b = float(row["adhoc_us"]), float(row["batches_us"])
        if not np.isfinite([a, b]).all() or min(a, b) <= 0:
            raise ValueError("timings must be finite and positive")
        ratio = np.float32(integers["subset"]) / np.float32(integers["n"])
        # Preserve the historical policy outside the leaf covered by this experiment.
        if not (30000 < integers["n"] <= 300000 and integers["dim"] > 75
                and integers["k"] > 12 and integers["m"] > 10 and ratio >= np.float32(0.07)):
            continue
        samples.append((row, integers, ratio, a, b))
    if not samples:
        raise ValueError("no measurements cover the affected leaf")
    x = np.asarray([[v["k"], v["n"], v["dim"], v["m"], r] for _, v, r, _, _ in samples], dtype=np.float32)
    a = np.asarray([a for _, _, _, a, _ in samples])
    b = np.asarray([b for _, _, _, _, b in samples])
    complete = np.asarray([v["batch_count"] == min(v["k"], v["subset"]) for _, v, _, _, _ in samples])
    training = np.asarray([row["split"] == "train" for row, *_ in samples])
    groups = np.asarray([f'{v["n"]}:{v["dim"]}:{v["m"]}' for _, v, *_ in samples])
    if set(groups[training]) & set(groups[~training]):
        raise ValueError("training and validation must use independent index shapes")
    if not training.any() or not (~training).any():
        raise ValueError("both training and validation measurements are required")
    return x, a, b, complete, training, groups


def metrics(predicted_adhoc, a, b, complete):
    """Latency ratios compare against the faster strategy with a complete result set."""
    oracle = np.where(complete, np.minimum(a, b), a)
    chosen = np.where(predicted_adhoc, a, b)
    ratios = chosen / oracle
    return {
        "samples": len(a),
        "incomplete_batch_choices": int(np.sum(~predicted_adhoc & ~complete)),
        "geometric_mean_slowdown": float(np.exp(np.mean(np.log(ratios)))),
        "p95_slowdown": float(np.quantile(ratios, 0.95)),
        "worst_slowdown": float(np.max(ratios)),
        "aggregate_latency_us": float(np.sum(chosen)),
        "batches_choices": int(np.sum(~predicted_adhoc)),
    }


def fit(x, a, b, complete, groups):
    """Choose complexity using grouped CV; validation shapes never influence model selection."""
    # Preserve the old decision when the measured gain is small enough to be timing noise.
    y = (a <= BATCHES_MARGIN * b) | ~complete
    # A wrong choice that doubles latency matters more than one within measurement noise.
    weights = np.where(complete, np.maximum(0.05, np.abs(np.log(a / b))), 20.0)
    trials = []
    for leaves in (4, 8, 12, 20):
        for min_leaf in (3, 6):
            prediction = np.zeros(len(x), dtype=bool)
            for train, test in GroupKFold(n_splits=4).split(x, y, groups):
                model = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=min_leaf, random_state=0)
                model.fit(x[train], y[train], sample_weight=weights[train])
                prediction[test] = model.predict(x[test])
            score = metrics(prediction, a, b, complete)
            trials.append((score["incomplete_batch_choices"], score["geometric_mean_slowdown"], leaves, min_leaf, score))
    _, _, leaves, min_leaf, _ = min(trials)
    model = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=min_leaf, random_state=0)
    model.fit(x, y, sample_weight=weights)
    return model, [{"max_leaf_nodes": t[2], "min_samples_leaf": t[3], **t[4]} for t in trials]


def export_header(model, digest, bounds=None):
    """Use float features on both sides of the Python/C++ boundary, including ratio rounding."""
    used = set(FEATURES) if bounds is not None else {FEATURES[index] for index in model.tree_.feature if index >= 0}
    parameters = ", ".join(("float" if name == "ratio" else "size_t") +
                           (f" {name}" if name in used else "") for name in FEATURES)
    lines = [LICENSE, "#pragma once", "#include <cstddef>", "",
             "// Generated by scripts/hnsw_policy/train.py; do not edit the thresholds by hand.",
             f"// Measurement SHA-256: {digest}",
             "// Called only from the affected leaf of HNSWIndex::preferAdHocSearch.",
             f"inline bool preferHnswAdhocMeasured({parameters}) {{"]
    if bounds is not None:
        lines.append("    // Keep the historical leaf outside the measured feature ranges.")
        conditions = []
        for name, (low, high) in zip(FEATURES, bounds):
            conditions.append(f"!(static_cast<float>({name}) >= {float(low):.17g} && "
                              f"static_cast<float>({name}) <= {float(high):.17g})")
        lines.append("    if (" + " ||\n        ".join(conditions) + ") {")
        lines.append("        return true;")
        lines.append("    }")

    def visit(node, depth):
        indent = "    " * depth
        tree = model.tree_
        if tree.children_left[node] == tree.children_right[node]:
            result = bool(model.classes_[np.argmax(tree.value[node][0])])
            lines.append(indent + f"return {'true' if result else 'false'};")
            return
        feature = FEATURES[tree.feature[node]]
        threshold = format(tree.threshold[node], ".17g")
        lines.append(indent + f"if (static_cast<float>({feature}) <= {threshold}) {{")
        visit(tree.children_left[node], depth + 1)
        lines.append(indent + "} else {")
        visit(tree.children_right[node], depth + 1)
        lines.append(indent + "}")

    visit(0, 1)
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    x, a, b, complete, training, groups = read_measurements(args.measurements)
    model, cv = fit(x[training], a[training], b[training], complete[training], groups[training])
    lower, upper = x[training].min(axis=0), x[training].max(axis=0)
    prediction = model.predict(x).astype(bool)
    prediction |= ~np.all((x >= lower) & (x <= upper), axis=1)
    report = {"cross_validation": cv, "leaves": int(model.get_n_leaves()),
              "parameters": model.get_params(), "numpy": np.__version__,
              "scikit_learn": sklearn.__version__, "batches_margin": BATCHES_MARGIN,
              "feature_bounds": {name: [float(low), float(high)]
                                 for name, low, high in zip(FEATURES, lower, upper)},
              "measurement_sha256": hashlib.sha256(args.measurements.read_bytes()).hexdigest()}
    for name, selection in (("train", training), ("validation", ~training)):
        report[name] = {
            "old": metrics(np.ones(np.sum(selection), dtype=bool), a[selection], b[selection], complete[selection]),
            "new": metrics(prediction[selection], a[selection], b[selection], complete[selection]),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    validation = report["validation"]
    if validation["new"]["incomplete_batch_choices"]:
        raise ValueError("candidate selects batches with incomplete results on validation workloads")
    if validation["new"]["aggregate_latency_us"] >= validation["old"]["aggregate_latency_us"]:
        raise ValueError("candidate does not improve aggregate validation latency")
    changed = (~training) & ~prediction
    if np.any(b[changed] > 2 * a[changed]):
        raise ValueError("candidate introduces a greater-than-2x validation regression")
    digest = hashlib.sha256(args.measurements.read_bytes()).hexdigest()
    bounds = list(zip(lower, upper))
    args.header.write_text(export_header(model, digest, bounds))
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
