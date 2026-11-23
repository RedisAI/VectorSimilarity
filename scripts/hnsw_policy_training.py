# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

"""Train and evaluate the RAM HNSW policy selector using paired measurements."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier


ADHOC = "ADHOC_BF"
BATCHES = "BATCHES"
AUTO = "AUTO"
FEATURES = ("k", "index_size", "dim", "m", "selectivity")
AUTO_MODES = {ADHOC, BATCHES, "BATCHES_TO_ADHOC_BF"}


@dataclass(frozen=True)
class Measurement:
    ms: float
    recall: float
    mode: str


@dataclass(frozen=True)
class Case:
    workload: str
    name: str
    cohort: str
    features: tuple
    subset_size: int
    actual_matches: int | None
    runs: dict


@dataclass(frozen=True)
class Outcome:
    case: Case
    qualified_ms: dict
    winner: str
    weight: float


@dataclass(frozen=True)
class FittedPolicy:
    tree: DecisionTreeClassifier
    leaf_choices: dict

    def predict(self, features):
        return [self.leaf_choices[node] for node in self.tree.apply(features)]


def integer(record, key, minimum=1):
    value = record[key]
    if type(value) is not int or not minimum <= value <= 2**64 - 1:
        raise ValueError(f"{key} must be an integer in [{minimum}, 2**64 - 1]")
    return value


def number(record, key, minimum, maximum=math.inf):
    value = record[key]
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be in [{minimum}, {maximum}]")
    return float(value)


def nonempty_string(record, key):
    value = record[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def measurement(record, policy):
    status = record["status"]
    if status == "timeout":
        # A timeout is a measured failure, with no fabricated latency or recall.
        return None
    if status != "ok":
        raise ValueError(f"{policy} status must be ok or timeout, got {status!r}")
    mode = nonempty_string(record, "mode")
    if mode not in (AUTO_MODES if policy == AUTO else {policy}):
        raise ValueError(f"{policy} ran in unexpected mode {mode!r}")
    ms = number(record, "ms", 0)
    if ms == 0:
        raise ValueError("ms must be positive")
    return Measurement(ms, number(record, "recall", 0, 1), mode)


def parse_case(record, estimate_source):
    workload = nonempty_string(record, "workload")
    name = nonempty_string(record, "case")
    cohort = nonempty_string(record, "cohort")
    if record["estimate_source"] != estimate_source:
        raise ValueError(f"expected estimate_source={estimate_source!r}")
    k = integer(record, "k")
    size = integer(record, "index_size")
    labels = integer(record, "label_count")
    dim = integer(record, "dim")
    m = integer(record, "m")
    subset = integer(record, "subset_size", minimum=0)
    actual = integer(record, "actual_matches", minimum=0) if "actual_matches" in record else None
    if labels > size or (actual is not None and actual > labels):
        raise ValueError("expected actual_matches <= label_count <= index_size")
    if estimate_source == "exact" and actual != subset:
        raise ValueError("exact selectivity requires subset_size == actual_matches")
    # Mirror preferAdHocSearch's clamp and float division. The denominator
    # counts labels, while index_size counts vectors, even for multi-value fields.
    selectivity = np.float32(min(subset, size)) / np.float32(labels)
    features = (k, size, dim, m, float(selectivity))

    pairs = record["runs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("runs must be a nonempty list of paired repetitions")
    policies = {ADHOC, BATCHES}
    if AUTO in pairs[0]:
        policies.add(AUTO)
    runs = {policy: [] for policy in sorted(policies)}
    for pair in pairs:
        if set(pair) != policies:
            raise ValueError("each repetition must contain ADHOC_BF and BATCHES, with optional AUTO")
        for policy in runs:
            runs[policy].append(measurement(pair[policy], policy))
    return Case(workload, name, cohort, features, subset, actual, runs)


def load_cases(paths, estimate_source):
    cases = {}
    sources = []
    for path in paths:
        raw = path.read_bytes()
        sources.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        for line_no, line in enumerate(raw.decode().splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = parse_case(json.loads(line), estimate_source)
                key = (case.workload, case.name)
                if key in cases:
                    raise ValueError(f"duplicate case {key!r}; combine its paired repetitions")
                cases[key] = case
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return [cases[key] for key in sorted(cases)], sources


def score_case(case, recall_floor, margin):
    qualified = {
        policy: median(run.ms for run in runs)
        for policy, runs in case.runs.items()
        if all(run is not None and run.recall >= recall_floor for run in runs)
    }
    if any(run is not None and run.recall < recall_floor for run in case.runs[ADHOC]):
        # Unexpectedly low exact recall suggests a bad comparison. A measured
        # timeout is different: BATCHES may meet the deadline and recall floor.
        return None
    forced = [policy for policy in (ADHOC, BATCHES) if policy in qualified]
    if not forced:
        return None
    best = min(forced, key=qualified.get)
    if len(forced) == 1:
        return Outcome(case, qualified, best, math.log(10))
    speedup = (max(qualified[ADHOC], qualified[BATCHES]) /
               min(qualified[ADHOC], qualified[BATCHES]))
    weight = min(math.log(speedup), math.log(10)) if speedup > 1 + margin else 0.0
    return Outcome(case, qualified, best, weight)


def fit(outcomes, max_leaves, min_leaf):
    decisive = [item for item in outcomes if item.weight > 0]
    if not decisive:
        raise ValueError("training set contains no decisive, recall-qualified comparisons")
    # Equal total influence per workload, then weight by the cost of a wrong
    # decision within that workload. Repetitions never become duplicate rows.
    workload_weights = defaultdict(float)
    for row in decisive:
        workload_weights[row.case.workload] += row.weight
    weights = [row.weight / workload_weights[row.case.workload] for row in decisive]
    model = DecisionTreeClassifier(
        max_leaf_nodes=max_leaves, min_samples_leaf=min_leaf, random_state=0
    )
    model.fit(
        [row.case.features for row in decisive], [row.winner for row in decisive],
        sample_weight=weights,
    )
    features = [row.case.features for row in outcomes]
    nodes = model.apply(features)
    failures = defaultdict(Counter)
    leaf_choices = dict(zip(nodes, model.predict(features), strict=True))
    for node, item in zip(nodes, outcomes, strict=True):
        for policy in (ADHOC, BATCHES):
            if policy not in item.qualified_ms:
                failures[node][policy] += 1
    for node, preferred in leaf_choices.items():
        # The weighted split finds speedups. Within a leaf, use the policy
        # with fewer observed quality/deadline failures first.
        leaf_choices[node] = min((ADHOC, BATCHES), key=lambda policy: (
            failures[node][policy], policy != preferred,
        ))
    return FittedPolicy(model, leaf_choices)


def describe_choices(outcomes, choices):
    slowdowns = []
    auto_ratios = []
    failures = 0
    for item, choice in zip(outcomes, choices, strict=True):
        selected = item.qualified_ms.get(choice)
        if selected is None:
            failures += 1
            continue
        slowdowns.append(selected / item.qualified_ms[item.winner])
        if AUTO in item.qualified_ms:
            auto_ratios.append(selected / item.qualified_ms[AUTO])

    return {
        "cases": len(outcomes),
        "selected": dict(sorted(Counter(choices).items())),
        "quality_or_timeout_failures": failures,
        "within_5pct_of_best_forced": sum(value <= 1.05 for value in slowdowns),
        "median_slowdown": median(slowdowns) if slowdowns else None,
        "p90_slowdown": float(np.quantile(slowdowns, 0.9)) if slowdowns else None,
        "worst_slowdown": max(slowdowns, default=None),
        "comparable_auto_cases": len(auto_ratios),
        "median_latency_over_measured_auto": median(auto_ratios) if auto_ratios else None,
    }


def evaluate(model, outcomes):
    if not outcomes:
        return None
    choices = model.predict([item.case.features for item in outcomes])
    cohorts = defaultdict(list)
    winners_by_features = defaultdict(set)
    for item, choice in zip(outcomes, choices, strict=True):
        cohorts[item.case.cohort].append((item, choice))
        if item.weight > 0:
            winners_by_features[tuple(np.float32(item.case.features))].add(item.winner)
    return {
        "candidate": describe_choices(outcomes, choices),
        "always_adhoc": describe_choices(outcomes, [ADHOC] * len(outcomes)),
        "always_batches": describe_choices(outcomes, [BATCHES] * len(outcomes)),
        "measured_auto": describe_choices(
            [item for item in outcomes if AUTO in item.case.runs],
            [AUTO for item in outcomes if AUTO in item.case.runs],
        ),
        "auto_recorded_cases": sum(AUTO in item.case.runs for item in outcomes),
        "auto_modes": dict(sorted(Counter(
            run.mode if run else "timeout"
            for item in outcomes for run in item.case.runs.get(AUTO, [])
        ).items())),
        "conflicting_decisive_feature_vectors": sum(
            len(winners) > 1 for winners in winners_by_features.values()
        ),
        "by_cohort": {
            cohort: describe_choices(*zip(*rows)) for cohort, rows in sorted(cohorts.items())
        },
        "decisions": [{
            "workload": item.case.workload,
            "case": item.case.name,
            "cohort": item.case.cohort,
            "features": dict(zip(FEATURES, item.case.features)),
            "subset_size": item.case.subset_size,
            "actual_matches": item.case.actual_matches,
            "best_forced": item.winner,
            "decisive": item.weight > 0,
            "choice": choice,
            "qualified_ms": item.qualified_ms,
        } for item, choice in zip(outcomes, choices, strict=True)],
    }


def export_rule(model, node=0):
    tree = model.tree.tree_
    left, right = tree.children_left[node], tree.children_right[node]
    if left == right:
        return {"policy": str(model.leaf_choices[node])}
    if_le, if_gt = export_rule(model, left), export_rule(model, right)
    if if_le == if_gt:
        return if_le
    return {
        "feature": FEATURES[tree.feature[node]],
        "threshold": float(tree.threshold[node]),
        "if_le": if_le,
        "if_gt": if_gt,
    }


def print_rule(rule, depth=0):
    indent = "  " * depth
    if "policy" in rule:
        print(f"{indent}{rule['policy']}")
        return
    print(f"{indent}if {rule['feature']} <= {rule['threshold']:.9g}")
    print_rule(rule["if_le"], depth + 1)
    print(f"{indent}else")
    print_rule(rule["if_gt"], depth + 1)


def export_cpp(rule):
    """Emit an offline candidate with the same float32 inputs used by sklearn."""
    expressions = {
        "k": "static_cast<float>(input.k)",
        "index_size": "static_cast<float>(input.indexSize)",
        "dim": "static_cast<float>(input.dimension)",
        "m": "static_cast<float>(input.m)",
        "selectivity": "selectivity",
    }

    def render(node, depth):
        indent = "    " * depth
        if "policy" in node:
            return [indent + ("return true;" if node["policy"] == ADHOC else "return false;")]
        expression = expressions[node["feature"]]
        # Preserve the double-precision threshold, including the midpoint
        # between adjacent values of a float32 feature.
        threshold = node["threshold"].hex()
        return [
            f"{indent}if ({expression} <= {threshold}) {{",
            *render(node["if_le"], depth + 1),
            f"{indent}}}",
            *render(node["if_gt"], depth),
        ]

    header = """// Generated by scripts/HNSW_batches_clf.py.
// Candidate for offline evaluation.
#pragma once

#include <algorithm>
#include <cstddef>

struct HnswPolicyInput {
    std::size_t k;
    std::size_t indexSize;
    std::size_t dimension;
    std::size_t m;
    std::size_t subsetSize;
    std::size_t labelCount;
};

// True selects ADHOC_BF, false selects BATCHES.
inline bool preferAdHocSearchCandidate(const HnswPolicyInput &input) {
    [[maybe_unused]] const float selectivity =
        input.indexSize == 0
            ? 0.0f
            : static_cast<float>(std::min(input.subsetSize, input.indexSize)) /
                  static_cast<float>(input.labelCount);
"""
    return header + "\n".join(render(rule, 1)) + "\n}\n"


def train(args):
    cases, sources = load_cases(args.measurements, args.estimate_source)
    workloads = sorted({case.workload for case in cases})
    held_out = set(args.holdout_workload)
    if missing := held_out.difference(workloads):
        raise ValueError(f"unknown holdout workloads: {', '.join(sorted(missing))}")
    by_workload = {name: [] for name in workloads}
    unusable = Counter()
    for case in cases:
        outcome = score_case(case, args.min_recall, args.min_speedup)
        if outcome:
            by_workload[case.workload].append(outcome)
        else:
            unusable[case.workload] += 1

    train_groups = [name for name in workloads if name not in held_out]
    training = [item for name in train_groups for item in by_workload[name]]
    model = fit(training, args.max_leaf_nodes, args.min_samples_leaf)
    cross_validation = {}
    for name in train_groups:
        other_cases = [item for item in training if item.case.workload != name]
        if not by_workload[name] or not any(item.weight > 0 for item in other_cases):
            cross_validation[name] = None
            continue
        fold_model = fit(other_cases, args.max_leaf_nodes, args.min_samples_leaf)
        cross_validation[name] = evaluate(fold_model, by_workload[name])

    return model, {
        "format_version": 1,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "sources": sources,
        "estimate_source": args.estimate_source,
        "features": FEATURES,
        "parameters": {
            "min_recall": args.min_recall,
            "min_speedup": args.min_speedup,
            "max_leaf_nodes": args.max_leaf_nodes,
            "min_samples_leaf": args.min_samples_leaf,
        },
        "workloads": {
            name: {
                "role": "holdout" if name in held_out else "train",
                "usable": len(by_workload[name]),
                "decisive": dict(sorted(Counter(
                    item.winner for item in by_workload[name] if item.weight > 0
                ).items())),
                "unusable_comparisons": unusable[name],
            } for name in workloads
        },
        "training": evaluate(model, training),
        "leave_one_workload_out": cross_validation,
        "holdout": {name: evaluate(model, by_workload[name]) for name in sorted(held_out)},
        "tree": export_rule(model),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", nargs="+", type=Path, required=True, metavar="JSONL")
    parser.add_argument("--report", type=Path, required=True, help="write checksummed inputs, evaluation and tree as JSON")
    parser.add_argument("--cpp", type=Path, metavar="HEADER", help="write a standalone C++17 candidate")
    parser.add_argument("--holdout-workload", action="append", default=[], metavar="NAME")
    parser.add_argument("--estimate-source", choices=("planner", "exact"), default="planner",
                        help="require planner estimates (default) or run a separate exact-count experiment")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-speedup", type=float, default=0.05,
                        help="required latency ratio minus one to use a comparison for fitting")
    parser.add_argument("--max-leaf-nodes", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.min_recall <= 1 or not math.isfinite(args.min_speedup) or args.min_speedup < 0:
        parser.error("min-recall must be between 0 and 1 and min-speedup must be finite and nonnegative")
    if args.max_leaf_nodes < 2 or args.min_samples_leaf < 1:
        parser.error("max-leaf-nodes must be at least 2 and min-samples-leaf must be positive")
    output_paths = [path.resolve() for path in (args.report, args.cpp) if path is not None]
    if len(output_paths) != len(set(output_paths)):
        parser.error("report and C++ output must be different files")
    if set(output_paths).intersection(path.resolve() for path in args.measurements):
        parser.error("outputs must not overwrite their inputs")
    try:
        _, report = train(args)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.cpp is not None:
            args.cpp.write_text(export_cpp(report["tree"]), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print_rule(report["tree"])
