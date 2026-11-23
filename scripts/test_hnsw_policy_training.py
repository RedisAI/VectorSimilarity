# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from hnsw_policy_training import (
    ADHOC, AUTO, BATCHES, export_cpp, export_rule, fit, parse_case, score_case, train,
)


def run(policy, ms, recall=1.0):
    return {"status": "ok", "mode": policy, "ms": ms, "recall": recall}


def case(workload="customer", name="q1", k=2000, subset=4000, adhoc_ms=10, batches_ms=2):
    return {
        "workload": workload,
        "case": name,
        "cohort": "numeric-and",
        "k": k,
        "index_size": 5700,
        "label_count": 5700,
        "dim": 1024,
        "m": 16,
        "subset_size": subset,
        "actual_matches": 2000,
        "estimate_source": "planner",
        "runs": [{
            ADHOC: run(ADHOC, adhoc_ms),
            BATCHES: run(BATCHES, batches_ms),
            AUTO: run("BATCHES_TO_ADHOC_BF", 15),
        }],
    }


class PolicyTrainingTests(unittest.TestCase):
    def test_high_k_reaches_model_and_estimate_matches_runtime_clamp(self):
        record = case(k=2000, subset=9999)
        record["label_count"] = 2850
        parsed = parse_case(record, "planner")
        self.assertEqual(parsed.features, (2000, 5700, 1024, 16, 2.0))
        self.assertEqual(parsed.actual_matches, 2000)

    def test_exact_counts_are_kept_out_of_planner_runs(self):
        record = case()
        record["estimate_source"] = "exact"
        with self.assertRaisesRegex(ValueError, "expected estimate_source"):
            parse_case(record, "planner")
        with self.assertRaisesRegex(ValueError, "subset_size == actual_matches"):
            parse_case(record, "exact")

    def test_runtime_switch_is_allowed_only_for_auto(self):
        record = case()
        record["runs"][0][BATCHES] = run("BATCHES_TO_ADHOC_BF", 1)
        with self.assertRaisesRegex(ValueError, "unexpected mode"):
            parse_case(record, "planner")

    def test_quality_overrides_speed_and_measured_timeouts(self):
        record = case()
        record["runs"][0][BATCHES]["recall"] = 0.90
        result = score_case(parse_case(record, "planner"), 0.95, 0.05)
        self.assertEqual(result.winner, ADHOC)
        self.assertNotIn(BATCHES, result.qualified_ms)
        record["runs"][0][BATCHES]["recall"] = 0.99
        record["runs"][0][ADHOC] = {"status": "timeout"}
        result = score_case(parse_case(record, "planner"), 0.95, 0.05)
        self.assertEqual(result.winner, BATCHES)
        record["runs"][0][ADHOC] = run(ADHOC, 10, recall=0.90)
        self.assertIsNone(score_case(parse_case(record, "planner"), 0.95, 0.05))

    def test_leaf_with_conflicting_evidence_uses_fewer_quality_failures(self):
        records = [case(name=f"q{i}") for i in range(6)]
        records[-1]["runs"][0][BATCHES]["recall"] = 0.90
        outcomes = [score_case(parse_case(record, "planner"), 0.95, 0.05) for record in records]
        model = fit(outcomes, max_leaves=4, min_leaf=1)
        self.assertEqual(model.tree.predict([outcomes[0].case.features]).tolist(), [BATCHES])
        self.assertEqual(model.predict([outcomes[0].case.features]), [ADHOC])
        self.assertEqual(export_rule(model), {"policy": ADHOC})

    def test_holdout_is_not_fit_and_rule_matches_model(self):
        records = [case("train-a", "low-k", k=25, adhoc_ms=1),
                   case("train-b", "high-k", k=2000),
                   case("held", "different-high-k", k=2000, adhoc_ms=1)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            args = argparse.Namespace(
                measurements=[path], estimate_source="planner", holdout_workload=["held"],
                min_recall=0.95, min_speedup=0.05, max_leaf_nodes=4, min_samples_leaf=1,
            )
            model, report = train(args)
        self.assertEqual(report["training"]["candidate"]["cases"], 2)
        self.assertEqual(set(report["leave_one_workload_out"]), {"train-a", "train-b"})
        held = report["holdout"]["held"]["decisions"][0]
        self.assertEqual(held["choice"], BATCHES)
        self.assertEqual(held["best_forced"], ADHOC)
        self.assertEqual(report["tree"]["feature"], "k")
        self.assertEqual(report["tree"]["if_le"]["policy"], ADHOC)
        self.assertEqual(report["tree"]["if_gt"]["policy"], BATCHES)
        predicted, = model.predict([parse_case(records[-1], "planner").features])
        self.assertEqual(predicted, held["choice"])

    @unittest.skipUnless(shutil.which("c++"), "a C++ compiler is needed to check header parity")
    def test_generated_cpp_preserves_float32_splits(self):
        lower = case(name="lower", k=16777216, adhoc_ms=1)
        upper = case(name="upper", k=16777218)
        outcomes = [score_case(parse_case(row, "planner"), 0.95, 0.05) for row in (lower, upper)]
        model = fit(outcomes, max_leaves=4, min_leaf=1)
        features = [parse_case(case(k=k), "planner").features
                    for k in (16777216, 16777217, 16777218)]
        expected = [policy == ADHOC for policy in model.predict(features)]
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "policy.h").write_text(export_cpp(export_rule(model)))
            source = folder / "test.cpp"
            cases = "\n".join(
                f"    assert(preferAdHocSearchCandidate({{{k}, 5700, 1024, 16, 4000, 5700}})"
                f" == {str(result).lower()});"
                for (k, *_), result in zip(features, expected, strict=True)
            )
            source.write_text(
                '#include "policy.h"\n#include <cassert>\nint main() {\n' + cases + "\n}\n"
            )
            executable = folder / "test"
            subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", source,
                            "-o", executable], check=True)
            subprocess.run([executable], check=True)


if __name__ == "__main__":
    unittest.main()
