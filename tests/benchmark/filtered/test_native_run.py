# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import filtered_bench as bench
import native_run as native


class NativeWrapperTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle_path = self.root / "bundle"
        self.bundle = bench.write_bundle(self.bundle_path,
            np.asarray([[0, 0], [1, 0], [-1, 0], [2, 0]], dtype=np.float32),
            np.asarray([[0, 0]], dtype=np.float32), "l2", 7, {"source": "native-test"})
        self.binary = self.root / "binary"
        self.binary.write_bytes(b"stand-in for binary provenance tests")

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, command, **changes):
        args = argparse.Namespace(command=command, bundle=str(self.bundle_path),
            binary=str(self.binary), output=str(self.root / command), process_timeout_seconds=10,
            m=16, ef_construction=200, policies=["NATIVE_SCAN"], repeats=1,
            ef_runtime=200, batch_size=100, timeout_ms=10000, graph=str(self.root / "build"))
        vars(args).update(changes)
        return args

    def rows(self, policies=None, repeats=1):
        truth = list(bench.oracle_rows(self.bundle))
        return [{"case_index": index, "repeat": repeat, "requested_policy": policy,
                 "effective_policy": policy, "status": "ok", "ids": row["ids"],
                 "distances": row["distances"], "latency_ms": 0.125,
                 "counters": {"distance_evaluations": row["eligible_count"]}}
                for index, row in enumerate(truth) for repeat in range(repeats)
                for policy in policies or ["NATIVE_SCAN"]]

    def write_rows(self, rows):
        path = self.root / "raw.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def invoke(self, command, output, timeout):
        if command[1] == "build":
            (output / "graph.hnsw").write_bytes(b"test graph")
        else:
            policies = command[command.index("--policies") + 1].split(",")
            repeats = int(command[command.index("--repeats") + 1])
            (output / "native-results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows(policies, repeats)))
        return {"compiler": "test", "build_type": "Release", "index_bytes": 128}, 0.1

    def build_graph(self):
        with patch.object(native, "binary_description", return_value={"policies": ["NATIVE_SCAN"]}), \
                patch.object(native, "invoke_native", side_effect=self.invoke):
            self.assertEqual(native.execute(self.arguments("build")), 0)

    def test_binary_eligibility_offsets_preserve_all_cases(self):
        output = self.root / "inputs"
        output.mkdir()
        native.write_native_inputs(self.bundle, output)
        ids = np.fromfile(output / "filters.u64", dtype="<u8")
        for line, case in zip((output / "cases.tsv").read_text().splitlines(), self.bundle.cases):
            index, query, k, offset, count = map(int, line.split("\t"))
            self.assertEqual(self.bundle.cases[index], case)
            self.assertEqual((query, k), (case["query_id"], case["k"]))
            self.assertEqual(ids[offset:offset + count].tolist(), self.bundle.filters[case["filter_id"]])

    def test_complete_native_matrix_and_timeout_records(self):
        rows = self.rows(["NATIVE_SCAN", "NATIVE_BATCHES"], repeats=2)
        rows[0].update(status="timeout", ids=[], distances=[], error="deadline exceeded")
        converted = native.convert_results(self.bundle, self.write_rows(rows),
                                           ["NATIVE_SCAN", "NATIVE_BATCHES"], 2)
        self.assertEqual(len(converted), len(self.bundle.cases) * 4)
        self.assertEqual(converted[0]["status"], "timeout")
        self.assertEqual(converted[0]["bundle_id"], self.bundle.bundle_id)
        self.assertEqual(converted[0]["case_id"], self.bundle.cases[0]["case_id"])
        self.assertNotIn("case_index", converted[0])

    def test_native_protocol_rejects_missing_duplicate_and_wrong_records(self):
        valid = self.rows()
        mutations = [valid[:-1], valid + [valid[0]]]
        for change in ({"case_index": True}, {"repeat": 1}, {"effective_policy": "AUTO"},
                       {"status": "missing"}, {"latency_ms": float("nan")},
                       {"ids": [0, 0], "distances": [0, 0]},
                       {"ids": [99], "distances": [0]}, {"counters": {"visits": -1}}):
            rows = copy.deepcopy(valid)
            rows[0].update(change)
            mutations.append(rows)
        for rows in mutations:
            with self.subTest(first_row=rows[0]), self.assertRaises(ValueError):
                native.convert_results(self.bundle, self.write_rows(rows), ["NATIVE_SCAN"], 1)

    def test_graph_integrity_and_same_corpus_reuse(self):
        self.build_graph()
        graph, metadata = native.validate_graph(self.root / "build", self.bundle)
        changed = copy.copy(self.bundle)
        changed.bundle_id = "different-query-and-filter-bundle"
        native.validate_graph(self.root / "build", changed)
        changed = copy.deepcopy(self.bundle)
        changed.manifest["metric"] = "cosine"
        with self.assertRaisesRegex(ValueError, "corpus"):
            native.validate_graph(self.root / "build", changed)
        graph.write_bytes(b"modified graph")
        with self.assertRaisesRegex(ValueError, "checksum"):
            native.validate_graph(self.root / "build", self.bundle)
        self.assertEqual(metadata["creator_bundle_id"], self.bundle.bundle_id)

    def test_registered_candidate_label_reaches_binary_and_scorer_contract(self):
        self.build_graph()
        with patch.object(native, "binary_description", return_value={"policies": ["CANDIDATE_TEST"]}), \
                patch.object(native, "invoke_native", side_effect=self.invoke):
            args = self.arguments("run", policies=["CANDIDATE_TEST"], repeats=2)
            self.assertEqual(native.execute(args), 0)
            with self.assertRaises(FileExistsError):
                native.execute(args)
        output = self.root / "run"
        metadata = bench.read_json(output / "run.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["binary_sha256"], bench.sha256(self.binary))
        truth = {row["case_id"]: row for row in bench.oracle_rows(self.bundle)}
        report = bench.score_results(self.bundle, bench.read_jsonl(output / "results.jsonl"), truth,
                                     ["CANDIDATE_TEST"], 2)
        self.assertTrue(all(row["recall"] in (None, 1) for row in report["queries"]))

    def test_unknown_adapter_fails_before_creating_output(self):
        with patch.object(native, "binary_description", return_value={"policies": ["NATIVE_SCAN"]}):
            with self.assertRaisesRegex(ValueError, "registered"):
                native.execute(self.arguments("run", policies=["NAVIX"]))
        self.assertFalse((self.root / "run").exists())

    def test_fatal_native_failure_retains_failed_metadata(self):
        with patch.object(native, "binary_description", return_value={"policies": ["NATIVE_SCAN"]}), \
                patch.object(native, "invoke_native", side_effect=OSError("injected native failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                native.execute(self.arguments("build"))
        metadata = bench.read_json(self.root / "build" / "build.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("injected native failure", metadata["error"])
        self.assertFalse((self.root / "build" / "graph.json").exists())

    def test_timeout_rows_are_retained_and_run_fails(self):
        self.build_graph()

        def timeout_run(command, output, timeout):
            rows = self.rows()
            rows[0].update(status="timeout", ids=[], distances=[])
            (output / "native-results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            return {"compiler": "test"}, 0.1

        with patch.object(native, "binary_description", return_value={"policies": ["NATIVE_SCAN"]}), \
                patch.object(native, "invoke_native", side_effect=timeout_run):
            self.assertEqual(native.execute(self.arguments("run")), 1)
        output = self.root / "run"
        rows = list(bench.read_jsonl(output / "results.jsonl"))
        self.assertEqual(len(rows), len(self.bundle.cases))
        self.assertEqual(rows[0]["status"], "timeout")
        metadata = bench.read_json(output / "run.json")
        self.assertEqual(metadata["status"], "completed_with_failures")
        self.assertEqual(metadata["failure_count"], 1)


@unittest.skipUnless(os.environ.get("VECSIM_FILTERED_BINARY"), "set VECSIM_FILTERED_BINARY for native integration")
class NativeIntegrationTest(unittest.TestCase):
    def test_frozen_graph_baselines_and_oracle_for_l2_and_cosine(self):
        binary = str(Path(os.environ["VECSIM_FILTERED_BINARY"]).resolve())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(20221)
            for metric in ("l2", "cosine"):
                with self.subTest(metric=metric):
                    bundle_path = root / metric
                    bundle = bench.write_bundle(bundle_path,
                        rng.normal(size=(40, 8)).astype(np.float32),
                        rng.normal(size=(2, 8)).astype(np.float32), metric, 20221, {"source": "integration"})
                    graph = root / (metric + "-graph")
                    output = root / (metric + "-run")
                    self.assertEqual(native.main(["build", "--bundle", str(bundle_path),
                        "--binary", binary, "--output", str(graph), "--m", "8"]), 0)
                    graph_hash = bench.sha256(graph / "graph.hnsw")
                    self.assertEqual(native.main(["run", "--bundle", str(bundle_path),
                        "--binary", binary, "--graph", str(graph), "--output", str(output),
                        "--repeats", "2", "--batch-size", "3"]), 0)
                    truth = {row["case_id"]: row for row in bench.oracle_rows(bundle)}
                    report = bench.score_results(bundle, bench.read_jsonl(output / "results.jsonl"),
                                                 truth, native.DEFAULT_POLICIES, 2)
                    self.assertTrue(all(row["status"] == "ok" and row["complete"] and
                                        row["recall"] in (None, 1) for row in report["queries"]))
                    self.assertEqual(graph_hash, bench.sha256(graph / "graph.hnsw"))
                    command = bench.read_json(output / "run.json")["native_command"]
                    command[command.index("--output") + 1] = str(root / "rejected.jsonl")
                    bad_cases = root / "bad-cases.tsv"
                    for field, value, message in (
                            ("--policies", "UNIMPLEMENTED_CANDIDATE", "Unknown or repeated"),
                            ("--cases", str(bad_cases), "Invalid query, filter slice")):
                        with self.subTest(rejected_field=field):
                            bad_cases.write_text("0\t999\t10\t0\t1\n")
                            invalid = command.copy()
                            invalid[invalid.index(field) + 1] = value
                            completed = subprocess.run(invalid, capture_output=True, text=True, timeout=30)
                            self.assertNotEqual(completed.returncode, 0)
                            self.assertIn(message, completed.stderr)
                            self.assertFalse((root / "rejected.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
