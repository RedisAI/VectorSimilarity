# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import filtered_bench as bench


class FilteredBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = bench.write_bundle(
            self.root / "bundle", np.array([[0, 0], [1, 0], [-1, 0], [2, 0]], dtype=np.float32),
            np.array([[0, 0]], dtype=np.float32), "l2", 7, {"source": "test"})

    def tearDown(self):
        self.temporary.cleanup()

    def custom_case(self, ids, k):
        self.bundle.filters = {"chosen": ids}
        self.bundle.cases = [{"case_id": "chosen", "query_id": 0, "filter_id": "chosen", "k": k,
                              "cohort": "test"}]
        return {row["case_id"]: row for row in bench.oracle_rows(self.bundle)}

    def result(self, ids, **changes):
        result = {"bundle_id": self.bundle.bundle_id, "case_id": "chosen", "requested_policy": "AUTO",
                  "effective_policy": None, "repeat": 0, "status": "ok", "ids": ids,
                  "distances": [0.0] * len(ids), "latency_ms": 1.0}
        result.update(changes)
        return result

    def score(self, truth, result):
        return bench.score_results(self.bundle, [result], truth, ["AUTO"], 1)["queries"][0]

    def test_known_squared_l2_and_cosine(self):
        np.testing.assert_array_equal(bench.distances_to(self.bundle.vectors, [0, 0], "l2"), [0, 1, 1, 4])
        vectors = np.array([[1, 0], [0, 2], [-3, 0]], dtype=np.float32)
        np.testing.assert_allclose(bench.distances_to(vectors, [1, 0], "cosine"), [0, 1, 2])

    def test_boundary_tie_does_not_replace_strict_neighbor(self):
        truth = self.custom_case([0, 1, 2, 3], 2)
        self.assertEqual(truth["chosen"]["strict_ids"], [0])
        self.assertEqual(truth["chosen"]["boundary_ids"], [1, 2])
        self.assertEqual(self.score(truth, self.result([0, 2]))["recall"], 1)
        self.assertEqual(self.score(truth, self.result([1, 2]))["recall"], 0.5)

    def test_fewer_than_k_and_zero_eligible(self):
        truth = self.custom_case([1], 10)
        self.assertEqual(self.score(truth, self.result([1]))["recall"], 1)
        self.assertFalse(self.score(truth, self.result([]))["complete"])
        truth = self.custom_case([], 10)
        scored = self.score(truth, self.result([]))
        self.assertIsNone(scored["recall"])
        self.assertTrue(scored["complete"])

    def test_bad_results_are_not_successes(self):
        truth = self.custom_case([0, 1], 2)
        for ids in ([0, 0], [0, 2], [-1], [False], [0, 1, 2]):
            with self.subTest(ids=ids):
                row = self.score(truth, self.result(ids))
                self.assertEqual(row["status"], "invalid")
                self.assertEqual(row["recall"], 0)
        self.assertEqual(self.score(truth, self.result([0], distances=[float("nan")]))["status"], "invalid")
        self.assertEqual(self.score(truth, self.result([0], latency_ms=-1))["status"], "invalid")

    def test_missing_runs_and_errors_remain_in_denominator(self):
        truth = self.custom_case([0, 1], 2)
        records = [self.result([], status="timeout")]
        report = bench.score_results(self.bundle, records, truth, ["AUTO", "BATCHES"], 2)
        self.assertEqual(len(report["queries"]), 4)
        self.assertEqual([row["status"] for row in report["queries"]], ["timeout", "missing", "missing", "missing"])
        self.assertTrue(all(row["recall"] == 0 for row in report["queries"]))

    def test_mixed_bundle_and_duplicate_records_rejected(self):
        truth = self.custom_case([0], 1)
        with self.assertRaisesRegex(ValueError, "bundle_id"):
            self.score(truth, self.result([0], bundle_id="other"))
        with self.assertRaisesRegex(ValueError, "duplicate result"):
            bench.score_results(self.bundle, [self.result([0])] * 2, truth, ["AUTO"], 1)

    def test_manifest_checksums_and_file_lengths(self):
        vectors = self.root / "bundle" / "vectors.f32"
        vectors.write_bytes(vectors.read_bytes() + b"extra")
        with self.assertRaisesRegex(ValueError, "checksum"):
            bench.load_bundle(self.root / "bundle")
        manifest_path = self.root / "bundle" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["vectors.f32"] = bench.sha256(vectors)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "byte length"):
            bench.load_bundle(self.root / "bundle")

    def test_bad_filter_and_case_ids_rejected(self):
        root = self.root / "bundle"
        path = root / "filters.json"
        filters = json.loads(path.read_text())
        filters["bad"] = [1, 1]
        path.write_text(json.dumps(filters))
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["files"]["filters.json"] = bench.sha256(path)
        (root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            bench.load_bundle(root)

    def test_oracle_checksum_and_no_overwrite(self):
        path = self.root / "truth.jsonl"
        bench.write_oracle(self.bundle, path)
        self.assertEqual(len(bench.load_oracle(self.bundle, path)), len(self.bundle.cases))
        with self.assertRaisesRegex(ValueError, "already exists"):
            bench.write_oracle(self.bundle, path)
        path.write_text(path.read_text() + "{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            bench.load_oracle(self.bundle, path)

    def test_fbin_header_and_payload_validation(self):
        path = self.root / "base.fbin"
        path.write_bytes(struct.pack("<II", 2, 2) + struct.pack("<ffff", 1, 2, 3, 4))
        np.testing.assert_array_equal(bench.read_matrix(path, "fbin", 2, 2), [[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "header"):
            bench.read_matrix(path, "fbin", 3, 2)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "byte length"):
            bench.read_matrix(path, "fbin", 2, 2)

    def test_nonfinite_and_zero_cosine_rejected(self):
        for vectors in (np.array([[np.inf, 1]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32)):
            with self.assertRaises(ValueError):
                bench.write_bundle(self.root / "bad", vectors, np.ones((1, 2), dtype=np.float32),
                                   "cosine", 1, {})

    def test_deterministic_prepare(self):
        args = ["--documents", "20", "--dimensions", "4", "--queries", "2", "--seed", "5"]
        bench.main(["prepare", "--output", str(self.root / "a"), *args])
        bench.main(["prepare", "--output", str(self.root / "b"), *args])
        self.assertEqual(bench.load_bundle(self.root / "a").bundle_id,
                         bench.load_bundle(self.root / "b").bundle_id)

    def test_import_records_held_out_query_interval(self):
        vectors = self.root / "base.raw"
        queries = self.root / "queries.raw"
        np.array([[0, 0], [1, 0], [2, 0]], dtype="<f4").tofile(vectors)
        np.array([[1, 1], [2, 2], [3, 3]], dtype="<f4").tofile(queries)
        args = ["import", "--output", str(self.root / "held-out"), "--vectors", str(vectors),
                "--queries", str(queries), "--format", "raw", "--documents", "3",
                "--query-count", "3", "--dimensions", "2", "--metric", "l2", "--source", "fixture",
                "--query-start", "1", "--limit-queries", "2"]
        bench.main(args)
        bundle = bench.load_bundle(self.root / "held-out")
        np.testing.assert_array_equal(bundle.queries, [[2, 2], [3, 3]])
        self.assertEqual(bundle.manifest["provenance"]["query_start"], 1)
        self.assertEqual(bundle.manifest["provenance"]["query_stop"], 3)

    def test_score_cli_reports_missing_case_and_fails(self):
        root = self.root / "bundle"
        bench.write_oracle(self.bundle, root / "ground_truth.jsonl")
        results = self.root / "empty.jsonl"
        results.write_text("")
        report = self.root / "report.json"
        code = bench.main(["score", str(root), str(results), "--policies", "AUTO", "--repeats", "1",
                           "--output", str(report)])
        self.assertEqual(code, 1)
        summaries = json.loads(report.read_text())["summaries"]
        self.assertEqual(summaries[0]["failures"], len(self.bundle.cases))

    def test_score_cli_fails_incomplete_results_without_recall_gate(self):
        root = self.root / "bundle"
        oracle_path = root / "ground_truth.jsonl"
        bench.write_oracle(self.bundle, oracle_path)
        truth = bench.load_oracle(self.bundle, oracle_path)
        records = [self.result(row["ids"], case_id=case_id) for case_id, row in truth.items()]
        incomplete = next(record for record in records if record["ids"])
        incomplete["ids"].pop()
        incomplete["distances"].pop()
        results = self.root / "incomplete.jsonl"
        results.write_text("".join(json.dumps(record) + "\n" for record in records))
        report = self.root / "report.json"
        code = bench.main(["score", str(root), str(results), "--policies", "AUTO", "--repeats", "1",
                           "--output", str(report)])
        self.assertEqual(code, 1)
        summary = json.loads(report.read_text())["summaries"][0]
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["incomplete"], 1)


if __name__ == "__main__":
    unittest.main()
