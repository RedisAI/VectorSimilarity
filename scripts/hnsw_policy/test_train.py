# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train import export_header, fit, metrics, read_measurements


class TrainingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "measurements.csv"
        self.rows = []
        for n in (35000, 50000, 75000, 100000, 161812):
            for k in (13, 50, 200, 500):
                for ratio in (0.1, 0.95):
                    self.rows.append(dict(n=n, dim=128, m=16, k=k, subset=int(n * ratio),
                                          bytes=4, batch_count=k,
                                          split="validation" if n == 161812 else "train",
                                          adhoc_us=1 if ratio == 0.1 else 10,
                                          batches_us=10 if ratio == 0.1 else 1))

    def write(self):
        with self.path.open("w") as target:
            writer = csv.DictWriter(target, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_preserves_measured_k_and_ratio(self):
        """Measured feature values must not be reconstructed from positional label arrays."""
        self.write()
        x, *_ = read_measurements(self.path)
        self.assertEqual(set(x[:, 0]), {13, 50, 200, 500})
        self.assertTrue(np.any(x[:, 4] == np.float32(0.95)))

    def test_rejects_shared_index_shapes(self):
        self.rows[-1]["split"] = "train"
        self.write()
        with self.assertRaisesRegex(ValueError, "independent index shapes"):
            read_measurements(self.path)

    def test_rejects_invalid_timings(self):
        for timing in (0, -1, "nan", "inf"):
            with self.subTest(timing=timing):
                self.rows[0]["adhoc_us"] = timing
                self.write()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    read_measurements(self.path)

    def test_learns_both_strategies_on_separate_shapes(self):
        self.write()
        x, a, b, complete, training, groups = read_measurements(self.path)
        model, _ = fit(x[training], a[training], b[training], complete[training], groups[training])
        np.testing.assert_array_equal(model.predict(x[~training]), a[~training] < b[~training])
        header = export_header(model, "test-input")
        self.assertIn("return true;", header)
        self.assertIn("return false;", header)

    def test_incomplete_batch_is_not_an_oracle_win(self):
        result = metrics(np.array([False]), np.array([10.0]), np.array([1.0]), np.array([False]))
        self.assertEqual(result["incomplete_batch_choices"], 1)

    def test_fast_incomplete_batches_train_as_adhoc(self):
        for row in self.rows:
            row.update(adhoc_us=10, batches_us=1)
            if row["k"] == 500:
                row["batch_count"] = 499
        self.write()
        x, a, b, complete, training, groups = read_measurements(self.path)
        model, _ = fit(x[training], a[training], b[training], complete[training], groups[training])
        np.testing.assert_array_equal(model.predict(x[~training]), ~complete[~training])

    def test_cli_writes_validated_header_and_report(self):
        for row in self.rows:
            if row["split"] == "validation":
                row["subset"] = round(row["subset"] / row["n"] * 65000)
                row["n"] = 65000
        self.write()
        directory = Path(self.directory.name)
        header, report = directory / "policy.h", directory / "report.json"
        subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")),
                        str(self.path), "--header", str(header), "--report", str(report)],
                       check=True, capture_output=True, text=True)
        self.assertIn("preferHnswAdhocMeasured", header.read_text())
        self.assertEqual(json.loads(report.read_text())["validation"]["new"]["batches_choices"], 4)

    @unittest.skipUnless(shutil.which("c++"), "requires a C++ compiler")
    def test_cpp_export_matches_classifier_at_thresholds(self):
        """Rounding near learned thresholds must not change the deployed decision."""
        for row in self.rows:
            if row["k"] == 500:
                row["adhoc_us"] = 0.5
        self.write()
        x, a, b, complete, training, groups = read_measurements(self.path)
        model, _ = fit(x[training], a[training], b[training], complete[training], groups[training])
        low, high = x[training].min(axis=0), x[training].max(axis=0)
        directory = Path(self.directory.name)
        header = directory / "policy.h"
        header.write_text(export_header(model, "test-input", list(zip(low, high))))
        source = directory / "check.cpp"
        source.write_text('''#include "policy.h"
#include <iostream>
int main() {
    size_t k, n, dim, m;
    float ratio;
    while (std::cin >> k >> n >> dim >> m >> ratio) {
        std::cout << preferHnswAdhocMeasured(k, n, dim, m, ratio) << '\\n';
    }
}
''')
        binary = directory / "check"
        subprocess.run(["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
                       check=True, capture_output=True, text=True)
        points = list(x)
        for feature, threshold in zip(model.tree_.feature, model.tree_.threshold):
            if feature < 0:
                continue
            values = ([np.floor(threshold), np.floor(threshold) + 1] if feature < 4 else
                      [np.nextafter(np.float32(threshold), np.float32(direction))
                       for direction in (-np.inf, np.inf)])
            for value in values:
                point = x[0].copy()
                point[feature] = value
                points.append(point)
        points.extend([low - 1, high + 1])
        points = np.asarray(points, dtype=np.float32)
        points[:, :4] = points[:, :4].astype(np.uint64)
        expected = model.predict(points).astype(bool)
        expected |= ~np.all((points >= low) & (points <= high), axis=1)
        inputs = "".join(" ".join(str(int(v)) for v in p[:4]) + f" {float(p[4]):.17g}\n" for p in points)
        result = subprocess.run([str(binary)], input=inputs, capture_output=True, text=True, check=True)
        self.assertEqual([bool(int(v)) for v in result.stdout.split()], expected.tolist())


if __name__ == "__main__":
    unittest.main()
