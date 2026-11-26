# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from download_dbpedia import DIMENSIONS, download_range, main


class Response(io.BytesIO):
    def __init__(self, status, headers, body=b""):
        super().__init__(body)
        self.status = status
        self.headers = headers


class DownloadTest(unittest.TestCase):
    def fetch(self, directory, mutation=None, truncated=False):
        payload = b"x" * (DIMENSIONS * 4)
        calls = []

        def opener(request, timeout):
            calls.append(request)
            headers = {"ETag": '"pinned"', "x-amz-version-id": "version-1"}
            if request.get_method() == "HEAD":
                return Response(200, dict(headers, **{"Content-Length": str(len(payload) * 10)}))
            self.assertEqual(request.get_header("If-match"), '"pinned"')
            self.assertEqual(request.get_header("Range"), f"bytes={len(payload)}-{2 * len(payload) - 1}")
            headers.update({"Content-Length": str(len(payload)), "Content-Range": f"bytes {len(payload)}-{2 * len(payload) - 1}/{10 * len(payload)}"})
            if mutation:
                headers.update(mutation)
            return Response(206, headers, payload[:-1] if truncated else payload)

        result = download_range("https://example.invalid/array", Path(directory) / "array.raw", 1, 1, 10, opener)
        self.assertEqual(len(calls), 2)
        return result, payload

    def test_pinned_range_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            result, payload = self.fetch(directory)
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(result["row_start"], 1)
            self.assertEqual(result["row_stop"], 2)
            self.assertEqual((Path(directory) / "array.raw").read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                self.fetch(directory)

    def test_range_version_and_size_must_match(self):
        for mutation in ({"Content-Range": "bytes 0-1/2"}, {"x-amz-version-id": "changed"}, {"ETag": '"changed"'}, {"Content-Length": "1"}):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.fetch(directory, mutation)
                self.assertFalse((Path(directory) / "array.raw").exists())

    def test_truncation_does_not_publish_completed_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.fetch(directory, truncated=True)
            self.assertFalse((Path(directory) / "array.raw").exists())

    def test_query_intervals_reject_invalid_and_mixed_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            for start, count in ((-1, 1), (0, 0), (9999, 2), (1999, 2)):
                with self.subTest(start=start, count=count), self.assertRaises(SystemExit):
                    main(["--output", str(Path(directory) / "output"), "--query-start", str(start), "--query-count", str(count)])
            self.assertFalse((Path(directory) / "output").exists())


if __name__ == "__main__":
    unittest.main()
