#!/usr/bin/env python3
# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

"""Download checked DBpedia row ranges without fetching the entire corpus."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
from urllib.request import Request, urlopen


DIMENSIONS = 768
DOCUMENTS = 1_000_000
QUERIES = 10_000
DEVELOPMENT_END = 2_000
BASE_URL = "https://s3.eu-west-1.amazonaws.com/dev.cto.redis/VectorSimilarity/"
VECTOR_URL = BASE_URL + "dbpedia-cosine-dim768-1M-vectors.raw"
QUERY_URL = BASE_URL + "dbpedia-cosine-dim768-test_vectors.raw"


def download_range(url, output, first_row, rows, source_rows, opener=urlopen):
    """Bind a conditional range read to the version observed by HEAD."""
    expected_total = source_rows * DIMENSIONS * 4
    start = first_row * DIMENSIONS * 4
    length = rows * DIMENSIONS * 4
    end = start + length - 1
    with opener(Request(url, method="HEAD"), timeout=120) as response:
        if response.status != 200:
            raise ValueError(f"unexpected HEAD status: {response.status}")
        if int(response.headers.get("Content-Length", "-1")) != expected_total:
            raise ValueError("source size does not match the registered FP32 array")
        etag = response.headers.get("ETag")
        version = response.headers.get("x-amz-version-id")
        if not etag or not version:
            raise ValueError("source does not expose both ETag and S3 version ID")
        modified = response.headers.get("Last-Modified")

    # Anonymous GetObjectVersion is unavailable on this bucket. If-Match pins the
    # object, and checking the returned version catches replacement with equal ETag.
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "If-Match": etag})
    digest = hashlib.sha256()
    received = 0
    part = output.with_suffix(output.suffix + ".part")
    if output.exists() or part.exists():
        raise FileExistsError(output)
    with opener(request, timeout=120) as response:
        if response.status != 206:
            raise ValueError("server did not honor the requested byte range")
        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected_total}":
            raise ValueError("response Content-Range does not match the requested array rows")
        if int(response.headers.get("Content-Length", "-1")) != length:
            raise ValueError("response Content-Length does not match the requested array rows")
        if response.headers.get("ETag") != etag or response.headers.get("x-amz-version-id") != version:
            raise ValueError("source object changed between HEAD and GET")
        with part.open("xb") as target:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > length:
                    raise ValueError("response contains more bytes than requested")
                target.write(block)
                digest.update(block)
    if received != length:
        raise ValueError(f"truncated array: received {received}, expected {length}")
    # Exclusive creation preserves the no-overwrite contract even if another
    # process creates the final path while the download is in flight.
    output.hardlink_to(part)
    part.unlink()
    return {
        "url": url, "etag": etag, "s3_version_id": version, "last_modified": modified,
        "source_rows": source_rows, "source_bytes": expected_total,
        "row_start": first_row, "row_stop": first_row + rows,
        "content_range": f"bytes {start}-{end}/{expected_total}",
        "bytes": received, "sha256": digest.hexdigest(), "file": output.name,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, choices=(100_000, 1_000_000), default=100_000)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--query-start", type=int, default=0)
    args = parser.parse_args(argv)
    query_end = args.query_start + args.query_count
    if args.query_start < 0 or args.query_count <= 0 or query_end > QUERIES:
        parser.error("query rows must be a nonempty interval inside [0, 10000)")
    if args.query_start < DEVELOPMENT_END < query_end:
        parser.error("a download must not mix development and held-out query rows")
    args.output.mkdir(parents=True, exist_ok=False)
    vectors = download_range(VECTOR_URL, args.output / "vectors.raw", 0, args.documents, DOCUMENTS)
    queries = download_range(QUERY_URL, args.output / "queries.raw", args.query_start, args.query_count, QUERIES)
    source = f"dbpedia-http-range-docs0:{args.documents}-queries{args.query_start}:{query_end}"
    command = [
        sys.executable, str(Path(__file__).with_name("filtered_bench.py")), "import",
        "--output", str(args.output.parent / (args.output.name + "-bundle")),
        "--vectors", str(args.output / "vectors.raw"), "--queries", str(args.output / "queries.raw"),
        "--format", "raw", "--documents", str(args.documents), "--query-count", str(args.query_count),
        "--dimensions", str(DIMENSIONS), "--metric", "cosine", "--source", source, "--seed", "20221",
    ]
    provenance = {
        "schema_version": 1, "dataset": "dbpedia-1m", "dtype": "<f4",
        "dimensions": DIMENSIONS, "metric": "cosine", "source": source,
        "query_split": "development" if query_end <= DEVELOPMENT_END else "held-out",
        "vectors": vectors, "queries": queries, "import_command": command,
        "note": "Importer row numbers refer to these subset files; original row intervals are recorded here.",
    }
    with (args.output / "provenance.json").open("x", encoding="utf-8") as target:
        json.dump(provenance, target, indent=2, sort_keys=True)
        target.write("\n")
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
