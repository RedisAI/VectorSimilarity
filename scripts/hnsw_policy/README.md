# HNSW hybrid policy measurements

The historical classifier routes every passing fraction to ad-hoc search when
`30000 < n <= 300000`, `dim > 75`, `k > 12`, and `M > 10`. Its training script,
`scripts/HNSW_batches_clf.py`, contains handwritten winner labels without the
underlying timings. Its comments describe `k=5,50,500`, while its feature arrays
encode `5,10,15`. Those labels cannot establish which representation is correct.

This experiment uses fresh paired timings to retrain that leaf. It keeps the
historical decisions elsewhere. Actual `k` and passing counts are recorded in
each row instead of reconstructed from the position of a handwritten label.

## Collect

Build the `bm_hnsw_policy` target in a Release VectorSimilarity build with tests
enabled. Run collection without other benchmarks, builds, or tests sharing the
machine:

```sh
cmake -S . -B build/policy -DCMAKE_BUILD_TYPE=Release -DVECSIM_BUILD_TESTS=ON
cmake --build build/policy --target bm_hnsw_policy -j 6
python3 scripts/hnsw_policy/collect.py build/policy/benchmark/bm_hnsw_policy /tmp/hnsw-policy-run
```

`workloads.json` specifies each configuration as
`[vector_count, dimension, M, seed, element_bytes, clustered]`. The validation
set uses separate index shapes and seeds. The collector retains raw samples,
median summaries, the executable and benchmark-source hashes, revision, and CPU
information. Reusing an output directory resumes completed configurations only
when the metadata matches; failed executions retain partial output and stderr.

The C++ executable creates deterministic isotropic or clustered synthetic
vectors, using cosine distance, `EF_CONSTRUCTION=100`, and `EF_RUNTIME=10`.
Filters are random subsets with exact cardinality, traversed in label order.
Ad-hoc search scores all passing labels into a bounded heap. Batches uses the
RediSearch batch-size and cardinality-refinement calculations, without a policy
switch. Each workload uses three query vectors and three paired measurements
after warming both strategies; execution order alternates. Result counts and
recall against exact filtered search accompany the timings.

## Train and validate

```sh
python3 -m venv /tmp/hnsw-policy-env
/tmp/hnsw-policy-env/bin/pip install -r scripts/hnsw_policy/requirements.txt
/tmp/hnsw-policy-env/bin/python scripts/hnsw_policy/train.py /tmp/hnsw-policy-run/measurements.csv \
  --header /tmp/hnsw_hybrid_policy.h --report /tmp/hnsw-policy-report.json
/tmp/hnsw-policy-env/bin/python -m unittest discover -s scripts/hnsw_policy -p 'test_*.py'
```

Tree complexity is selected by cross-validation grouped by index shape, using
only the training set. Timing differences weight the labels, and small gains
preserve ad-hoc search to avoid learning timing noise. The independent validation
set must improve aggregate latency, introduce no incomplete batches choices,
and contain no newly selected batches case slower than ad-hoc by more than 2x.
A failed validation writes the report but does not emit a production header.

The exported function preserves ad-hoc outside the training feature ranges;
the caller must also restrict it to the index kinds represented by the data.
Float conversion is explicit in the export to match scikit-learn's input
precision. Retain the measurement CSV and metadata with a generated policy so
reviewers can reproduce its thresholds and inspect per-workload regressions.

## Included policy

`data/samples.csv` retains all 9,720 paired samples; `data/measurements.csv`
contains their 1,080 per-workload summaries. `data/metadata.json` identifies the
machine, collector source, executable, and workloads. `data/report.json` records
model selection and validation results. From the VectorSimilarity repository
root, regenerate the deployed header with:

```sh
/tmp/hnsw-policy-env/bin/python scripts/hnsw_policy/train.py scripts/hnsw_policy/data/measurements.csv \
  --header src/VecSim/algorithms/hnsw/hnsw_hybrid_policy.h --report /tmp/hnsw-policy-report.json
clang-format -i src/VecSim/algorithms/hnsw/hnsw_hybrid_policy.h
```

The production header is `src/VecSim/algorithms/hnsw/hnsw_hybrid_policy.h`.

Of the held-out workloads, 96 reach the affected leaf. Their summed median
latency falls from 2,132,118 to 337,693 microseconds (6.3x). The candidate chooses
batches for 83 of them, with no incomplete result sets and no newly selected
batches case more than 2x slower than ad-hoc. Its worst slowdown against the
faster strategy is still 6.38x on a case where it retains ad-hoc; this is not an
oracle policy or an end-to-end Redis speedup claim.

The caller limits this replacement to unquantized cosine FLOAT32/FLOAT64
indexes with equal vector and label counts. The generated bounds are:

| Feature | Measured range |
| --- | --- |
| `n` | 35,000–300,000 |
| Dimension | 76–1,536 |
| `M` | 11–64 |
| `k` | 13–500 |
| Passing fraction | 0.1–1.0 |

These bounds contain interpolated configurations as well as measured points.
Outside them, the historical decision is preserved. Explicit query policy
overrides remain authoritative.

## Limits

These are warm, in-process vector-library measurements on synthetic data, not
end-to-end Redis latency measurements. They exclude document loading, filter
iterator costs, deleted-ID traversal, and distributed execution. The sampled
indexes use unquantized FLOAT32/FLOAT64 vectors and one vector per label.
Other hardware, data distributions, filter/vector correlations, metrics,
quantization, and query `EF_RUNTIME` values require additional measurements.

Batches is approximate. A faster complete result set can still have lower
recall than ad-hoc; inspect the recorded recall alongside latency. Result
completeness is a validation gate, not a recall guarantee.
