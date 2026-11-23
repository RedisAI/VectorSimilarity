# Training the RAM HNSW hybrid policy tree

`scripts/HNSW_batches_clf.py` fits a fixed decision tree that predicts the
initial choice between `ADHOC_BF` and `BATCHES` for filtered KNN. Use paired
measurements to make a candidate tree, inspect workload-level validation, and
optionally export a standalone C++17 header.

## Run the trainer

Run these commands from the root of the VectorSimilarity checkout. The trainer
requires Python 3.10 or newer, NumPy, and scikit-learn.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy scikit-learn

python scripts/HNSW_batches_clf.py \
    --measurements controlled.jsonl customer-replay.jsonl \
    --holdout-workload customer-replay \
    --report policy-evaluation.json \
    --cpp hnsw_policy_candidate.h
```

The script prints the tree and writes a JSON report with the input SHA-256
checksums, library versions, arguments, validation results, and tree. The
optional C++ header exposes `preferAdHocSearchCandidate(input)`. It returns
`true` for `ADHOC_BF` and `false` for `BATCHES`. Count features are converted
to float32 before evaluation, and the emitted constants preserve the fitted
thresholds. A runtime integration can call this helper and retain the
existing mode reporting and switching logic.

Run the script with no arguments to fit and plot the historical labels. That
mode requires Matplotlib. Its archived feature arrays encode `k` as 5, 10, and
15, while the comments describe 5, 50, and 500. The historical mode preserves
the archived encoding. Measurement mode uses the actual query `k`.

## Record paired measurements

Collect both forced policies on the same RAM HNSW snapshot, query vector,
predicate, `k`, search settings, and deadline. Rotate the order across
repetitions. Verify the effective policy outside the timed region. Use exact
filtered top-k over the same stored vectors to compute recall. Also verify
result completeness, eligibility, and document uniqueness. When only three
documents qualify for `k=10`, the reference contains those three documents.
Use vector-stage results from customer replays before application grouping
or ranking.

Store one object per line in JSONL. The following object is formatted across
lines for readability. Its timings are illustrative.

```json
{
  "workload": "customer-replay",
  "case": "query-01",
  "cohort": "numeric-and",
  "estimate_source": "planner",
  "k": 2000,
  "index_size": 57000,
  "label_count": 57000,
  "dim": 1024,
  "m": 16,
  "subset_size": 8500,
  "actual_matches": 3400,
  "runs": [{
    "ADHOC_BF": {"status": "ok", "mode": "ADHOC_BF", "ms": 12.1, "recall": 1.0},
    "BATCHES": {"status": "ok", "mode": "BATCHES", "ms": 15.3, "recall": 0.97},
    "AUTO": {"status": "ok", "mode": "BATCHES_TO_ADHOC_BF", "ms": 18.7, "recall": 1.0}
  }]
}
```

- **`workload`** identifies an independent corpus or replay family. Keep
  variants of the same corpus, including different `k` values and index
  sizes, in the same workload. Use `cohort` to break those variants out in the
  report. Use a unique `case` within each workload.
- **`index_size` and `label_count`** count vectors and distinct vector
  labels, respectively. The five model features are `k`, index size, vector
  dimension, graph `M`, and estimated selectivity. The selectivity calculation
  mirrors RAM HNSW by clamping `subset_size` to index size and dividing by
  label count.
- **`subset_size`** is the initial filter estimate supplied to the selector.
  Record `actual_matches` when available to audit estimation error. For a
  separate exact-count experiment, set both fields to the actual count, set
  `estimate_source` to `exact`, and pass `--estimate-source exact`.
  The trainer rejects mixed estimate sources.
- **`runs`** contains all paired repetitions for the case. Include both
  forced policies and, optionally, AUTO in every repetition. Use
  `{"status": "timeout"}` for a measured timeout. A successful forced policy
  must report its requested mode. AUTO can report `ADHOC_BF`, `BATCHES`, or
  `BATCHES_TO_ADHOC_BF`.

## How the tree is fitted

| Choice | Default and rationale |
| --- | --- |
| Recall floor | `--min-recall 0.95`. A policy qualifies only when every paired repetition completes and reaches the floor. An exact run with low recall invalidates the case. An exact timeout can favor BATCHES if BATCHES completes and qualifies. |
| Latency | Take the median across repetitions for each qualified policy. Each query is one training case, regardless of repetition count. |
| Minimum speedup | `--min-speedup 0.05`. When both policies qualify, require the slower-to-faster latency ratio to exceed 1.05 to use the case in fitting. Close cases contribute to evaluation. |
| Training weights | Give more weight to a costly wrong decision using the logarithm of the latency ratio, capped at a 10× ratio. Give each workload the same total weight so a large synthetic sweep cannot dominate a smaller replay by case count. |
| Tree size | `--max-leaf-nodes 20` preserves the historical size cap. `--min-samples-leaf 3` limits splits that depend on a few measurements. Use smaller trees when held-out quality and latency are comparable. |
| Leaf policy | Prefer the policy with fewer training quality or deadline failures in that leaf. Break ties using the fitted decision. |

Use `--holdout-workload NAME` once per reserved workload. The final fit
excludes those measurements. The report also evaluates leave-one-workload-out
folds among the training workloads. A single training workload cannot supply
a workload-level fold on its own.

## Review the candidate

Start with the held-out reports. Check recall or timeout failures separately
from latency ratios. The latency metrics compare the selected forced policy
with the best qualifying forced policy and with measured AUTO when available.
Measured AUTO may switch mid-query. Validate an integrated candidate to
measure its own switching costs and end-to-end latency.

Use the cohort and conflicting-feature reports to decide what to measure next.
For a wider sweep, vary index sizes and `k` around existing thresholds and
customer values, then vary selectivity and predicate shape. Include filters
with the same selectivity whose matches are near, far from, or independent of
the query vector. If they favor different policies despite identical model
inputs, the current features cannot distinguish those cases.
