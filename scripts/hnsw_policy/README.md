# HNSW hybrid policy evidence

The production changes are in [VectorSimilarity #1086](https://github.com/RedisAI/VectorSimilarity/pull/1086) and [RediSearch #12054](https://github.com/RediSearch/RediSearch/pull/12054). Benchmark tooling and results are kept on this separate evidence branch.

- [AWS end-to-end comparison](aws-2026-11-01/README.md): baseline versus candidate through real two-shard coordinators, using real DBpedia embeddings and both fresh and document-ID-hole histories. This is the current validation.
- [Native reserved-query evidence](native-2026-11-01/): earlier policy revisions, six index shapes, and high-K/continuation experiments. These timings exclude Search filter and document-ID traversal costs.
- [Original synthetic experiment](synthetic-experiment.md): historical 9,720 paired measurements and the original learned tree. The reported 6.3× sum-of-medians improvement does not describe the final policy or the customer's query.

Every experiment retains its source revisions and limitations. Batches is approximate: inspect recall and per-workload p95 alongside median latency. None of these datasets is the customer's actual data.
