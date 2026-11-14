# Phase 2: DBpedia100k pruning screen

Continue on the already selected dorer-arm-big, using unchanged binaries and
prepared input. Preserve the first-phase evidence. This is exploratory tuning
on the existing queries, not validation on unseen queries.

1. Verify host idleness, free disk, binary/data identities and effective flags.
2. Screen PiPNN alpha 1.0, 1.1, 1.4 with one tree, retaining alpha 1.2 as reference.
   Also test alpha 1.2 with two trees to distinguish pruning from candidate coverage.
3. Test HNSW M16 efConstruction=100 twice alongside existing efConstruction=200.
4. At measured recall >=95% and >=99%, select the PiPNN setting with the lowest
   p95 among settings with construction time no greater than half the fastest
   feasible HNSW baseline and resident memory <=125% of it. If none qualifies,
   report that and repeat the best latency candidate within the HNSW construction
   budget. Repeat each selected new configuration once; retain both targets.
5. Compare repeat medians/ranges and memory, including peak build RSS. Obtain
   separate review against raw evidence and document the decision.

Plan review found no execution blocker. Construction boundaries and compiler
arithmetic flags differ, so the 2x ratio is only a screening proxy, not an
end-to-end or algorithm-isolated claim. Fixed randomness does not test graph-seed
variation. Search effort uses the existing finite sweep with no interpolation.
Parameter tuning is the only change in this phase; aligned compiler settings,
common ready-to-query instrumentation and larger-scale runs remain separate work.

## Selection adjustment after screening

The first HNSW efConstruction=100 run takes 59.6 seconds, below every PiPNN
candidate (minimum 60.0 seconds). Therefore no candidate meets even the fallback
construction budget. Retain that failure explicitly. Repeat alpha=1.1 anyway
to measure the potentially useful near-equal-build, high-recall tradeoff; do not
present it as passing the original selection gate. Alpha=1.2 remains the best
95%-recall PiPNN point and already has two runs. No further parameter expansion.
