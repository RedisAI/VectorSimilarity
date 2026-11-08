# Shared quantized storage contexts

StorageType selects a typed context. The same QuantPreprocessor, QuantizedCalculator,
component factory and backend lifecycle serve scalar and transformed quantizers.
The scalar-family default context retains SQ8 layout hooks and existing encoding,
correction arithmetic and optimized kernel selection. A non-scalar storage type
can supply `template <typename Input, VecSimMetric Metric, bool WithMean> using Context`.

The context owns encoding, prepared-query layout, dispatch, mean installation,
validation and fitted-state persistence. It has a stable address. Component wrappers
share ownership through VecSimAllocator-backed shared_ptrs; the index keeps a non-owning
runtime alias. A scoring call uses the existing cached dispatch, without a new
per-candidate virtual call. The shared control allocation and mean buffer are included
in the component estimate; shared allocation size is measured before context
construction, without fitting data or allocating a rotation.

Tiered construction requests a backend prepared for the regular training phase.
It does not interpret storage-specific parameters. Mean installation occurs with
queries excluded and before the first stored vector. Failed preparation retains
previous fitted state. Input validation precedes destructive replacements.

This foundation introduces no new accepted quantization mode. Existing scalar V5
payloads and nonquantized V4 payloads retain their layouts. Serialization remains
available only with test helpers. Existing SQ4 layout/code traits can use the scalar
default context; TQ2/TQ3 are supplied separately by the stacked draft.

Regression coverage checks SQ8 stored bytes against independent expected bytes,
legacy distance results and stateless function pointers, common component allocation
estimates, combined preprocessing failure cleanup and allocated buffer lifetime.
See the stacked TQ draft's validation receipt for current ARM execution evidence.
