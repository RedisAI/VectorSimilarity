# Correctly rounded SQ8-FP32 L2 experiment

The target is the exact real value of `sum((min + delta*q[i] - query[i])^2)` for the
stored FP32 metadata, byte codes, and FP32 query. The result is rounded once to FP32,
nearest with ties to even. This does not undo storage quantization or earlier centering.
Storage/query layouts and the distance-function signatures are unchanged.

## Exact fallback

Every finite FP32 number is an integer times `2^-149`. Products therefore fit on a
fixed-point grid with unit `2^-298`. The fallback expands the distance identity into
integer products, keeping separate nonnegative positive and negative accumulators.
Unlike the former FP32 identity, its cancellation is exact.

For any dimension representable by a 64-bit `size_t`, the sum of the absolute terms
is bounded by `n*(|min| + 255*|delta| + max|query|)^2 < 2^337`. Ten 64-bit limbs cover
values below `2^342` on this grid. The 128-bit coefficients fit too: the largest is
the 48-bit squared delta significand times a byte-square sum smaller than `2^80`.
There are no heap allocations in the distance routine. The implementation requires
GCC/Clang-style unsigned 128-bit integers, as available on the tested 64-bit targets.

The final integer is rounded using retained bits, a guard bit, and all sticky bits.
FP32 subnormal results and overflow to positive infinity are constructed by their
bit patterns, independent of flush-to-zero settings and the ambient rounding mode.
Nonfinite metadata returns NaN. A NaN query returns NaN, even if another coordinate
is infinite. Otherwise an infinite query returns positive infinity. Empty distance
returns positive zero without reading either operand.

## Certified fast path

This path is disabled under fast-math, non-nearest rounding, dimensions above `2^24`,
and subnormal/nonfinite inputs. It also falls back when nonzero minimum/query values
differ by more than 28 exponents. In the remaining domain their subtraction is exact
in FP64: at most 24 significand bits plus 28 alignment bits and one carry bit.
The FP32 delta times an 8-bit code also fits exactly in FP64 (at most 32 bits).

Each residual therefore has just one FP64 rounding. Its square has at most three
rounding factors, and the nonnegative sum adds at most `n-1`. The usual
`gamma_(n+2)` bound is strictly smaller than the implemented
`4*(n+4)*epsilon_double`, expressed relative to the computed sum. That conservative
factor also absorbs the rounding of the bound itself. No FP64 intermediate can
underflow or overflow for this input domain. Outward `nextafter` calls enclose the
exact result; the fast path returns only if both endpoints round to the same FP32.
Output subnormal/overflow boundaries use the integer fallback.

## Integration and verification

All existing architecture-specific SQ8-FP32 L2 entries forward to the shared
certified/exact implementation. Other metrics and input types are unchanged. This
first implementation does not retain the old SIMD arithmetic, so a performance
regression versus the PR is expected and must be measured, not assumed negligible.

The branch-local CI workflow compares against PR head
`838a30de4940f0794f84213e38c1b802d3f1db7c`. Verification includes production
quantization regressions, midpoint/sticky-bit and output range tests, an independent
Python arbitrary-integer oracle that directly squares reconstructed residuals and
rounds by binary-searching FP32 encodings, GCC/Clang comparisons, ASan/UBSan, and the
spaces/components/HNSW-SQ8 suites. The oracle includes cases beyond FP64 precision,
multiple rounding modes, and flush-to-zero controls.

Benchmarking runs after all builds/tests, on a pinned CI CPU, alternating baseline
and candidate order for seven measured rounds after warmup. Both receive the same
vectors through their own production preprocessing. Measurements are hot-cache
distance-kernel thread CPU time, not end-to-end search latency. No build, test, or
benchmark is run locally.
