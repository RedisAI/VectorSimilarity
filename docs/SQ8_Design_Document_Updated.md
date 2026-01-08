# Naive SQ8 Design Document (Updated)

## Overview

**Scalar Quantization (SQ8)** compresses each FP32 vector component into an 8-bit unsigned integer (uint8). This document describes the implementation as reflected in the current codebase (`src/VecSim/spaces/`).

---

## 1. Data Structures - Quantized Vectors

### 1.1 Asymmetric Distance (FP32 Query vs SQ8 Storage)

**Storage Vector Layout** (stored in index):

| Field | Type | Size | Applies For | Description |
|-------|------|------|-------------|-------------|
| **Quantized values** | `uint8[dim]` | `dim` B | IP, Cosine, L2 | Scalar-quantized components (0–255) |
| **min** | `float32` | 4 B | IP, Cosine, L2 | Minimum original FP32 value |
| **delta** | `float32` | 4 B | IP, Cosine, L2 | Quantization step: `(max − min) / 255` |
| **sum** | `float32` | 4 B | IP, Cosine | Sum of original FP32 values: `Σx_i` |
| **sum_of_squares** | `float32` | 4 B | L2 only | Sum of squared original values: `Σx_i²` |

**Storage Blob Size**:
- **IP/Cosine**: `dim + 3 * sizeof(float)` = `dim + 12` bytes
- **L2**: `dim + 4 * sizeof(float)` = `dim + 16` bytes

**Query Vector Layout** (asymmetric - remains FP32):

| Field | Type | Size | Applies For | Description |
|-------|------|------|-------------|-------------|
| **Query values** | `float32[dim]` | `dim * 4` B | IP, Cosine, L2 | Original FP32 query values |
| **y_sum** | `float32` | 4 B | IP, Cosine | Precomputed sum: `Σy_i` |
| **y_sum_squares** | `float32` | 4 B | L2 only | Precomputed sum of squares: `Σy_i²` |

**Query Blob Size**: `(dim + 1) * sizeof(float)`

### 1.2 Symmetric Distance (SQ8 vs SQ8)

Both vectors use the same storage layout with precomputed metadata.

**SQ8-SQ8 Storage Vector Layout**:

| Field | Type | Size | Applies For | Description |
|-------|------|------|-------------|-------------|
| **Quantized values** | `uint8[dim]` | `dim` B | All | Scalar-quantized components (0–255) |
| **min** | `float32` | 4 B | All | Minimum original FP32 value |
| **delta** | `float32` | 4 B | All | Quantization step: `(max − min) / 255` |
| **sum** | `float32` | 4 B | All | Precomputed sum: `Σx_i` |
| **sum_of_squares** | `float32` | 4 B | L2 only | Precomputed sum of squares: `Σx_i²` |

**SQ8-SQ8 Blob Size**:
- **IP/Cosine**: `dim + 3 * sizeof(float)` = `dim + 12` bytes
- **L2**: `dim + 4 * sizeof(float)` = `dim + 16` bytes

---

## 2. Quantization Formula

**Encoding** (FP32 → uint8):
```
q_i = round((x_i - min) / delta)
```

Where:
- `min = min(x_0, x_1, ..., x_{dim-1})`
- `max = max(x_0, x_1, ..., x_{dim-1})`
- `delta = (max - min) / 255` (or `1.0` if `max == min`)

**Decoding** (uint8 → FP32):
```
x_i ≈ min + delta * q_i
```

---

## 3. Distance Computation Formulas

### 3.1 Asymmetric Inner Product (FP32 query × SQ8 storage)

**Algebraic optimization** avoids dequantization in the hot loop:

```
IP(x, y) = Σ(x_i * y_i)
         ≈ Σ((min + delta * q_i) * y_i)
         = min * Σy_i + delta * Σ(q_i * y_i)
         = min * y_sum + delta * quantized_dot_product
```

Where:
- `y_sum = Σy_i` is precomputed and stored in the query blob
- `quantized_dot_product = Σ(q_i * y_i)` is computed efficiently

**Distance returned**: `1.0 - IP`

### 3.2 Asymmetric L2 Squared (FP32 query × SQ8 storage)

Uses dequantization approach:
```
L2²(x, y) = Σ(x_i - (min + delta * q_i))²
```

**Distance returned**: L2² value directly

### 3.3 Symmetric Inner Product (SQ8 × SQ8)

**Algebraic optimization** using precomputed sums:

```
IP(x, y) = Σ((min_x + δ_x * qx_i) * (min_y + δ_y * qy_i))
         = min_x * sum_y + min_y * sum_x - dim * min_x * min_y
           + δ_x * δ_y * Σ(qx_i * qy_i)
```

Where:
- `sum_x`, `sum_y` are precomputed sums of original values
- `Σ(qx_i * qy_i)` is computed using efficient integer dot product (VNNI/DOTPROD)

**Distance returned**: `1.0 - IP`

### 3.4 Symmetric L2 Squared (SQ8 × SQ8)

Uses identity: `||x - y||² = ||x||² + ||y||² - 2*IP(x, y)`

```
L2²(x, y) = sum_sq_x + sum_sq_y - 2 * IP(x, y)
```

Where:
- `sum_sq_x = Σx_i²`, `sum_sq_y = Σy_i²` are precomputed
- `IP(x, y)` uses the symmetric IP formula above

**Distance returned**: L2² value directly

---

## 4. Preprocessor Pipeline

### 4.1 Architecture

The preprocessor system uses a container pattern with chainable preprocessors:

```
PreprocessorsContainerAbstract
    └── MultiPreprocessorsContainer<DataType, n_preprocessors>
            └── PreprocessorInterface (abstract)
                    ├── CosinePreprocessor<DataType>
                    └── QuantPreprocessor<DataType, Metric>
```

### 4.2 CosinePreprocessor

Normalizes vectors in-place using L2 normalization.

**Operations**:
- `preprocessForStorage`: Allocate (if needed), copy, normalize
- `preprocessQuery`: Allocate aligned (if needed), copy, normalize
- `preprocessStorageInPlace`: Normalize existing blob

### 4.3 QuantPreprocessor

Quantizes storage vectors to SQ8, query vectors remain FP32 with precomputed sums.

**Template Parameters**:
- `DataType`: Source data type (float)
- `Metric`: VecSimMetric (L2, IP, or Cosine)

**Metric-specific behavior**:
- `extra_storage_values_count`: 2 for L2 (sum + sum_squares), 1 for IP/Cosine (sum only)

**Storage Processing**:
1. Find min/max values
2. Calculate delta = (max - min) / 255
3. Quantize each value: `q_i = round((x_i - min) / delta)`
4. Compute sum and sum_squares while quantizing (single pass)
5. Store metadata: `[min, delta, sum, (sum_squares for L2)]`

**Query Processing**:
1. Copy original FP32 values
2. Compute and append precomputed value:
   - IP/Cosine: `y_sum = Σy_i`
   - L2: `y_sum_squares = Σy_i²`

---

## 5. SIMD Optimizations

### 5.1 Asymmetric Distance Functions (FP32 × SQ8)

**AVX512 (512-bit)**:
- Process 16 uint8 values at a time
- Zero-extend uint8 → int32 using `_mm512_cvtepu8_epi32`
- Convert int32 → float32 using `_mm512_cvtepi32_ps`
- FMA: `_mm512_fmadd_ps` for accumulation

**AVX2 (256-bit)**:
- Process 8 uint8 values at a time
- Zero-extend uint8 → int32 using `_mm256_cvtepu8_epi32`
- Convert int32 → float32 using `_mm256_cvtepi32_ps`
- FMA: `_mm256_fmadd_ps` for accumulation

**SSE (128-bit)**:
- Process 4 uint8 values at a time
- Zero-extend uint8 → int32 using `_mm_cvtepu8_epi32`
- Convert int32 → float32 using `_mm_cvtepi32_ps`
- FMA: `_mm_fmadd_ps` for accumulation

**NEON (ARM 128-bit)**:
- Process 16 uint8 values at a time
- Zero-extend uint8 → uint16 using `vmovl_u8`
- Zero-extend uint16 → uint32 using `vmovl_u16`
- Convert uint32 → float32 using `vcvtq_f32_u32`
- FMA: `vfmaq_f32` for accumulation

### 5.2 Symmetric Distance Functions (SQ8 × SQ8)

**AVX512-VNNI** (Intel Ice Lake+):
- Uses `_mm512_dpbusd_epi32` for 4× uint8 dot product per lane
- Processes 64 uint8 pairs per instruction
- Accumulates directly to int32

**AVX512 (without VNNI)**:
- Zero-extend uint8 → int16 using `_mm512_cvtepu8_epi16`
- Multiply using `_mm512_mullo_epi16`
- Horizontal add pairs using `_mm512_madd_epi16`
- Accumulate to int32

**AVX2**:
- Zero-extend uint8 → int16 using `_mm256_cvtepu8_epi16`
- Multiply using `_mm256_mullo_epi16`
- Horizontal add pairs using `_mm256_madd_epi16`
- Accumulate to int32

**NEON with DOTPROD** (ARM v8.2+):
- Uses `vdotq_u32` for 4× uint8 dot product per lane
- Processes 16 uint8 pairs per instruction
- Accumulates directly to uint32

**NEON (without DOTPROD)**:
- Multiply using `vmull_u8` (8-bit → 16-bit)
- Pairwise add using `vpaddlq_u16` (16-bit → 32-bit)
- Accumulate to uint32

### 5.3 Function Resolution

Functions are resolved at runtime based on CPU capabilities:

```cpp
// Example: SQ8_InnerProductSQ8_GetDistFunc
if (features.avx512_vnni) return SQ8_InnerProductSQ8_AVX512_VNNI;
if (features.avx512f)     return SQ8_InnerProductSQ8_AVX512F;
if (features.avx2)        return SQ8_InnerProductSQ8_AVX2;
if (features.sse4_1)      return SQ8_InnerProductSQ8_SSE4;
return SQ8_InnerProductSQ8_Scalar;
```

---

## 6. File Organization

### 6.1 Core Implementation Files

| File | Description |
|------|-------------|
| `spaces/SQ8/sq8_quant.h` | Quantization utilities and blob accessors |
| `spaces/SQ8/sq8_spaces.h` | Space factory and function resolution |
| `spaces/SQ8/sq8_spaces.cpp` | Space factory implementation |

### 6.2 Asymmetric Distance Functions (FP32 × SQ8)

| File | Description |
|------|-------------|
| `spaces/SQ8/SQ8_IP_FP32.h` | Inner product function declarations |
| `spaces/SQ8/SQ8_IP_FP32_AVX512.cpp` | AVX512 implementation |
| `spaces/SQ8/SQ8_IP_FP32_AVX.cpp` | AVX2 implementation |
| `spaces/SQ8/SQ8_IP_FP32_SSE.cpp` | SSE4 implementation |
| `spaces/SQ8/SQ8_IP_FP32_NEON.cpp` | ARM NEON implementation |
| `spaces/SQ8/SQ8_L2_FP32.h` | L2 distance function declarations |
| `spaces/SQ8/SQ8_L2_FP32_AVX512.cpp` | AVX512 implementation |
| `spaces/SQ8/SQ8_L2_FP32_AVX.cpp` | AVX2 implementation |
| `spaces/SQ8/SQ8_L2_FP32_SSE.cpp` | SSE4 implementation |
| `spaces/SQ8/SQ8_L2_FP32_NEON.cpp` | ARM NEON implementation |

### 6.3 Symmetric Distance Functions (SQ8 × SQ8)

| File | Description |
|------|-------------|
| `spaces/SQ8/SQ8_IP_SQ8.h` | SQ8×SQ8 inner product declarations |
| `spaces/SQ8/SQ8_IP_SQ8_AVX512_VNNI.cpp` | AVX512-VNNI implementation |
| `spaces/SQ8/SQ8_IP_SQ8_AVX512.cpp` | AVX512 implementation |
| `spaces/SQ8/SQ8_IP_SQ8_AVX.cpp` | AVX2 implementation |
| `spaces/SQ8/SQ8_IP_SQ8_SSE.cpp` | SSE4 implementation |
| `spaces/SQ8/SQ8_IP_SQ8_NEON.cpp` | ARM NEON implementation |
| `spaces/SQ8/SQ8_L2_SQ8.h` | SQ8×SQ8 L2 distance declarations |
| `spaces/SQ8/SQ8_L2_SQ8_AVX512_VNNI.cpp` | AVX512-VNNI implementation |
| `spaces/SQ8/SQ8_L2_SQ8_AVX512.cpp` | AVX512 implementation |
| `spaces/SQ8/SQ8_L2_SQ8_AVX.cpp` | AVX2 implementation |
| `spaces/SQ8/SQ8_L2_SQ8_SSE.cpp` | SSE4 implementation |
| `spaces/SQ8/SQ8_L2_SQ8_NEON.cpp` | ARM NEON implementation |

### 6.4 Preprocessor Files

| File | Description |
|------|-------------|
| `spaces/preprocessors.h` | Preprocessor interface and container |
| `spaces/SQ8/sq8_quant_preprocessor.h` | SQ8 quantization preprocessor |
| `spaces/normalize/cosine_preprocessor.h` | Cosine normalization preprocessor |

---


## 7. Memory Layout Examples

### 7.1 IP/Cosine Storage Vector (dim=128)

```
Offset 0-127:   uint8[128]  - Quantized values
Offset 128-131: float32     - min
Offset 132-135: float32     - delta
Offset 136-139: float32     - sum
Total: 140 bytes
```

### 7.2 L2 Storage Vector (dim=128)

```
Offset 0-127:   uint8[128]  - Quantized values
Offset 128-131: float32     - min
Offset 132-135: float32     - delta
Offset 136-139: float32     - sum
Offset 140-143: float32     - sum_of_squares
Total: 144 bytes
```

### 7.3 Query Vector (dim=128)

```
Offset 0-511:   float32[128] - Original FP32 values
Offset 512-515: float32      - Precomputed sum (y_sum or y_sum_squares)
Total: 516 bytes
```

---

## 8. Cosine Similarity Handling

Cosine similarity is implemented as normalized inner product:

1. **CosinePreprocessor** normalizes vectors to unit length
2. **QuantPreprocessor** quantizes the normalized vectors
3. **IP distance function** computes inner product
4. Result is `1.0 - IP` (distance from similarity)

This approach reuses IP infrastructure while achieving cosine semantics.

---

## 9. Key Implementation Details

### 9.1 Blob Accessors (sq8_quant.h)

```cpp
// Storage blob accessors
inline uint8_t *GetQuantizedValues(void *blob) { return (uint8_t *)blob; }
inline float *GetMin(void *blob, size_t dim) { return (float *)((uint8_t *)blob + dim); }
inline float *GetDelta(void *blob, size_t dim) { return GetMin(blob, dim) + 1; }
inline float *GetSum(void *blob, size_t dim) { return GetDelta(blob, dim) + 1; }
inline float *GetSumOfSquares(void *blob, size_t dim) { return GetSum(blob, dim) + 1; }

// Query blob accessors
inline float *GetQueryValues(void *blob) { return (float *)blob; }
inline float *GetQueryPrecomputedValue(void *blob, size_t dim) { return (float *)blob + dim; }
```

### 9.2 Quantization Function

```cpp
inline void QuantizeVector(const float *input, size_t dim, void *output) {
    float min_val = *std::min_element(input, input + dim);
    float max_val = *std::max_element(input, input + dim);
    float delta = (max_val != min_val) ? (max_val - min_val) / 255.0f : 1.0f;
    float inv_delta = 1.0f / delta;

    uint8_t *quant = GetQuantizedValues(output);
    for (size_t i = 0; i < dim; i++) {
        quant[i] = (uint8_t)roundf((input[i] - min_val) * inv_delta);
    }

    *GetMin(output, dim) = min_val;
    *GetDelta(output, dim) = delta;
}
```

---

## 10. Performance Considerations

1. **Asymmetric vs Symmetric**: Asymmetric (FP32×SQ8) provides better accuracy; Symmetric (SQ8×SQ8) provides better performance for re-ranking.

2. **VNNI/DOTPROD**: Hardware integer dot product instructions provide significant speedup for symmetric operations.

3. **Precomputed Values**: Storing sums and sum-of-squares avoids redundant computation during distance calculations.

4. **Memory Bandwidth**: SQ8 reduces memory bandwidth by 4× compared to FP32, which often dominates performance in large-scale similarity search.

5. **Alignment**: Query blobs are allocated with alignment for optimal SIMD performance.