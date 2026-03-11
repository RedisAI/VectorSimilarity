BRANCH: meiravg_sq_comp

# SQ8 Naive Scalar Quantization - RAM Index Design

## 1. Overview

This document describes the design for adding naive scalar quantization (SQ8) support to VectorSimilarity's in-memory indexes (HNSW and BruteForce).

### Goals
- **Memory reduction**: ~4x memory savings (FP32 → SQ8 + metadata)
- **Minimal accuracy loss**: Naive min-max quantization without training
- **Reuse existing infrastructure**: Leverage existing index implementations, preprocessors, and distance functions

### Non-Goals (Future Work)
- Trained quantization (PQ, OPQ)
- SQ4 or other bit-widths
- Input types other than FLOAT32 (initially)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User API                                     │
│   HNSWParams { type=FLOAT32, quantization=SQ8, metric, dim, ... }   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Index Factory                                   │
│  - Detects quantization=SQ8                                         │
│  - Creates SQDistanceCalculator (2-mode)                            │
│  - Creates QuantPreprocessor                                        │
│  - Computes correct storedDataSize                                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HNSW / BruteForce Index                          │
│  - Uses calcDistance<DistanceMode::Query> for search                │
│  - Uses calcDistance<DistanceMode::Data> for graph operations       │
│  - Preprocessor quantizes FP32 → SQ8 for storage                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design (Bottom-Up)

### 3.1 Distance Calculator Interface Changes

**Location**: `src/VecSim/spaces/computer/calculator.h`

The base `IndexCalculatorInterface` is modified to have two explicit distance methods:

```cpp
template <typename DistType>
class IndexCalculatorInterface : public VecsimBaseObject {
public:
    explicit IndexCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    virtual ~IndexCalculatorInterface() = default;

    /**
     * Calculate distance between a query-format vector and a stored-format vector.
     * Used for: search operations, insertion (finding neighbors for new vector)
     *
     * @param query Vector in query format (e.g., FP32 + metadata for SQ8)
     * @param stored Vector in storage format (e.g., SQ8 + metadata)
     */
    virtual DistType calcQueryToStoredDistance(const void* query, const void* stored, size_t dim) const = 0;

    /**
     * Calculate distance between two stored-format vectors.
     * Used for: graph maintenance, neighbor selection among existing vectors
     *
     * @param v1 First vector in storage format
     * @param v2 Second vector in storage format
     */
    virtual DistType calcStoredToStoredDistance(const void* v1, const void* v2, size_t dim) const = 0;
};
```

### 3.2 Calculator Implementations

#### Non-SQ Calculator (both methods use same function)

```cpp
template <typename DistType>
class DistanceCalculatorCommon : public IndexCalculatorInterface<DistType> {
public:
    DistanceCalculatorCommon(std::shared_ptr<VecSimAllocator> allocator,
                             spaces::dist_func_t<DistType> dist_func)
        : IndexCalculatorInterface<DistType>(allocator), dist_func_(dist_func) {}

    DistType calcQueryToStoredDistance(const void* query, const void* stored, size_t dim) const override {
        return dist_func_(query, stored, dim);
    }

    DistType calcStoredToStoredDistance(const void* v1, const void* v2, size_t dim) const override {
        return dist_func_(v1, v2, dim);  // Same function for non-SQ
    }

private:
    spaces::dist_func_t<DistType> dist_func_;
};
```

#### SQ Calculator (different functions for each mode)

```cpp
template <typename DistType>
class SQDistanceCalculator : public IndexCalculatorInterface<DistType> {
public:
    SQDistanceCalculator(std::shared_ptr<VecSimAllocator> allocator,
                         spaces::dist_func_t<DistType> queryToStoredFunc,  // SQ8-FP32
                         spaces::dist_func_t<DistType> storedToStoredFunc) // SQ8-SQ8
        : IndexCalculatorInterface<DistType>(allocator),
          queryToStoredFunc_(queryToStoredFunc),
          storedToStoredFunc_(storedToStoredFunc) {}

    DistType calcQueryToStoredDistance(const void* query, const void* stored, size_t dim) const override {
        return queryToStoredFunc_(query, stored, dim);  // SQ8_FP32_* functions
    }

    DistType calcStoredToStoredDistance(const void* v1, const void* v2, size_t dim) const override {
        return storedToStoredFunc_(v1, v2, dim);  // SQ8_SQ8_* functions
    }

private:
    spaces::dist_func_t<DistType> queryToStoredFunc_;   // SQ8_FP32_* functions
    spaces::dist_func_t<DistType> storedToStoredFunc_;  // SQ8_SQ8_* functions
};
```

**Existing distance functions to use**:
- `SQ8_SQ8_L2Sqr`, `SQ8_FP32_L2Sqr` (in `src/VecSim/spaces/L2/L2.cpp`)
- `SQ8_SQ8_InnerProduct`, `SQ8_FP32_InnerProduct` (in `src/VecSim/spaces/IP/IP.cpp`)
- Cosine variants

### 3.3 Preprocessor: `QuantPreprocessor` (Existing)

**Location**: `src/VecSim/spaces/computer/preprocessors.h`

The existing `QuantPreprocessor<DataType>` already implements SQ8 quantization:

**Quantization algorithm**:
```
quantized[i] = round((input[i] - min_val) / delta)
delta = (max_val - min_val) / 255.0
```

**Storage layout**:
```
| quantized_values[dim] | min_val | delta | sum | (sum_squares for L2) |
```
- L2: `dim` bytes + 4 floats (16 bytes metadata)
- IP/Cosine: `dim` bytes + 3 floats (12 bytes metadata)

**Query layout** (for asymmetric distance):
```
| query_values[dim] | sum | (sum_squares for L2) |
```

### 3.4 Components Factory

#### 3.3.1 SQ8 Preprocessor Factory (Shared)

**Location**: `src/VecSim/index_factories/components/preprocessors_factory.h`

New function for SQ8 preprocessor creation that can be **reused by RediSearchDisk**:

```cpp
/**
 * Creates a preprocessor container for SQ8 quantization.
 *
 * This function is designed to be reusable by both:
 * - VectorSimilarity SQ8 RAM indexes (may need normalization)
 * - RediSearchDisk (always receives pre-normalized vectors from tiered frontend)
 *
 * @param is_normalized If true, skips normalization (vectors already normalized).
 *                      RediSearchDisk should always pass true.
 */
template <typename DataType, VecSimMetric Metric>
PreprocessorsContainerAbstract* CreateSQ8PreprocessorsContainer(
    std::shared_ptr<VecSimAllocator> allocator,
    size_t dim,
    bool is_normalized,
    unsigned char alignment) {

    if constexpr (Metric == VecSimMetric_Cosine) {
        if (!is_normalized) {
            // Normalize then quantize
            constexpr size_t n_preprocessors = 2;
            auto* container = new (allocator)
                MultiPreprocessorsContainer<DataType, n_preprocessors>(allocator, alignment);
            container->addPreprocessor(
                new (allocator) NormalizePreprocessor<DataType>(allocator, dim));
            container->addPreprocessor(
                new (allocator) QuantPreprocessor<DataType, Metric>(allocator, dim));
            return container;
        }
    }
    // Just quantize (L2, IP, or already normalized Cosine)
    constexpr size_t n_preprocessors = 1;
    auto* container = new (allocator)
        MultiPreprocessorsContainer<DataType, n_preprocessors>(allocator, alignment);
    container->addPreprocessor(
        new (allocator) QuantPreprocessor<DataType, Metric>(allocator, dim));
    return container;
}

// Runtime metric dispatch wrapper
template <typename DataType>
PreprocessorsContainerAbstract* CreateSQ8PreprocessorsForMetric(
    std::shared_ptr<VecSimAllocator> allocator,
    VecSimMetric metric,
    size_t dim,
    bool is_normalized,
    unsigned char alignment) {
    switch (metric) {
    case VecSimMetric_L2:
        return CreateSQ8PreprocessorsContainer<DataType, VecSimMetric_L2>(
            allocator, dim, is_normalized, alignment);
    case VecSimMetric_IP:
        return CreateSQ8PreprocessorsContainer<DataType, VecSimMetric_IP>(
            allocator, dim, is_normalized, alignment);
    case VecSimMetric_Cosine:
        return CreateSQ8PreprocessorsContainer<DataType, VecSimMetric_Cosine>(
            allocator, dim, is_normalized, alignment);
    }
    throw std::invalid_argument("Unsupported metric for SQ8 preprocessors");
}
```

**RediSearchDisk integration**: The disk repo can replace its `CreateDiskPreprocessorsContainer`
with a call to `CreateSQ8PreprocessorsForMetric(..., is_normalized=true)`.

#### 3.3.2 SQ8 Index Components Factory

**Location**: `src/VecSim/index_factories/components/components_factory.h`

New function to create complete SQ8 components:

```cpp
template <typename DataType, typename DistType>
IndexComponents<DataType, DistType> CreateSQ8IndexComponents(
    std::shared_ptr<VecSimAllocator> allocator,
    VecSimMetric metric,
    size_t dim,
    bool is_normalized);
```

This function:
1. Gets SQ8-SQ8 and SQ8-FP32 distance functions based on metric
2. Creates `SQDistanceCalculator` with both functions
3. Calls `CreateSQ8PreprocessorsForMetric` for preprocessor chain
4. Returns `IndexComponents` struct

#### 3.3.3 Reusability Summary

| Component | VectorSimilarity | RediSearchDisk | Shared? |
|-----------|------------------|----------------|---------|
| `SQDistanceCalculator` (2-mode) | ✅ Uses | ❌ N/A | No |
| `DiskDistanceCalculator` (3-mode) | ❌ N/A | ✅ Uses | No |
| `CreateSQ8PreprocessorsContainer` | ✅ Uses | ✅ Can reuse | **Yes** |
| `CreateSQ8IndexComponents` | ✅ Uses | ❌ Has own | No |

**Note**: RediSearchDisk needs `DiskDistanceCalculator` with the additional `Full` mode
(FP32-FP32) for reranking operations, so it cannot use `SQDistanceCalculator`.

---

## 4. API Design

### 4.1 New Quantization Enum

**Location**: `src/VecSim/vec_sim_common.h`

```c
typedef enum {
    VecSimQuant_NONE = 0,  // No quantization (default)
    VecSimQuant_SQ8 = 1,   // Scalar quantization to 8-bit
    // Future: VecSimQuant_SQ4, VecSimQuant_PQ, etc.
} VecSimQuantization;
```

### 4.2 Updated HNSWParams

```c
typedef struct {
    VecSimType type;              // Input datatype (FLOAT32 for SQ8)
    size_t dim;
    VecSimMetric metric;
    bool multi;
    size_t blockSize;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
    double epsilon;
    VecSimQuantization quantization;  // NEW: Storage quantization (default: NONE)
} HNSWParams;
```

### 4.3 Updated BFParams

```c
typedef struct {
    VecSimType type;
    size_t dim;
    VecSimMetric metric;
    bool multi;
    size_t blockSize;
    VecSimQuantization quantization;  // NEW
} BFParams;
```

### 4.4 Validation Rules

- `quantization=SQ8` requires `type=FLOAT32` (initially)
- All metrics supported: L2, IP, Cosine

---

## 5. Index Modifications

### 5.1 Distance Calculation Call Sites

The index implementations use the two new interface methods:

#### **`calcQueryToStoredDistance`** - Search & insertion operations
- `searchLayer()` - comparing query to stored vectors
- `searchBottomLayerEP()` - entry point search
- `getDistanceFrom()` - user-facing distance query
- `addVector()` - finding neighbors for new vector

#### **`calcStoredToStoredDistance`** - Graph maintenance operations
- `revisitNeighborConnections()` - comparing stored vectors
- `repairConnectionsForDeletion()` - graph repair
- `selectNeighbors()` - neighbor selection among existing vectors

### 5.2 Modified Distance Calls

With the new interface, all indexes (SQ and non-SQ) use the same method calls:

```cpp
// In search operations and insertion (query/new vector vs stored)
void HNSWIndex<DataType, DistType>::searchLayer(...) {
    dist = indexCalculator->calcQueryToStoredDistance(query, stored, dim);
}

// In graph operations (stored vs stored)
void HNSWIndex<DataType, DistType>::revisitNeighborConnections(...) {
    dist = indexCalculator->calcStoredToStoredDistance(stored1, stored2, dim);
}
```

**No `if constexpr` branching needed** - the calculator implementation handles the difference:
- Non-SQ calculator: both methods call the same distance function
- SQ calculator: each method calls the appropriate function (SQ8-FP32 or SQ8-SQ8)

### 5.3 Migration: Replacing `calcDistance` Calls

All existing `calcDistance(v1, v2, dim)` calls need to be replaced:

| Context | Old call | New call |
|---------|----------|----------|
| Search (query vs stored) | `calcDistance(query, stored, dim)` | `calcQueryToStoredDistance(query, stored, dim)` |
| Insertion (new vs stored) | `calcDistance(newVec, stored, dim)` | `calcQueryToStoredDistance(newVec, stored, dim)` |
| Graph ops (stored vs stored) | `calcDistance(v1, v2, dim)` | `calcStoredToStoredDistance(v1, v2, dim)` |

---

## 6. Factory Modifications

### 6.1 HNSW Factory

**Location**: `src/VecSim/index_factories/hnsw_factory.cpp`

```cpp
VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized) {
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;

    // Validate SQ8 constraints
    if (hnswParams->quantization == VecSimQuant_SQ8) {
        assert(hnswParams->type == VecSimType_FLOAT32 &&
               "SQ8 currently only supports FLOAT32 input");
    }

    AbstractIndexInitParams abstractInitParams =
        VecSimFactory::NewAbstractInitParams(hnswParams, params->logCtx, is_normalized);

    if (hnswParams->type == VecSimType_FLOAT32) {
        IndexComponents<float, float> indexComponents;

        if (hnswParams->quantization == VecSimQuant_SQ8) {
            // SQ8 path - creates SQDistanceCalculator
            indexComponents = CreateSQ8IndexComponents<float, float>(
                abstractInitParams.allocator,
                hnswParams->metric,
                hnswParams->dim,
                is_normalized);
        } else {
            // Regular FP32 path - creates DistanceCalculatorCommon
            indexComponents = CreateIndexComponents<float, float>(...);
        }

        // Same index instantiation for both - no IsSQ template parameter needed
        return NewIndex_ChooseMultiOrSingle<float, float>(
            hnswParams, abstractInitParams, indexComponents);
    }
    // ... other types
}
```

**Note**: No `IsSQ` template parameter needed on the index. The calculator's virtual methods
handle the difference between SQ and non-SQ distance calculations.

### 6.2 Stored Data Size Calculation

**Location**: `src/VecSim/utils/vec_utils.cpp`

Update `VecSimParams_GetStoredDataSize` or create new function:

```cpp
size_t GetSQ8StoredDataSize(size_t dim, VecSimMetric metric) {
    size_t metadata_floats = (metric == VecSimMetric_L2) ? 4 : 3;
    return dim * sizeof(uint8_t) + metadata_floats * sizeof(float);
}
```

### 6.3 Factory Utils Update

**Location**: `src/VecSim/index_factories/factory_utils.h`

```cpp
template <typename IndexParams>
static AbstractIndexInitParams NewAbstractInitParams(const IndexParams *algo_params,
                                                      void *logCtx,
                                                      bool is_input_preprocessed) {
    size_t storedDataSize;
    if (algo_params->quantization == VecSimQuant_SQ8) {
        storedDataSize = GetSQ8StoredDataSize(algo_params->dim, algo_params->metric);
    } else {
        storedDataSize = VecSimParams_GetStoredDataSize(
            algo_params->type, algo_params->dim, algo_params->metric);
    }

    size_t inputBlobSize = is_input_preprocessed
        ? storedDataSize
        : algo_params->dim * VecSimType_sizeof(algo_params->type);

    // ... rest unchanged
}
```

---

## 7. Serialization

### 7.1 File Format Changes

Add `quantization` field to saved index metadata:

**Location**: `src/VecSim/algorithms/hnsw/hnsw_serializer_impl.h`

```cpp
void HNSWIndex::saveIndexFields(std::ofstream &output) const {
    // ... existing fields
    writeBinaryPOD(output, this->quantization);  // NEW
}

void HNSWIndex::restoreIndexFields(std::ifstream &input) {
    // ... existing fields
    readBinaryPOD(input, this->quantization);  // NEW
}
```

### 7.2 Version Bump

Increment `EncodingVersion` to V5 for backward compatibility detection.

---

## 8. TieredIndex Integration

### 8.1 Configuration

- **Frontend (flat buffer)**: FP32, no quantization, uses FP32-FP32 distance
- **Backend (HNSW SQ8)**: SQ8 storage, uses SQ8-FP32 for search, SQ8-SQ8 for graph

### 8.2 Vector Transfer Flow

When vectors move from frontend to backend:
1. Frontend stores FP32 (possibly normalized for cosine)
2. Backend's `addVector()` receives FP32 blob
3. Backend's preprocessor quantizes FP32 → SQ8
4. Backend stores SQ8

No special handling needed - each index has its own preprocessor.

---

## 9. Implementation Plan

### Phase 1: Core Components
1. Add `VecSimQuantization` enum to `vec_sim_common.h`
2. Add `quantization` field to `HNSWParams` and `BFParams`
3. Modify `IndexCalculatorInterface` to have two methods:
   - `calcQueryToStoredDistance()`
   - `calcStoredToStoredDistance()`
4. Update `DistanceCalculatorCommon` to implement both methods (same function)
5. Create `SQDistanceCalculator` class with different functions per method
6. Create `CreateSQ8IndexComponents` in components factory

### Phase 2: Index Modifications
1. Replace all `calcDistance()` calls with appropriate method:
   - Search/insertion operations → `calcQueryToStoredDistance()`
   - Graph maintenance operations → `calcStoredToStoredDistance()`
2. Apply to HNSW, BruteForce, and TieredIndex

### Phase 3: Factory Updates
1. Update HNSW factory to detect `quantization=SQ8`
2. Update BruteForce factory similarly
3. Update `NewAbstractInitParams` for correct `storedDataSize`

### Phase 4: Serialization
1. Add `quantization` to saved/restored fields
2. Bump encoding version
3. Handle backward compatibility

### Phase 5: Testing
1. Unit tests for `SQDistanceCalculator`
2. Integration tests for HNSW SQ8 index
3. Integration tests for BruteForce SQ8 index
4. Accuracy validation (recall tests)
5. Memory usage validation

---

## 10. Memory Analysis

### Per-Vector Storage

| Type | Storage per vector (dim=128) |
|------|------------------------------|
| FP32 | 512 bytes |
| SQ8 (L2) | 128 + 16 = 144 bytes |
| SQ8 (IP/Cosine) | 128 + 12 = 140 bytes |

**Savings**: ~3.5x for typical dimensions

### Query Blob Size

Query blobs remain FP32 with metadata:
- L2: `(dim + 2) * sizeof(float)`
- IP/Cosine: `(dim + 1) * sizeof(float)`

---

## 11. Limitations

1. **Input type**: Only FLOAT32 supported initially
2. **Naive quantization**: No training, per-vector min-max scaling
3. **Accuracy**: Some recall loss compared to FP32 (typically <5% for reasonable data)

---

## 12. Future Extensions

1. **FLOAT64 input**: Quantize FP64 → SQ8
2. **SQ4**: 4-bit quantization for more compression
3. **Trained quantization**: Learn optimal quantization parameters from data
4. **Mixed precision**: Different quantization for different index levels
