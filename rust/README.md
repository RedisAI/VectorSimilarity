# VecSim Rust Implementation

High-performance vector similarity search library written in Rust, with Python and C bindings.

## Crate Structure

```
rust/
├── vecsim/           # Core library - vector indices and algorithms
├── vecsim-python/    # Python bindings (PyO3/maturin)
└── vecsim-c/         # C-compatible FFI layer
```

### vecsim (Core Library)

The main library providing:
- **Index Types**: BruteForce, HNSW, SVS (Vamana), Tiered, Disk-based
- **Data Types**: f32, f64, Float16, BFloat16, Int8, UInt8, Int32, Int64
- **Distance Metrics**: L2 (Euclidean), Inner Product, Cosine
- **Features**: Multi-value indices, SIMD optimization, serialization, parallel operations

### vecsim-python

Python bindings using PyO3. Must be built with `maturin`, not `cargo build` directly.

### vecsim-c

C-compatible API matching the C++ VecSim interface for drop-in replacement.

## Building

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- For Python bindings: Python 3.8+ and [maturin](https://github.com/PyO3/maturin)

### Build Core Libraries

```bash
cd rust

# Build vecsim and vecsim-c (release mode)
cargo build --release
```

The workspace is configured to exclude `vecsim-python` from default builds since it requires `maturin`.

### Build Python Bindings

Python bindings require `maturin` because they link against the Python interpreter at runtime:

```bash
# Install maturin
pip install maturin

# Build the wheel
cd rust/vecsim-python
maturin build --release

# Or install directly into current Python environment
maturin develop --release
```

The wheel will be created in `rust/target/wheels/`.

### Clean Build

```bash
cd rust
cargo clean
cargo build --release -p vecsim -p vecsim-c
```

## Testing

### Run All Tests

```bash
cd rust

# Run all tests for core library
cargo test -p vecsim

# Run tests with output displayed
cargo test -p vecsim -- --nocapture

# Run a specific test
cargo test -p vecsim test_hnsw_basic

# Run tests matching a pattern
cargo test -p vecsim hnsw
```

### Test Categories

The test suite includes:

- **Unit tests**: Inline tests throughout the codebase
- **End-to-end tests** (`e2e_tests.rs`): Complete workflow tests, serialization, persistence
- **Data type tests** (`data_type_tests.rs`): Coverage for all 8 vector types
- **Parallel stress tests** (`parallel_stress_tests.rs`): Concurrency and thread safety

### Run Tests with Features

```bash
# Run tests with all features enabled
cargo test -p vecsim --all-features
```

## Benchmarking

Benchmarks use [Criterion](https://github.com/bheisler/criterion.rs) for accurate measurements.

### Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `hnsw_bench` | HNSW index operations (add, query, multi-value) |
| `brute_force_bench` | BruteForce index performance |
| `tiered_bench` | Two-tier index benchmarks |
| `svs_bench` | Single-layer Vamana graph performance |
| `comparison_bench` | Cross-algorithm comparisons |
| `dbpedia_bench` | Real-world dataset benchmarks with recall measurement |

### Run Benchmarks

```bash
cd rust

# Run all benchmarks
cargo bench -p vecsim

# Run a specific benchmark
cargo bench -p vecsim --bench hnsw_bench

# Run benchmarks matching a pattern
cargo bench -p vecsim -- "hnsw"
```

### Real-World Dataset Benchmarks

The `dbpedia_bench` benchmark can use real DBPedia embeddings for realistic performance testing:

```bash
# Download benchmark data (from project root)
bash tests/benchmark/bm_files.sh benchmarks-all

# Run DBPedia benchmark
cargo bench -p vecsim --bench dbpedia_bench
```

If the dataset is not available, the benchmark falls back to random data.

### Benchmark Output

Results are saved to `rust/target/criterion/` with HTML reports. Open `rust/target/criterion/report/index.html` to view.

## Examples

### Run the Profiling Example

```bash
cd rust
cargo run --release -p vecsim --example profile_insert
```

### Quick Start Code

```rust
use vecsim::prelude::*;

fn main() {
    // Create an HNSW index for 128-dimensional f32 vectors with cosine similarity
    let params = HnswParams::new(128, Metric::Cosine)
        .with_m(16)
        .with_ef_construction(200);

    let mut index: HnswSingle<f32> = HnswSingle::new(params);

    // Add vectors
    let vector = vec![0.1f32; 128];
    index.add(&vector, 1).unwrap();

    // Query
    let query = vec![0.1f32; 128];
    let results = index.search(&query, 10);

    for result in results {
        println!("Label: {}, Distance: {}", result.label, result.distance);
    }
}
```

## Project Structure

```
rust/vecsim/
├── src/
│   ├── lib.rs              # Public API and prelude
│   ├── index/              # Index implementations
│   │   ├── brute_force.rs  # Exact search
│   │   ├── hnsw.rs         # Hierarchical NSW
│   │   ├── svs.rs          # Vamana graph
│   │   ├── tiered.rs       # Two-tier hybrid
│   │   └── disk.rs         # Memory-mapped indices
│   ├── distance/           # SIMD-optimized distance functions
│   ├── quantization/       # Scalar quantization (SQ8)
│   ├── storage/            # Vector storage backends
│   └── types/              # Core type definitions
├── benches/                # Criterion benchmarks
├── tests/                  # Integration tests
└── examples/               # Usage examples
```

## Common Issues

### Python bindings fail with "Undefined symbols"

Python extensions must be built with `maturin`, not `cargo build`:

```bash
# Wrong - will fail with linker errors
cargo build -p vecsim-python

# Correct
cd vecsim-python && maturin build --release
```

### Building the Python crate explicitly

If you need to build `vecsim-python` with cargo (not recommended), you must use maturin:

```bash
cd vecsim-python
maturin build --release
```

Direct `cargo build -p vecsim-python` will fail with linker errors.

## License

BSD-3-Clause
