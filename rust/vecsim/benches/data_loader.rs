//! Data loader for benchmark datasets.
//!
//! Loads binary vector files in the same format as C++ benchmarks.
//! The format is simple: n_vectors * dim * sizeof(T) contiguous bytes.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Load f32 vectors from a binary file.
///
/// Format: contiguous f32 values, each vector is `dim` floats.
pub fn load_f32_vectors(path: &Path, n_vectors: usize, dim: usize) -> std::io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::with_capacity(n_vectors);
    let mut buffer = vec![0u8; dim * std::mem::size_of::<f32>()];

    for _ in 0..n_vectors {
        reader.read_exact(&mut buffer)?;
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Load f64 vectors from a binary file.
pub fn load_f64_vectors(path: &Path, n_vectors: usize, dim: usize) -> std::io::Result<Vec<Vec<f64>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::with_capacity(n_vectors);
    let mut buffer = vec![0u8; dim * std::mem::size_of::<f64>()];

    for _ in 0..n_vectors {
        reader.read_exact(&mut buffer)?;
        let vector: Vec<f64> = buffer
            .chunks_exact(8)
            .map(|bytes| {
                f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                    bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            })
            .collect();
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Dataset configuration matching C++ benchmarks.
pub struct DatasetConfig {
    pub name: &'static str,
    pub dim: usize,
    pub n_vectors: usize,
    pub n_queries: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub vectors_file: &'static str,
    pub queries_file: &'static str,
}

/// DBPedia single-value dataset (fp32, cosine, 768 dim, 1M vectors).
/// Download with: bash tests/benchmark/bm_files.sh benchmarks-all
pub const DBPEDIA_SINGLE_FP32: DatasetConfig = DatasetConfig {
    name: "dbpedia_single_fp32",
    dim: 768,
    n_vectors: 1_000_000,
    n_queries: 10_000,
    m: 64,
    ef_construction: 512,
    vectors_file: "tests/benchmark/data/dbpedia-cosine-dim768-1M-vectors.raw",
    queries_file: "tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw",
};

/// Fashion images multi-value dataset (fp32, cosine, 512 dim).
/// Download with: bash tests/benchmark/bm_files.sh benchmarks-all
pub const FASHION_MULTI_FP32: DatasetConfig = DatasetConfig {
    name: "fashion_multi_fp32",
    dim: 512,
    n_vectors: 1_111_025,
    n_queries: 10_000,
    m: 64,
    ef_construction: 512,
    vectors_file: "tests/benchmark/data/fashion_images_multi_value-cosine-dim512-1M-vectors.raw",
    queries_file: "tests/benchmark/data/fashion_images_multi_value-cosine-dim512-test_vectors.raw",
};

/// Try to find the repository root by looking for benchmark data directory.
pub fn find_repo_root() -> Option<std::path::PathBuf> {
    let mut current = std::env::current_dir().ok()?;

    // Try current directory and parents
    for _ in 0..5 {
        let test_path = current.join("tests/benchmark/data");
        if test_path.exists() {
            return Some(current);
        }
        current = current.parent()?.to_path_buf();
    }

    None
}

/// Load dataset vectors, returning None if files don't exist.
pub fn try_load_dataset_vectors(config: &DatasetConfig, max_vectors: usize) -> Option<Vec<Vec<f32>>> {
    let repo_root = find_repo_root()?;
    let vectors_path = repo_root.join(config.vectors_file);

    if !vectors_path.exists() {
        eprintln!("Dataset vectors not found: {:?}", vectors_path);
        eprintln!("Run: bash tests/benchmark/bm_files.sh benchmarks-all");
        return None;
    }

    let n = max_vectors.min(config.n_vectors);
    load_f32_vectors(&vectors_path, n, config.dim).ok()
}

/// Load dataset queries, returning None if files don't exist.
pub fn try_load_dataset_queries(config: &DatasetConfig, max_queries: usize) -> Option<Vec<Vec<f32>>> {
    let repo_root = find_repo_root()?;
    let queries_path = repo_root.join(config.queries_file);

    if !queries_path.exists() {
        eprintln!("Dataset queries not found: {:?}", queries_path);
        return None;
    }

    let n = max_queries.min(config.n_queries);
    load_f32_vectors(&queries_path, n, config.dim).ok()
}

/// Generate random f32 vectors (fallback when real data is unavailable).
pub fn generate_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Generate normalized random f32 vectors (for cosine similarity).
pub fn generate_normalized_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}
