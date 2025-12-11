use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::convert::TryInto;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{ArgAction, Parser};

/// Dimensionality of the vectors in the deep.*.fbin files.
const DIM: usize = 96;

/// Convenience alias for fallible results in this binary.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Command-line configuration parsed via clap.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "groundtruth_calc",
    version,
    about = "Compute exact k-nearest-neighbor groundtruth for ANN benchmarking.",
    long_about = "Compute exact k-nearest-neighbor groundtruth for ANN benchmarking.\n\
\n\
This tool reads query and base vector datasets in .fbin format, computes exact\n\
k-nearest neighbors for each query against the full base set, and writes\n\
per-query results as .ibin files.\n\
\n\
The computation is parallelized over queries and supports checkpoint/resume by\n\
skipping queries that already have result files in the output directory.\n\
Optionally, it can merge all per-query results into a single groundtruth .ibin\n\
file that is compatible with the existing Python read_ibin() helper.\n\
\n\
Example:\n\
  groundtruth_calc -q deep.query.public.10K.fbin -b deep.base.100M.fbin \\\n+    -o results --k 100 -j 16 --merge-output deep.groundtruth.10K.ibin\n\
\n\
Checkpoint/resume:\n\
  Existing per-query result files in the output directory (query_*.ibin) are\n\
  detected and skipped on subsequent runs.",
)]
struct Config {
    /// Path to the query vectors .fbin file.
    #[arg(short = 'q', long = "queries", value_name = "PATH")]
    query_path: PathBuf,

    /// Path to the base vectors .fbin file (e.g., 100M base vectors).
    #[arg(short = 'b', long = "base", value_name = "PATH")]
    base_path: PathBuf,

    /// Output directory for per-query groundtruth .ibin files.
    #[arg(short = 'o', long = "output", value_name = "DIR")]
    results_dir: PathBuf,

    /// Number of nearest neighbors to compute per query (k).
    #[arg(long = "k", value_name = "NUMBER", default_value_t = 100)]
    k: usize,

    /// Number of worker threads to use (defaults to number of CPU cores).
    #[arg(short = 'j', long = "workers", value_name = "NUMBER")]
    num_workers: Option<usize>,

    /// How often to print progress updates, in number of completed queries.
    #[arg(long = "progress-interval", value_name = "NUMBER", default_value_t = 100)]
    progress_interval: usize,

    /// Optional chunk size for processing base vectors in batches (reserved).
    #[arg(long = "chunk-size", value_name = "NUMBER")]
    chunk_size: Option<usize>,

    /// Optional path to write a single merged groundtruth .ibin file.
    #[arg(long = "merge-output", value_name = "PATH")]
    merge_output: Option<PathBuf>,

    /// Increase verbosity (currently a placeholder, reserved for future use).
    #[arg(long = "verbose", short = 'v', action = ArgAction::Count)]
    _verbose: u8,
}

/// Neighbor in the top-k heap. We keep the *largest* distance at the top
/// of the BinaryHeap so it can be evicted when a closer neighbor is found.
#[derive(Debug, Clone)]
struct Neighbor {
    dist2: f32,
    index: u32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.dist2 == other.dist2 && self.index == other.index
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Larger distance means "greater" in the heap ordering.
        self.dist2.partial_cmp(&other.dist2)
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // We assume distances are finite (no NaNs) in the dataset.
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Format a Duration as HH:mm:ss.SSS.
fn format_hhmmss_millis(d: Duration) -> String {
    let total_millis = d.as_millis();
    let total_secs = total_millis / 1000;
    let millis = (total_millis % 1000) as u32;
    let hours = (total_secs / 3600) % 24;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}

/// Current absolute time-of-day (UTC, modulo 24h) formatted as HH:mm:ss.SSS.
fn current_time_of_day_formatted() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    format_hhmmss_millis(now)
}

/// Log a message to stdout with absolute time and elapsed time since `start`.
fn log_with_times(start: Instant, msg: &str) {
    let abs = current_time_of_day_formatted();
    let elapsed = format_hhmmss_millis(start.elapsed());
    println!("[{}][{}] {}", abs, elapsed, msg);
}

/// Log a message to stderr with absolute time and elapsed time since `start`.
fn log_error_with_times(start: Instant, msg: &str) {
    let abs = current_time_of_day_formatted();
    let elapsed = format_hhmmss_millis(start.elapsed());
    eprintln!("[{}][{}] {}", abs, elapsed, msg);
}

fn main() {
    let start = Instant::now();
    if let Err(e) = run(start) {
        log_error_with_times(start, &format!("Error: {e}"));
        std::process::exit(1);
    }
}

fn run(start: Instant) -> Result<()> {
    let mut config = Config::parse();

    if config.num_workers.is_none() {
        let default_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        config.num_workers = Some(default_workers);
    }

    log_with_times(
        start,
        &format!("Loading query vectors from {:?}...", config.query_path),
    );
    let queries = load_fbin_vectors(&config.query_path)?;
    log_with_times(start, &format!("Loaded {} query vectors", queries.len()));

    log_with_times(
        start,
        &format!("Loading base vectors from {:?}...", config.base_path),
    );
    let base = load_fbin_vectors(&config.base_path)?;
    log_with_times(start, &format!("Loaded {} base vectors", base.len()));

    if base.is_empty() || queries.is_empty() {
        return Err("Base or query vectors are empty".into());
    }

    // Pre-compute squared L2 norms for queries and base vectors.
    log_with_times(start, "Precomputing vector norms...");
    let query_norms = Arc::new(compute_norms(&queries));
    let base_norms = Arc::new(compute_norms(&base));

    // Detect already-completed queries from existing result files.
    log_with_times(
        start,
        &format!(
            "Scanning results directory {:?} for completed queries...",
            config.results_dir
        ),
    );
    let completed = detect_completed_queries(&config.results_dir)?;
    let total_queries = queries.len();
    let already_completed = completed.len();
    log_with_times(
        start,
        &format!(
            "{} / {} queries already have result files",
            already_completed, total_queries
        ),
    );

    // Build the list of pending query IDs.
    let pending_ids: Vec<usize> = (0..total_queries)
        .filter(|qid| !completed.contains(qid))
        .collect();

    if pending_ids.is_empty() {
        log_with_times(start, "All queries have been processed. Nothing to do.");
        return Ok(());
    }

    log_with_times(start, &format!("Pending queries: {}", pending_ids.len()));

    // Shared data across worker threads.
    let base = Arc::new(base);
    let queries = Arc::new(queries);
    let pending = Arc::new(pending_ids);

    let completed_counter = Arc::new(AtomicUsize::new(already_completed));
    let next_index = Arc::new(AtomicUsize::new(0));

    let num_workers = config
        .num_workers
        .unwrap_or(1)
        .min(queries.len().max(1))
        .max(1);
    log_with_times(start, &format!("Using {} worker threads", num_workers));

    // Choose a progress interval that gives roughly ~20 updates by default
    // (unless overridden by the user).
    let desired_updates: usize = 20;
    let auto_interval = (total_queries / desired_updates).max(1);
    let progress_interval_effective = if config.progress_interval == 0 {
        auto_interval
    } else {
        config.progress_interval
    };

    fs::create_dir_all(&config.results_dir)?;

    let mut handles = Vec::with_capacity(num_workers);

    for _worker_id in 0..num_workers {
        let base = Arc::clone(&base);
        let base_norms = Arc::clone(&base_norms);
        let queries = Arc::clone(&queries);
        let query_norms = Arc::clone(&query_norms);
        let pending = Arc::clone(&pending);
        let next_index = Arc::clone(&next_index);
        let completed_counter = Arc::clone(&completed_counter);
        let results_dir = config.results_dir.clone();
        let k = config.k;
        let progress_interval = progress_interval_effective;
        let start_for_thread = start;

        let total_queries_for_progress = total_queries;

        let handle = thread::spawn(move || {
            loop {
                let idx = next_index.fetch_add(1, AtomicOrdering::SeqCst);
                if idx >= pending.len() {
                    break;
                }

                let qid = pending[idx];
                if let Err(e) = process_query(
                    qid,
                    &queries,
                    &query_norms,
                    &base,
                    &base_norms,
                    k,
                    &results_dir,
                ) {
                    log_error_with_times(
                        start_for_thread,
                        &format!("Worker error while processing query {qid}: {e}"),
                    );
                    continue;
                }

                let done = completed_counter.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                if done % progress_interval == 0 || done == total_queries_for_progress {
                    let percent = (done as f64) * 100.0
                        / (total_queries_for_progress as f64);
                    log_with_times(
                        start_for_thread,
                        &format!(
                            "Progress: {}/{} ({percent:.2}%)",
                            done, total_queries_for_progress
                        ),
                    );
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        if let Err(e) = handle.join() {
            log_error_with_times(start, &format!("A worker thread panicked: {:?}", e));
        }
    }

    log_with_times(start, "All pending queries processed.");

    if let Some(merge_path) = &config.merge_output {
        log_with_times(
            start,
            &format!("Merging per-query results into {:?}...", merge_path),
        );
        merge_results(
            &config.results_dir,
            merge_path,
            total_queries,
            config.k,
        )?;
        log_with_times(
            start,
            &format!("Merged groundtruth written to {:?}", merge_path),
        );
    }

    Ok(())
}

/// Load an .fbin file into a Vec of fixed-size vectors.
fn load_fbin_vectors(path: &Path) -> Result<Vec<[f32; DIM]>> {
    let mut file = File::open(path)?;

    // Get file size to validate header
    let file_size = file.metadata()?.len() as usize;

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    let nvecs_header = i32::from_le_bytes(header[0..4].try_into()?) as usize;
    let dim_in_file = i32::from_le_bytes(header[4..8].try_into()?) as usize;

    if dim_in_file != DIM {
        return Err(format!(
            "Unexpected dimension in {:?}: got {}, expected {}",
            path, dim_in_file, DIM
        )
        .into());
    }

    // Calculate actual number of vectors from file size
    let data_size = file_size - 8; // subtract header
    let bytes_per_vector = DIM * std::mem::size_of::<f32>();
    let nvecs_actual = data_size / bytes_per_vector;

    eprintln!("DEBUG: File {:?}", path);
    eprintln!("DEBUG: File size: {} bytes ({:.2} MB)", file_size, file_size as f64 / 1024.0 / 1024.0);
    eprintln!("DEBUG: Header claims: nvecs={}, dim={}", nvecs_header, dim_in_file);
    eprintln!("DEBUG: Actual vectors based on file size: {}", nvecs_actual);

    // Use actual count from file size, not header (header may be wrong)
    let nvecs = nvecs_actual;

    if nvecs_header != nvecs_actual {
        eprintln!("WARNING: Header nvecs ({}) doesn't match file size ({}). Using file size.",
                  nvecs_header, nvecs_actual);
    }

    // Allocate the final vector storage directly
    let mut vectors = Vec::with_capacity(nvecs);

    // Read vectors one at a time to avoid allocating a huge temporary buffer
    let mut vec_buf = [0u8; DIM * 4];
    for _ in 0..nvecs {
        file.read_exact(&mut vec_buf)?;

        let mut v = [0f32; DIM];
        for (i, chunk) in vec_buf.chunks_exact(4).enumerate() {
            v[i] = f32::from_le_bytes(chunk.try_into()?);
        }
        vectors.push(v);
    }

    Ok(vectors)
}

/// Compute squared L2 norms for a set of vectors.
fn compute_norms(vectors: &[[f32; DIM]]) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum())
        .collect()
}

/// Compute squared L2 distance between two vectors using their precomputed
/// squared norms and dot product.
fn squared_l2_distance(
    q: &[f32; DIM],
    q_norm: f32,
    b: &[f32; DIM],
    b_norm: f32,
) -> f32 {
    let dot = dot_product(q, b);
    q_norm + b_norm - 2.0 * dot
}

/// Process a single query: compute its top-k nearest neighbors and write
/// results to an .ibin file.
fn process_query(
    qid: usize,
    queries: &[[f32; DIM]],
    query_norms: &[f32],
    base: &[[f32; DIM]],
    base_norms: &[f32],
    k: usize,
    results_dir: &Path,
) -> Result<()> {
    let q_vec = &queries[qid];
    let q_norm = query_norms[qid];

    let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);

    for (idx, b_vec) in base.iter().enumerate() {
        let b_norm = base_norms[idx];
        let dist2 = squared_l2_distance(q_vec, q_norm, b_vec, b_norm);

        if heap.len() < k {
            heap.push(Neighbor {
                dist2,
                index: idx as u32,
            });
        } else if let Some(worst) = heap.peek() {
            if dist2 < worst.dist2 {
                heap.pop();
                heap.push(Neighbor {
                    dist2,
                    index: idx as u32,
                });
            }
        }
    }

    // Convert heap into a sorted list of neighbor indices, closest first.
    let mut neighbors = heap.into_sorted_vec();
    // into_sorted_vec returns ascending order (smallest distance first).

    let mut indices: Vec<i32> = Vec::with_capacity(neighbors.len());
    for n in neighbors.drain(..) {
        indices.push(n.index as i32);
    }

    write_query_result(results_dir, qid, &indices)?;
    Ok(())
}

fn dot_product(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..DIM {
        sum += a[i] * b[i];
    }
    sum
}

/// Detect already-completed queries by scanning the results directory for
/// files named in the pattern `query_<id>.ibin`.
fn detect_completed_queries(results_dir: &Path) -> Result<HashSet<usize>> {
    let mut completed = HashSet::new();

    match fs::read_dir(results_dir) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }
                if let Some(qid) = parse_query_id_from_filename(&entry.file_name()) {
                    completed.insert(qid);
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            // Directory does not exist yet; no completed queries.
        }
        Err(e) => return Err(e.into()),
    }

    Ok(completed)
}

fn parse_query_id_from_filename(name: &std::ffi::OsStr) -> Option<usize> {
    let s = name.to_str()?;
    if !s.starts_with("query_") || !s.ends_with(".ibin") {
        return None;
    }

    let inner = &s[6..s.len() - 5];
    inner.parse::<usize>().ok()
}

/// Write a single-query result file in .ibin format, with shape (1, k).
fn write_query_result(results_dir: &Path, qid: usize, neighbors: &[i32]) -> Result<()> {
    let filename = format!("query_{:05}.ibin", qid);
    let path = results_dir.join(filename);

    let mut file = File::create(path)?;

    let nvecs: i32 = 1;
    let dim: i32 = neighbors.len() as i32;

    file.write_all(&nvecs.to_le_bytes())?;
    file.write_all(&dim.to_le_bytes())?;

    for &idx in neighbors {
        file.write_all(&idx.to_le_bytes())?;
    }

    Ok(())
}

/// Merge per-query result files from `results_dir` into a single groundtruth
/// .ibin file at `merge_path` with shape (n_queries, k).
fn merge_results(
    results_dir: &Path,
    merge_path: &Path,
    num_queries: usize,
    k: usize,
) -> Result<()> {
    // Prepare output file.
    let mut out = File::create(merge_path)?;

    let nvecs = num_queries as i32;
    let dim = k as i32;
    out.write_all(&nvecs.to_le_bytes())?;
    out.write_all(&dim.to_le_bytes())?;

    // For each query ID in order, read its per-query .ibin and append the
    // k neighbor indices. We expect exactly (1, k) per file.
    for qid in 0..num_queries {
        let filename = format!("query_{:05}.ibin", qid);
        let path = results_dir.join(filename);

        let mut file = File::open(&path).map_err(|e| {
            format!(
                "Failed to open per-query result for qid {} at {:?}: {}",
                qid, path, e
            )
        })?;

        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;
        let nvecs_file = i32::from_le_bytes(header[0..4].try_into()?) as usize;
        let dim_file = i32::from_le_bytes(header[4..8].try_into()?) as usize;

        if nvecs_file != 1 || dim_file != k {
            return Err(format!(
                "Unexpected shape in per-query result {:?}: got ({}, {}), expected (1, {})",
                path, nvecs_file, dim_file, k
            )
            .into());
        }

        let mut buf = vec![0u8; k * 4];
        file.read_exact(&mut buf)?;
        out.write_all(&buf)?;
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::fs;
    use std::io::Write as _;
    use tempfile::tempdir;

    #[test]
    fn neighbor_ordering_prefers_smaller_distance() {
        let mut heap = BinaryHeap::new();
        heap.push(Neighbor { dist2: 10.0, index: 1 });
        heap.push(Neighbor { dist2: 5.0, index: 2 });
        heap.push(Neighbor { dist2: 7.5, index: 3 });

        // BinaryHeap is a max-heap by Ord, so the largest distance should be on top.
        let top = heap.peek().unwrap();
        assert_eq!(top.dist2, 10.0);
    }

    #[test]
    fn dot_product_and_squared_l2_distance_match_manual_computation() {
        // Build small vectors whose non-zero entries are in the first 3
        // dimensions only. The remaining coordinates are zero.
        let mut a = [0.0f32; DIM];
        let mut b = [0.0f32; DIM];
        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        b[0] = 2.0;
        b[2] = -1.0;

        let dot = dot_product(&a, &b);
        assert!((dot - (1.0 * 2.0 + 2.0 * 0.0 + 3.0 * -1.0)).abs() < 1e-6);

        let a_norm = compute_norms(&[a])[0];
        let b_norm = compute_norms(&[b])[0];
        let dist2 = squared_l2_distance(&a, a_norm, &b, b_norm);

        // Manual squared L2: ||a - b||^2 over the first three non-zero dims.
        let manual_dist2: f32 = (1.0f32 - 2.0).powi(2)
            + (2.0f32 - 0.0).powi(2)
            + (3.0f32 - -1.0).powi(2);
        assert!((dist2 - manual_dist2).abs() < 1e-4);
    }

    #[test]
    fn parse_query_id_from_valid_and_invalid_filenames() {
        assert_eq!(
            parse_query_id_from_filename(OsStr::new("query_00042.ibin")),
            Some(42)
        );
        assert_eq!(
            parse_query_id_from_filename(OsStr::new("query_abc.ibin")),
            None
        );
        assert_eq!(
            parse_query_id_from_filename(OsStr::new("not_a_query_00001.ibin")),
            None
        );
    }

    #[test]
    fn write_and_load_fbin_roundtrip_small_vectors() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("small.fbin");

        // Create a small synthetic fbin file with DIM-dim vectors.
        let mut file = File::create(&path).unwrap();
        let nvecs: i32 = 2;
        let dim: i32 = DIM as i32;
        file.write_all(&nvecs.to_le_bytes()).unwrap();
        file.write_all(&dim.to_le_bytes()).unwrap();

        let v0: Vec<f32> = (0..DIM).map(|i| i as f32).collect();
        let v1: Vec<f32> = (0..DIM).map(|i| (i * 2) as f32).collect();
        for v in &[v0, v1] {
            for &x in v {
                file.write_all(&x.to_le_bytes()).unwrap();
            }
        }

        drop(file);

        let loaded = load_fbin_vectors(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!((loaded[0][0] - 0.0).abs() < 1e-6);
        assert!((loaded[0][1] - 1.0).abs() < 1e-6);
        assert!((loaded[1][0] - 0.0).abs() < 1e-6);
        assert!((loaded[1][1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn write_query_result_and_detect_completed_queries() {
        let tmp = tempdir().unwrap();
        let results_dir = tmp.path();

        // No completed queries initially.
        let completed = detect_completed_queries(results_dir).unwrap();
        assert!(completed.is_empty());

        // Write two query result files.
        write_query_result(results_dir, 0, &[1, 2, 3]).unwrap();
        write_query_result(results_dir, 5, &[4, 5, 6]).unwrap();

        let completed = detect_completed_queries(results_dir).unwrap();
        assert!(completed.contains(&0));
        assert!(completed.contains(&5));
        assert_eq!(completed.len(), 2);
    }

    #[test]
    fn merge_results_produces_expected_ibin_layout() {
        let tmp = tempdir().unwrap();
        let results_dir = tmp.path().join("results");
        fs::create_dir_all(&results_dir).unwrap();

        // Create three per-query files with k=2.
        write_query_result(&results_dir, 0, &[10, 11]).unwrap();
        write_query_result(&results_dir, 1, &[20, 21]).unwrap();
        write_query_result(&results_dir, 2, &[30, 31]).unwrap();

        let merged_path = tmp.path().join("merged.ibin");
        merge_results(&results_dir, &merged_path, 3, 2).unwrap();

        // Read back merged file and validate header and contents.
        let mut f = File::open(&merged_path).unwrap();
        let mut header = [0u8; 8];
        f.read_exact(&mut header).unwrap();
        let nvecs = i32::from_le_bytes(header[0..4].try_into().unwrap());
        let dim = i32::from_le_bytes(header[4..8].try_into().unwrap());
        assert_eq!(nvecs, 3);
        assert_eq!(dim, 2);

        let mut buf = vec![0u8; 3 * 2 * 4];
        f.read_exact(&mut buf).unwrap();
        let mut vals = Vec::new();
        for chunk in buf.chunks_exact(4) {
            vals.push(i32::from_le_bytes(chunk.try_into().unwrap()));
        }
        assert_eq!(vals, vec![10, 11, 20, 21, 30, 31]);
    }
}