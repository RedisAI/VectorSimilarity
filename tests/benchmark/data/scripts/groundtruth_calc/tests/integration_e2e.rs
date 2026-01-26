use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;

const DIM: usize = 96;

/// Helper to get the path to the built binary `groundtruth_calc`.
fn binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_groundtruth_calc"));
    if !path.exists() {
        // Fallback: assume debug target from workspace root when running `cargo test` from crate dir.
        path = PathBuf::from("target/debug/groundtruth_calc");
    }
    path
}

/// Helper to create an fbin file with synthetic vectors.
fn create_fbin_file(path: &PathBuf, vectors: &[[f32; DIM]]) {
    let mut file = File::create(path).unwrap();
    let nvecs = vectors.len() as i32;
    let dim = DIM as i32;
    file.write_all(&nvecs.to_le_bytes()).unwrap();
    file.write_all(&dim.to_le_bytes()).unwrap();
    for v in vectors {
        for &x in v {
            file.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

/// Helper to read a groundtruth ibin file.
fn read_groundtruth(path: &PathBuf) -> (usize, usize, Vec<Vec<i32>>) {
    let mut file = File::open(path).unwrap();
    let mut header = [0u8; 8];
    file.read_exact(&mut header).unwrap();
    let num_queries = i32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let k = i32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut groundtruth = Vec::with_capacity(num_queries);
    let mut row_buf = vec![0u8; k * 4];
    for _ in 0..num_queries {
        file.read_exact(&mut row_buf).unwrap();
        let row: Vec<i32> = row_buf
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        groundtruth.push(row);
    }
    (num_queries, k, groundtruth)
}

#[test]
fn end_to_end_computes_some_results_and_merge_output() {
    let bin = binary_path();
    assert!(bin.exists(), "groundtruth_calc binary not found at {:?}", bin);

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let queries = crate_dir.join("deep.query.public.10K.fbin");
    let base = crate_dir.join("deep.base.10K.fbin");

    // Use a small subset of queries via output dir and trust program to respect all queries in file.
    let tmpdir = tempfile::tempdir().unwrap();
    let results_dir = tmpdir.path().join("results");
    fs::create_dir_all(&results_dir).unwrap();
    let merged = tmpdir.path().join("merged.ibin");

    let status = Command::new(&bin)
        .current_dir(&crate_dir)
        .args([
            "-q",
            queries.to_str().unwrap(),
            "-b",
            base.to_str().unwrap(),
            "-o",
            results_dir.to_str().unwrap(),
            "--k",
            "10",
            "--workers",
            "2",
            "--merge-output",
            merged.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run groundtruth_calc binary");

    assert!(status.success());

    // Check that some per-query files and the merged file exist.
    let entries: Vec<_> = fs::read_dir(&results_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(!entries.is_empty(), "no per-query result files were created");
    assert!(merged.exists(), "merged groundtruth file was not created");
}

/// Test that extend mode produces correct groundtruth.
///
/// Strategy:
/// 1. Create synthetic query and base vectors with known distances
/// 2. Run normal mode on first 10 base vectors to get initial groundtruth
/// 3. Run extend mode with combined 20 vectors (first 10 + 10 new)
/// 4. Run normal mode on all 20 vectors (ground truth)
/// 5. Verify extend result matches full computation
#[test]
fn extend_mode_produces_correct_groundtruth() {
    let bin = binary_path();
    assert!(bin.exists(), "groundtruth_calc binary not found at {:?}", bin);

    let tmpdir = tempfile::tempdir().unwrap();

    // Create 3 query vectors (all zeros)
    let queries: Vec<[f32; DIM]> = (0..3).map(|_| [0.0f32; DIM]).collect();
    let query_path = tmpdir.path().join("queries.fbin");
    create_fbin_file(&query_path, &queries);

    // Create 10 "original" base vectors with distances 1, 4, 9, 16, 25, 36, 49, 64, 81, 100
    let mut original_base: Vec<[f32; DIM]> = Vec::new();
    for i in 0..10 {
        let mut v = [0.0f32; DIM];
        v[0] = (i + 1) as f32; // distance = (i+1)^2
        original_base.push(v);
    }
    let original_base_path = tmpdir.path().join("original_base.fbin");
    create_fbin_file(&original_base_path, &original_base);

    // Create 10 "new" base vectors with distances 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5
    // (squared: 0.25, 2.25, 6.25, 12.25, 20.25, 30.25, 42.25, 56.25, 72.25, 90.25)
    let mut new_base: Vec<[f32; DIM]> = Vec::new();
    for i in 0..10 {
        let mut v = [0.0f32; DIM];
        v[0] = (i as f32 + 0.5).sqrt(); // distance = i + 0.5
        new_base.push(v);
    }

    // Create combined base (original + new)
    let mut combined_base = original_base.clone();
    combined_base.extend(new_base);
    let combined_base_path = tmpdir.path().join("combined_base.fbin");
    create_fbin_file(&combined_base_path, &combined_base);

    // Step 1: Run normal mode on original base to get initial groundtruth
    let original_results_dir = tmpdir.path().join("original_results");
    fs::create_dir_all(&original_results_dir).unwrap();
    let original_groundtruth = tmpdir.path().join("original_groundtruth.ibin");

    let status = Command::new(&bin)
        .args([
            "-q", query_path.to_str().unwrap(),
            "-b", original_base_path.to_str().unwrap(),
            "-o", original_results_dir.to_str().unwrap(),
            "--k", "5",
            "--workers", "1",
            "--merge-output", original_groundtruth.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run groundtruth_calc for original base");
    assert!(status.success(), "Normal mode on original base failed");

    // Step 2: Run extend mode
    let extend_results_dir = tmpdir.path().join("extend_results");
    fs::create_dir_all(&extend_results_dir).unwrap();
    let extended_groundtruth = tmpdir.path().join("extended_groundtruth.ibin");

    let status = Command::new(&bin)
        .args([
            "--extend",
            "-q", query_path.to_str().unwrap(),
            "--existing-groundtruth", original_groundtruth.to_str().unwrap(),
            "--combined-base", combined_base_path.to_str().unwrap(),
            "--original-base-count", "10",
            "-o", extend_results_dir.to_str().unwrap(),
            "--k", "5",
            "--workers", "1",
            "--merge-output", extended_groundtruth.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run groundtruth_calc in extend mode");
    assert!(status.success(), "Extend mode failed");

    // Step 3: Run normal mode on combined base (ground truth)
    let full_results_dir = tmpdir.path().join("full_results");
    fs::create_dir_all(&full_results_dir).unwrap();
    let full_groundtruth = tmpdir.path().join("full_groundtruth.ibin");

    let status = Command::new(&bin)
        .args([
            "-q", query_path.to_str().unwrap(),
            "-b", combined_base_path.to_str().unwrap(),
            "-o", full_results_dir.to_str().unwrap(),
            "--k", "5",
            "--workers", "1",
            "--merge-output", full_groundtruth.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run groundtruth_calc for full base");
    assert!(status.success(), "Normal mode on full base failed");

    // Step 4: Compare extended groundtruth with full groundtruth
    let (ext_nq, ext_k, ext_gt) = read_groundtruth(&extended_groundtruth);
    let (full_nq, full_k, full_gt) = read_groundtruth(&full_groundtruth);

    assert_eq!(ext_nq, full_nq, "Query count mismatch");
    assert_eq!(ext_k, full_k, "k mismatch");

    for qid in 0..ext_nq {
        assert_eq!(
            ext_gt[qid], full_gt[qid],
            "Query {} groundtruth mismatch: extended={:?}, full={:?}",
            qid, ext_gt[qid], full_gt[qid]
        );
    }
}
