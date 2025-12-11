use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Helper to get the path to the built binary `groundtruth_calc`.
fn binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_groundtruth_calc"));
    if !path.exists() {
        // Fallback: assume debug target from workspace root when running `cargo test` from crate dir.
        path = PathBuf::from("target/debug/groundtruth_calc");
    }
    path
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

