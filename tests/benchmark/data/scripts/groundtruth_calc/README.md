# groundtruth_calc

A Rust tool for computing exact k-nearest neighbor groundtruth for vector similarity benchmarks.

## Features

- **Normal mode**: Compute groundtruth from scratch for query and base vector sets
- **Extend mode**: Efficiently update existing groundtruth when base vectors are appended
- **Merge-only mode**: Combine per-query result files into a single groundtruth file
- **Validation mode**: Verify groundtruth file correctness
- **Checkpoint/resume**: Automatically resumes from where it left off if interrupted
- **Parallel processing**: Uses all CPU cores by default

## Building

```bash
cargo build --release
```

## File Formats

- **Query/Base vectors**: `.fbin` format (4-byte header: nvecs, dim; then float32 vectors)
- **Groundtruth**: `.ibin` format (4-byte header: nqueries, k; then int32 neighbor indices)

## Usage

### Normal Mode

Compute groundtruth from scratch:

```bash
groundtruth_calc \
    -q queries.fbin \
    -b base.fbin \
    -o results_dir \
    --k 100 \
    --merge-output groundtruth.ibin
```

### Extend Mode

Efficiently update groundtruth when new vectors are appended to the base set:

```bash
groundtruth_calc --extend \
    -q queries.fbin \
    --existing-groundtruth old_groundtruth.ibin \
    --combined-base combined_base.fbin \
    --original-base-count 100000000 \
    -o results_dir \
    --merge-output new_groundtruth.ibin
```

This is much faster than recomputing from scratch because it:
1. Loads existing groundtruth and computes distances only for those neighbors
2. Only scans the new vectors (skips the original base set)
3. Merges results to find the new top-k

### Merge-Only Mode

Combine per-query result files into a single groundtruth file:

```bash
groundtruth_calc --merge-only \
    -o results_dir \
    --merge-output groundtruth.ibin \
    --k 100 \
    --num-queries 10000
```

### Validation Mode

Verify a groundtruth file is correct:

```bash
groundtruth_calc --validate \
    --groundtruth groundtruth.ibin \
    -q queries.fbin \
    -b base.fbin
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-q, --queries` | Path to query vectors (.fbin) | Required* |
| `-b, --base` | Path to base vectors (.fbin) | Required* |
| `-o, --output` | Output directory for per-query results | Required* |
| `--k` | Number of nearest neighbors | 100 |
| `-j, --workers` | Number of worker threads | CPU cores |
| `--merge-output` | Path for merged groundtruth file | None |
| `--progress-interval` | Progress update frequency | 100 |

### Extend Mode Options

| Option | Description |
|--------|-------------|
| `--extend` | Enable extend mode |
| `--existing-groundtruth` | Path to existing groundtruth (.ibin) |
| `--combined-base` | Path to combined base vectors (.fbin) |
| `--original-base-count` | Number of vectors in original base |

### Merge-Only Mode Options

| Option | Description |
|--------|-------------|
| `--merge-only` | Enable merge-only mode |
| `--num-queries` | Total number of queries |

### Validation Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--validate` | Enable validation mode | |
| `--groundtruth` | Path to groundtruth file to validate | |
| `--validate-samples` | Number of queries to sample | 10 |
| `--validate-non-neighbors` | Non-neighbors to check per query | 1000 |

*Required unless using a special mode that doesn't need them.

## Examples

### Full workflow for extending groundtruth

1. You have existing groundtruth for 100M base vectors
2. You want to add 100M more vectors (total 200M)

```bash
# Concatenate base files (outside this tool)
cat base.100M.fbin base.new.100M.fbin > base.200M.fbin

# Extend groundtruth (only scans the new 100M vectors)
groundtruth_calc --extend \
    -q queries.fbin \
    --existing-groundtruth groundtruth.100M.ibin \
    --combined-base base.200M.fbin \
    --original-base-count 100000000 \
    -o extend_results \
    --merge-output groundtruth.200M.ibin \
    -j 16
```

### Resume interrupted computation

Just run the same command again - it will skip already-completed queries:

```bash
# If interrupted, just run again
groundtruth_calc -q queries.fbin -b base.fbin -o results --merge-output gt.ibin
```

## Output

- **Per-query files**: `results_dir/query_00000.ibin`, `query_00001.ibin`, etc.
- **Merged file**: Single `.ibin` file with all queries (if `--merge-output` specified)
- **Backup**: Existing output files are renamed to `.old` before overwriting

