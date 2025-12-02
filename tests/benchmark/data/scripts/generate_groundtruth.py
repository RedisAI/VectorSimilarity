#!/usr/bin/env python3
"""
Generate groundtruth file for deep.base.10M.fbin dataset.
Computes the 100 nearest neighbors for each of the 10K query vectors.
Supports checkpointing to allow interruption and resumption.
"""

import numpy as np
from fbin_reader import read_fbin, write_ibin, read_ibin
import time
import os

CHECKPOINT_DIR = 'tests/benchmark/data/scripts/.groundtruth_checkpoint'
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'groundtruth_checkpoint.ibin')
PROGRESS_FILE = os.path.join(CHECKPOINT_DIR, 'progress.txt')

def ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(groundtruth, completed_batches, total_batches):
    """Save progress checkpoint as ibin file."""
    ensure_checkpoint_dir()

    # Save the groundtruth computed so far as ibin
    write_ibin(CHECKPOINT_FILE, groundtruth)

    # Save progress metadata
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"Completed: {completed_batches}/{total_batches} batches ({100*completed_batches/total_batches:.1f}%)\n")
        f.write(f"Timestamp: {time.ctime()}\n")

    print(f"Checkpoint saved: {completed_batches}/{total_batches} batches")

def load_checkpoint():
    """Load progress checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(PROGRESS_FILE):
        try:
            # Read progress metadata
            with open(PROGRESS_FILE, 'r') as f:
                progress_line = f.readline()

            # Parse completed batches and total batches from progress file
            # Format: "Completed: X/Y batches (Z%)"
            parts = progress_line.split('/')
            if len(parts) >= 2:
                completed_batches = int(parts[0].split()[-1])
                total_batches_in_checkpoint = int(parts[1].split()[0])

                # Load the groundtruth ibin file
                groundtruth = read_ibin(CHECKPOINT_FILE)

                print(f"Loaded checkpoint: {completed_batches}/{total_batches_in_checkpoint} batches completed")
                print(f"Checkpoint shape: {groundtruth.shape}")

                return {
                    'groundtruth': groundtruth,
                    'completed_batches': completed_batches,
                    'total_batches': total_batches_in_checkpoint
                }
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    return None

def compute_groundtruth_chunked(base_vectors, query_vectors, k=100, batch_size=100, resume=True):
    """
    Compute k nearest neighbors for each query vector with checkpointing.

    Args:
        base_vectors: (N, D) array of base vectors
        query_vectors: (Q, D) array of query vectors
        k: number of nearest neighbors to find
        batch_size: process queries in batches to save memory
        resume: whether to resume from checkpoint if available

    Returns:
        (Q, k) array of indices of nearest neighbors
    """
    n_queries = query_vectors.shape[0]
    n_base = base_vectors.shape[0]
    total_batches = (n_queries + batch_size - 1) // batch_size

    # Try to load checkpoint
    checkpoint = None
    start_batch = 0
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            # Check if checkpoint is compatible (same number of queries and k)
            if checkpoint['groundtruth'].shape == (n_queries, k):
                groundtruth = checkpoint['groundtruth']
                # Recalculate start_batch based on current batch_size
                # The checkpoint may have been created with a different batch_size
                start_batch = checkpoint['completed_batches']
                print(f"Resuming from batch {start_batch} (checkpoint was created with different batch size)")
            else:
                print(f"Checkpoint shape {checkpoint['groundtruth'].shape} doesn't match expected shape {(n_queries, k)}")
                print("Starting fresh...")
                groundtruth = np.zeros((n_queries, k), dtype=np.int32)
        else:
            groundtruth = np.zeros((n_queries, k), dtype=np.int32)
    else:
        groundtruth = np.zeros((n_queries, k), dtype=np.int32)

    print(f"Computing groundtruth for {n_queries} queries against {n_base} base vectors")
    print(f"Finding {k} nearest neighbors per query")
    print(f"Total batches: {total_batches}, Starting from batch: {start_batch}\n")

    start_time = time.time()

    for batch_idx in range(start_batch, total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_queries)
        batch_queries = query_vectors[batch_start:batch_end]

        # Compute L2 distances: ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q·b
        # Compute squared norms
        query_norms = np.sum(batch_queries ** 2, axis=1, keepdims=True)  # (batch_size, 1)
        base_norms = np.sum(base_vectors ** 2, axis=1)  # (n_base,)

        # Compute dot products: q · b
        dot_products = np.dot(batch_queries, base_vectors.T)  # (batch_size, n_base)

        # Compute squared distances: ||q - b||^2
        distances = query_norms + base_norms - 2 * dot_products  # (batch_size, n_base)

        # Find k nearest neighbors (smallest distances)
        nearest_indices = np.argsort(distances, axis=1)[:, :k]
        groundtruth[batch_start:batch_end] = nearest_indices

        elapsed = time.time() - start_time
        progress = ((batch_idx + 1) / total_batches) * 100
        rate = (batch_idx + 1 - start_batch) / elapsed if elapsed > 0 else 0
        eta = (total_batches - batch_idx - 1) / rate if rate > 0 else 0

        print(f"Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - "
              f"Elapsed: {elapsed:.1f}s - Rate: {rate:.2f} batches/s - ETA: {eta:.1f}s")

        # Save checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            save_checkpoint(groundtruth, batch_idx + 1, total_batches)

    return groundtruth


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate groundtruth for deep.base.10M.fbin')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start fresh, ignore checkpoint')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch size for processing queries (default: 5)')
    args = parser.parse_args()

    print("Loading data files...")
    start_time = time.time()

    # Load query and base vectors
    query_vectors = read_fbin('tests/benchmark/data/deep.query.public.10K.fbin')
    print(f"Loaded query vectors: {query_vectors.shape}")

    base_vectors = read_fbin('tests/benchmark/data/deep.base.100K.fbin')
    print(f"Loaded base vectors: {base_vectors.shape}")

    load_time = time.time() - start_time
    print(f"Data loading took {load_time:.1f}s\n")

    # Compute groundtruth with checkpointing
    start_time = time.time()
    groundtruth = compute_groundtruth_chunked(base_vectors, query_vectors, k=100,
                                              batch_size=args.batch_size, resume=args.resume)
    compute_time = time.time() - start_time
    print(f"\nGroundtruth computation took {compute_time:.1f}s")

    # Write groundtruth to file
    print(f"\nWriting groundtruth to file...")
    output_file = 'tests/benchmark/data/deep.groundtruth.100K.10K.ibin'
    write_ibin(output_file, groundtruth)
    print(f"Groundtruth written to: {output_file}")
    print(f"Groundtruth shape: {groundtruth.shape}")
    print(f"First 5 rows (first 10 neighbors):")
    print(groundtruth[:5, :10])

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        os.remove(PROGRESS_FILE)
        print(f"\nCheckpoint files cleaned up")

