# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is a template file for converting an existing HNSW index file
# (*.hnsw_v3) to SQ8 quantized HNSW index (*-sq8.hnsw_v5).
# Usage: poetry run python tests/benchmark/data/scripts/convert_to_sq8.py

import numpy as np
from VecSim import *
import os

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'tests', 'benchmark', 'data')

# Index parameters (must match the source index)

'''
# For dbpedia-cosine-dim768-M64-efc512.hnsw_v3
INPUT_INDEX = os.path.join(DATA_DIR, 'dbpedia-cosine-dim768-M64-efc512.hnsw_v3')
#INPUT_INDEX = os.path.join(DATA_DIR, 'dbpedia-cosine-dim768-M64-efc512-fp16.hnsw_v3')
DIM = 768
M = 64
EF_CONSTRUCTION = 512
METRIC = VecSimMetric_Cosine
MULTI = False
TYPE = VecSimType_FLOAT16
#TYPE = VecSimType_FLOAT32
'''

# For fashion_images_multi_value-cosine-dim512-M64-efc512.hnsw_v3
#INPUT_INDEX = os.path.join(DATA_DIR, 'fashion_images_multi_value-cosine-dim512-M64-efc512.hnsw_v3')
INPUT_INDEX = os.path.join(DATA_DIR, 'fashion_images_multi_value-cosine-dim512-M64-efc512-fp16.hnsw_v3')
DIM = 512
M = 64
EF_CONSTRUCTION = 512
METRIC = VecSimMetric_Cosine
MULTI = True
TYPE = VecSimType_FLOAT16
#TYPE = VecSimType_FLOAT32
N_LABELS = 44441  # the number of unique labels

OUTPUT_INDEX = INPUT_INDEX.replace('.hnsw_v3', '-sq8.hnsw_v5')

def convert():
    print(f'Loading source index: {INPUT_INDEX}')
    source = HNSWIndex(INPUT_INDEX)
    n_vectors = source.index_size()
    print(f'  Loaded {n_vectors} vectors, dim={DIM}, multi={MULTI}')

    # Extract all vectors from the source index
    print('Extracting vectors...')

    if MULTI:
        # For multi-value indices, each label can have multiple vectors.
        # Discover labels by iterating and collecting non-empty results.
        n_labels = N_LABELS

        # Collect all vectors grouped by label
        label_vectors = {}  # label -> list of vectors
        total_extracted = 0
        for label in range(n_labels):
            vecs = source.get_vector(label)
            if vecs is not None and len(vecs) > 0:
                label_vectors[label] = vecs  # shape (num_vecs_for_label, dim)
                total_extracted += len(vecs)
            if label % 100000 == 0:
                print(f'  Extracted label {label}/{n_labels} ({total_extracted} vectors so far)')
        print(f'  Extracted {n_labels} labels, {total_extracted} vectors total')

        # Gather all vectors into a single array for computing the mean
        all_vectors = np.vstack(list(label_vectors.values()))
    else:
        # Single-value index: one vector per label, labels are 0..n_vectors-1
        all_vectors = np.zeros((n_vectors, DIM), dtype=np.float32)
        for label in range(n_vectors):
            vecs = source.get_vector(label)
            all_vectors[label] = vecs[0]  # get_vector returns array of shape (1, dim)
            if label % 100000 == 0:
                print(f'  Extracted {label}/{n_vectors}')
        print(f'  Extracted {n_vectors}/{n_vectors}')

    del source

    # Compute mean vector for SQ8 quantization
    print('Computing mean vector...')
    mean = all_vectors.mean(axis=0).astype(np.float32)

    # Create SQ8 HNSW index
    print('Creating SQ8 HNSW index...')
    params = HNSWParams()
    params.dim = DIM
    params.metric = METRIC
    params.multi = MULTI
    params.type = TYPE
    params.M = M
    params.efConstruction = EF_CONSTRUCTION
    params.quantType = VecSimQuant_SQ8
    params.quantParams = mean

    sq8_index = HNSWIndex(params)

    # Add vectors
    print('Indexing vectors...')
    if MULTI:
        added = 0
        for label, vecs in label_vectors.items():
            for vec in vecs:
                sq8_index.add_vector(vec.astype(np.float16), label)
                #sq8_index.add_vector(vec, label)
                added += 1
            if label % 100000 == 0:
                print(f'  label {label}/{n_labels} ({added} vectors added)')
        print(f'  Done: {added} vectors added across {len(label_vectors)} labels')
    else:
        for label in range(n_vectors):
            sq8_index.add_vector(all_vectors[label].astype(np.float16), label)
            #sq8_index.add_vector(all_vectors[label], label)
            if label % 100000 == 0:
                print(f'  {label}/{n_vectors}')
        print(f'  {n_vectors}/{n_vectors}')

    # Save
    print(f'Saving SQ8 index to: {OUTPUT_INDEX}')
    sq8_index.save_index(OUTPUT_INDEX)

    # Verify
    print('Verifying saved index...')
    loaded = HNSWIndex(OUTPUT_INDEX)
    loaded.check_integrity()
    expected_vectors = n_vectors
    assert loaded.index_size() == expected_vectors, \
        f'Expected {expected_vectors} vectors, got {loaded.index_size()}'
    file_size = os.path.getsize(OUTPUT_INDEX)
    print(f'  File size: {file_size / (1024**3):.2f} GB')
    print(f'  Vectors: {loaded.index_size()}')
    print('Done!')

if __name__ == '__main__':
    convert()
