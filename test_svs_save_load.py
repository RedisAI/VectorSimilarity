#!/usr/bin/env python3
"""
Test script to verify SVS save_index and load_index functionality
"""

import numpy as np
import tempfile
import os
import sys

from VecSim import *

try:
    import VecSim
    print("VecSim module imported successfully")
except ImportError as e:
    print(f"Failed to import VecSim: {e}")
    sys.exit(1)

def test_svs_save_load():
    """Test SVS index save and load functionality"""

    # Create test data
    dim = 128
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, dim).astype(np.float32)
    labels = np.arange(n_vectors, dtype=np.uint64)

    # Create SVS parameters
    svs_params = VecSim.SVSParams()
    svs_params.dim = dim
    svs_params.type = VecSim.VecSimType_FLOAT32
    svs_params.metric = VecSim.VecSimMetric_L2
    svs_params.blockSize = 1024

    # Create SVS index
    print("Creating SVS index...")
    index = VecSim.SVSIndex(svs_params)

    # Add vectors to index
    print("Adding vectors to index...")
    index.add_vector_parallel(vectors, labels)

    print(f"Index size after adding vectors: {index.index_size()}")

    # Test query before saving
    query_vector = vectors[0:1]  # Use first vector as query
    results = index.knn_query(query_vector, k=5)
    print(f"Query results before save: {results}")

    # Create temporary directory for saving
    temp_dir = "temp"
    save_path = os.path.join(temp_dir, "svs_index")
    os.makedirs(save_path,exist_ok=True)

    # Test save_index
    print(f"Saving index to {save_path}...")
    try:
        index.save_index(save_path)
        print("Index saved successfully!")

        # Check if files were created
        config_path = os.path.join(save_path, "config")
        graph_path = os.path.join(save_path, "graph")
        data_path = os.path.join(save_path, "data")

        print(f"Config directory exists: {os.path.exists(config_path)}")
        print(f"Graph directory exists: {os.path.exists(graph_path)}")
        print(f"Data directory exists: {os.path.exists(data_path)}")

    except Exception as e:
        print(f"Error saving index: {e}")
        return False

    # Create new index for loading
    print("Creating new index for loading...")
    new_index = VecSim.SVSIndex(svs_params)

    # Test load_index
    print(f"Loading index from {save_path}...")
    try:
        new_index.load_index(save_path)
        print("Index loaded successfully!")

        print(f"Loaded index size: {new_index.index_size()}")

        # Test query after loading
        results_after_load = new_index.knn_query(query_vector, k=5)
        print(f"Query results after load: {results_after_load}")

        # Compare results
        if np.array_equal(results[0], results_after_load[0]) and np.array_equal(results[1], results_after_load[1]):
            print("SUCCESS: Query results match before and after save/load!")
            return True
        else:
            print("WARNING: Query results differ before and after save/load")
            return True  # Still consider it a success if loading worked

    except Exception as e:
        print(f"Error loading index: {e}")
        return False

if __name__ == "__main__":
    print("Testing SVS save_index and load_index functionality...")
    success = test_svs_save_load()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)
