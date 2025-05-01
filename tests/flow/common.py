# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

from VecSim import *
import numpy as np
from scipy import spatial
from numpy.testing import assert_allclose, assert_equal
import time
import math
from ml_dtypes import bfloat16

def create_hnsw_params(dim, num_elements, metric, data_type, ef_construction=200, m=16, ef_runtime=10, epsilon=0.01,
                      is_multi=False):
    hnsw_params = HNSWParams()

    hnsw_params.dim = dim
    hnsw_params.metric = metric
    hnsw_params.type = data_type
    hnsw_params.M = m
    hnsw_params.efConstruction = ef_construction
    hnsw_params.efRuntime = ef_runtime
    hnsw_params.epsilon = epsilon
    hnsw_params.multi = is_multi

    return hnsw_params

# Helper function for creating an index,uses the default HNSW parameters if not specified.
def create_hnsw_index(dim, num_elements, metric, data_type, ef_construction=200, m=16, ef_runtime=10, epsilon=0.01,
                      is_multi=False):
    hnsw_params = HNSWParams()

    hnsw_params.dim = dim
    hnsw_params.metric = metric
    hnsw_params.type = data_type
    hnsw_params.M = m
    hnsw_params.efConstruction = ef_construction
    hnsw_params.efRuntime = ef_runtime
    hnsw_params.epsilon = epsilon
    hnsw_params.multi = is_multi

    return HNSWIndex(hnsw_params)

# Helper function for creating an index, uses the default flat parameters if not specified.
def create_flat_index(dim, metric, data_type, is_multi=False):
    bfparams = BFParams()

    bfparams.dim = dim
    bfparams.type = data_type
    bfparams.metric = metric
    bfparams.multi = is_multi

    return BFIndex(bfparams)

def create_add_vectors(index, vectors):
    label_to_vec_list = []
    for i, vector in enumerate(vectors):
        index.add_vector(vector, i)
        label_to_vec_list.append((i, vector))
    return label_to_vec_list

# Compute the expected speedup as a function of the expected parallel section rate of the code by Amdahl's law
def expected_speedup(expected_parallel_rate, n_threads):
    return 1 / ((1-expected_parallel_rate) + expected_parallel_rate/n_threads)

def bytes_to_mega(bytes, ndigits = 3):
    return round(bytes/pow(10,6), ndigits)

def round_(f_value, ndigits = 2):
    return round(f_value, ndigits)


def round_ms(f_value, ndigits = 2):
    return round(f_value * 1000, ndigits)

def vec_to_bfloat16(vec):
    return vec.astype(bfloat16)

def vec_to_float16(vec):
    return vec.astype(np.float16)

def create_int8_vectors(shape, rng: np.random.Generator = None):
    rng = np.random.default_rng(seed=42) if rng is None else rng
    return rng.integers(low=-128, high=127, size=shape, dtype=np.int8)

def create_uint8_vectors(shape, rng: np.random.Generator = None):
    rng = np.random.default_rng(seed=42) if rng is None else rng
    return rng.integers(low=0, high=255, size=shape, dtype=np.uint8)

def get_ground_truth_results(dist_func, query, vectors, k):
    results = [{"dist": dist_func(query, vec), "label": key} for key, vec in vectors]
    results = sorted(results, key=lambda x: x["dist"])
    keys = [res["label"] for res in results[:k]]

    return results, keys

def fp32_expand_and_calc_cosine_dist(a, b):
    # stupid numpy doesn't make any intermediate conversions when handling small types
    # so we might get overflow. We need to convert to float32 ourselves.
    a_float32 = a.astype(np.float32)
    b_float32 = b.astype(np.float32)
    return spatial.distance.cosine(a_float32, b_float32)
