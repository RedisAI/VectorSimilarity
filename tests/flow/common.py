# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

from VecSim import *
import numpy as np
from scipy import spatial
from numpy.testing import assert_allclose, assert_equal
import time
import math

def create_hnsw_params(dim, num_elements, metric, data_type, ef_construction=200, m=16, ef_runtime=10, epsilon=0.01,
                      is_multi=False):
    hnsw_params = HNSWParams()

    hnsw_params.dim = dim
    hnsw_params.metric = metric
    hnsw_params.type = data_type
    hnsw_params.M = m
    hnsw_params.efConstruction = ef_construction
    hnsw_params.initialCapacity = num_elements
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
    hnsw_params.initialCapacity = num_elements
    hnsw_params.efRuntime = ef_runtime
    hnsw_params.epsilon = epsilon
    hnsw_params.multi = is_multi

    return HNSWIndex(hnsw_params)


# Compute the expected speedup as a function of the expected parallel section rate of the code by Amdahl's law
def expected_speedup(expected_parallel_rate, n_threads):
    return 1 / ((1-expected_parallel_rate) + expected_parallel_rate/n_threads)

def bytes_to_mega(bytes, ndigits = 3):
    return round(bytes/pow(10,6), ndigits)

def round_(f_value, ndigits = 2):
    return round(f_value, ndigits)


def round_ms(f_value, ndigits = 2):
    return round(f_value * 1000, ndigits)  
