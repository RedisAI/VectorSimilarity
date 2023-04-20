# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

from VecSim import *
import numpy as np
from scipy import spatial
from numpy.testing import assert_allclose
import time

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

