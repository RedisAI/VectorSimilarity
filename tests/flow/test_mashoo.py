# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import VecSim
from common import *
import subprocess
import hnswlib
result = subprocess.run(
            "grep CMAKE_BUILD_TYPE build/temp*/CMakeCache.txt",
            shell=True,
            capture_output=True,
            text=True
        )

print(f"\n{result.stdout}")
metric_to_string_map = {
    VecSimMetric_L2: "L2",
    VecSimMetric_IP: "IP",
    VecSimMetric_Cosine: "Cosine",}

datatype_to_string_map = {
    VecSimType_INT8: "int8",
    VecSimType_FLOAT32: "float32",
    }

def create_tiered_hnsw_params(swap_job_threshold = 0):
    tiered_hnsw_params = TieredHNSWParams()
    tiered_hnsw_params.swapJobThreshold = swap_job_threshold
    return tiered_hnsw_params
# def test_test():
#     print("Test")
# compare results with the original version of hnswlib - do not use elements deletion.
class INT8SetUp():
    def __init__(self,
                 metric,
                 dim = 1024,
                    num_elements = 1_000_000,
                    M = 32,
                    ef_c = 256,
                    ef_r = 50,
                    data_type = VecSimType_INT8,
                    ):
        hnsw_params = create_hnsw_params(dim = dim,
                                        num_elements = num_elements,
                                                metric = metric,
                                                data_type = data_type,
                                                ef_construction = ef_c,
                                                m = M,
                                                ef_runtime = ef_r,
                                                is_multi = False)
        tiered_hnsw_params = create_tiered_hnsw_params()

        self.index = Tiered_HNSWIndex(hnsw_params, tiered_hnsw_params, flat_buffer_size=1024)

        rng = np.random.default_rng(seed=42)
        #### Create vectors
        self.data = create_int8_vectors((num_elements, dim), rng)

        self.index.disable_logs()

        self.data_type = data_type
        self.num_elements = num_elements
        self.M = M
        self.ef_c = ef_c


def build(metric, data_type):
    int8_setup = INT8SetUp(metric = metric, data_type = data_type)
    index = int8_setup.index
    start = time.time()
    print(f"\nindex settings: data type: {datatype_to_string_map[data_type]}, M: {int8_setup.M}, ef_c: {int8_setup.ef_c}, metric: {metric_to_string_map[metric]}")
    for i, vector in enumerate(int8_setup.data):
        index.add_vector(vector, i)
    build_time = time.time() - start
    print(f"current hnsw size: {index.hnsw_label_count()}")
    threads_num = index.get_threads_num()
    print(f"indexing {int8_setup.num_elements} vectors with {threads_num} threads took {build_time:.2f} seconds")


    index.wait_for_index()

    assert index.hnsw_label_count() == int8_setup.num_elements
def test_cosine_int8():
    build(VecSimMetric_Cosine, VecSimType_INT8)
# def test_l2_int8():
#     build(VecSimMetric_L2, VecSimType_INT8)
# def test_l2_FLOAT32():
#     build(VecSimMetric_L2, VecSimType_FLOAT32)
# def test_cosine_FLOAT32():
#     build(VecSimMetric_Cosine, VecSimType_FLOAT32)


        #### Create queries
        # measure performance
