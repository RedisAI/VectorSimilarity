#pragma once

#include "bm_common.h"

template <typename index_type_t>
class BM_VecSimBF16dIndex : public BM_VecSimCommon<index_type_t> {
public:
    using data_t = BM_VecSimCommon<index_type_t>::data_t;
    const static unsigned short bf16_index_offset = 2;

    // The constructor is called after we have already registered the tests residing in
    // BM_VecSimCommon, (and not in this class) so `ref_count` is not zero at the first time
    // BM_VecSimBF16dIndex Ctor is called, and we can't rely on it to decide whether we should
    // initialize the indices or not. This is why we use the `is_initialized` flag. Also, we keep
    // the value of ref_count at the moment of initialization in first_bf16BM_ref_count to free
    // the indices when ref_count is decreased to this value. Reminder: ref_count is updated in
    // BM_VecSimIndex ctor (and dtor).
    static bool is_initialized;

    static size_t first_bf16BM_ref_count;
    BM_VecSimBF16dIndex() {
        if (!is_initialized) {
            // Initialize the updated indexes as well, if this is the first instance.
            Initialize();
            is_initialized = true;
            first_bf16BM_ref_count = REF_COUNT;
        }
    }

    ~BM_VecSimBF16dIndex() {
        if (REF_COUNT == first_bf16BM_ref_count) {
            VecSimIndex_Free(INDICES[VecSimAlgo_BF + bf16_index_offset]);
        }
    }

private:
    static void loadRawVectors(const std::string &raw_file, data_t *raw_vectors_output) {
        std::ifstream input(raw_file, std::ios::binary);

        if (!input.is_open()) {
            throw std::runtime_error("Raw vectors file was not found in path. Exiting...");
        }
        input.seekg(0, std::ifstream::beg);
        std::cout << "start loadRawVectors" << std::endl;
        for (size_t i = 0; i < N_VECTORS; ++i) {
            // data_t *blob = raw_vectors + DIM * i;
            data_t blob[DIM];
            input.read((char *)blob, DIM * sizeof(data_t));

            VecSimIndex_AddVector(INDICES[VecSimAlgo_BF], blob, i);
            VecSimIndex_AddVector(INDICES[VecSimAlgo_HNSWLIB], blob, i);
            VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + bf16_index_offset], blob, i);
        }

        std::cout << "done loadRawVectors" << std::endl;

        //    assert(input.eof());
        //   input.read((char *)raw_vectors_output, DIM * N_VECTORS * sizeof(data_t));
    }

    static const char *raw_vectors_file;
    static void Initialize();
};

template <typename index_type_t>
bool BM_VecSimBF16dIndex<index_type_t>::is_initialized = false;

template <typename index_type_t>
size_t BM_VecSimBF16dIndex<index_type_t>::first_bf16BM_ref_count = 0;

template <typename index_type_t>
void BM_VecSimBF16dIndex<index_type_t>::Initialize() {

    // Load raw vectors
    //  data_t *raw_vectors = new data_t[N_VECTORS * DIM];
    // loadRawVectors(BM_VecSimGeneral::AttachRootPath(raw_vectors_file), raw_vectors);
    loadRawVectors(BM_VecSimGeneral::AttachRootPath(raw_vectors_file), NULL);

    // for (size_t i = 0; i < N_VECTORS; ++i) {
    //     data_t *blob = raw_vectors + DIM * i;
    //     VecSimIndex_AddVector(INDICES[VecSimAlgo_BF], blob, i);
    //     VecSimIndex_AddVector(INDICES[VecSimAlgo_HNSWLIB], blob, i);
    //     VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + bf16_index_offset], blob, i);
    // }
    // delete[] raw_vectors;
}
