#include <benchmark/benchmark.h>
#include "OpenBLAS/cblas.h"

size_t dim = 128;
size_t n = 1000000;
size_t k = 10;

static void BruteForceIndex_InternalProduct(float *vectors, float *queryBlob, float *scores) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, dim, -1, vectors, dim, queryBlob, 1, 1, scores, 1);
}

static void BruteForceIndex_L2(float *vectors, float *queryBlob, float *scores) {
    float tmp_vector[dim];
    for (size_t i = 0; i < n; i++) {
        cblas_scopy(dim, vectors + (i * dim), 1, tmp_vector, 1);
        cblas_saxpy(dim, -1.0f, queryBlob, 1, tmp_vector, 1);
        scores[i] = cblas_sdot(dim, tmp_vector, 1, tmp_vector, 1);
    }
}

static void OpenBlas_normalize_vector(benchmark::State &state) {

    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)rand() / (float)(RAND_MAX / 100);
    }
    for (auto _ : state) {
        // This code gets timed
        float norm = cblas_snrm2(dim, v, 1);
        norm = norm == 0.0 ? 0 : 1.0f / norm;
        cblas_sscal(dim, norm, v, 1);
    }
}

static void OpenBlas_sgemv(benchmark::State &state) {
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)rand() / (float)(RAND_MAX / 100);
    }

    float *vectors = (float *)malloc(n * dim * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            (vectors + i * dim)[j] = (float)rand() / (float)(RAND_MAX / 100);
        }
    }
    // This code gets timed
    float scores[n];
    std::fill_n(scores, n, 1.0);
    for (auto _ : state) {
        BruteForceIndex_InternalProduct(vectors, v, scores);
    }
}

static void OpenBlas_matrix_vector_l2(benchmark::State &state) {
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)rand() / (float)(RAND_MAX / 100);
    }

    float *vectors = (float *)malloc(n * dim * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            (vectors + i * dim)[j] = (float)rand() / (float)(RAND_MAX / 100);
        }
    }
    // This code gets timed
    float scores[n];
    std::fill_n(scores, n, 1.0);
    for (auto _ : state) {
        BruteForceIndex_L2(vectors, v, scores);
    }
}

static void OpenBlas_find_min(benchmark::State &state) {

    float v[n];
    for (size_t i = 0; i < n; i++) {
        v[i] = (float)rand() / (float)(RAND_MAX / 100);
    }
    for (auto _ : state) {
        // This code gets timed
        cblas_ismin(n, v, 1);
    }
}

static void OpenBlas_top_k_ip(benchmark::State &state) {
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)rand() / (float)(RAND_MAX / 100);
    }

    float *vectors = (float *)malloc(n * dim * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            (vectors + i * dim)[j] = (float)rand() / (float)(RAND_MAX / 100);
        }
    }
    float scores[n];
    std::fill_n(scores, n, 1.0);
    for (auto _ : state) {
        // This code gets timed
        BruteForceIndex_InternalProduct(vectors, v, scores);
        for (size_t i = 0; i < k; i++) {
            size_t min_index = cblas_ismin(n, scores, 1);
            scores[min_index] = std::numeric_limits<float>::max();
        }
    }
}

static void OpenBlas_top_k_l2(benchmark::State &state) {
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)rand() / (float)(RAND_MAX / 100);
    }

    float *vectors = (float *)malloc(n * dim * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            (vectors + i * dim)[j] = (float)rand() / (float)(RAND_MAX / 100);
        }
    }
    float scores[n];
    std::fill_n(scores, n, 1.0);
    for (auto _ : state) {
        // This code gets timed
        BruteForceIndex_L2(vectors, v, scores);
        for (size_t i = 0; i < k; i++) {
            size_t min_index = cblas_ismin(n, scores, 1);
            scores[min_index] = std::numeric_limits<float>::max();
        }
    }
}

// Register the function as a benchmark
BENCHMARK(OpenBlas_normalize_vector);
BENCHMARK(OpenBlas_sgemv);
BENCHMARK(OpenBlas_matrix_vector_l2);
BENCHMARK(OpenBlas_find_min);
BENCHMARK(OpenBlas_top_k_ip);
BENCHMARK(OpenBlas_top_k_l2);
// Run the benchmark
BENCHMARK_MAIN();
