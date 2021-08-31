#include <benchmark/benchmark.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define __builtin_popcount(t) __popcnt(t)
#else

#include <x86intrin.h>

#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#define USE_AVX512
#include <immintrin.h>

size_t dim = 128;
size_t n = 1000000;
size_t k = 10;

static float InnerProductSIMD16Ext(float *pVect1, float *pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[16];

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return 1.0f - sum;
}

static float L2SqrSIMD16Ext(float *pVect1, float *pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

static void BruteForceIndex_InternalProduct(float *vectors, float *queryBlob, float *scores) {
    for (size_t i = 0; i < n; i++) {
        scores[i] = InnerProductSIMD16Ext(vectors + (i * dim), queryBlob, dim);
    }
}

static void BruteForceIndex_L2(float *vectors, float *queryBlob, float *scores) {
    for (size_t i = 0; i < n; i++) {
        scores[i] = L2SqrSIMD16Ext(vectors + (i * dim), queryBlob, dim);
    }
}

static void avx512_matrix_vector_ip(benchmark::State &state) {
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
    for (auto _ : state) {
        // This code gets timed
        BruteForceIndex_InternalProduct(vectors, v, scores);
    }
}

static void avx512_matrix_vector_l2(benchmark::State &state) {
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
    for (auto _ : state) {
        BruteForceIndex_L2(vectors, v, scores);
    }
}

// Register the function as a benchmark
BENCHMARK(avx512_matrix_vector_ip);
BENCHMARK(avx512_matrix_vector_l2);
// Run the benchmark
BENCHMARK_MAIN();
