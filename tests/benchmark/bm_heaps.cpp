#include <benchmark/benchmark.h>
#include <random>

#include "VecSim/utils/updatable_heap.h"

#include "VecSim/memory/vecsim_malloc.h"

class BM_UpdatableHeaps : public benchmark::Fixture {
public:
    std::shared_ptr<VecSimAllocator> allocator;
    std::default_random_engine priorityGenerator;
    std::uniform_real_distribution<double> distribution;
    BM_UpdatableHeaps() : allocator(VecSimAllocator::newVecsimAllocator()), distribution(0.0, 2.0) {}

    void SetUp(const ::benchmark::State &state) {
        priorityGenerator.seed(0); // fixed seed for reproducibility
    }

    void insertToHeap(vecsim_stl::abstract_priority_queue<float, size_t> &heap, float f, size_t n, size_t k) {
        if (heap.size() < k) {
            heap.emplace(f, n);
        } else if (f > heap.top().first) {
            heap.pop();
            heap.emplace(f, n);
        }
    }
};

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Insert_UpdatableHeap)(benchmark::State &state) {
    vecsim_stl::updatable_max_heap<float, size_t> heap(this->allocator);
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        this->insertToHeap(heap, this->distribution(this->priorityGenerator), c++, k);
    }
}

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Insert_AltUpdatableHeap)(benchmark::State &state) {
    vecsim_stl::alt_updatable_max_heap<float, size_t> heap(this->allocator);
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        this->insertToHeap(heap, this->distribution(this->priorityGenerator), c++, k);
    }
}

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Insert_BoostUpdatableHeap)(benchmark::State &state) {
    vecsim_stl::updatable_max_fibonacci_heap<float, size_t> heap(this->allocator);
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        this->insertToHeap(heap, this->distribution(this->priorityGenerator), c++, k);
    }
}

#define ITERATIONS 1 << 15
#define DATA_SIZE 1 << 20
#define RANGE 1 << 3, 1 << 12 // Keep diff of 9
#if 0
#define ITER
#else
#define ITER ->Iterations(ITERATIONS)
#endif

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Update_UpdatableHeap)(benchmark::State &state) {
    vecsim_stl::updatable_max_heap<float, size_t> heap(this->allocator);
    size_t labels = DATA_SIZE / 8;
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        this->insertToHeap(heap, this->distribution(this->priorityGenerator), c%labels, k);
        c++;
    }
}

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Update_AltUpdatableHeap)(benchmark::State &state) {
    vecsim_stl::alt_updatable_max_heap<float, size_t> heap(this->allocator);
    size_t labels = DATA_SIZE / 8;
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        this->insertToHeap(heap, this->distribution(this->priorityGenerator), c%labels, k);
        c++;
    }
}

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Update_BoostUpdatableHeap)(benchmark::State &state) {
    vecsim_stl::updatable_max_fibonacci_heap<float, size_t> heap(this->allocator);
    size_t labels = DATA_SIZE / 8;
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        this->insertToHeap(heap, this->distribution(this->priorityGenerator), c%labels, k);
        c++;
    }
}


BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Heaps_Boost_fib)(benchmark::State &state) {
    boost::heap::fibonacci_heap<std::pair<float, size_t>> heap;
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        float f = this->distribution(this->priorityGenerator);
        if (heap.size() < k) {
            heap.emplace(f, c);
        } else if (f > heap.top().first) {
            heap.pop();
            heap.emplace(f, c);
        }
        c++;
    }
}

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Heaps_Boost_std)(benchmark::State &state) {
    boost::heap::priority_queue<std::pair<float, size_t>> heap;
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        float f = this->distribution(this->priorityGenerator);
        if (heap.size() < k) {
            heap.emplace(f, c);
        } else if (f > heap.top().first) {
            heap.pop();
            heap.emplace(f, c);
        }
        c++;
    }
}

BENCHMARK_DEFINE_F(BM_UpdatableHeaps, Heaps_Std)(benchmark::State &state) {
    std::priority_queue<std::pair<float, size_t>> heap;
    size_t k = state.range(0);
    size_t c = 0;
    for (auto _ : state) {
        float f = this->distribution(this->priorityGenerator);
        if (heap.size() < k) {
            heap.emplace(f, c);
        } else if (f > heap.top().first) {
            heap.pop();
            heap.emplace(f, c);
        }
        c++;
    }
}

// BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Insert_UpdatableHeap)->RangeMultiplier(2)->Range(RANGE)ITER;
// BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Insert_AltUpdatableHeap)->RangeMultiplier(2)->Range(RANGE)ITER;
// BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Insert_BoostUpdatableHeap)->RangeMultiplier(2)->Range(RANGE)ITER;

// BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Update_UpdatableHeap)->RangeMultiplier(2)->Range(RANGE)ITER;
// BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Update_AltUpdatableHeap)->RangeMultiplier(2)->Range(RANGE)ITER;
// BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Update_BoostUpdatableHeap)->RangeMultiplier(2)->Range(RANGE)ITER;

BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Heaps_Boost_fib)->RangeMultiplier(2)->Range(RANGE)ITER;
BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Heaps_Boost_std)->RangeMultiplier(2)->Range(RANGE)ITER;
BENCHMARK_REGISTER_F(BM_UpdatableHeaps, Heaps_Std)->RangeMultiplier(2)->Range(RANGE)ITER;

BENCHMARK_MAIN();
