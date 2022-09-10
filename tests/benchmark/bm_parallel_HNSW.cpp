#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"

#include <unistd.h>
#include <omp.h>
#include <thread>
#include <random>
#include <atomic>
#include <iostream>
#include <cassert>

using std::cout;
using std::endl;
using std::atomic_int;

void run_parallel_search_benchmark() {
	size_t n = 100000;
	size_t dim = 32;
	size_t n_threads = 8;
	cout << "\nStarting parallel searching benchmark using " << n_threads << " threads" << endl;

	VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
			.hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n,
					.efRuntime = 200
			}};
	VecSimIndex *index = VecSimIndex_New(&params);

	VecSimParams bf_params{.algo = VecSimAlgo_BF,
			.bfParams = BFParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n}};
	VecSimIndex *bf_index = VecSimIndex_New(&bf_params);

	std::mt19937 rng;
	rng.seed(47);
	std::uniform_real_distribution<float> distrib(-1, 1);

	// Generate random queries
	size_t num_queries = 20000;
	size_t k = 10;
	float queries[num_queries][dim];

	for (size_t i = 0; i < num_queries; i++) {
		for (size_t j = 0; j < dim; j++) {
			queries[i][j] = distrib(rng);
		}
	}

	cout << "Creating an HNSW index of size " << n << " with dim=" << dim << " sequentially" << endl;
	auto started = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < n; i++) {
		float f[dim];
		for (size_t j = 0; j < dim; j++) {
			f[j] = distrib(rng);
		}
		VecSimIndex_AddVector(index, (const void *) f, i);
	}
	assert(VecSimIndex_IndexSize(index) == n);
	auto done = std::chrono::high_resolution_clock::now();
	std::cout << "Total build time is "
	          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << " ms" << std::endl;

	VecSimQueryResult_List total_res[num_queries];
	atomic_int q_counter(0);
	started = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(n_threads) shared(n, index, dim, distrib, rng, num_queries, queries, total_res, k, cout, q_counter) default(none)
	{
		int myID = omp_get_thread_num();
#pragma omp critical
		cout << "Thread " << myID << " start running queries, index size upon search starts: " << VecSimIndex_IndexSize(index) << endl;
		while (true) {
			size_t next_val = q_counter++;
			if (next_val >= num_queries) {
				break;
			}
			auto hnsw_results = VecSimIndex_TopKQuery(index, queries[next_val],
				                                          k, nullptr, BY_SCORE);
			total_res[next_val] = hnsw_results;
			}
#pragma omp critical
		cout << "Thread " << myID << " done running queries, index size upon search ends: " << VecSimIndex_IndexSize(index) << endl;
	}
	done = std::chrono::high_resolution_clock::now();
	std::cout << "Total search time is "
	          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << " ms" << std::endl;
	std::cout << "Max gap between number of vectors whose indexing began to the number"
	             "of vectors whose indexing finished and available for search is :"
	          << reinterpret_cast<HNSWIndex *>(index)->getHNSWIndex()->max_gap << std::endl;

	for (size_t i = 0; i < n; i++) {
		VecSimIndex_AddVector(bf_index,
		                      (const void *) reinterpret_cast<HNSWIndex *>(index)->getHNSWIndex()->getDataByLabel(i),
		                      i);
	}

// Measure recall:
	size_t total_correct = 0;
	for (size_t q = 0; q < num_queries; q++) {
		size_t correct = 0;
		auto hnsw_results = total_res[q];
		auto bf_results = VecSimIndex_TopKQuery(bf_index, queries[q], k,
		                                        nullptr, BY_SCORE);
		auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
		while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
			auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
			auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
			while (VecSimQueryResult_IteratorHasNext(bf_it)) {
				auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
				if (VecSimQueryResult_GetId(hnsw_res_item) ==
				    VecSimQueryResult_GetId(bf_res_item)) {
					correct++;
					break;
				}
			}
			VecSimQueryResult_IteratorFree(bf_it);
		}
		VecSimQueryResult_IteratorFree(hnsw_it);

		VecSimQueryResult_Free(bf_results);
		VecSimQueryResult_Free(hnsw_results);
		if (q % 1000 == 1 && q > 1) {
			std::cout << "Accumulated recall after " << q << " queries is " << float(total_correct) / (k * q)
			          << std::endl;
		}
		total_correct += correct;
	}
	std::cout << "Total recall is " << float(total_correct) / (k * num_queries) << std::endl;

	VecSimIndex_Free(index);
}

void run_parallel_indexing_benchmark() {
	size_t n = 100000;
	size_t dim = 32;
	size_t n_threads = 8;
	cout << "\nStarting parallel indexing benchmark using " << n_threads << " threads" << endl;

	VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
			.hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n,
					.efRuntime = 200
			}};
	VecSimIndex *index = VecSimIndex_New(&params);

	VecSimParams bf_params{.algo = VecSimAlgo_BF,
			.bfParams = BFParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n}};
	VecSimIndex *bf_index = VecSimIndex_New(&bf_params);

	std::mt19937 rng;
	rng.seed(47);
	std::uniform_real_distribution<float> distrib(-1, 1);

	// Generate random queries
	size_t num_queries = 20000;
	size_t k = 10;
	float queries[num_queries][dim];

	for (size_t i = 0; i < num_queries; i++) {
		for (size_t j = 0; j < dim; j++) {
			queries[i][j] = distrib(rng);
		}
	}
	atomic_int q_counter(0);

	cout << "Creating an HNSW index of size " << n << " with dim=" << dim << " with parallel indexing" << endl;
	auto started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(n_threads) shared(n, index, dim, distrib, rng, num_queries, queries, k, cout, q_counter) default(none)
	{
		int myID = omp_get_thread_num();
#pragma omp critical
			cout << "Thread " << myID << " started indexing vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
			for (size_t i = 0; i < n; i++) {
				float f[dim];
				for (size_t j = 0; j < dim; j++) {
					f[j] = distrib(rng);
				}
				VecSimIndex_AddVector(index, (const void *) f, i);
			}
#pragma omp critical
			cout << "Thread " << myID << " done indexing vectors" << endl;

	}
	assert(VecSimIndex_IndexSize(index) == n);
	auto done = std::chrono::high_resolution_clock::now();
	std::cout << "Total build time is "
	          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << " ms" << std::endl;
	std::cout << "Max gap between number of vectors whose indexing began to the number"
	             "of vectors whose indexing finished and available for search is :"
	          << reinterpret_cast<HNSWIndex *>(index)->getHNSWIndex()->max_gap << std::endl;

	for (size_t i = 0; i < n; i++) {
		VecSimIndex_AddVector(bf_index,
		                      (const void *) reinterpret_cast<HNSWIndex *>(index)->getHNSWIndex()->getDataByLabel(i),
		                      i);
	}
	cout << "Running queries (sequentially)" << endl;

// Measure recall:
	size_t total_correct = 0;
	long total_time = 0;
	for (size_t q = 0; q < num_queries; q++) {
		started = std::chrono::high_resolution_clock::now();
		auto hnsw_results = VecSimIndex_TopKQuery(index, queries[q],
		                                          k, nullptr, BY_SCORE);
		done = std::chrono::high_resolution_clock::now();
		total_time += std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
		size_t correct = 0;
		auto bf_results = VecSimIndex_TopKQuery(bf_index, queries[q], k,
		                                        nullptr, BY_SCORE);
		auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
		while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
			auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
			auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
			while (VecSimQueryResult_IteratorHasNext(bf_it)) {
				auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
				if (VecSimQueryResult_GetId(hnsw_res_item) ==
				    VecSimQueryResult_GetId(bf_res_item)) {
					correct++;
					break;
				}
			}
			VecSimQueryResult_IteratorFree(bf_it);
		}
		VecSimQueryResult_IteratorFree(hnsw_it);

		VecSimQueryResult_Free(bf_results);
		VecSimQueryResult_Free(hnsw_results);
		if (q % 1000 == 1 && q > 1) {
			std::cout << "Accumulated recall after " << q << " queries is " << float(total_correct) / (k * q)
			          << std::endl;
		}
		total_correct += correct;
	}
	std::cout << "Total search time is " << total_time << " ms" << std::endl;
	std::cout << "Total recall is " << float(total_correct) / (k * num_queries) << std::endl;

	VecSimIndex_Free(index);
}

void run_all_parallel_benchmark() {
	size_t n = 100000;
	size_t dim = 32;
	size_t n_threads = 8;
	cout << "\nStarting parallel indexing + searching benchmark using " << n_threads << " threads" << endl;

	VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
			.hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n,
					.efRuntime = 200
			}};
	VecSimIndex *index = VecSimIndex_New(&params);

	VecSimParams bf_params{.algo = VecSimAlgo_BF,
			.bfParams = BFParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n}};
	VecSimIndex *bf_index = VecSimIndex_New(&bf_params);

	std::mt19937 rng;
	rng.seed(47);
	std::uniform_real_distribution<float> distrib(-1, 1);

	// Generate random queries
	size_t num_queries = 20000;
	size_t k = 10;
	float queries[num_queries][dim];

	for (size_t i = 0; i < num_queries; i++) {
		for (size_t j = 0; j < dim; j++) {
			queries[i][j] = distrib(rng);
		}
	}
	VecSimQueryResult_List total_res[num_queries];
	atomic_int q_counter(0);

	cout << "Creating an HNSW index of size " << n << " with dim=" << dim << " with parallel indexing. After 4 seconds, "
																			 "we start running queries in parallel" << endl;
	auto started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(n_threads) shared(n, index, dim, distrib, rng, num_queries, queries, total_res, k, cout, q_counter) default(none)
	{
		int myID = omp_get_thread_num();
		if (myID % 2) {
#pragma omp critical
			cout << "Thread " << myID << " started indexing vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
			for (size_t i = 0; i < n; i++) {
				float f[dim];
				for (size_t j = 0; j < dim; j++) {
					f[j] = distrib(rng);
				}
				VecSimIndex_AddVector(index, (const void *) f, i);
			}
#pragma omp critical
			cout << "Thread " << myID << " done indexing vectors" << endl;
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(4000));
#pragma omp critical
			cout << "Thread " << myID << " start running queries, index size upon search starts: " << VecSimIndex_IndexSize(index) << endl;
			while (true) {

				size_t next_val = q_counter++;
				if (next_val >= num_queries) {
					break;
				}
				auto hnsw_results = VecSimIndex_TopKQuery(index, queries[next_val],
				                                          k, nullptr, BY_SCORE);
				total_res[next_val] = hnsw_results;
			}
#pragma omp critical
			cout << "Thread " << myID << " done running queries, index size upon search ends: " << VecSimIndex_IndexSize(index) << endl;
		}
	}
	assert(VecSimIndex_IndexSize(index) == n);
	auto done = std::chrono::high_resolution_clock::now();
	std::cout << "Total build/search time is "
	          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << " ms" << std::endl;
	std::cout << "Max gap between number of vectors whose indexing began to the number"
	             "of vectors whose indexing finished and available for search is :"
	          << reinterpret_cast<HNSWIndex *>(index)->getHNSWIndex()->max_gap << std::endl;

	for (size_t i = 0; i < n; i++) {
		VecSimIndex_AddVector(bf_index,
		                      (const void *) reinterpret_cast<HNSWIndex *>(index)->getHNSWIndex()->getDataByLabel(i),
		                      i);
	}

// Measure recall:
	size_t total_correct = 0;
	for (size_t q = 0; q < num_queries; q++) {
		size_t correct = 0;
		auto hnsw_results = total_res[q];
		auto bf_results = VecSimIndex_TopKQuery(bf_index, queries[q], k,
		                                        nullptr, BY_SCORE);
		auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
		while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
			auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
			auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
			while (VecSimQueryResult_IteratorHasNext(bf_it)) {
				auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
				if (VecSimQueryResult_GetId(hnsw_res_item) ==
				    VecSimQueryResult_GetId(bf_res_item)) {
					correct++;
					break;
				}
			}
			VecSimQueryResult_IteratorFree(bf_it);
		}
		VecSimQueryResult_IteratorFree(hnsw_it);

		VecSimQueryResult_Free(bf_results);
		VecSimQueryResult_Free(hnsw_results);
		if (q % 1000 == 1 && q > 1) {
			std::cout << "Accumulated recall after " << q << " queries is " << float(total_correct) / (k * q)
			          << std::endl;
		}
		total_correct += correct;
	}
	std::cout << "Total recall is " << float(total_correct) / (k * num_queries) << std::endl;

	VecSimIndex_Free(index);
}

int main() {
	run_all_parallel_benchmark();
	run_parallel_indexing_benchmark();
	run_parallel_search_benchmark();
}
