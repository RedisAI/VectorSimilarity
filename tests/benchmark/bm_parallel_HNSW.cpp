#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/algorithms/hnsw/serialization.h"

#include <unistd.h>
#include <omp.h>
#include <thread>
#include <random>
#include <atomic>
#include <iostream>
#include <cassert>

using std::atomic_int;
using std::cout;
using std::endl;

class BM_ParallelHNSW  {
private:
	std::vector<std::vector<float>> data;
	size_t n_threads;

	VecSimParams bfParams;
	VecSimParams hnswParams;

	std::vector<std::vector<float>> queries;
	size_t k;

	float computeRecall(const std::vector<VecSimQueryResult_List> &total_res, VecSimIndex *bf_index, bool show_trace=false) {
		size_t total_correct = 0;
		for (size_t q = 0; q < queries.size(); q++) {
			size_t correct = 0;
			auto hnsw_results = total_res[q];
			auto bf_results = VecSimIndex_TopKQuery(bf_index, queries[q].data(), k, nullptr, BY_SCORE);
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
			total_correct += correct;
			if (show_trace && q % 1000 == 999) {
				std::cout << "Accumulated recall after " << q+1 << " queries is "
				          << float(total_correct) / (k * q) << std::endl;
			}
		}
		return (float)total_correct / ((float)k*(float)queries.size());
	}

public:
	BM_ParallelHNSW(VecSimParams hnsw_params, VecSimParams bf_params, size_t n_threads_, size_t n_queries, size_t k_) {
		n_threads = n_threads_;
		k = k_;
		hnswParams = hnsw_params;
		bfParams = bf_params;

		// Generate random vectors data and queries
		size_t n_vectors = hnsw_params.hnswParams.initialCapacity;
		data.reserve(n_vectors);
		std::mt19937 rng;
		rng.seed(47);
		std::uniform_real_distribution<float> distrib(-1, 1);
		for (size_t i=0; i<n_vectors; i++) {
			std::vector<float> vec(hnsw_params.hnswParams.dim);
			for (size_t j=0; j < hnsw_params.hnswParams.dim; j++) {
				vec[j] = (distrib(rng));
			}
			data.push_back(vec);
		}
		queries.reserve(n_queries);
		for (size_t i=0; i<n_queries; i++) {
			std::vector<float> vec(hnsw_params.hnswParams.dim);
			for (size_t j=0; j < hnsw_params.hnswParams.dim; j++) {
				vec[j] = (distrib(rng));
			}
			queries.push_back(vec);
		}
	}
	~BM_ParallelHNSW() = default;

	void run_parallel_search_benchmark() {
		cout << "\nStarting parallel searching benchmark using " << n_threads << " threads" << endl;

		auto hnsw_index = VecSimIndex_New(&hnswParams);
		auto bf_index = VecSimIndex_New(&bfParams);

		cout << "Creating an HNSW index of size " << data.size() << " with dim=" << hnswParams.hnswParams.dim
		<< " sequentially..." << endl;

		auto started = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(hnsw_index, (const void *)data[i].data(), i);
		}
		auto done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size());
		std::cout << "Total build time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;

		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
					(const void *)reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->getDataByLabel(i), i);
		}

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		atomic_int q_counter(0);

		cout << "\nRunning " << queries.size() << " queries using " << n_threads << " threads..." << endl;
		started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(n_threads)                                                        \
    shared(hnsw_index, queries, total_res, k, cout, q_counter) default(none)
		{
			while (true) {
				size_t next_val = q_counter++;
				if (next_val >= queries.size()) {
					break;
				}
				auto hnsw_results =
						VecSimIndex_TopKQuery(hnsw_index, queries[next_val].data(), k, nullptr, BY_SCORE);
				total_res[next_val] = hnsw_results;
			}
		}

		done = std::chrono::high_resolution_clock::now();
		std::cout << "Total search time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;

		cout << "Total recall is: " << computeRecall(total_res, bf_index) << endl;

		VecSimIndex_Free(hnsw_index);
		VecSimIndex_Free(bf_index);
	}

	void run_parallel_indexing_benchmark() {
		cout << "\nStarting parallel indexing benchmark using " << n_threads << " threads" << endl;

		auto hnsw_index = VecSimIndex_New(&hnswParams);
		auto bf_index = VecSimIndex_New(&bfParams);

		cout << "Creating an HNSW index of size " << data.size() << " with dim=" << hnswParams.hnswParams.dim
		     << " with parallel indexing" << endl;
		auto started = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(n_threads) shared(hnsw_index, queries, k, cout) default(none)
		{
			int myID = omp_get_thread_num();
#pragma omp critical
			cout << "Thread " << myID << " started indexing vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
			for (size_t i = 0; i < data.size(); i++) {
				VecSimIndex_AddVector(hnsw_index, (const void *)data[i].data(), i);
			}
#pragma omp critical
			cout << "Thread " << myID << " done indexing vectors" << endl;
		}

		auto done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size());
		std::cout << "Total build time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		std::cout << "Max gap between number of vectors whose indexing began to the number "
		             "of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;

		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
					(const void *)reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getDataByLabel(i),
					i);
		}

		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		cout << "Running " << queries.size() << " queries (sequentially)..." << endl;
		started = std::chrono::high_resolution_clock::now();

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		for (size_t i=0; i < queries.size(); i++) {
			total_res[i] = VecSimIndex_TopKQuery(hnsw_index, queries[i].data(), k, nullptr, BY_SCORE);
		}

		done = std::chrono::high_resolution_clock::now();
		std::cout << "Total search time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		cout << "Total recall is: " << computeRecall(total_res, bf_index) << endl;

		VecSimIndex_Free(hnsw_index);
		VecSimIndex_Free(bf_index);
	}

	void run_all_parallel_benchmark() {
		cout << "\nStarting parallel indexing + searching benchmark using " << n_threads << " threads"
		     << endl;
		auto hnsw_index = VecSimIndex_New(&hnswParams);
		auto bf_index = VecSimIndex_New(&bfParams);

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		atomic_int q_counter(0);

		cout << "Creating an HNSW index of size " << data.size() << " with dim=" << hnswParams.hnswParams.dim
		     << " with parallel indexing. After 4 seconds, we start running " << queries.size()
			 << " queries in parallel"  << endl;
		auto started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(n_threads) shared(hnsw_index, queries, total_res, k, cout, q_counter) default(none)
		{
			int myID = omp_get_thread_num();
			if (myID % 2) {
#pragma omp critical
				cout << "Thread " << myID << " started indexing vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
				for (size_t i = 0; i < data.size(); i++) {
					VecSimIndex_AddVector(hnsw_index, (const void *)data[i].data(), i);
				}
#pragma omp critical
				cout << "Thread " << myID << " done indexing vectors" << endl;
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(4000));
#pragma omp critical
				cout << "Thread " << myID << " start running queries, index size upon search starts: "
				     << VecSimIndex_IndexSize(hnsw_index) << endl;
				while (true) {
					size_t next_val = q_counter++;
					if (next_val >= queries.size()) {
						break;
					}
					auto hnsw_results =
							VecSimIndex_TopKQuery(hnsw_index, queries[next_val].data(), k, nullptr, BY_SCORE);
					total_res[next_val] = hnsw_results;
				}
#pragma omp critical
				cout << "Thread " << myID << " done running queries, index size upon search ends: "
				     << VecSimIndex_IndexSize(hnsw_index) << endl;
			}
		}
		auto done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size());
		std::cout << "Total build/search time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		std::cout << "Max gap between number of vectors whose indexing began to the number"
		             "of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;

		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
					(const void *)reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getDataByLabel(i),
					i);
		}
		auto total_recall = computeRecall(total_res, bf_index, true);
		cout << "Total recall is: " << total_recall << endl;

		VecSimIndex_Free(hnsw_index);
		VecSimIndex_Free(bf_index);
	}
};


int main() {
	size_t n = 100000;
	size_t dim = 32;
	size_t n_threads = 8;
	size_t k = 10;
	size_t n_queries = 20000;

	VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
			.hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n,
					.efRuntime = 200}};

	VecSimParams bf_params{.algo = VecSimAlgo_BF,
			.bfParams = BFParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n}};

	auto bm = BM_ParallelHNSW(params, bf_params, n_threads, n_queries, k);
	bm.run_parallel_search_benchmark();
    bm.run_parallel_indexing_benchmark();
	bm.run_all_parallel_benchmark();
}
