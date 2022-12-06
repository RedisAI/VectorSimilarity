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
#include <pthread.h>
#include <shared_mutex>

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
		size_t total_correct_cur = 0;
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
			total_correct_cur += correct;
			if (show_trace && q % 1000 == 999) {
				std::cout << "After running " << q+1 << " queries, recall of the last 1000 queries is "
				          << float(total_correct_cur) / (k * 1000) << std::endl;
				total_correct_cur = 0;
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
		cout << "\n***Starting parallel searching benchmark using " << n_threads << " threads***" << endl;

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
		cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory <<
		     " and the capacity is " << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getIndexCapacity() << endl;

		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
					(const void *)reinterpret_cast<HNSWIndex_Single<float, float> *>(hnsw_index)->getDataByLabel(i), i);
		}

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		std::atomic_int q_counter(0);

		cout << "Running " << queries.size() << " queries using " << n_threads << " threads..." << endl;
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
		cout << "\n***Starting parallel indexing benchmark using " << n_threads << " threads***" << endl;

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
		std::cout << "Max gap between number of vectors whose indexing began to the number"
		             " of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;
		cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory <<
		     " and the capacity is " << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getIndexCapacity() << endl;
		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
					(const void *)reinterpret_cast<HNSWIndex_Single<float, float>  *>(hnsw_index)->getDataByLabel(i),
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
		cout << "\n***Starting parallel indexing + searching benchmark using " << n_threads << " threads***"
		     << endl;
		auto hnsw_index = VecSimIndex_New(&hnswParams);
		auto bf_index = VecSimIndex_New(&bfParams);

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		std::atomic_int q_counter(0);

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
				std::this_thread::sleep_for(std::chrono::milliseconds(2000));
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
		             " of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;

		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		for (size_t i = 0; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
					(const void *)reinterpret_cast<HNSWIndex_Single<float, float> *>(hnsw_index)->getDataByLabel(i),
					i);
		}
		auto total_recall = computeRecall(total_res, bf_index, true);
		cout << "Total recall is: " << total_recall << endl;

		VecSimIndex_Free(hnsw_index);
		VecSimIndex_Free(bf_index);
	}

	/*
	 * 1. insert 500K vectors (in parallel)
	 * 2. do parallel search/update - 4 threads are searching, 4 threads are updating. recall is measured w.r.t updated
	 * data.
	 */
	void run_parallel_update_benchmark_delete_alone() {
		// Set the global r/w lock to prefer writers.
		std::shared_mutex guard;
		auto *handle = (pthread_rwlock_t *)guard.native_handle();
		pthread_rwlock_t rwlock_prefer_writers;
		pthread_rwlockattr_t attr;
		pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
		pthread_rwlock_init(&rwlock_prefer_writers, &attr);
		*handle = rwlock_prefer_writers;

		cout << "\n***Starting parallel updating + searching benchmark where delete operations are executed alone"
				" using " << n_threads << " threads***" << endl;
		auto hnsw_index = VecSimIndex_New(&hnswParams);
		auto bf_index = VecSimIndex_New(&bfParams);

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		std::atomic_int q_counter(0);

		cout << "Creating an HNSW index of size " << data.size() / 2 << " with dim=" << hnswParams.hnswParams.dim
		     << " with parallel indexing"  << endl;
		auto started = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(n_threads) shared(hnsw_index, queries, k, cout) default(none)
		{
			int myID = omp_get_thread_num();
#pragma omp critical
			cout << "Thread " << myID << " started indexing vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
			for (size_t i = 0; i < data.size() / 2; i++) {
				VecSimIndex_AddVector(hnsw_index, (const void *)data[i].data(), i);
			}
#pragma omp critical
			cout << "Thread " << myID << " done indexing vectors" << endl;
		}

		auto done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size() / 2);
		std::cout << "Total build time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		std::cout << "Max gap between number of vectors whose indexing began to the number"
		             " of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;
		cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory  <<
		" and the capacity is " << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getIndexCapacity() << endl;
		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;

		// Staring phase 2 - parallel read/update
		size_t sleep_time = 20;
		cout << "\nUpdating " << data.size() / 2 << " vectors with dim=" << hnswParams.hnswParams.dim
		     << " in parallel. After " << sleep_time << " seconds, we start running " << queries.size()
		     << " queries in parallel"  << endl;
		size_t last_inserted_label = data.size()/2;

		started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(n_threads) shared(hnsw_index, queries, total_res, k, cout, q_counter, last_inserted_label, sleep_time, guard) default(none)
		{
			int myID = omp_get_thread_num();
			if (myID % 2) {
#pragma omp critical
				cout << "Thread " << myID << " started updating vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
				for (size_t i = data.size() / 2; i < data.size(); i++) {
#pragma omp critical
					{
						last_inserted_label = i;
					}
					// delete job runs alone
					guard.lock();
					VecSimIndex_DeleteVector(hnsw_index, i - data.size() / 2);
					guard.unlock();
					guard.lock_shared();
					VecSimIndex_AddVector(hnsw_index, (const void *) data[i].data(), i);
					guard.unlock_shared();
				}
#pragma omp critical
				cout << "Thread " << myID << " done updating vectors" << endl;
			} else {
				std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
#pragma omp critical
				cout << "Thread " << myID << " start running queries, index size upon search starts: "
				     << VecSimIndex_IndexSize(hnsw_index) << " and max label is: " <<
					 last_inserted_label << endl;
				while (true) {
					size_t next_val = q_counter++;
					if (next_val >= queries.size()) {
						break;
					}
					guard.lock_shared();
					auto hnsw_results =
							VecSimIndex_TopKQuery(hnsw_index, queries[next_val].data(), k, nullptr, BY_SCORE);
					guard.unlock_shared();
					total_res[next_val] = hnsw_results;
				}
#pragma omp critical
				cout << "Thread " << myID << " done running queries, index size upon search ends: "
				     << VecSimIndex_IndexSize(hnsw_index) << " and max label is: " <<
					 last_inserted_label << endl;
			}
		}
		done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size() / 2);
		std::cout << "Total update/search time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		std::cout << "Max gap between number of vectors whose indexing began to the number"
		             " of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;
		cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory <<
		" and the capacity is " << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getIndexCapacity() << endl;

		serializer.reset(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		for (size_t i = data.size() / 2; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
			                      (const void *)reinterpret_cast<HNSWIndex_Single<float, float> *>(hnsw_index)->getDataByLabel(i),
			                      i);
		}
		auto total_recall = computeRecall(total_res, bf_index, true);
		cout << "Total recall is: " << total_recall << endl;

		VecSimIndex_Free(hnsw_index);
		VecSimIndex_Free(bf_index);
	}

	/*
	 * 1. insert 500K vectors (in parallel)
	 * 2. do parallel search/update - 4 threads are searching, 4 threads are updating. recall is measured w.r.t updated
	 * data.
	 */
	void run_parallel_update_benchmark_with_repair_jobs() {
		cout << "\n***Starting parallel updating + searching benchmark where delete operations are executed with repair jobs"
		        " using " << n_threads << " threads***" << endl;

		// Set the global r/w lock to prefer writers.
		std::shared_mutex guard;
		auto *handle = (pthread_rwlock_t *)guard.native_handle();
		pthread_rwlock_t rwlock_prefer_writers;
		pthread_rwlockattr_t attr;
		pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
		pthread_rwlock_init(&rwlock_prefer_writers, &attr);
		*handle = rwlock_prefer_writers;

		auto hnsw_index = VecSimIndex_New(&hnswParams);
		auto bf_index = VecSimIndex_New(&bfParams);

		std::vector<VecSimQueryResult_List> total_res(queries.size());
		std::atomic_int q_counter(0);

		cout << "Creating an HNSW index of size " << data.size() / 2 << " with dim=" << hnswParams.hnswParams.dim
		     << " with parallel indexing"  << endl;
		auto started = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(n_threads) shared(hnsw_index, queries, k, cout) default(none)
		{
			int myID = omp_get_thread_num();
#pragma omp critical
			cout << "Thread " << myID << " started indexing vectors..." << endl;
#pragma omp for schedule(dynamic) nowait
			for (size_t i = 0; i < data.size()/2; i++) {
				VecSimIndex_AddVector(hnsw_index, (const void *)data[i].data(), i);
			}
#pragma omp critical
			cout << "Thread " << myID << " done indexing vectors" << endl;
		}
		auto done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size()/2);
		std::cout << "Total build time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		std::cout << "Max gap between number of vectors whose indexing began to the number"
		             " of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;
		auto serializer =
				HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory << " and the capacity is "
		<< reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->getIndexCapacity() <<endl;
		serializer.reset();

		// Staring phase 2 - parallel read/update
		size_t sleep_time = 7;
		cout << "\nDeleting " << data.size() / 2 << " vectors with dim=" << hnswParams.hnswParams.dim
		     << " in parallel"  << endl;

		size_t last_inserted_label = data.size()/2;

		// Holds an indicator for whether the deleted item is ready for eviction.
		std::unordered_map<idType, long> swap_jobs;
		std::deque<repairJob> all_jobs;
		std::mutex jobs_queue_guard;

		started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(n_threads) shared(hnsw_index, queries, total_res, k, cout, q_counter, swap_jobs, \
		sleep_time, last_inserted_label, guard, serializer, all_jobs, jobs_queue_guard) default(none)
		{
			int myID = omp_get_thread_num();
			if (myID == 0) {
				for (size_t i = 0; i < data.size() / 2; i++) {
					guard.lock_shared();
					auto repair_jobs = reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->removeVector_POC(i);
					VecSimIndex_AddVector(hnsw_index, (const void *) data[i + data.size() / 2].data(),
					                      i + data.size() / 2);
					guard.unlock_shared();
					swap_jobs.insert({i, repair_jobs.size()});

					jobs_queue_guard.lock();
					last_inserted_label++;
					all_jobs.insert(all_jobs.end(), repair_jobs.begin(), repair_jobs.end());
					jobs_queue_guard.unlock();

					if (i % 1000 == 999) {
						guard.lock();
						std::vector<idType> to_remove;
						for (auto &it : swap_jobs) {
							if (it.second == 0) {
								to_remove.push_back(it.first);
								reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->SwapJob_POC(it.first);
							}
						}
//						serializer.reset(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
//						cout << "Checking index integrity after " << i << " deletions and swapping: " << serializer.checkIntegrity().valid_state << endl;
						for (auto it : to_remove) {
							swap_jobs.erase(it);
						}
						guard.unlock();
					}
				}
			} else if (myID >= 1 && myID < 6) {
#pragma omp critical
				cout << "Thread " << myID << " repairing vectors" << endl;
				while (true) {
					jobs_queue_guard.lock();
					if (swap_jobs.empty() && last_inserted_label == data.size()-1) {
						break;
					}
					auto job = all_jobs.front();
					all_jobs.pop_front();
					jobs_queue_guard.unlock();

					guard.lock_shared();
					reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->repairConnectionsForDeletion_POC(job.internal_id, job.level);
					__atomic_fetch_sub(&(swap_jobs[job.associated_deleted_id]), 1, __ATOMIC_RELAXED);
					guard.unlock_shared();
				}
				cout << "Thread " << myID << " done repairing nodes" << endl;
			} else {
				std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
#pragma omp critical
				cout << "Thread " << myID << " start running queries, index size upon search starts: "
			        << VecSimIndex_IndexSize(hnsw_index) << " and max label is: " <<
			        last_inserted_label << endl;
			while (true) {
				size_t next_val = q_counter++;
				if (next_val >= queries.size()) {
					break;
				}
				guard.lock_shared();
				auto hnsw_results =
						VecSimIndex_TopKQuery(hnsw_index, queries[next_val].data(), k, nullptr, BY_SCORE);
				guard.unlock_shared();
				total_res[next_val] = hnsw_results;
			}
#pragma omp critical
			cout << "Thread " << myID << " done running queries, index size upon search ends: "
			     << VecSimIndex_IndexSize(hnsw_index) << " and max label is: " <<
			     last_inserted_label << endl;
		}
	}
		done = std::chrono::high_resolution_clock::now();
		assert(VecSimIndex_IndexSize(hnsw_index) == data.size()/2);
		std::cout << "Index size is " << VecSimIndex_IndexSize(hnsw_index) << " with "
		<< VecSimIndex_Info(hnsw_index).hnswInfo.numDeleted << " marked deleted elements" << std::endl;
		std::cout << "Total update time is "
		          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
		          << " ms" << std::endl;
		std::cout << "Max gap between number of vectors whose indexing began to the number"
		             " of vectors whose indexing finished and available for search is: "
		          << reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index)->max_gap << std::endl;
		cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory << endl;

		if (!swap_jobs.empty()) {
			started = std::chrono::high_resolution_clock::now();
			for (auto &it: swap_jobs) {
				reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->SwapJob_POC(it.first);
			}
			done = std::chrono::high_resolution_clock::now();
			std::cout << "Index size is " << VecSimIndex_IndexSize(hnsw_index) << " with "
			          << VecSimIndex_Info(hnsw_index).hnswInfo.numDeleted << " marked deleted elements" << std::endl;
			std::cout << "Total swap and clean time of " << swap_jobs.size() << "leftovers is "
			          << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count()
			          << " ms" << std::endl;
			cout << "Index memory is : " << VecSimIndex_Info(hnsw_index).hnswInfo.memory <<
			     " and the capacity is " << reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->getIndexCapacity()
			     << endl;
		}
		serializer.reset(reinterpret_cast<HNSWIndex<float, float>  *>(hnsw_index));
		cout << "Checking index integrity: " << serializer.checkIntegrity().valid_state << endl;
		serializer.reset();

		for (size_t i = data.size() / 2; i < data.size(); i++) {
			VecSimIndex_AddVector(bf_index,
			                      (const void *)reinterpret_cast<HNSWIndex_Single<float, float> *>(hnsw_index)->getDataByLabel(i),
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
					.blockSize = 1024,
					.efRuntime = 200}};

	VecSimParams bf_params{.algo = VecSimAlgo_BF,
			.bfParams = BFParams{.type = VecSimType_FLOAT32,
					.dim = dim,
					.metric = VecSimMetric_Cosine,
					.initialCapacity = n,
					.blockSize = n}};

	auto bm = BM_ParallelHNSW(params, bf_params, n_threads, n_queries, k);
//    bm.run_parallel_indexing_benchmark();
//	bm.run_parallel_search_benchmark();
//	bm.run_all_parallel_benchmark();
//	bm.run_parallel_update_benchmark_delete_alone();
	bm.run_parallel_update_benchmark_with_repair_jobs();
}
