/*
 * HNSW Disk Index Serializer
 *
 * This program creates serialized HNSW disk indexes from binary vector files.
 * It supports both .raw files (no header) and .fbin files (with header).
 *
 * Usage:
 *   ./hnsw_disk_serializer <input_file> <output_name> <dim> <metric> <type> [M] [efConstruction] [threads]
 *
 * Arguments:
 *   input_file      - Binary file containing vectors (.raw or .fbin)
 *                     .raw: Raw binary data (no header, dim must be specified)
 *                     .fbin: Binary file with header (dim can be auto-detected or verified)
 *   output_name     - Base name for output (will create output_name.hnsw_disk_v1 and output_name_rocksdb/)
 *   dim             - Vector dimension (required for .raw, optional for .fbin - use 0 for auto-detect)
 *   metric          - Distance metric: L2, IP, or Cosine
 *   type            - Data type: FLOAT32, FLOAT64, BFLOAT16, FLOAT16, INT8, UINT8
 *   M               - HNSW M parameter (default: 64)
 *   efConstruction  - HNSW efConstruction parameter (default: 512)
 *   threads         - Number of threads for parallel indexing (default: 4, use 0 for single-threaded)
 *
 * Examples:
 *   # Using .raw file (dimension required)
 *   ./hnsw_disk_serializer dbpedia-cosine-dim768-test_vectors.raw \
 *       dbpedia-cosine-dim768-M64-efc512 768 Cosine FLOAT32 64 512
 *
 *   # Using .fbin file (auto-detect dimension) with 8 threads
 *   ./hnsw_disk_serializer vectors.fbin output 0 Cosine FLOAT32 64 512 8
 *
 *   # Using .fbin file (verify dimension)
 *   ./hnsw_disk_serializer vectors.fbin output 768 Cosine FLOAT32 64 512
 */

#include "VecSim/vec_sim.h"
#include "VecSim/index_factories/hnsw_disk_factory.h"
#include "VecSim/algorithms/hnsw/hnsw_disk.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "mock_thread_pool.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <unistd.h>
#include <thread>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;



// Helper to parse metric
VecSimMetric parseMetric(const std::string &metric) {
    if (metric == "L2" || metric == "euclidean") return VecSimMetric_L2;
    if (metric == "IP") return VecSimMetric_IP;
    if (metric == "Cosine" || metric == "cosine") return VecSimMetric_Cosine;
    throw std::runtime_error("Unknown metric: " + metric + ". Use L2, IP, or Cosine");
}

// Helper to parse type
VecSimType parseType(const std::string &type) {
    if (type == "FLOAT32" || type == "float32") return VecSimType_FLOAT32;
    if (type == "FLOAT64" || type == "float64") return VecSimType_FLOAT64;
    if (type == "BFLOAT16" || type == "bfloat16") return VecSimType_BFLOAT16;
    if (type == "FLOAT16" || type == "float16") return VecSimType_FLOAT16;
    if (type == "INT8" || type == "int8") return VecSimType_INT8;
    if (type == "UINT8" || type == "uint8") return VecSimType_UINT8;
    throw std::runtime_error("Unknown type: " + type);
}

// Helper to get type size in bytes
size_t getTypeSize(VecSimType type) {
    switch (type) {
        case VecSimType_FLOAT32: return 4;
        case VecSimType_FLOAT64: return 8;
        case VecSimType_BFLOAT16: return 2;
        case VecSimType_FLOAT16: return 2;
        case VecSimType_INT8: return 1;
        case VecSimType_UINT8: return 1;
        default: throw std::runtime_error("Unknown type size");
    }
}

// Helper to get type name
const char* getTypeName(VecSimType type) {
    switch (type) {
        case VecSimType_FLOAT32: return "FLOAT32";
        case VecSimType_FLOAT64: return "FLOAT64";
        case VecSimType_BFLOAT16: return "BFLOAT16";
        case VecSimType_FLOAT16: return "FLOAT16";
        case VecSimType_INT8: return "INT8";
        case VecSimType_UINT8: return "UINT8";
        default: return "UNKNOWN";
    }
}

// Helper to detect file type based on extension
enum class FileType {
    RAW,   // Raw binary file (no header)
    FBIN   // Binary file with header (DiskANN format)
};

FileType detectFileType(const std::string &filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    if (ext == "fbin") {
        return FileType::FBIN;
    }
    return FileType::RAW;
}

// Helper to get file info for fbin files
struct FileInfo {
    size_t num_vectors;
    size_t dim;
};

// Helper to read fbin header from an open stream
FileInfo readFbinHeader(std::ifstream &file, const std::string &filename) {
    uint32_t num_vectors = 0;
    uint32_t dim = 0;
    file.read(reinterpret_cast<char *>(&num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));

    if (!file) {
        throw std::runtime_error("Failed to read header from file: " + filename);
    }

    return {num_vectors, dim};
}

FileInfo getFbinInfo(const std::string &filename, VecSimType type) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    auto info = readFbinHeader(file, filename);
    return info;
}

// Template function to load vectors from fbin file
template<typename T>
size_t loadVectorsFromFbin(const std::string &filename, VecSimIndex *index,
                           size_t dim, size_t num_vectors, size_t report_interval) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    // Read and skip header
    readFbinHeader(input, filename);

    size_t vector_size_bytes = dim * sizeof(T);
    std::vector<char> vector_buffer(vector_size_bytes);
    size_t vectors_added = 0;

    while (vectors_added < num_vectors && input.read(vector_buffer.data(), vector_size_bytes)) {
        VecSimIndex_AddVector(index, vector_buffer.data(), vectors_added);
        vectors_added++;

        if (vectors_added % report_interval == 0 || vectors_added == num_vectors) {
            std::cout << "\rProgress: " << vectors_added << "/" << num_vectors
                      << " (" << (vectors_added * 100 / num_vectors) << "%)" << std::flush;
        }
    }
    return vectors_added;
}

// Template function to load vectors from raw file
template<typename T>
size_t loadVectorsFromRaw(const std::string &filename, VecSimIndex *index,
                          size_t dim, size_t num_vectors, size_t report_interval) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    size_t vector_size_bytes = dim * sizeof(T);
    std::vector<char> vector_buffer(vector_size_bytes);
    size_t vectors_added = 0;

    while (input.read(vector_buffer.data(), vector_size_bytes)) {
        VecSimIndex_AddVector(index, vector_buffer.data(), vectors_added);
        vectors_added++;

        if (vectors_added % report_interval == 0 || vectors_added == num_vectors) {
            std::cout << "\rProgress: " << vectors_added << "/" << num_vectors
                      << " (" << (vectors_added * 100 / num_vectors) << "%)" << std::flush;
        }
    }

    input.close();
    return vectors_added;
}

// Helper macro to dispatch to the appropriate loader based on data type
#define DISPATCH_LOADER(LOADER_FUNC, data_type, ...) \
    switch (data_type) { \
        case VecSimType_FLOAT32: \
            return LOADER_FUNC<float>(__VA_ARGS__); \
        case VecSimType_FLOAT64: \
            return LOADER_FUNC<double>(__VA_ARGS__); \
        case VecSimType_BFLOAT16: \
            return LOADER_FUNC<bfloat16>(__VA_ARGS__); \
        case VecSimType_FLOAT16: \
            return LOADER_FUNC<float16>(__VA_ARGS__); \
        case VecSimType_INT8: \
            return LOADER_FUNC<int8_t>(__VA_ARGS__); \
        case VecSimType_UINT8: \
            return LOADER_FUNC<uint8_t>(__VA_ARGS__); \
        default: \
            throw std::runtime_error("Unsupported data type"); \
    }

// Helper to load vectors based on file type and data type
size_t loadVectors(FileType file_type, VecSimType data_type, const std::string &filename,
                   VecSimIndex *index, size_t dim, size_t num_vectors, size_t report_interval) {
    if (file_type == FileType::FBIN) {
        DISPATCH_LOADER(loadVectorsFromFbin, data_type, filename, index, dim, num_vectors, report_interval);
    } else {
        DISPATCH_LOADER(loadVectorsFromRaw, data_type, filename, index, dim, num_vectors, report_interval);
    }
}

// Helper template to get the distance type for a given data type
template<typename T> struct DistType { using type = float; };
template<> struct DistType<float> { using type = float; };
template<> struct DistType<double> { using type = double; };

// Helper to save index based on type
template<typename DataT>
void saveIndexTyped(VecSimIndex *index, const std::string &output_file) {
    using DistT = typename DistType<DataT>::type;
    auto *hnsw = dynamic_cast<HNSWDiskIndex<DataT, DistT> *>(index);
    hnsw->saveIndex(output_file);
}

void saveIndexByType(VecSimIndex *index, const std::string &output_file) {
    VecSimType type = VecSimIndex_BasicInfo(index).type;

    switch (type) {
        case VecSimType_FLOAT32:
            saveIndexTyped<float>(index, output_file);
            break;
        case VecSimType_FLOAT64:
            saveIndexTyped<double>(index, output_file);
            break;
        case VecSimType_BFLOAT16:
            saveIndexTyped<bfloat16>(index, output_file);
            break;
        case VecSimType_FLOAT16:
            saveIndexTyped<float16>(index, output_file);
            break;
        case VecSimType_INT8:
            saveIndexTyped<int8_t>(index, output_file);
            break;
        case VecSimType_UINT8:
            saveIndexTyped<uint8_t>(index, output_file);
            break;
        default:
            throw std::runtime_error("Invalid index data type");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_name> <dim> <metric> <type> [M] [efConstruction] [threads]\n";
        std::cerr << "\nArguments:\n";
        std::cerr << "  input_file      - Binary file (.raw or .fbin)\n";
        std::cerr << "  output_name     - Base name for output files\n";
        std::cerr << "  dim             - Vector dimension (required for .raw, use 0 for .fbin auto-detect)\n";
        std::cerr << "  metric          - Distance metric: L2, IP, or Cosine\n";
        std::cerr << "  type            - Data type: FLOAT32, FLOAT64, BFLOAT16, FLOAT16, INT8, UINT8\n";
        std::cerr << "  M               - HNSW M parameter (default: 64)\n";
        std::cerr << "  efConstruction  - HNSW efConstruction parameter (default: 512)\n";
        std::cerr << "  threads         - Number of threads for parallel indexing (default: 4, use 0 for single-threaded)\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_name = argv[2];
    size_t dim = std::stoull(argv[3]);
    VecSimMetric metric = parseMetric(argv[4]);
    VecSimType type = parseType(argv[5]);
    size_t M = (argc > 6) ? std::stoull(argv[6]) : 64;
    size_t efConstruction = (argc > 7) ? std::stoull(argv[7]) : 512;
    size_t num_threads = (argc > 8) ? std::stoull(argv[8]) : 4;

    // Check if input file exists
    if (!std::filesystem::exists(input_file)) {
        std::cerr << "Error: Input file does not exist: " << input_file << "\n";
        return 1;
    }

    // Detect file type
    FileType file_type = detectFileType(input_file);
    std::string file_type_str = (file_type == FileType::FBIN) ? "fbin" : "raw";

    // Get file info based on type
    size_t num_vectors = 0;
    if (file_type == FileType::FBIN) {
        try {
            FileInfo info = getFbinInfo(input_file, type);
            num_vectors = info.num_vectors;

            if (dim == 0) {
                // Auto-detect dimension from fbin file
                dim = info.dim;
                std::cout << "Auto-detected dimension from fbin file: " << dim << "\n";
            } else if (dim != info.dim) {
                std::cerr << "Error: Specified dimension (" << dim
                          << ") does not match fbin file dimension (" << info.dim << ")\n";
                return 1;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error reading fbin file: " << e.what() << "\n";
            return 1;
        }
    } else {
        // RAW file
        if (dim == 0) {
            std::cerr << "Error: Dimension must be specified for .raw files\n";
            return 1;
        }
        size_t file_size = std::filesystem::file_size(input_file);
        size_t type_size = getTypeSize(type);
        size_t vector_size_bytes = dim * type_size;
        num_vectors = file_size / vector_size_bytes;

        if (file_size % vector_size_bytes != 0) {
            std::cerr << "Warning: File size (" << file_size << " bytes) is not a multiple of vector size ("
                      << vector_size_bytes << " bytes). Some data may be truncated.\n";
        }
    }

    std::cout << "=== HNSW Disk Index Serializer ===\n";
    std::cout << "Input file:     " << input_file << "\n";
    std::cout << "File type:      " << file_type_str << "\n";
    std::cout << "Output name:    " << output_name << "\n";
    std::cout << "Dimension:      " << dim << "\n";
    std::cout << "Metric:         " << argv[4] << "\n";
    std::cout << "Type:           " << getTypeName(type) << "\n";
    std::cout << "M:              " << M << "\n";
    std::cout << "efConstruction: " << efConstruction << "\n";
    std::cout << "Threads:        " << (num_threads > 0 ? std::to_string(num_threads) : "single-threaded") << "\n";
    std::cout << "Number of vectors: " << num_vectors << "\n";
    std::cout << "==================================\n\n";

    // Create temporary RocksDB directory
    std::string rocksdb_path = "/tmp/hnsw_disk_serializer_" + std::to_string(getpid());
    std::filesystem::create_directories(rocksdb_path);
    std::cout << "Creating temporary RocksDB at: " << rocksdb_path << "\n";

    // Create HNSW disk index parameters
    HNSWDiskParams params;
    params.type = type;
    params.dim = dim;
    params.metric = metric;
    params.multi = false;
    params.initialCapacity = 0;
    params.blockSize = DEFAULT_BLOCK_SIZE;
    params.M = M;
    params.efConstruction = efConstruction;
    params.efRuntime = 10;  // Default, can be changed at load time
    params.epsilon = HNSW_DEFAULT_EPSILON;
    params.dbPath = rocksdb_path.c_str();

    VecSimParams vecsimParams;
    vecsimParams.algo = VecSimAlgo_HNSWLIB_DISK;
    vecsimParams.algoParams.hnswDiskParams = params;

    std::cout << "Creating HNSW disk index...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    VecSimIndex *index = VecSimIndex_New(&vecsimParams);
    if (!index) {
        std::cerr << "Error: Failed to create index\n";
        std::filesystem::remove_all(rocksdb_path);
        return 1;
    }

    // Set up multi-threaded processing if requested
    std::unique_ptr<tieredIndexMock> mock_thread_pool;
    if (num_threads > 0) {
        mock_thread_pool = std::make_unique<tieredIndexMock>();
        mock_thread_pool->ctx->index_strong_ref.reset(index, [](VecSimIndex*) {
            // Custom deleter that does nothing - we manage index lifetime separately
        });
        mock_thread_pool->thread_pool_size = num_threads;
        mock_thread_pool->init_threads();

        // Configure the disk index to use the job queue
        if (type == VecSimType_FLOAT32) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<float, float> *>(index);
            if (disk_index) {
                disk_index->setJobQueue(&mock_thread_pool->jobQ, mock_thread_pool->ctx,
                                        tieredIndexMock::submit_callback);
            }
        } else if (type == VecSimType_FLOAT64) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<double, double> *>(index);
            if (disk_index) {
                disk_index->setJobQueue(&mock_thread_pool->jobQ, mock_thread_pool->ctx,
                                        tieredIndexMock::submit_callback);
            }
        } else if (type == VecSimType_BFLOAT16) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<bfloat16, float> *>(index);
            if (disk_index) {
                disk_index->setJobQueue(&mock_thread_pool->jobQ, mock_thread_pool->ctx,
                                        tieredIndexMock::submit_callback);
            }
        } else if (type == VecSimType_FLOAT16) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<float16, float> *>(index);
            if (disk_index) {
                disk_index->setJobQueue(&mock_thread_pool->jobQ, mock_thread_pool->ctx,
                                        tieredIndexMock::submit_callback);
            }
        } else if (type == VecSimType_INT8) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<int8_t, float> *>(index);
            if (disk_index) {
                disk_index->setJobQueue(&mock_thread_pool->jobQ, mock_thread_pool->ctx,
                                        tieredIndexMock::submit_callback);
            }
        } else if (type == VecSimType_UINT8) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<uint8_t, float> *>(index);
            if (disk_index) {
                disk_index->setJobQueue(&mock_thread_pool->jobQ, mock_thread_pool->ctx,
                                        tieredIndexMock::submit_callback);
            }
        }
        std::cout << "Multi-threaded indexing enabled with " << num_threads << " threads\n";
    }

    std::cout << "Index created successfully\n";
    std::cout << "Loading vectors from file...\n";

    // Read and insert vectors using appropriate loader
    size_t report_interval = std::max(num_vectors / 100, size_t(1)); // Report every 1%
    std::cout << "Report interval: " << report_interval << " vectors\n";
    size_t vectors_added = 0;

    try {
        vectors_added = loadVectors(file_type, type, input_file, index, dim, num_vectors, report_interval);
    } catch (const std::exception &e) {
        std::cerr << "\nError loading vectors: " << e.what() << "\n";
        VecSimIndex_Free(index);
        std::filesystem::remove_all(rocksdb_path);
        return 1;
    }

    std::cout << "\n";

    if (vectors_added != num_vectors) {
        std::cerr << "Warning: Expected " << num_vectors << " vectors but added " << vectors_added << "\n";
    }

    // Wait for all background jobs to complete if using multi-threaded indexing
    if (mock_thread_pool) {
        std::cout << "Waiting for background indexing jobs to complete...\n";
        mock_thread_pool->thread_pool_wait();

        // Flush any remaining pending vectors and wait for those jobs too
        if (type == VecSimType_FLOAT32) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<float, float> *>(index);
            if (disk_index) disk_index->flushBatch();
        } else if (type == VecSimType_FLOAT64) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<double, double> *>(index);
            if (disk_index) disk_index->flushBatch();
        } else if (type == VecSimType_BFLOAT16) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<bfloat16, float> *>(index);
            if (disk_index) disk_index->flushBatch();
        } else if (type == VecSimType_FLOAT16) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<float16, float> *>(index);
            if (disk_index) disk_index->flushBatch();
        } else if (type == VecSimType_INT8) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<int8_t, float> *>(index);
            if (disk_index) disk_index->flushBatch();
        } else if (type == VecSimType_UINT8) {
            auto *disk_index = dynamic_cast<HNSWDiskIndex<uint8_t, float> *>(index);
            if (disk_index) disk_index->flushBatch();
        }

        // Wait again for the flush batch jobs to complete
        mock_thread_pool->thread_pool_wait();
        std::cout << "All background jobs completed.\n";

        // Stop the thread pool before saving
        mock_thread_pool->thread_pool_join();
    }

    auto index_time = std::chrono::high_resolution_clock::now();
    auto indexing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(index_time - start_time).count();
    std::cout << "Indexing completed in " << indexing_duration << " ms\n";
    std::cout << "Index size: " << VecSimIndex_IndexSize(index) << " vectors\n\n";

    // Save the index
    std::cout << "Saving index to: " << output_name << ".hnsw_disk_v1\n";

    // Create output directory structure:
    // output_name/
    //   ├── index.hnsw_disk_v1
    //   └── rocksdb/
    //
    // Note: We pass the folder path to saveIndex, which will:
    // 1. Create the metadata file at folder_path/index.hnsw_disk_v1
    // 2. Create the checkpoint at folder_path/rocksdb/
    std::filesystem::create_directories(output_name);
    saveIndexByType(index, output_name);

    auto save_time = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_time - index_time).count();
    std::cout << "Saving completed in " << save_duration << " ms\n";

    // Get output file sizes
    std::string metadata_file = std::string(output_name) + "/index.hnsw_disk_v1";
    size_t metadata_size = std::filesystem::file_size(metadata_file);
    // Checkpoint directory is at output_name/rocksdb
    std::string checkpoint_dir = std::string(output_name) + "/rocksdb";
    size_t checkpoint_size = 0;
    if (std::filesystem::exists(checkpoint_dir)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(checkpoint_dir)) {
            if (entry.is_regular_file()) {
                checkpoint_size += entry.file_size();
            }
        }
    }

    std::cout << "\n=== Output Files ===\n";
    std::cout << "Output folder:  " << output_name << "/\n";
    std::cout << "  - index.hnsw_disk_v1 (" << metadata_size << " bytes)\n";
    std::cout << "  - rocksdb/ (" << checkpoint_size / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "Total size:     " << (metadata_size + checkpoint_size) / (1024.0 * 1024.0) << " MB\n";

    // Cleanup
    if (mock_thread_pool) {
        mock_thread_pool.reset();
    }
    VecSimIndex_Free(index);
    std::filesystem::remove_all(rocksdb_path);

    auto total_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_time - start_time).count();
    std::cout << "\n=== Summary ===\n";
    std::cout << "Total time: " << total_duration << " seconds\n";
    std::cout << "Serialization complete!\n";

    return 0;
}

