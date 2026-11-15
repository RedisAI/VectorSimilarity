/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "pipnn.h"
#include "VecSim/index_factories/pipnn_factory.h"
#include "VecSim/vec_sim.h"

#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unistd.h>

namespace {
static_assert(sizeof(idType) == 4, "PiPNN format v1 requires 32-bit internal IDs");
constexpr char magic[] = "VSPIPE01";
constexpr uint64_t hashBasis = 14695981039346656037ULL;
constexpr uint64_t hashPrime = 1099511628211ULL;

size_t checkedProduct(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b)
        throw std::runtime_error("PiPNN file size overflow");
    return a * b;
}
size_t checkedSum(size_t a, size_t b) {
    if (a > SIZE_MAX - b)
        throw std::runtime_error("PiPNN file size overflow");
    return a + b;
}

// Integers and IEEE floating-point bit patterns use little-endian encoding.
class Writer {
    FILE *file;
    uint64_t hash = hashBasis;

public:
    explicit Writer(FILE *f) : file(f) {}
    void bytes(const void *data, size_t size, bool checksum = true) {
        auto p = static_cast<const unsigned char *>(data);
        if (std::fwrite(data, 1, size, file) != size)
            throw std::runtime_error("PiPNN write failed");
        if (checksum)
            for (size_t i = 0; i < size; ++i)
                hash = (hash ^ p[i]) * hashPrime;
    }
    void integer(uint64_t value, unsigned width = 8, bool checksum = true) {
        unsigned char bytes_[8];
        for (unsigned i = 0; i < width; ++i)
            bytes_[i] = value >> (8 * i);
        bytes(bytes_, width, checksum);
    }
    void finish() { integer(hash, 8, false); }
};

class Reader {
    std::ifstream file;
    uint64_t hash = hashBasis;
    size_t remaining_;

public:
    explicit Reader(const std::string &path) : file(path, std::ios::binary | std::ios::ate) {
        if (!file)
            throw std::runtime_error("PiPNN open failed");
        auto size = file.tellg();
        if (size < 0 || static_cast<uint64_t>(size) > SIZE_MAX)
            throw std::runtime_error("PiPNN invalid file length");
        remaining_ = static_cast<size_t>(size);
        file.seekg(0);
    }
    size_t remaining() const { return remaining_; }
    void bytes(void *data, size_t size, bool checksum = true) {
        if (size > remaining_)
            throw std::runtime_error("PiPNN truncated file");
        file.read(static_cast<char *>(data), size);
        if (!file)
            throw std::runtime_error("PiPNN read failed");
        remaining_ -= size;
        auto p = static_cast<const unsigned char *>(data);
        if (checksum)
            for (size_t i = 0; i < size; ++i)
                hash = (hash ^ p[i]) * hashPrime;
    }
    uint64_t integer(unsigned width = 8, bool checksum = true) {
        unsigned char data[8];
        bytes(data, width, checksum);
        uint64_t value = 0;
        for (unsigned i = 0; i < width; ++i)
            value |= uint64_t(data[i]) << (8 * i);
        return value;
    }
    size_t size() {
        auto value = integer();
        if (value > SIZE_MAX)
            throw std::runtime_error("PiPNN size is not representable");
        return static_cast<size_t>(value);
    }
    void finish() {
        if (integer(8, false) != hash || remaining_ != 0)
            throw std::runtime_error("PiPNN checksum or trailing data error");
    }
};
} // namespace

void PiPNNIndex::saveIndex(const std::string &location) const {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    if (location.empty())
        throw std::invalid_argument("PiPNN empty save path");
    std::string temporary = location + ".tmp.XXXXXX";
    int fd = mkstemp(temporary.data());
    if (fd < 0)
        throw std::runtime_error("PiPNN create temporary file failed");
    FILE *file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        unlink(temporary.c_str());
        throw std::runtime_error("PiPNN open temporary file failed");
    }
    try {
        Writer out(file);
        out.bytes(magic, 8);
        out.integer(1, 4);
        out.integer(params_.dim);
        out.integer(params_.blockSize);
        out.integer(params_.maxDegree);
        out.integer(params_.leafSize);
        out.integer(params_.fanout);
        out.integer(params_.hashBits);
        out.integer(params_.candidateLimit);
        out.integer(std::bit_cast<uint64_t>(params_.alpha));
        out.integer(params_.efRuntime);
        out.integer(params_.seed);
        out.integer(this->indexSize());
        out.integer(graphReady_ ? 1 : 0, 4);
        for (size_t i = 0; i < this->indexSize(); ++i) {
            out.integer(this->getVectorLabel(i));
            const float *data = this->getDataByInternalId(i);
            for (size_t d = 0; d < params_.dim; ++d)
                out.integer(std::bit_cast<uint32_t>(data[d]), 4);
        }
        if (graphReady_) {
            if (degrees_.size() != this->indexSize() ||
                graph_.size() != checkedProduct(this->indexSize(), params_.maxDegree))
                throw std::runtime_error("PiPNN invalid graph shape");
            for (auto degree : degrees_)
                out.integer(degree);
            for (auto edge : graph_)
                out.integer(edge, 4);
        }
        out.finish();
        if (fflush(file) || fsync(fileno(file)))
            throw std::runtime_error("PiPNN flush failed");
        int result = fclose(file);
        file = nullptr;
        if (result)
            throw std::runtime_error("PiPNN close failed");
        if (rename(temporary.c_str(), location.c_str()))
            throw std::runtime_error("PiPNN atomic rename failed");
    } catch (...) {
        if (file)
            fclose(file);
        unlink(temporary.c_str());
        throw;
    }
}

PiPNNIndex *PiPNNIndex::loadIndex(const std::string &location, void *logCtx) {
    Reader in(location);
    char signature[8];
    in.bytes(signature, sizeof(signature));
    if (std::memcmp(signature, magic, 8) || in.integer(4) != 1)
        throw std::runtime_error("PiPNN unsupported file format");
    PiPNNParams params{};
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.dim = in.size();
    params.blockSize = in.size();
    params.maxDegree = in.size();
    params.leafSize = in.size();
    params.fanout = in.size();
    params.hashBits = in.size();
    params.candidateLimit = in.size();
    params.alpha = std::bit_cast<double>(in.integer());
    params.efRuntime = in.size();
    params.seed = in.integer();
    const size_t count = in.size();
    const auto ready = in.integer(4);
    if (ready > 1 || count > std::numeric_limits<idType>::max())
        throw std::runtime_error("PiPNN invalid index size or state");
    params = PiPNNFactory::ResolveParams(params);
    const size_t graphSize = ready ? checkedProduct(count, params.maxDegree) : 0;
    size_t expected = checkedProduct(count, checkedSum(8, checkedProduct(params.dim, 4)));
    expected = checkedSum(expected, ready ? checkedProduct(count, 8) : 0);
    expected = checkedSum(expected, checkedProduct(graphSize, 4));
    expected = checkedSum(expected, 8);
    if (in.remaining() != expected)
        throw std::runtime_error("PiPNN invalid payload length");

    VecSimParams config{};
    config.algo = VecSimAlgo_PIPNN;
    config.algoParams.pipnnParams = params;
    config.logCtx = logCtx;
    std::unique_ptr<PiPNNIndex, decltype(&VecSimIndex_Free)> index(
        static_cast<PiPNNIndex *>(PiPNNFactory::NewIndex(&config)), VecSimIndex_Free);
    if (!index)
        throw std::runtime_error("PiPNN index creation failed");
    vecsim_stl::vector<float> data(index->getAllocator());
    data.resize(params.dim);
    for (size_t i = 0; i < count; ++i) {
        const auto label = in.integer();
        if (label > std::numeric_limits<labelType>::max())
            throw std::runtime_error("PiPNN label is not representable");
        for (size_t d = 0; d < params.dim; ++d) {
            data[d] = std::bit_cast<float>(static_cast<uint32_t>(in.integer(4)));
            if (!std::isfinite(data[d]))
                throw std::runtime_error("PiPNN nonfinite vector");
        }
        if (index->addVector(data.data(), static_cast<labelType>(label)) != 1)
            throw std::runtime_error("PiPNN duplicate label in file");
    }
    if (ready) {
        index->degrees_.resize(count);
        index->graph_.resize(graphSize);
        for (auto &degree : index->degrees_) {
            degree = in.size();
            if (degree > params.maxDegree || degree >= count)
                throw std::runtime_error("PiPNN invalid degree");
        }
        for (auto &edge : index->graph_) {
            edge = static_cast<idType>(in.integer(4));
        }
        for (size_t i = 0; i < count; ++i) {
            bool hasSuccessor = count <= 1;
            for (size_t j = 0; j < index->degrees_[i]; ++j) {
                const auto edge = index->graph_[i * params.maxDegree + j];
                if (edge >= count || edge == i)
                    throw std::runtime_error("PiPNN invalid edge");
                hasSuccessor |= edge == (i + 1) % count;
                for (size_t k = 0; k < j; ++k)
                    if (index->graph_[i * params.maxDegree + k] == edge)
                        throw std::runtime_error("PiPNN duplicate edge");
            }
            if (!hasSuccessor)
                throw std::runtime_error("PiPNN missing connectivity edge");
        }
    }
    in.finish();
    index->graphReady_ = ready != 0;
    return index.release();
}
