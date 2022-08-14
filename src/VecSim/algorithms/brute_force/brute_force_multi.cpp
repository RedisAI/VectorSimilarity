#include "brute_force_multi.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"

BruteForceIndex_Multi::BruteForceIndex_Multi(const BFParams *params,
                                               std::shared_ptr<VecSimAllocator> allocator)
    : BruteForceIndex(params, allocator), labelToIdsLookup(allocator) {}

BruteForceIndex_Multi::~BruteForceIndex_Multi() {}

int BruteForceIndex_Multi::addVector(const void *vector_data, size_t label) {

    float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, vector_data, this->dim * sizeof(float));
        float_vector_normalize(normalized_data, this->dim);
        vector_data = normalized_data;
    }

    // Give the vector new id and increase count.
    idType id = count++;

    auto optionalIDs = this->labelToIdsLookup.find(label);

    if (optionalIDs == this->labelToIdsLookup.end()) {
        this->labelToIdsLookup.emplace(label, id);
    } else {
        optionalIDs->second.push_back(id);
    }

    int res = insertVector(vector_data, id);

    // add label to idToLabelMapping
    setVectorLabel(id, label);

    // add id to label:id map
    addIdToLabel(label, id);

    return res;
}

int BruteForceIndex_Multi::deleteVector(size_t label) {

    // Find the id to delete.
    auto deleted_label_id_pair = this->labelToIdLookup.find(label);
    if (deleted_label_id_pair == this->labelToIdLookup.end()) {
        // Nothing to delete.
        return true;
    }

    // Get deleted vector id.
    idType id_to_delete = deleted_label_id_pair->second;

    // Remove the pair of the deleted vector.
    labelToIdLookup.erase(label);

    return removeVector(id_to_delete);
}

double BruteForceIndex_Multi::getDistanceFrom(size_t label, const void *vector_data) {

    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = optionalId->second;

    // Get the vectorBlock and the relative index of the required id.
    VectorBlock *req_vectorBlock = getVectorVectorBlock(id);
    size_t req_rel_idx = getVectorRelativeIndex(id);

    return this->dist_func(req_vectorBlock->getVector(req_rel_idx), vector_data, &this->dim);
}

// inline definitions

void BruteForceIndex_Multi::addIdToLabel(labelType label, idType id) {
    labelToIdLookup.emplace(label, id);
}

void BruteForceIndex_Multi::replaceIdOfLabel(labelType label, idType new_id, idType old_id) {
    labelToIdLookup.at(label) = new_id;
}
