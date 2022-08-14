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
    int res = insertVector(vector_data, id);

    // add label to idToLabelMapping
    setVectorLabel(id, label);

    // add id to label:id map
    addIdToLabel(label, id);

    return res;
}

int BruteForceIndex_Multi::deleteVector(size_t label) {

    // Find the id to delete.
    auto deleted_label_ids_pair = this->labelToIdsLookup.find(label);
    if (deleted_label_ids_pair == this->labelToIdsLookup.end()) {
        // Nothing to delete.
        return true;
    }

    int ret = true;

    // Deletes all vectors under the given label.
    for (auto id_to_delete : deleted_label_ids_pair->second) {
        ret = (removeVector(id_to_delete) && ret);
    }

    // Remove the pair of the deleted vector.
    labelToIdsLookup.erase(label);
    return ret;
}

double BruteForceIndex_Multi::getDistanceFrom(size_t label, const void *vector_data) {

    auto IDs = this->labelToIdsLookup.find(label);
    if (IDs == this->labelToIdsLookup.end()) {
        return INVALID_SCORE;
    }

    float dist = std::numeric_limits<float>::infinity();
    for (auto id : IDs->second) {
        VectorBlock *req_vectorBlock = getVectorVectorBlock(id);
        size_t req_rel_idx = getVectorRelativeIndex(id);
        float d = this->dist_func(req_vectorBlock->getVector(req_rel_idx), vector_data, &this->dim);
        dist = (dist < d) ? dist : d;
    }

    return dist;
}

// inline definitions

void BruteForceIndex_Multi::addIdToLabel(labelType label, idType id) {
    auto labelKey = labelToIdsLookup.find(label);
    if (labelKey != labelToIdsLookup.end()) {
        labelKey->second.push_back(id);
    } else {
        labelToIdsLookup.emplace(
            label, vecsim_stl::vector<idType>{std::initializer_list<idType>{id}, this->allocator});
    }
}

void BruteForceIndex_Multi::replaceIdOfLabel(labelType label, idType new_id, idType old_id) {
    auto &labelKey = labelToIdsLookup.at(label);
    for (size_t i = 0; i < labelKey.size(); i++) {
        if (labelKey[i] == old_id) {
            labelKey[i] = new_id;
            return;
        }
    }
}
