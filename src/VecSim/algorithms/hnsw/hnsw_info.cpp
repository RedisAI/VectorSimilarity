/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "hnsw_info.h"
#include "VecSim/utils/vec_utils.h"

VecSimInfoIterator *HNSWInfo::getIterator() {
    VecSimInfoIterator *infoIterator = VecSimIndexInfo::getIterator();

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_M_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.M}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.efConstruction}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EF_RUNTIME_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.efRuntime}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_MAX_LEVEL,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.max_level}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_ENTRYPOINT,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.entrypoint}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EPSILON_STRING,
                         .fieldType = INFOFIELD_FLOAT64,
                         .fieldValue = {FieldValue{.floatingPointValue = info.hnswInfo.epsilon}}});

    return infoIterator;
}
