/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "tiered_index_info.h"
#include "VecSim/utils/vec_utils.h"

virtual VecSimInfoIterator *TieredIndexInfo::getIterator() {
    VecSimInfoIterator *iterator = VecSimInfoIterator_New(9);
        infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimAlgo_ToString(this->algo)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->isMulti}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->indexSize}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->indexLabelCount}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->memory}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimSearchMode_ToString(this->last_mode)}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::FRONTEND_INDEX_STRING,
        .fieldType = INFOFIELD_ITERATOR,
        .fieldValue = {FieldValue{.iteratorValue = this->frontendIndexInfo->getIterator()}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::BACKEND_INDEX_STRING,
        .fieldType = INFOFIELD_ITERATOR,
        .fieldValue = {FieldValue{.iteratorValue = this->backendIndexInfo->getIterator()}}});

    return infoIterator;

}