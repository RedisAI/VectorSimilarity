#include "vec_sim_info.h"
#include "utils/vec_utils.h"

VecSimInfoIterator *VecSimIndexInfo::getIterator() {
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(7);

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimAlgo_ToString(this->algo)}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TYPE_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(this->type)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->dim}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::METRIC_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimMetric_ToString(this->metric)}}});
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
}
