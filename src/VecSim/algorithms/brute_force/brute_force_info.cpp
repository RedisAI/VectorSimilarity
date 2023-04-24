/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "brute_force_info.h"
#include "VecSim/utils/vec_utils.h"
BruteForceInfo::BruteForceInfo(VecSimInfo *info) : VecSimInfo(*info) {}

VecSimInfoIterator *BruteForceInfo::getIterator() {
    VecSimInfoIterator *infoIterator = VecSimInfo::getIterator();
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->blockSize}}});
}
