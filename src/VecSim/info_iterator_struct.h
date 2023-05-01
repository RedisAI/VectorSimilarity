/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "info_iterator.h"
#include "VecSim/utils/arr_cpp.h"

struct VecSimInfoIterator {
private:
    VecSim_InfoField *fields;
    size_t currentIndex;

public:
    VecSimInfoIterator(size_t len) : fields(array_new<VecSim_InfoField>(len)), currentIndex(0) {}

    inline void addInfoField(VecSim_InfoField infoField) {
        this->fields = array_append(this->fields, infoField);
    }

    inline bool hasNext() { return this->currentIndex < array_len(this->fields); }

    inline VecSim_InfoField *next() { return this->fields + (this->currentIndex++); }

    inline size_t numberOfFields() { return array_len(this->fields); }

    virtual ~VecSimInfoIterator() {
        for (size_t i = 0; i < array_len(this->fields); i++) {
            if (this->fields[i].fieldType == INFOFIELD_ITERATOR) {
                delete this->fields[i].fieldValue.iteratorValue;
            }
        }
        array_free(this->fields);
    }
};
