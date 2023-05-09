/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A struct to hold an index information for generic purposes. Each information field is of
 * the type VecSim_InfoFieldType. This struct exposes an iterator-like API to iterate over the
 * information fields.
 */
typedef struct VecSimInfoIterator VecSimInfoIterator;

typedef enum {
    INFOFIELD_STRING,
    INFOFIELD_INT64,
    INFOFIELD_UINT64,
    INFOFIELD_FLOAT64,
    INFOFIELD_ITERATOR
} VecSim_InfoFieldType;

typedef union {
    double floatingPointValue;         // Floating point value. 64 bits float.
    int64_t integerValue;              // Integer value. Signed 64 bits integer.
    u_int64_t uintegerValue;           // Unsigned value. Unsigned 64 buts integer.
    const char *stringValue;           // String value.
    VecSimInfoIterator *iteratorValue; // Iterator value.
} FieldValue;

/**
 * @brief A struct to hold field information. This struct contains three members:
 *  fieldType - Enum describing the content of the value.
 *  fieldName - Field name.
 *  fieldValue - A union of string/integer/float values.
 */
typedef struct {
    const char *fieldName;          // Field name.
    VecSim_InfoFieldType fieldType; // Field type (in {STR, INT64, FLOAT64})
    FieldValue fieldValue;
} VecSim_InfoField;

/**
 * @brief Returns the number of fields in the info iterator.
 *
 * @param infoIterator Given info iterator.
 * @return size_t Number of fields.
 */
size_t VecSimInfoIterator_NumberOfFields(VecSimInfoIterator *infoIterator);

/**
 * @brief Returns if the fields iterator is depleted.
 *
 * @param infoIterator Given info iterator.
 * @return true Iterator is not depleted.
 * @return false Otherwise.
 */
bool VecSimInfoIterator_HasNextField(VecSimInfoIterator *infoIterator);

/**
 * @brief Returns a pointer to the next info field.
 *
 * @param infoIterator Given info iterator.
 * @return VecSim_InfoField* A pointer to the next info field.
 */
VecSim_InfoField *VecSimInfoIterator_NextField(VecSimInfoIterator *infoIterator);

/**
 * @brief Free an info iterator.
 *
 * @param infoIterator Given info iterator.
 */
void VecSimInfoIterator_Free(VecSimInfoIterator *infoIterator);

#ifdef __cplusplus
}
#endif
