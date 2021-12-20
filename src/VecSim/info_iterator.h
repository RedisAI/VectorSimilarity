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

typedef enum { STR, INT64, FLOAT64 } VecSim_InfoFieldType;

/**
 * @brief A struct to hold field information. This struct contains three members:
 *  fieldType - Enum describing the contenet of the value.
 *  fieldName - Field name.
 *  fieldValue - A union of string/integer/float values.
 */
typedef struct {
    VecSim_InfoFieldType fieldType; // Field type (in {STR, INT64, FLOAT64})
    const char *fieldName;          // Field name.
    union {
        double floatingPointValue; // Floating point value. Signed 64 bits float.
        int64_t integerValue;      // Integer value. Signed 64 bits integer.
        const char *stringValue;   // String value.
    } fieldValue;
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
