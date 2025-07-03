/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#define SVS_INVALID_META_DATA SIZE_MAX

typedef struct {
    bool valid_state;
    long memory_usage; // in bytes
    size_t index_size;
    size_t storage_size;
    size_t label_count;
    size_t capacity;
    size_t changes_count;
    bool is_compressed;
    bool is_multi;
} SVSIndexMetaData;
