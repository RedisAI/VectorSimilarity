/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <functional>

namespace MemoryUtils {

using alloc_deleter_t = std::function<void(void *)>;
using unique_blob = std::unique_ptr<void, alloc_deleter_t>;

} // namespace MemoryUtils
