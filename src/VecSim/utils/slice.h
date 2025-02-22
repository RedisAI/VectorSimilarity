#pragma once

#include "rocksdb/slice.h"

template <typename T>
static inline rocksdb::Slice as_slice(T &t) {
    return rocksdb::Slice(reinterpret_cast<const char *>(&t), sizeof(t));
}

template <typename T>
static inline rocksdb::Slice as_slice(T t[], size_t size) {
    return rocksdb::Slice(reinterpret_cast<const char *>(t), sizeof(T) * size);
}
