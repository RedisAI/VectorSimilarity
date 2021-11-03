#pragma once
#include <sys/param.h>

template <typename T>
struct array_hdr_t {
    size_t len;
    size_t cap;
    T buf[];
};

typedef void *array_t;

template <typename T>
array_hdr_t<T> *array_hdr(T *arr) {
    return (array_hdr_t<T> *)(((char *)arr) - sizeof(array_hdr_t<T>));
}

template <typename T>
T *array_new_sz(int32_t cap, size_t len) {
    auto *hdr = (array_hdr_t<T> *)malloc(sizeof(array_hdr_t<T>) + cap * sizeof(T));
    hdr->cap = cap;
    hdr->len = len;
    return hdr->buf;
}

template <typename T>
T *array_new(size_t cap) {
    return array_new_sz<T>(cap, 0);
}

template <typename T>
T *array_new_len(size_t cap, size_t len) {
    return array_new_sz<T>(cap, len);
}

template <typename T>
T *array_ensure_cap(T *arr, size_t cap) {
    array_hdr_t<T> *hdr = array_hdr(arr);
    if (cap > hdr->cap) {
        hdr->cap = MAX(hdr->cap * 2, cap);
        hdr = (array_hdr_t<T> *)realloc(hdr, sizeof(array_hdr_t<T>) + hdr->cap * sizeof(T));
    }
    return hdr->buf;
}

template <typename T>
T *array_grow(T *arr, size_t cap) {
    return array_ensure_cap(arr, ++array_hdr(arr)->len);
}

template <typename T>
T *array_append(T *arr, T val) {
    arr = array_grow(arr, array_hdr(arr)->len);
    arr[array_hdr(arr)->len - 1] = val;
    return arr;
}

template <typename T>
size_t array_len(T *arr) {
    return arr ? array_hdr(arr)->len : 0;
}

template <typename T>
void array_free(T *arr) {
    array_hdr_t<T> *arr_hdr = array_hdr(arr);
    free(arr_hdr);
}
