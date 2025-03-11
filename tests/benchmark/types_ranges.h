#pragma once
#include <array>
#include "bm_definitions.h"

#define DEFAULT_RANGE_RADII                                                                        \
    { 20, 35, 50 }
#define DEFAULT_RANGE_EPSILONS                                                                     \
    { 1, 10, 11 }

// This template struct methods returns the default values for radii and epsilons
// To specify different values for a certain type, use template specialization
template <typename t>
struct benchmark_range {
    static std::array<unsigned int, 3> get_radii() { return DEFAULT_RANGE_RADII; }
    static std::array<unsigned int, 3> get_epsilons() { return DEFAULT_RANGE_EPSILONS; }
};

// Larger Range query values are required for int8 wikipedia dataset.
// Default values gives 0 results
#define INT8_RANGE_RADII                                                                           \
    { 50, 65, 80 }

template <>
struct benchmark_range<int8_index_t> {
    static std::array<unsigned int, 3> get_radii() { return INT8_RANGE_RADII; }
    static std::array<unsigned int, 3> get_epsilons() { return DEFAULT_RANGE_EPSILONS; }
};
