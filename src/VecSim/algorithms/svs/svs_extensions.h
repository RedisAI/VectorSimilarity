#pragma once
#include "VecSim/algorithms/svs/svs_utils.h"

#if HAVE_SVS_LVQ
#include SVS_LVQ_HEADER

namespace svs_details {
template <size_t Primary>
struct LVQSelector {
    using strategy = svs::quantization::lvq::Sequential;
};

template <>
struct LVQSelector<4> {
    using strategy = svs::quantization::lvq::Turbo<16, 8>;
};
} // namespace svs_details

template <typename DataType, size_t QuantBits, size_t ResidualBits>
struct SVSStorageTraits<DataType, QuantBits, ResidualBits, std::enable_if_t<(QuantBits > 0)>> {
    using allocator_type = svs_details::SVSAllocator<std::byte>;
    using blocked_type = svs::data::Blocked<svs::AllocatorHandle<std::byte>>;
    using strategy_type = typename svs_details::LVQSelector<QuantBits>::strategy;
    using index_storage_type =
        svs::quantization::lvq::LVQDataset<QuantBits, ResidualBits, svs::Dynamic, strategy_type,
                                           blocked_type>;

    template <svs::data::ImmutableMemoryDataset Dataset, svs::threads::ThreadPool Pool>
    static index_storage_type create_storage(const Dataset &data, size_t block_size, Pool &pool,
                                             std::shared_ptr<VecSimAllocator> allocator) {
        const auto dim = data.dimensions();
        auto svs_bs = svs_details::SVSBlockSize(block_size, element_size(dim));

        allocator_type data_allocator{std::move(allocator)};
        auto blocked_alloc = svs::make_blocked_allocator_handle({svs_bs}, data_allocator);

        return index_storage_type::compress(data, pool, 0, blocked_alloc);
    }

    static constexpr size_t element_size(size_t dims, size_t alignment = 0) {
        using primary_type = typename index_storage_type::primary_type;
        using layout_type = typename primary_type::helper_type;
        using layout_dims_type = svs::lib::MaybeStatic<index_storage_type::extent>;
        const auto layout_dims = layout_dims_type{dims};
        return primary_type::compute_data_dimensions(layout_type{layout_dims}, alignment);
    }

    static size_t storage_capacity(const index_storage_type &storage) {
        // LVQDataset does not provide a capacity method
        return storage.size();
    }
};
#else
#pragma message "SVS LVQ is not available"
#endif // HAVE_SVS_LVQ
