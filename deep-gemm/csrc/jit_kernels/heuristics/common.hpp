#pragma once

#include <unordered_set>
#include <deep_gemm/common/types.cuh>

#include "config.hpp"
#include "runtime.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // Choose the best layout
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
        const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
        if (ArchSpec::compare(candidate_info, layout_info))
            layout = layout_candidates[i], layout_info = candidate_info;
    }

    // Infer other configs
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
                                     static_cast<int>(desc.gemm_type), static_cast<int>(desc.kernel_type),
                                     desc.m, desc.n, desc.k, desc.num_groups,
                                     static_cast<int>(desc.a_dtype), static_cast<int>(desc.b_dtype),
                                     static_cast<int>(desc.cd_dtype), static_cast<int>(desc.major_a),
                                     static_cast<int>(desc.major_b), static_cast<int>(desc.with_accumulation),
                                     desc.num_sms, desc.tc_util, desc.compiled_dims,
                                     desc.expected_m, desc.expected_n, desc.expected_k,
                                     desc.expected_num_groups);

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << desc << ": " << gemm_config << ", " << layout_info << std::endl;
            printed.insert(key);
        }
    }
    return gemm_config;
}

} // namespace deep_gemm
