#pragma once

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"

namespace deep_gemm::runtime {

static void deep_gemm_set_num_sms(int64_t new_num_sms) {
    device_runtime->set_num_sms(static_cast<int>(new_num_sms));
}

static int64_t deep_gemm_get_num_sms() {
    return device_runtime->get_num_sms();
}

static void deep_gemm_set_tc_util(int64_t new_tc_util) {
    device_runtime->set_tc_util(static_cast<int>(new_tc_util));
}

static int64_t deep_gemm_get_tc_util() {
    return device_runtime->get_tc_util();
}

static void deep_gemm_init(const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
    Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
    KernelRuntime::prepare_init(cuda_home_path_by_python);
#endif
}

#ifdef DG_USE_PYBIND11
static void register_apis(pybind11::module_& m) {
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        device_runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return device_runtime->get_tc_util();
    });
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
#endif
    });
}
#endif

} // namespace deep_gemm::runtime
