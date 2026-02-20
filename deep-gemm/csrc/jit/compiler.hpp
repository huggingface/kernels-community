#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <cstring>
#include <string>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"

namespace deep_gemm {

// Lazy-load NVRTC to avoid link-time dependency on libnvrtc.so.
// kernel-builder doesn't support linking extra CUDA libs yet, so we dlopen
// at runtime — same pattern as the CUDA driver API in jit/handle.hpp.
static void* get_nvrtc_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libnvrtc.so", RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr)
            handle = dlopen("libnvrtc.so.12", RTLD_LAZY | RTLD_LOCAL);
        DG_HOST_ASSERT(handle != nullptr and "Failed to load NVRTC library");
    }
    return handle;
}

#define DECL_LAZY_NVRTC_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_nvrtc_handle(), #name)); \
        DG_HOST_ASSERT(func != nullptr and "Failed to load NVRTC function"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

DECL_LAZY_NVRTC_FUNCTION(nvrtcVersion);
DECL_LAZY_NVRTC_FUNCTION(nvrtcCreateProgram);
DECL_LAZY_NVRTC_FUNCTION(nvrtcCompileProgram);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetProgramLogSize);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetProgramLog);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetPTXSize);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetPTX);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetCUBINSize);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetCUBIN);
DECL_LAZY_NVRTC_FUNCTION(nvrtcDestroyProgram);
DECL_LAZY_NVRTC_FUNCTION(nvrtcGetErrorString);

// Redirect nvrtc calls to lazy-loaded versions so NVRTCCompiler is unchanged
#define nvrtcVersion lazy_nvrtcVersion
#define nvrtcCreateProgram lazy_nvrtcCreateProgram
#define nvrtcCompileProgram lazy_nvrtcCompileProgram
#define nvrtcGetProgramLogSize lazy_nvrtcGetProgramLogSize
#define nvrtcGetProgramLog lazy_nvrtcGetProgramLog
#define nvrtcGetPTXSize lazy_nvrtcGetPTXSize
#define nvrtcGetPTX lazy_nvrtcGetPTX
#define nvrtcGetCUBINSize lazy_nvrtcGetCUBINSize
#define nvrtcGetCUBIN lazy_nvrtcGetCUBIN
#define nvrtcDestroyProgram lazy_nvrtcDestroyProgram
#define nvrtcGetErrorString lazy_nvrtcGetErrorString

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path cuda_home;
    static std::string library_version;
    static std::filesystem::path cuobjdump_path;

    static std::string get_library_version() {
        const auto dg_include = library_include_path / "deep_gemm";
        if (not std::filesystem::exists(dg_include)) {
            // Fallback: hash the root path itself
            std::string fallback(library_root_path.string());
            return get_hex_digest(std::vector<char>(fallback.begin(), fallback.end()));
        }
        std::vector<char> buffer;
        for (const auto& f: collect_files(dg_include)) {
            std::ifstream in(f, std::ios::binary);
            DG_HOST_ASSERT(in.is_open());
            buffer.insert(buffer.end(),
                          std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>());
        }
        return get_hex_digest(buffer);
    }

    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::library_version = get_library_version();
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not cuda_home.empty());
        DG_HOST_ASSERT(not library_version.empty());
        DG_HOST_ASSERT(not cuobjdump_path.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto& env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_JIT_CPP_STANDARD", 20));
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))
            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    std::filesystem::path get_tmp_file_path() const {
        return make_tmp_dir() / get_uuid();
    }

    void put(const std::filesystem::path& path, const std::string& data) const {
        const auto tmp_file_path = get_tmp_file_path();

        // Write into the temporary file
        std::ofstream out(tmp_file_path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // Atomically replace
        std::filesystem::rename(tmp_file_path, path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}$${}", name, library_version, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Create the kernel directory
        make_dirs(dir_path);

        // Compile into a temporary CUBIN
        const auto tmp_cubin_path = get_tmp_file_path();
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_PTX")) {
            // Dump PTX if needed
            const auto tmp_ptx_path = get_tmp_file_path();
            compile(code, dir_path, tmp_cubin_path, tmp_ptx_path);

            // Replace into the cache directory
            std::filesystem::rename(tmp_ptx_path, dir_path / "kernel.ptx");
        } else {
            compile(code, dir_path, tmp_cubin_path);
        }

        // Replace into the cache directory
        const auto cubin_path = dir_path / "kernel.cubin";
        std::filesystem::rename(tmp_cubin_path, cubin_path);

        // Disassemble if needed
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_SASS")) {
            // Dump into a temporary SASS
            const auto tmp_sass_path = get_tmp_file_path();
            disassemble(cubin_path, tmp_sass_path);

            // Replace into the current directory
            std::filesystem::rename(tmp_sass_path, dir_path / "kernel.sass");
        }

        // Put into the runtime cache
        const auto runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {
        // Disassemble the CUBIN file to SASS
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            fprintf(stderr, "Running cuobjdump command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            fprintf(stderr, "cuobjdump failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "cuobjdump failed");
        }
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_version);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // Parse "release X.Y" without std::regex
        int major = 0, minor = 0;
        const char* release_pos = std::strstr(output.c_str(), "release ");
        DG_HOST_ASSERT(release_pos != nullptr and "Could not find 'release' in nvcc --version output");
        std::sscanf(release_pos + 8, "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            fprintf(stderr, "Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // Override the compiler signature
        nvcc_path = cuda_home / "bin" / "nvcc";
        if (const auto& env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        const auto& arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        // DG_CUTLASS_INCLUDE is set by Python _find_cutlass_include() before ops.init()
        const auto& cutlass_include = get_env<std::string>("DG_CUTLASS_INCLUDE");
        std::string cutlass_flag = cutlass_include.empty() ? "" : fmt::format(" -I{}", cutlass_include);
        flags = fmt::format("{} -I{}{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), cutlass_flag, arch);

        // print flags if ENV is set
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_FLAGS", 0))
            fprintf(stderr, "NVCC compiler flags: %s\n", flags.c_str());
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        const auto& command = fmt::format("{} {} -cubin -o {} {}", nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            fprintf(stderr, "Running NVCC command: %s\n", command.c_str());
        const auto& [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            fprintf(stderr, "NVCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Compile to PTX if needed
        if (ptx_path.has_value()) {
            const auto ptx_command = fmt::format("{} {} -ptx -o {} {}", nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
                fprintf(stderr, "Running NVCC PTX command: %s\n", ptx_command.c_str());
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            if (ptx_return_code != 0) {
                fprintf(stderr, "NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                DG_HOST_ASSERT(false and "NVCC PTX compilation failed");
            }
        }

        // Check local memory usage (without std::regex — avoids ABI issues)
        if (get_env("DG_JIT_PTXAS_CHECK", 0))
            DG_HOST_ASSERT(output.find("Local memory used") == std::string::npos);

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            fprintf(stderr, "%s", output.c_str());
    }
};

class NVRTCCompiler final: public Compiler {
public:
    NVRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

        // Build include directories list
        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());
        // DG_CUTLASS_INCLUDE is set by Python _find_cutlass_include() before ops.init()
        if (const auto& cutlass_include = get_env<std::string>("DG_CUTLASS_INCLUDE"); not cutlass_include.empty())
            include_dirs += fmt::format("-I{} ", cutlass_include);

        // Add PCH support for version 12.8 and above
        // NOTES: PCH is vital for compilation speed
        std::string pch_flags;
        if (major > 12 or minor >= 8) {
            pch_flags = "--pch ";
            if (get_env<int>("DG_JIT_DEBUG", 0))
                pch_flags += "--pch-verbose=true ";
        }

        // Override the compiler flags
        // Only NVRTC >= 12.9 supports arch-specific family suffix
        const auto& arch = device_runtime->get_arch(false, major > 12 or minor >= 9);
        flags = fmt::format("{} {}--gpu-architecture=sm_{} -default-device {} --device-int128",
                            flags, include_dirs, arch, pch_flags);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Split flags by whitespace (without std::istringstream — avoids ABI issues)
        std::vector<std::string> options;
        {
            size_t i = 0;
            while (i < flags.size()) {
                while (i < flags.size() && (flags[i] == ' ' || flags[i] == '\t')) ++i;
                if (i >= flags.size()) break;
                size_t start = i;
                while (i < flags.size() && flags[i] != ' ' && flags[i] != '\t') ++i;
                options.push_back(flags.substr(start, i - start));
            }
        }

        // Convert to C-style string array for NVRTC
        std::vector<const char*> option_cstrs;
        for (const auto& opt: options)
            option_cstrs.push_back(opt.c_str());

        // Print compiler command if requested
        if (get_env<int>("DG_JIT_DEBUG", 0) or get_env<int>("DG_JIT_PRINT_COMPILER_COMMAND", 0)) {
            fprintf(stderr, "Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt: options)
                fprintf(stderr, "%s ", opt.c_str());
            fprintf(stderr, "\n");
        }

        // Create NVRTC program and compile
        nvrtcProgram program;
        DG_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result = nvrtcCompileProgram(program, static_cast<int>(option_cstrs.size()), option_cstrs.data());

        // Get and print compiler log
        size_t log_size;
        DG_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                fprintf(stderr, "NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        if (ptx_path.has_value()) {
            // Get PTX size and data if needed
            size_t ptx_size;
            DG_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
            std::string ptx_data(ptx_size, '\0');
            DG_NVRTC_CHECK(nvrtcGetPTX(program, ptx_data.data()));

            // Write into the file system
            put(ptx_path.value(), ptx_data);
        }

        // Get CUBIN size and data
        size_t cubin_size;
        DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write into the file system
        put(cubin_path, cubin_data);

        // Cleanup
        DG_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    }
    return std::make_shared<NVCCCompiler>();
});

} // namespace deep_gemm
