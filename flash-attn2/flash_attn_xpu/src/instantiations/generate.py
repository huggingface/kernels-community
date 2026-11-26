#!/usr/bin/env python3
# Regenerates the explicit-instantiation translation units in this directory.
#
# Each file holds the instantiations of one (kernel, head dim, dtype[, mode]),
# so the build compiles them as independent, parallel jobs; kernel-builder
# needs them as real files listed in build.toml. Layout:
#
#   <arch>/<kernel>/hdim<HD>_[<mode>_]<dtype>[_contig].cpp
#
# <arch> is the device IP version: xe20 = Xe2 (BMG), [kernel.flash_attn_xpu];
# xe35 = Xe3P (CRI),
# [kernel.flash_attn_xpu_cri]. Run from anywhere; prints the build.toml lists.
#
#   python3 flash_attn_xpu/src/instantiations/generate.py

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DTYPES = {"fp16": "cute::half_t", "bf16": "cute::bfloat16_t"}
ARCHS = {"xe20": 20, "xe35": 35}

# FMHA forward / backward (fmha_fwd.cpp, fmha_bwd.cpp), Xe2 image only.
FMHA_HDIMS = (32, 64, 96, 128, 160, 192, 256, 512)

# Split-KV decode, chunk prefill and paged decode.
HDIMS = (64, 96, 128, 192, 256, 512)
# Contiguous-K/V chunk prefill (see chunk_prefill_hd in chunk_prefill_args.hpp).
CONTIG_HDIMS = {"xe20": (128,), "xe35": (64, 128, 256)}


def fwd_dispatch(hd, dt, policy, stages, flags):
    return (f"template void policy_dispatch_{dt}<\n"
            f"    {policy}_policy_head{hd},\n"
            f"    PipelineStages_{stages},\n"
            f"    {flags}>(\n"
            "    sycl::queue& queue,\n"
            "    const fmha_fwd_args_t& args);\n")


def fwd(mode):
    head = {
        "varlen": "Varlen mode: IsVarLen=1, dtype={dt}",
        "fix": "Fixed mode: IsVarLen=0, IsPaged=0, dtype={dt}",
        "kvcache_paged": "Non-varlen + paged: IsVarLen=0, IsPaged=1, dtype={dt}\n"
                         "// Used by mha_fwd_kvcache when block_table is provided.",
    }[mode]
    blocks = {
        "varlen": [
            ("Varlen prefill + non-paged", "prefill", "Prefill", "1, 0"),
            ("Varlen prefill + paged", "prefill", "Prefill", "1, 1"),
            ("Varlen decode + non-paged", "decode", "Decode", "1, 0"),
            ("Varlen decode + paged", "decode_paged", "Decode", "1, 1"),
        ],
        "fix": [
            ("Decode fixed mode", "decode", "Decode", "0, 0"),
            ("Prefill fixed mode", "prefill", "Prefill", "0, 0"),
        ],
        "kvcache_paged": [
            ("Prefill paged", "prefill", "Prefill", "0, 1"),
            ("Decode paged (smaller K-tile to fit page boundaries)",
             "decode_paged", "Decode", "0, 1"),
            ("Rotary kvcache variants keep rotary code out of non-rotary kernels.",
             "prefill", "Prefill", "0, 0, true"),
            (None, "decode", "Decode", "0, 0, true"),
            (None, "prefill", "Prefill", "0, 1, true"),
            (None, "decode_paged", "Decode", "0, 1, true"),
        ],
    }[mode]

    def body(hd, dt):
        out = '#include "../../../fmha_fwd_impl.hpp"\n\n'
        out += "// " + head.format(dt=dt) + "\n"
        for comment, policy, stages, flags in blocks:
            out += "\n"
            if comment:
                out += f"// {comment}\n"
            out += fwd_dispatch(hd, dt, policy, stages, flags)
        return out
    return body


def bwd(mode):
    varlen = int(mode == "varlen")
    title = {"varlen": "Varlen", "fix": "Fixed"}[mode]

    def body(hd, dt):
        out = ('#include "../../../fmha_bwd_impl.hpp"\n\n'
               f"// {title} mode backward for head_dim={hd}, dtype={dt}\n")
        for a in (0, 1):
            for b in (0, 1):
                out += (f"\ntemplate void bwd_policy_dispatch_{dt}"
                        f"<bwd_policy_head{hd}, {a}, {b}, {varlen}>(\n"
                        "    sycl::queue& queue, const fmha_bwd_args_t& args);\n")
        return out
    return body


def decode(header, macro):
    def body(arch, hd, dt):
        return (f'#include "../../../decode/{header}"\n\n'
                f"{macro}({arch}, {hd}, {dt}, {DTYPES[dt]})\n")
    return body


def chunk_prefill(paged):
    def body(arch, hd, dt):
        return ('#include "../../../prefill/chunk_prefill.hpp"\n\n'
                "template bool flash_xpu_prefill::chunk_prefill_impl<"
                f"{arch}, {hd}, {str(dt == 'bf16').lower()}, "
                f"{str(paged).lower()}>(\n"
                "    sycl::queue&, const flash_xpu_prefill::chunk_prefill_args_t&);\n")
    return body


def fmha_files():
    files = []
    for mode in ("varlen", "fix", "kvcache_paged"):
        for hd in FMHA_HDIMS:
            for dt in DTYPES:
                files.append((f"fwd/hdim{hd}_{mode}_{dt}.cpp", fwd(mode)(hd, dt)))
    for mode in ("varlen", "fix"):
        for hd in FMHA_HDIMS:
            for dt in DTYPES:
                files.append((f"bwd/hdim{hd}_{mode}_{dt}.cpp", bwd(mode)(hd, dt)))
    return files


KERNELS = {
    "splitkv_decode": decode("splitkv_launch.hpp", "FLASH_XPU_SPLITKV_INSTANTIATE"),
    "chunk_prefill": chunk_prefill(paged=True),
    "paged_decode": decode("paged_decode.hpp", "FLASH_XPU_PAGED_DECODE_INSTANTIATE"),
}


def kernel_files(arch_dir, arch):
    files = []
    for kernel, body in KERNELS.items():
        for hd in HDIMS:
            for dt in DTYPES:
                files.append((f"{kernel}/hdim{hd}_{dt}.cpp", body(arch, hd, dt)))
                if kernel == "chunk_prefill" and hd in CONTIG_HDIMS[arch_dir]:
                    files.append((f"{kernel}/hdim{hd}_{dt}_contig.cpp",
                                  chunk_prefill(paged=False)(arch, hd, dt)))
    return files


def main():
    for arch_dir, arch in ARCHS.items():
        files = kernel_files(arch_dir, arch)
        if arch_dir == "xe20":
            files = fmha_files() + files
        print(f"# {arch_dir}: {len(files)} files")
        for name, text in files:
            path = ROOT / arch_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f'    "{path.relative_to(ROOT.parents[2]).as_posix()}",')


if __name__ == "__main__":
    main()
