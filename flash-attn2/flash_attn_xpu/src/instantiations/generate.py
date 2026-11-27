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
# [kernel.flash_attn_xpu_cri]. Run from anywhere. The default mode writes the
# files, deletes obsolete ones and prints the build.toml lists; --check only
# verifies that the files and the build.toml lists match this script.
#
#   python3 flash_attn_xpu/src/instantiations/generate.py [--check]

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT.parents[2]
BUILD_TOML = KERNEL_DIR / "build.toml"
SECTIONS = {"xe20": "flash_attn_xpu", "xe35": "flash_attn_xpu_cri"}
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


def expected_files():
    out = {}
    for arch_dir, arch in ARCHS.items():
        files = kernel_files(arch_dir, arch)
        if arch_dir == "xe20":
            files = fmha_files() + files
        out[arch_dir] = {ROOT / arch_dir / name: text for name, text in files}
    return out


def rel(path):
    return path.relative_to(KERNEL_DIR).as_posix()


def obsolete_files(expected):
    wanted = {path for files in expected.values() for path in files}
    return sorted(p for p in ROOT.rglob("*.cpp") if p not in wanted)


def check(expected):
    errors = []
    for files in expected.values():
        for path, text in files.items():
            if not path.is_file():
                errors.append(f"missing file: {rel(path)}")
            elif path.read_text() != text:
                errors.append(f"out-of-date file: {rel(path)}")
    errors += [f"obsolete file: {rel(p)}" for p in obsolete_files(expected)]

    kernels = tomllib.loads(BUILD_TOML.read_text())["kernel"]
    prefix = rel(ROOT) + "/"
    for arch_dir, files in expected.items():
        section = f"[kernel.{SECTIONS[arch_dir]}]"
        listed = [s for s in kernels[SECTIONS[arch_dir]]["src"]
                  if s.startswith(prefix)]
        wanted = {rel(p) for p in files}
        errors += [f"build.toml {section} missing: {s}"
                   for s in sorted(wanted - set(listed))]
        errors += [f"build.toml {section} not generated: {s}"
                   for s in sorted(set(listed) - wanted)]
        errors += [f"build.toml {section} duplicate: {s}"
                   for s in sorted({s for s in listed if listed.count(s) > 1})]
    return errors


def generate(expected):
    for arch_dir, files in expected.items():
        print(f"# {arch_dir}: {len(files)} files")
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f'    "{rel(path)}",')
    for path in obsolete_files(expected):
        path.unlink()
        print(f"# removed obsolete {rel(path)}")
    for path in sorted(ROOT.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify generated files and build.toml, write nothing")
    args = parser.parse_args()

    expected = expected_files()
    if not args.check:
        generate(expected)
    errors = check(expected)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
