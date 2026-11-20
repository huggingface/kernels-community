#!/usr/bin/env python3
"""Check the device code in a flash-attn2 XPU build.

Usage: check_xpu_build.py <build-dir-or-.so>... [--verbose]

The kernels are only fast when the build carries ahead-of-time (AOT) native
code for every device, and when each kernel runs with the GRF (register file)
size it was tuned for. Neither shows up as a build or test failure: a
JIT-only library or a kernel at the wrong GRF size runs correctly, just
slower. This script reads the native device images (zebin) from the
libraries and fails when an image is missing or a kernel has an unexpected
GRF size.

The GRF sizes come from flash_attn_xpu/src/decode/grf_mode.hpp, from the
kernel properties at each launch site, and from `[general.xpu]
cri-grf-size` in build.toml.
"""

import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ELF_MAGIC = b"\x7fELF\x02\x01\x01"
EM_INTELGT = 205
SHT_NOTE = 7
SHT_NOBITS = 8
NT_INTELGT_PRODUCT_FAMILY = 1

PRODUCTS = {1271: "pvc", 1274: "bmg", 1380: "cri"}

# Kernel families, matched on the mangled name in this order.
FAMILIES = [
    ("bwd", "mhabwdDeviceNameBwd"),
    ("bwd_odo", "mhaodoDeviceNameBwd"),
    ("bwd_cvt", "mhacvtDeviceNameBwd"),
    ("append_kv", "splitkv_append_kv"),
    ("xe3", "xe3_xe35"),  # Xe35 paged decode + chunk prefill (CRI only)
    ("splitkv_xe35", "xe35"),
    ("splitkv_xe20", "xe20"),
    ("paged_decode_xe2", "paged3xe2"),
    ("fwd", "fmha"),
]

# Families every image of a device must contain.
REQUIRED = {
    "pvc": {"fwd", "bwd", "bwd_odo", "bwd_cvt", "splitkv_xe20", "append_kv"},
    "bmg": {"fwd", "bwd", "bwd_odo", "bwd_cvt", "splitkv_xe20", "append_kv"},
    "cri": {"fwd", "bwd", "bwd_odo", "bwd_cvt", "splitkv_xe20", "append_kv",
            "splitkv_xe35", "xe3"},
}
# Families only built for CRI.
CRI_ONLY = {"xe3", "splitkv_xe35"}
# Families built for every device: each device must have the same kernels.
COMMON = REQUIRED["pvc"]

# Head sizes whose backward kernel has no GRF property (512 on CRI).
BWD_LARGE_GRF_HEAD_DIMS = {160, 192, 512}


def head_dim(name):
    m = re.search(r"FABwdKernelIN7cutlass\d+\w+?_tELi(\d+)E", name)
    if m is None:
        raise ValueError(f"no head size in {name}")
    return int(m.group(1))


def expected_grf(device, family, name):
    """GRF size a kernel must have, or None if it is not checked."""
    if family == "bwd":
        if head_dim(name) in BWD_LARGE_GRF_HEAD_DIMS:
            return 512 if device == "cri" else None
        return 256
    if family in ("bwd_odo", "bwd_cvt"):
        if head_dim(name) <= 128:
            return 512 if device == "cri" else None
        return 256
    if family == "append_kv":
        return 512 if device == "cri" else None
    if family == "xe3":
        return 512
    if family in ("splitkv_xe35", "splitkv_xe20", "paged_decode_xe2", "fwd"):
        return 256
    return None


def elf_sections(b, base=0):
    shoff = struct.unpack_from("<Q", b, base + 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, base + 0x3A)
    hdrs = []
    for i in range(shnum):
        sh = base + shoff + i * shentsize
        name, typ = struct.unpack_from("<II", b, sh)
        off, size = struct.unpack_from("<QQ", b, sh + 0x18)
        hdrs.append((name, typ, off, size))
    strtab = base + hdrs[shstrndx][2]
    out = []
    for name, typ, off, size in hdrs:
        n = b[strtab + name:b.index(b"\0", strtab + name)].decode()
        out.append((n, typ, base + off, size))
    return out


def elf_end(b, base):
    shoff = struct.unpack_from("<Q", b, base + 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", b, base + 0x3A)
    end = shoff + shentsize * shnum
    for i in range(shnum):
        sh = base + shoff + i * shentsize
        typ = struct.unpack_from("<I", b, sh + 4)[0]
        off, size = struct.unpack_from("<QQ", b, sh + 0x18)
        if typ != SHT_NOBITS:
            end = max(end, off + size)
    return base + end


def zstd_frame_len(b, off):
    p = off + 4
    fhd = b[p]
    p += 1
    single_segment = (fhd >> 5) & 1
    if not single_segment:
        p += 1
    p += [0, 1, 2, 4][fhd & 3]
    p += [1 if single_segment else 0, 2, 4, 8][fhd >> 6]
    while True:
        bh = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16)
        p += 3
        p += 1 if (bh >> 1) & 3 == 1 else bh >> 3
        if bh & 1:
            break
    if (fhd >> 2) & 1:
        p += 4
    return p - off


def zstd_decompress(frame):
    return subprocess.run(["zstd", "-dc"], input=frame, capture_output=True,
                          check=True).stdout


def section_blobs(sec):
    """The (possibly zstd-compressed) device images in an offload section."""
    p = 0
    while p < len(sec):
        if sec.startswith(ZSTD_MAGIC, p):
            n = zstd_frame_len(sec, p)
            yield zstd_decompress(sec[p:p + n])
            p += n
        else:
            nxt = sec.find(ZSTD_MAGIC, p)
            yield sec[p:nxt if nxt >= 0 else len(sec)]
            if nxt < 0:
                return
            p = nxt


def zebins(blob):
    for m in re.finditer(re.escape(ELF_MAGIC), blob):
        o = m.start()
        if struct.unpack_from("<H", blob, o + 0x12)[0] == EM_INTELGT:
            yield blob[o:elf_end(blob, o)]


def product_family(zb):
    for name, typ, off, size in elf_sections(zb):
        if typ != SHT_NOTE:
            continue
        p = off
        while p < off + size:
            namesz, descsz, ntype = struct.unpack_from("<III", zb, p)
            p += 12
            owner = zb[p:p + namesz].rstrip(b"\0")
            p += (namesz + 3) & ~3
            if owner == b"IntelGT" and ntype == NT_INTELGT_PRODUCT_FAMILY:
                return struct.unpack_from("<I", zb, p)[0]
            p += (descsz + 3) & ~3
    return None


def kernels(zb):
    """(name, grf_count) for each kernel in the .ze_info of a zebin."""
    secs = {n: (o, s) for n, t, o, s in elf_sections(zb)}
    o, s = secs[".ze_info"]
    text = zb[o:o + s].decode(errors="replace")
    # The top-level `kernels:` list, up to the next top-level key.
    start = text.index("\nkernels:")
    text = re.split(r"\n(?=\S)", text[start + 1:])[0]
    for m in re.finditer(r"\n  - name:\s+(\S+)(.*?)(?=\n  - name:|\Z)",
                         text, re.S):
        grf = re.search(r"\n\s+grf_count:\s*(\d+)", m.group(2))
        yield m.group(1), int(grf.group(1)) if grf else None


def family_of(name):
    for family, needle in FAMILIES:
        if needle in name:
            return family
    return "other"


def check_library(path, verbose):
    errors = []
    so = path.read_bytes()
    sections = [(n, o, s) for n, t, o, s in elf_sections(so)
                if n.startswith("__CLANG_OFFLOAD_BUNDLE__sycl-")]
    if not sections:
        return [f"{path}: no SYCL device code"]

    found = defaultdict(Counter)
    names = defaultdict(set)  # device -> kernel names of the COMMON families
    histogram = defaultdict(Counter)
    mismatches = {}  # (device, family, grf, want) -> [count, example name]
    for sec_name, off, size in sections:
        for blob in section_blobs(so[off:off + size]):
            for zb in zebins(blob):
                pf = product_family(zb)
                device = PRODUCTS.get(pf)
                if device is None:
                    errors.append(f"{sec_name}: zebin for unknown product "
                                  f"family {pf}")
                    continue
                for name, grf in kernels(zb):
                    family = family_of(name)
                    found[device][family] += 1
                    if family in COMMON:
                        names[device].add(name)
                    histogram[(device, family)][grf] += 1
                    want = expected_grf(device, family, name)
                    if want is not None and grf != want:
                        key = (device, family, grf, want)
                        mismatches.setdefault(key, [0, name])[0] += 1

    for (device, family, grf, want), (n, name) in sorted(
            mismatches.items(), key=str):
        errors.append(f"{device}: {n} {family} kernel(s) have GRF {grf}, "
                      f"expected {want} (e.g. {name})")

    for device, families in REQUIRED.items():
        if device not in found:
            errors.append(f"no AOT image for {device} (the kernels would be "
                          f"JIT-compiled at run time)")
            continue
        missing = families - set(found[device])
        if missing:
            errors.append(f"{device}: no {', '.join(sorted(missing))} kernels")
        if device != "cri":
            extra = CRI_ONLY & set(found[device])
            if extra:
                errors.append(f"{device}: has CRI-only "
                              f"{', '.join(sorted(extra))} kernels")

    # A kernel IGC did not compile for a device is left out of that device's
    # zebin (which then only carries SPIR-V), so it is JIT-compiled at run
    # time. Catch it by comparing the kernels across devices.
    all_names = set().union(*names.values()) if names else set()
    for device in sorted(names):
        missing = all_names - names[device]
        if missing:
            errors.append(f"{device}: {len(missing)} kernel(s) have no AOT "
                          f"code (e.g. {min(missing)})")

    print(f"{path}:")
    for (device, family), grfs in sorted(histogram.items()):
        if verbose or family != "other":
            sizes = ", ".join(f"{g}: {n}" for g, n in sorted(
                grfs.items(), key=lambda x: (x[0] is None, x[0] or 0)))
            print(f"  {device:4} {family:17} GRF {{{sizes}}}")
    return [f"{path.name}: {e}" for e in errors]


def main():
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    verbose = "--verbose" in sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    libs = []
    for arg in map(Path, args):
        libs += sorted(arg.rglob("_flash_attn2_*.so")) if arg.is_dir() else [arg]
    # Only the XPU variants carry SYCL device code.
    libs = [lib for lib in libs if "xpu" in str(lib)]
    if not libs:
        sys.exit(f"no XPU flash-attn2 library in {' '.join(args)}")

    errors = []
    for lib in libs:
        errors += check_library(lib, verbose)
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print("XPU device code OK")


if __name__ == "__main__":
    main()
