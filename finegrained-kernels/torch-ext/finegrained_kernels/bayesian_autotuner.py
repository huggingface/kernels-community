# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bayesian-optimization autotuner: benches a budgeted sample of the config grid instead of
the full grid, via a Tree-structured Parzen Estimator (TPE) over the config *dimensions*
(``num_warps`` / ``num_stages`` / tile sizes / flags). After a short random seed phase it
models the good (top-``gamma`` by measured time) vs bad per-dimension value densities and
benches the unmeasured config that maximizes ``l(x)/g(x)`` — the Expected-Improvement proxy
— then a coordinate-descent pass polishes the best. Subclasses Triton's ``Autotuner`` and
uses only the Python stdlib (no external optimizer dependency).

Each new key's search is warm-started from the most recently tuned key's best config (nearby
workloads share tile-shape preferences). Grids smaller than ``n_trials`` defer to the stock
exhaustive bench-all.
"""

from __future__ import annotations

import hashlib
import logging
import json
import math
import os
import traceback
import random
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from typing import Dict, List

from triton.runtime.autotuner import (
    Autotuner,
    Config,
    JITFunction,
    driver,
    get_cache_invalidating_env_vars,
    get_cache_manager,
    knobs,
    triton_key,
)

logger = logging.getLogger(__name__)

# Env-gated host-execution counter (FINEGRAINED_AUTOTUNE_CALLCOUNT=<path>): every
# BayesianAutotuner.run() call increments its kernel's tally, dumped to <path> at exit.
# Discriminates cudagraph REPLAY (host code runs only at capture) from per-token
# re-execution — the decode-gap forensics tool.
_CALL_COUNTS = None
if os.environ.get("FINEGRAINED_AUTOTUNE_CALLCOUNT"):
    import atexit

    _CALL_COUNTS = {}

    def _dump_call_counts(path=os.environ["FINEGRAINED_AUTOTUNE_CALLCOUNT"]):
        with open(path, "w") as f:
            json.dump(_CALL_COUNTS, f, indent=1, sort_keys=True)

    atexit.register(_dump_call_counts)


class BayesianAutotuner(Autotuner):
    """Drop-in replacement for ``triton.runtime.autotuner.Autotuner`` that
    benches ~``n_trials`` configs per key via TPE Bayesian optimization +
    coordinate-descent refinement, instead of the full grid."""

    def __init__(
        self,
        *args,
        n_trials: int = 80,
        max_failures: int | None = None,
        n_startup_trials: int = 12,
        gamma: float = 0.25,
        refine: bool = True,
        max_refine_iters: int = 5,
        log_path: str | None = None,
        path_anchor_axes: tuple[str, ...] = (),
        finite_check_args: tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fn_name = getattr(self.fn, "__name__", str(self.fn))
        # Bayesian trial budget — the per-decorator default, overridable via the
        # FINEGRAINED_AUTOTUNE_TRIALS env var (quick sweeps / exhaustive runs without touching
        # the decorators; set it >= grid size to fall back to stock exhaustive bench-all).
        self.n_trials = int(os.environ.get("FINEGRAINED_AUTOTUNE_TRIALS") or n_trials)
        # A tune aborts once this many configs have failed to compile/run without the measured
        # budget being met. Defaults to n_trials: a grid that cannot land its measurements without
        # that many rejects has a pruner gap, and the fix is to fence the dead region rather than
        # to tolerate the compiles. Its own axis so a kernel can raise it deliberately.
        self.max_failures = int(
            os.environ.get("FINEGRAINED_AUTOTUNE_MAX_FAILURES")
            or max_failures
            or self.n_trials
        )
        self.n_startup_trials = n_startup_trials
        # top fraction of measured configs the TPE treats as "good"
        self.gamma = gamma
        self.refine = refine
        self.max_refine_iters = max_refine_iters
        # Branch axes (by kwarg name) that relocate the TILE optimum — the caller's
        # declaration, since it is configuration knowledge (the tuner never hardcodes
        # axis names). Each distinct value combination of these axes gets one guaranteed
        # max-tile anchor; empty = median anchors only.
        self.path_anchor_axes = tuple(path_anchor_axes)
        # Output arg NAMES (a declaration — the tuner never hardcodes kernel details) whose
        # values are checked for NaN/inf after each config's bench: a config that writes
        # non-finite output scores inf and can never be crowned. The tuner times, it never
        # validates — this veto closes the NaN half of the "wrong-answer configs win
        # tunes" class at negligible cost (one isfinite pass per benched config).
        self.finite_check_args = tuple(finite_check_args)
        # JSONL log of every benched (key, config, ms) — set here or via the
        # FINEGRAINED_AUTOTUNE_LOG env var. Analyse offline to prune bad configs.
        self.log_path = log_path or os.environ.get("FINEGRAINED_AUTOTUNE_LOG")
        # steady-state launch support: key extraction without dict builds (eager decode
        # is host-bound; the cache-hit path of run() is on the per-token critical path)
        self._arg_position = {name: i for i, name in enumerate(self.arg_names)}
        self._arg_name_set = frozenset(self.arg_names)
        self._dtype_tag_cache: Dict[tuple, str] = {}
        self._dtype_str_cache: Dict[torch.dtype, str] = {}

    def _dtype_str(self, dtype) -> str:
        s = self._dtype_str_cache.get(dtype)
        if s is None:
            s = self._dtype_str_cache[dtype] = str(dtype)
        return s

    # substrings marking a COMPILE-stage failure — the only class safe to memoize on
    # disk (a benching/CUDA error can be transient or sticky-context contamination;
    # persisting one would permanently fence a healthy config for this source version)
    _COMPILE_FAILURE_MARKS = (
        "PassManager",
        "CompilationError",
        "MLIR",
        "ConvertTritonGPUToLLVM",
        "TritonGPUAccelerateMatmul",
    )

    def _bench(self, *args, config, **meta):
        """Score any failing config as inf instead of raising — a compile failure is data
        for the search, not a fatal error. Stock Triton forgives only OutOfResources, but
        e.g. Triton 3.7.1's ``warp_specialize`` raises RuntimeError at unsupported
        (shape, config) combos; unguarded, one such config kills the whole tune when a
        small grid falls through to the stock exhaustive path below. Every failure is
        recorded and ``_report_bench_failures`` reports every distinct failure —
        inf-scoring must not silently hide a broken path behind a healthy one.
        (Stock already inf's OutOfResources / CompileTimeAssertionFailure / PTXASError
        internally without reaching this handler — those are deliberate guard classes,
        e.g. our own ``tl.static_assert`` fences; the reporter covers the UNEXPECTED
        failure classes stock would otherwise let kill the tune.)

        Compile-stage failures are memoized on disk keyed by everything that determines
        compilation (kernel source hash + config + constexpr arg values + tensor dtypes +
        invalidating env), so the next fresh tune of ANY shape skips the doomed compile
        outright (~40-66 wasted compiles per nvfp4 tune before this). The memo dies with
        the kernel source (``fn.cache_key``); bench-stage errors are never persisted."""
        sig = self._compile_signature(config, meta)
        memo = self._failed_compile_memo()
        if sig is not None and sig in memo:
            self._failures.append((config, memo[sig] + "  [memoized]"))
            return [float("inf")] * 3
        try:
            timing = super()._bench(*args, config=config, **meta)
            bad = self._nonfinite_output(meta)
            if bad is not None:
                self._failures.append((config, f"non-finite output in {bad} (numerics veto)"))
                return [float("inf")] * 3
            return timing
        except Exception as e:
            # triton's CompilationError carries only the location + source snippet in str(e);
            # the actual reason lives in the chained cause, so without it every compile failure
            # reads as an anonymous caret. Keep the head of the location and the cause verbatim.
            msg = str(e)[:250]
            # triton nests a CompilationError per @jit frame, each carrying only its own location
            # and source — the actual reason is at the BOTTOM of the chain. Walk to it, else every
            # compile failure reads as an anonymous caret and cannot be fenced.
            root, seen = e, 0
            while (root.__cause__ or root.__context__) is not None and seen < 12:
                root = root.__cause__ or root.__context__
                seen += 1
            if root is not e:
                msg += f" || root: {type(root).__name__}: {str(root)[-300:]}"
            err = f"{type(e).__name__}: {msg}"
            if os.environ.get("FINEGRAINED_AUTOTUNE_TRACEBACKS"):
                logger.warning(
                    "[autotune] %s failed on %s:\n%s",
                    self.fn_name,
                    config,
                    "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                )
            self._failures.append((config, err))
            if sig is not None and any(m in err for m in self._COMPILE_FAILURE_MARKS):
                memo[sig] = err
                self._persist_failed_compile_memo()
            return [float("inf")] * 3

    def _nonfinite_output(self, meta=None) -> str | None:
        """Name of the first declared output arg holding NaN/inf after a bench, else None.
        fp8 tensors byte-check the E4M3 NaN pattern (no ``isfinite`` kernel for fp8);
        integer/packed outputs are skipped (every bit pattern is a value)."""
        if not self.finite_check_args:
            return None
        nargs = {**(getattr(self, "nargs", None) or {}), **(meta or {})}
        for name in self.finite_check_args:
            v = nargs.get(name)
            if not isinstance(v, torch.Tensor) or v.numel() == 0:
                continue
            if v.dtype == torch.float8_e4m3fn:
                flat = v.reshape(-1).view(torch.uint8)
                bad = bool((((flat & 0x7F) == 0x7F)).any())
            elif v.is_floating_point():
                bad = bool((~v.reshape(-1).float().isfinite()).any())
            else:
                continue
            if bad:
                return name
        return None

    def _compile_signature(self, config, meta=None) -> str | None:
        """Hash of the compile determinants for one config: the config itself plus the
        constexpr argument VALUES and tensor argument dtypes from this launch (both feed
        specialization — e.g. GATE flips arms, a uint8 A packs the loads). Source/env
        live in the memo FILE's key, not here.

        ``meta`` carries the launch kwargs: every constexpr is kwarg-passed at the call
        sites, so ``self.nargs`` (positional-only) alone would hash GATE/recipe flags as
        None and collide memo entries across arms."""
        nargs = {**(getattr(self, "nargs", None) or {}), **(meta or {})}
        if not nargs:
            return None
        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        parts = [str(sorted(config.all_kwargs().items()))]
        for param in fn.params:
            v = nargs.get(param.name)
            if hasattr(v, "dtype"):
                parts.append(f"{param.name}:{v.dtype}")
            elif param.is_constexpr and isinstance(
                v, (bool, int, float, str, type(None))
            ):
                parts.append(f"{param.name}={v}")
            elif isinstance(v, int) and not isinstance(v, bool):
                # non-constexpr ints feed Triton's specialization (==1 / %16 divisibility),
                # which is (shape, config)-dependent for WS lowering failures — hashing the
                # bucket keeps one shape's PassManager failure from fencing every shape
                parts.append(
                    f"{param.name}:i{1 if v == 1 else (16 if v % 16 == 0 else 0)}"
                )
        return hashlib.sha256("-".join(parts).encode("utf-8")).hexdigest()

    def _failed_compile_memo(self) -> dict:
        """The per-(kernel source, backend, env) failed-compile dict, loaded from
        Triton's on-disk cache once per autotuner instance."""
        if getattr(self, "_failed_memo", None) is not None:
            return self._failed_memo
        try:
            from triton.compiler.compiler import make_backend

            fn = self.fn
            while not isinstance(fn, JITFunction):
                fn = fn.fn
            group = hashlib.sha256(
                "-".join(
                    [
                        triton_key(),
                        make_backend(driver.active.get_current_target()).hash(),
                        fn.cache_key,
                        str(sorted(get_cache_invalidating_env_vars().items())),
                    ]
                ).encode("utf-8")
            ).hexdigest()
            self._failed_memo_cache = get_cache_manager(group)
            self._failed_memo_file = f"{fn.__name__[:150]}.failed_compiles.json"
            path = self._failed_memo_cache.get_file(self._failed_memo_file)
            self._failed_memo = json.load(open(path)) if path else {}
        except Exception:
            self._failed_memo_cache = None
            self._failed_memo = {}
        return self._failed_memo

    def _persist_failed_compile_memo(self):
        if getattr(self, "_failed_memo_cache", None) is None:
            return
        try:
            self._failed_memo_cache.put(
                json.dumps(self._failed_memo), self._failed_memo_file, binary=False
            )
        except Exception:
            pass  # persistence is best-effort; the in-memory memo still holds

    def _report_bench_failures(self):
        """After every tune, report every UNIQUE failure — a failure is never silent:
        inf-scoring keeps the search alive, but a human must see what broke (e.g. a code
        change that kills one compute path would otherwise silently degrade into "the other
        path wins"). Distinct errors are deduped with a count and an example config; the
        JSONL autotune log has per-config detail."""
        if not self._failures:
            return
        by_err = defaultdict(list)
        for c, err in self._failures:
            by_err[err].append(c)
        for err, cfgs in by_err.items():
            c = cfgs[0]
            example = ", ".join(f"{k}={v}" for k, v in c.kwargs.items())
            logger.warning(
                "[autotune] %s: %d config(s) failed to compile/run — %s  (e.g. %s, w%d s%d)",
                self.fn_name,
                len(cfgs),
                err,
                example,
                c.num_warps,
                c.num_stages,
            )

    def run(self, *args, **kwargs):
        if _CALL_COUNTS is not None:
            _CALL_COUNTS[self.fn_name] = _CALL_COUNTS.get(self.fn_name, 0) + 1
        self._failures = []
        # Small grid → defer to parent (stock exhaustive bench-all). Steady state
        # still short-circuits the parent's per-call dict builds: a crowned config
        # with no pre_hook launches directly. The lookup key must be STOCK format
        # (key values + untagged dtype strings) — it is the parent that populates
        # this branch's cache entries.
        if len(self.configs) <= 1 or self.n_trials >= len(self.configs):
            config = self.configs[0] if len(self.configs) == 1 else None
            if config is None and self.cache:
                n_positional = len(args)
                key = []
                for k in self.keys:
                    i = self._arg_position.get(k)
                    if i is not None and i < n_positional:
                        key.append(args[i])
                    elif k in kwargs and k in self._arg_name_set:
                        key.append(kwargs[k])
                for i in range(n_positional):
                    dtype = getattr(args[i], "dtype", None)
                    if dtype is not None:
                        key.append(self._dtype_str(dtype))
                for k, v in kwargs.items():
                    if k in self._arg_name_set and self._arg_position[k] >= n_positional:
                        dtype = getattr(v, "dtype", None)
                        if dtype is not None:
                            key.append(self._dtype_str(dtype))
                config = self.cache.get(tuple(key))
            if config is not None and config.pre_hook is None:
                self.best_config = config
                config_kwargs = getattr(config, "_all_kwargs_memo", None)
                if config_kwargs is None:
                    config_kwargs = config._all_kwargs_memo = config.all_kwargs()
                return self.fn.run(*args, **kwargs, **config_kwargs)
            ret = super().run(*args, **kwargs)
            self._report_bench_failures()
            return ret

        # Key extraction runs on every launch and eager decode is host-bound, so it
        # avoids dict builds: positional args by precomputed index, then kwargs —
        # the same (arg_names-prefix, kwargs-order) iteration the dict merge gave,
        # keeping keys byte-compatible with existing disk crown caches.
        arg_names = self.arg_names
        n_positional = len(args)
        tags = self._dtype_tag_cache
        key = []
        for k in self.keys:
            i = self._arg_position.get(k)
            if i is not None and i < n_positional:
                key.append(args[i])
            elif k in kwargs and k in self._arg_name_set:
                key.append(kwargs[k])
        # name-tagged, with an explicit token for absent optional tensors: a positional
        # dtype list lets a gather-only and a scatter-only launch alias to one entry,
        # while descriptor-arm legality (descriptor_box_pruner) hangs off GatherIdx
        # presence — the aliased replay would read wrong rows without ever re-pruning
        for i in range(n_positional):
            v = args[i]
            dtype = getattr(v, "dtype", None) if v is not None else None
            if v is not None and dtype is None:
                continue
            tag_key = (arg_names[i], dtype)
            tag = tags.get(tag_key)
            if tag is None:
                tag = tags[tag_key] = f"{tag_key[0]}:{dtype}"
            key.append(tag)
        for k, v in kwargs.items():
            if k not in self._arg_name_set or self._arg_position[k] < n_positional:
                continue
            dtype = getattr(v, "dtype", None) if v is not None else None
            if v is not None and dtype is None:
                continue
            tag_key = (k, dtype)
            tag = tags.get(tag_key)
            if tag is None:
                tag = tags[tag_key] = f"{tag_key[0]}:{dtype}"
            key.append(tag)
        key = tuple(key)

        first_launch = key not in self.cache
        if first_launch:
            self.nargs = dict(zip(self.arg_names, args))
            pruned = self.prune_configs(kwargs)

            def benchmark():
                t0 = time.time()
                self.cache[key] = self._bayesian_search(pruned, args, kwargs, key)
                self._report_bench_failures()
                self.bench_time = time.time() - t0
                if knobs.autotuning.print:
                    fn_name = getattr(self.fn, "__name__", str(self.fn))
                    print(
                        f"[bayesian-autotune] {fn_name} tuned "
                        f"{len(self.configs_timings)} configs in {self.bench_time:.1f}s — "
                        f"key={key}, best={self.cache[key].all_kwargs()}"
                    )
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)

            if self.cache_results:
                self.check_disk_cache(key, pruned, benchmark)
            else:
                benchmark()

        config = self.cache[key]
        self.best_config = config
        config_kwargs = getattr(config, "_all_kwargs_memo", None)
        if config_kwargs is None:
            config_kwargs = config._all_kwargs_memo = config.all_kwargs()
        if config.pre_hook is not None:
            if not first_launch:  # tuned path already built nargs for prune_configs
                self.nargs = dict(zip(self.arg_names, args))
            config.pre_hook({**self.nargs, **kwargs, **config_kwargs})
        ret = self.fn.run(*args, **kwargs, **config_kwargs)
        if first_launch and self.finite_check_args:
            # First launch after a key miss: the one regime where a crowned launch has been
            # observed returning non-finite output while every subsequent identical launch
            # is correct (large multi-device models; root cause open — see the DSV3
            # investigation record). Detect via the declared output args and relaunch once,
            # loudly. Sentinel/padding rows can false-trigger (uninitialized by contract);
            # the cost is one duplicate launch per key, never a dropped config.
            bad = self._nonfinite_output(kwargs)
            if bad is not None:
                logger.warning(
                    "[autotune] %s: non-finite %s on the FIRST launch of key=%s — relaunching once",
                    self.fn_name, bad, key,
                )
                ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret

    def _bayesian_search(self, configs: List[Config], args, kwargs, key) -> Config:
        timings: Dict[int, float] = {}
        sigs = [tuple(sorted(c.all_kwargs().items())) for c in configs]

        def bench_idx(idx: int) -> float:
            if idx in timings:
                return timings[idx]
            try:
                ms = self._bench(*args, config=configs[idx], **kwargs)
                if isinstance(ms, (tuple, list)):
                    ms = ms[0]
                timings[idx] = float(ms)
            except Exception:
                timings[idx] = float("inf")
            self._log_result(key, configs[idx], timings[idx])
            return timings[idx]

        # Distinct values per config dimension, for the TPE's Laplace smoothing.
        dim_vals: Dict = defaultdict(set)
        for sig in sigs:
            for d, v in sig:
                dim_vals[d].add(v)

        # Seed phase: one BASIN ANCHOR per (COMPUTE_MODE, SWAP_AB) group, the most recent
        # cached key's best (warm start), then seeded-random fill. The anchors guarantee every
        # categorical basin gets at least one real measurement — without them the TPE's
        # per-dimension model can write off a whole axis it never saw succeed (two dsv4 tunes
        # shipped 25-60% slow winners because their random seeds only sampled a basin's dead
        # configs), and coordinate descent can't recover a winner two coupled flips away.
        n_startup = max(2, min(self.n_startup_trials, self.n_trials))
        anchors = self._basin_anchor_indices(configs)
        order = list(range(len(configs)))
        random.Random(0).shuffle(order)
        warm_idx = self._warm_start_index(configs)
        head = anchors + ([warm_idx] if warm_idx is not None else [])
        order = list(dict.fromkeys(head + order))
        for idx in order[: max(n_startup, len(head))]:
            bench_idx(idx)

        # TPE: split measured configs into good (top-gamma) / bad, build per-dimension value
        # densities for each, and bench the unmeasured config maximizing log l(x) - log g(x)
        # (Expected-Improvement proxy), updating the model after each measurement.
        # inf (failed-to-compile) configs are EXCLUDED from the densities: a compile failure is
        # evidence about that one joint shape (usually shared memory), not about its dimension
        # values — counting them as "bad" buried SWAP_AB under a wall of BN=256 smem failures
        # and made the tuner ship a 53µs winner while the 41µs swap config sat unbenched.
        # They don't consume the trial budget either (n_trials = MEASURED configs; a failure
        # stays in ``timings`` only as a skip-list entry) — the compile it burned is the one
        # cost that can't be refunded here, which is what the smem/compile-guard pruners avoid.
        warned_failures = False
        while sum(1 for t in timings.values() if t != float("inf")) < self.n_trials:
            # Failures don't consume the trial budget (above), which is right for a handful of
            # smem/regs rejects but unbounded if a launch is broadly broken: the loop keeps
            # picking, each pick costs an O(configs) TPE rescan, and the tune degenerates into a
            # silent multi-hour spin instead of a diagnosable error. Cap the failures at the same
            # budget as the measurements and fail loudly.
            if (
                not warned_failures
                and sum(1 for t in timings.values() if t == float("inf")) >= self.max_failures
            ):
                # Loud, once, and non-fatal. Aborting here would break tunes that DO find a winner
                # after many rejects (mxfp4 W4A4 at a non-128 N burns >100), trading a silent
                # inefficiency for a hard failure. Warning keeps the waste visible — this is how
                # ~100 dead compiles per mx tune (an IndexError in load_act_mx's inline-quant arm)
                # were found at all — without gating the kernel on a diagnostic.
                warned_failures = True
                self._report_bench_failures()
                logger.warning(
                    "[autotune] %s: %d configs failed to compile/run before the trial budget was "
                    "met — that is dead compile time every tune, and a pruner gap worth closing.",
                    self.fn_name,
                    self.max_failures,
                )
            ranked = sorted(
                (i for i, t in timings.items() if t != float("inf")), key=timings.get
            )
            if not ranked:  # nothing compiled yet — keep seeding in shuffled order
                nxt = next((i for i in order if i not in timings), None)
                if nxt is None:
                    break
                bench_idx(nxt)
                continue
            n_good = max(1, round(self.gamma * len(ranked)))
            good_c: Dict = defaultdict(lambda: defaultdict(int))
            bad_c: Dict = defaultdict(lambda: defaultdict(int))
            for j, i in enumerate(ranked):
                tgt = good_c if j < n_good else bad_c
                for d, v in sigs[i]:
                    tgt[d][v] += 1
            n_bad = len(ranked) - n_good
            best_i, best_score = None, -math.inf
            for i in range(len(configs)):
                if i in timings:
                    continue
                score = 0.0
                for d, v in sigs[i]:
                    V = len(dim_vals[d])
                    lp = (good_c[d][v] + 1.0) / (
                        n_good + V
                    )  # P(v | good), Laplace-smoothed
                    gp = (bad_c[d][v] + 1.0) / (n_bad + V)  # P(v | bad)
                    score += math.log(lp) - math.log(gp)
                if score > best_score:
                    best_score, best_i = score, i
            if best_i is None:
                break
            bench_idx(best_i)

        # Coordinate-descent refinement: try single-dim perturbations around
        # the current best until no neighbor improves.
        if self.refine:
            for _ in range(self.max_refine_iters):
                best_idx = min(timings, key=timings.get)
                best_sig = sigs[best_idx]
                best_ms = timings[best_idx]
                improved = False
                for i, s in enumerate(sigs):
                    if i in timings:
                        continue
                    diff = sum(1 for (_, a), (_, b) in zip(best_sig, s) if a != b)
                    if diff != 1:
                        continue
                    if bench_idx(i) < best_ms:
                        improved = True
                if not improved:
                    break

        # Crown RUNOFF: the search's single-shot timings carry enough noise to flip crowns
        # between same-code tune rolls (gpt-oss prefill grouped rolled 1727-2150us on identical
        # grids; the tuner's one RNG site is seeded, so noise is the only roll source). Re-time
        # the top candidates back-to-back — same warmed process, adjacent in time, so they see
        # the same clocks — and crown the winner of THAT comparison. Costs ~3 extra benches.
        finite = sorted((t, i) for i, t in timings.items() if t != float("inf"))
        if len(finite) > 1:
            for _, i in finite[:3]:
                timings.pop(i, None)
                bench_idx(i)  # memoizing closure: re-benches and re-records the popped entry

        self.configs_timings = {configs[i]: t for i, t in timings.items()}
        best = min(timings, key=timings.get)
        if timings[best] == float("inf"):
            # crowning an arbitrary broken config would persist it as `best` on disk and
            # every later healthy process would replay it without re-tuning — the exact
            # sticky-contaminated-context scenario inf-forgiveness exists to survive
            self._report_bench_failures()
            raise RuntimeError(
                f"[bayesian-autotune] {self.fn_name}: every benched config failed "
                "(all-inf tune) — nothing correct to crown or persist; the context may "
                "be trap-contaminated (retry in a fresh process) or the grid is broken "
                "for this launch (see the failure report above)."
            )
        return configs[best]

    def _log_result(self, key, config: Config, ms: float):
        """Append one ``(key, config, ms)`` record as JSONL for offline analysis —
        e.g. pruning configs that are consistently far off the per-key best. ``inf`` ms
        marks a config that failed to compile/run (out of resources, etc.)."""
        if not self.log_path:
            return
        try:
            rec = {
                "timestamp": time.time(),
                "fn_name": getattr(self.fn, "__name__", str(self.fn)),
                "key": list(key),
                "kwargs": config.kwargs,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "ms": ms,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def _basin_anchor_indices(self, configs: List[Config]) -> List[int]:
        """One representative config index per (COMPUTE_MODE, SWAP_AB) basin — the MEDIAN in
        tile-sort order, a mid-sized tile with mid warps/stages. Not the smallest: a basin's
        minimal corner (min BK x 2 warps) can be latency-bound pathological (a 131µs anchor in
        a basin whose peak is 41µs re-poisons the axis it was meant to protect). Coordinate
        descent climbs BN/BK/warps/stages from wherever the TPE lands within the basin.
        Returns [] when the grid has no such axes (single basin)."""
        # Basin axes are DERIVED, not declared: a config kwarg with string or boolean
        # values is a branch axis (different code path — compute mode, operand swap,
        # warp specialization), which partitions the grid into disjoint performance
        # basins. Numeric kwargs (tiles/warps/stages) are ordinal — the TPE's densities
        # and coordinate descent handle those.
        basin_axes = sorted(
            {
                k
                for c in configs
                for k, v in c.kwargs.items()
                if isinstance(v, (bool, str))
            }
        )
        if not basin_axes:
            return []

        # A basin = the compute-path axes (constexprs selecting different compiled code).
        groups: Dict = {}
        for i, c in enumerate(configs):
            basin = tuple(c.kwargs.get(k) for k in basin_axes)
            groups.setdefault(basin, []).append(i)

        if len(groups) <= 1:
            return []

        def tile_order(i):
            return (
                configs[i].kwargs.get("BLOCK_SIZE_N", 0),
                configs[i].kwargs.get("BLOCK_SIZE_K", 0),
                configs[i].num_warps,
                configs[i].num_stages,
            )

        # Median anchor per basin, PLUS one max-tile anchor per declared PATH-axis combo
        # (``path_anchor_axes``). The median alone left the big-tile end unsampled: TPE
        # densities lock onto the first-timed mid-tile region and coordinate descent cannot
        # cross DIAGONAL ridges — dsv4 mxfp8 prefill gate_up's BN=128/BK=256 winner (840us)
        # needs BN and warps to move together, so descent from the BN=64 crown rejects the
        # lone BN step and n_trials 100/200/400 all missed it; one guaranteed big-tile
        # sample fixed it (crown 958 -> 880 at the default budget). The max anchor rides
        # the declared COARSE grouping, NOT the full branch-axis cross: per-basin max
        # anchors on the weight-only grouped grid (32 basins: modes x WS x PS x 2 memory
        # modes each way) put 64 forced trials into a 100-trial budget and starved the
        # search — gpt-oss prefill regressed 3463 -> 4099us on a quiet box before this was
        # scoped. Which axes qualify is the KERNEL's declaration (configuration knowledge);
        # axes absent from this grid are ignored.
        anchors = []
        coarse_seen = set()
        coarse_axes = [k for k in self.path_anchor_axes if k in basin_axes]
        for basin, idxs in groups.items():
            ordered = sorted(idxs, key=tile_order)
            anchors.append(ordered[len(ordered) // 2])
            if not self.path_anchor_axes:
                continue  # no declaration -> median anchors only
            # declared axes absent from THIS grid still coarse-group to () — one global
            # max-tile anchor, the big-tile guarantee without per-basin anchor spam
            coarse = tuple(basin[basin_axes.index(k)] for k in coarse_axes)
            if coarse not in coarse_seen and ordered[-1] != anchors[-1]:
                coarse_seen.add(coarse)
                anchors.append(ordered[-1])
        return anchors

    def _warm_start_index(self, configs: List[Config]):
        """Return the index in ``configs`` matching the most recently cached
        key's best config (or ``None`` if no prior tune or no match in the
        current pruned list)."""
        if not self.cache:
            return None
        # Python 3.7+ dicts preserve insertion order; last entry = most recent tune.
        prev_best = next(reversed(self.cache.values()))
        prev_kwargs = prev_best.all_kwargs()
        for i, c in enumerate(configs):
            if c.all_kwargs() == prev_kwargs:
                return i
        return None

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        """Persist the tuned best config to Triton's on-disk cache so a later run (or process)
        skips the whole minutes-long search+compile — most of which is Triton codegen, not
        benching. Keyed exactly like Triton's own autotune cache (triton version + backend +
        kernel source hash + invalidating env + key + the full config grid), so any of those
        changing re-tunes.

        Unlike Triton's stock version we do NOT bail when configs carry a ``pre_hook`` (ours
        always do): we persist each config's ``all_kwargs()`` — serializable and pre_hook-free —
        and on a hit re-match it to the live ``Config`` object, so the pre_hook is preserved from
        code rather than lost in (de)serialization. A hit launches only the winning config."""
        if not tuning_key:
            bench_fn()
            return False

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        env_vars = get_cache_invalidating_env_vars()
        cache_key = [
            triton_key(),
            make_backend(driver.active.get_current_target()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            # Key NAMES and the search budget are determinants too: tuning_key holds
            # values only (renaming a key or swapping two same-typed keys would collide),
            # and FINEGRAINED_AUTOTUNE_TRIALS is not in Triton's invalidating-env list —
            # without n_trials a 3-trial exploratory winner gets replayed at 100 trials.
            str(self.keys),
            f"n_trials={self.n_trials},n_startup_trials={self.n_startup_trials},gamma={self.gamma}",
            str(tuning_key),
        ] + [str(c) for c in configs]  # str(Config) is pre_hook-free and process-stable
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.bayes_autotune.json"
        keylog = os.environ.get("FINEGRAINED_AUTOTUNE_KEYLOG")
        if keylog:
            with open(keylog, "a") as f:
                f.write(json.dumps({
                    "fn": fn.__name__,
                    "hit": cache.get_file(file_name) is not None,
                    "tuning_key": str(tuning_key),
                    "keys": str(self.keys),
                    "n_configs": len(configs),
                    "env": str(sorted(env_vars.items())),
                    "fn_cache_key": fn.cache_key[:16],
                    "hash": cache_key[:16],
                }) + "\n")

        # signature -> live Config (carries the pre_hook); used to re-match the cached winner
        by_sig = {tuple(sorted(c.all_kwargs().items())): c for c in configs}

        def load_crown() -> bool:
            path = cache.get_file(file_name)
            if not path:
                return False
            try:
                with open(path) as f:
                    data = json.load(f)
                best = by_sig.get(tuple(sorted(data["best"].items())))
                if best is None:
                    return False
                self.cache[tuning_key] = best
                self.configs_timings = {
                    by_sig[s]: t
                    for kw, t in data["timings"]
                    for s in (tuple(sorted(kw.items())),)
                    if s in by_sig
                }
                return True
            except Exception:
                return False  # corrupt/stale cache file → re-tune

        if load_crown():
            return True

        # Cross-process tuning lock. Without it, TP/EP ranks sharing this cache dir all tune the
        # same key CONCURRENTLY and each crowns its own winner: the search is seeded-random and
        # near-ties break on measurement noise, so ranks end up on different tiles, round
        # differently, and greedy decoding turns that into different tokens (measured: an 8-rank
        # MXFP8 run where 3 ranks emitted a different token). Serializing means one rank tunes and
        # the rest read the same crown — identical configs, and ~world_size less tuning wall-clock.
        # Best-effort: any lock failure falls through to tuning locally (correct, just redundant).
        with _tuning_lock(cache, file_name):
            if load_crown():  # another rank tuned this key while we waited
                return True
            bench_fn()
            # publish INSIDE the lock: a rank released to an empty cache would tune again
            try:
                cache.put(
                    json.dumps(
                        {
                            "best": self.cache[tuning_key].all_kwargs(),
                            "timings": [
                                (c.all_kwargs(), t) for c, t in self.configs_timings.items()
                            ],
                        }
                    ),
                    file_name,
                    binary=False,
                )
            except Exception:
                pass
        return False


@contextmanager
def _tuning_lock(cache, file_name: str):
    """Serialize tuning of ONE key across every process sharing the Triton cache directory.

    Held while a key is tuned and its crown published, so concurrent TP/EP ranks converge on the
    same config instead of each crowning its own (see ``check_disk_cache``). Advisory ``flock`` on
    a sidecar file next to the crown; entirely best-effort — no cache dir, no ``fcntl``, or a lock
    that cannot be taken within ``FINEGRAINED_TUNE_LOCK_TIMEOUT`` (default 30 min, since the holder
    is doing a real tune) all fall through to tuning locally. Losing the lock costs redundant work,
    never correctness."""
    lock_file = None
    try:
        import fcntl

        cache_dir = getattr(cache, "cache_dir", None)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            deadline = time.time() + float(
                os.environ.get("FINEGRAINED_TUNE_LOCK_TIMEOUT") or 1800
            )
            lock_file = open(os.path.join(cache_dir, f"{file_name}.lock"), "w")
            while True:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        logger.warning(
                            "[autotune] %s: tuning lock held elsewhere for >%.0fs — tuning locally",
                            file_name,
                            time.time() - (deadline - 1800),
                        )
                        lock_file.close()
                        lock_file = None
                        break
                    time.sleep(1.0)
    except Exception:
        lock_file = None
    try:
        yield
    finally:
        if lock_file is not None:
            try:
                import fcntl

                fcntl.flock(lock_file, fcntl.LOCK_UN)
            finally:
                lock_file.close()



def bayesian_autotune(
    configs,
    key,
    *,
    n_trials: int = 80,
    max_failures: int | None = None,
    n_startup_trials: int = 12,
    gamma: float = 0.25,
    refine: bool = True,
    max_refine_iters: int = 5,
    log_path: str | None = None,
    path_anchor_axes: tuple[str, ...] = (),
    finite_check_args: tuple[str, ...] = (),
    cache_results: bool = True,
    reset_to_zero=None,
    restore_value=None,
    **kwargs,
):
    """Decorator mirroring ``@triton.autotune``. Extra kwargs:
    n_trials:                 successfully measured configs per key (TPE budget; configs
                              that fail to compile/run are skipped without consuming it)
    n_startup_trials:         random seed configs before the TPE model kicks in
    gamma:                    top fraction of measured configs treated as "good"
    refine, max_refine_iters: coordinate-descent refinement after the TPE
    path_anchor_axes:         branch axes (kwarg names) whose values relocate the tile
                              optimum — one guaranteed max-tile anchor per value combo
                              (see _basin_anchor_indices); the kernel declares them, the
                              tuner stays independent of configuration details
    max_failures:             abort the tune once this many configs fail to compile/run before
                              the measured budget is met (default n_trials; env override
                              FINEGRAINED_AUTOTUNE_MAX_FAILURES). Without it a broadly broken
                              grid spins: failures don't consume the trial budget, so the search
                              keeps picking and each pick costs an O(configs) rescan.
    log_path:                 JSONL of benched configs (or FINEGRAINED_AUTOTUNE_LOG)
    cache_results:            persist the tuned best config to disk (on by default) so later
                              runs skip the search+compile — see BayesianAutotuner.check_disk_cache"""

    def decorator(fn):
        return BayesianAutotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            n_trials=n_trials,
            max_failures=max_failures,
            n_startup_trials=n_startup_trials,
            gamma=gamma,
            refine=refine,
            max_refine_iters=max_refine_iters,
            log_path=log_path,
            path_anchor_axes=path_anchor_axes,
            finite_check_args=finite_check_args,
            cache_results=cache_results,
            **kwargs,
        )

    return decorator
