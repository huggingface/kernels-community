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
import random
import time
from collections import defaultdict
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


class BayesianAutotuner(Autotuner):
    """Drop-in replacement for ``triton.runtime.autotuner.Autotuner`` that
    benches ~``n_trials`` configs per key via TPE Bayesian optimization +
    coordinate-descent refinement, instead of the full grid."""

    def __init__(
        self,
        *args,
        n_trials: int = 80,
        n_startup_trials: int = 12,
        gamma: float = 0.25,
        refine: bool = True,
        max_refine_iters: int = 5,
        log_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fn_name = getattr(self.fn, "__name__", str(self.fn))
        # Bayesian trial budget — the per-decorator default, overridable via the
        # FINEGRAINED_AUTOTUNE_TRIALS env var (quick sweeps / exhaustive runs without touching
        # the decorators; set it >= grid size to fall back to stock exhaustive bench-all).
        self.n_trials = int(os.environ.get("FINEGRAINED_AUTOTUNE_TRIALS") or n_trials)
        self.n_startup_trials = n_startup_trials
        # top fraction of measured configs the TPE treats as "good"
        self.gamma = gamma
        self.refine = refine
        self.max_refine_iters = max_refine_iters
        # JSONL log of every benched (key, config, ms) — set here or via the
        # FINEGRAINED_AUTOTUNE_LOG env var. Analyse offline to prune bad configs.
        self.log_path = log_path or os.environ.get("FINEGRAINED_AUTOTUNE_LOG")

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
        sig = self._compile_signature(config)
        memo = self._failed_compile_memo()
        if sig is not None and sig in memo:
            self._failures.append((config, memo[sig] + "  [memoized]"))
            return [float("inf")] * 3
        try:
            return super()._bench(*args, config=config, **meta)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            self._failures.append((config, err))
            if sig is not None and any(m in err for m in self._COMPILE_FAILURE_MARKS):
                memo[sig] = err
                self._persist_failed_compile_memo()
            return [float("inf")] * 3

    def _compile_signature(self, config) -> str | None:
        """Hash of the compile determinants for one config: the config itself plus the
        constexpr argument VALUES and tensor argument dtypes from this launch (both feed
        specialization — e.g. GATE flips arms, a uint8 A packs the loads). Source/env
        live in the memo FILE's key, not here."""
        nargs = getattr(self, "nargs", None)
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
        self._failures = []
        # Small grid → defer to parent (stock exhaustive bench-all).
        if len(self.configs) <= 1 or self.n_trials >= len(self.configs):
            ret = super().run(*args, **kwargs)
            self._report_bench_failures()
            return ret

        self.nargs = dict(zip(self.arg_names, args))
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in self.arg_names}
        key = [_args[k] for k in self.keys if k in _args]
        key.extend(str(v.dtype) for v in _args.values() if hasattr(v, "dtype"))
        key = tuple(key)

        if key not in self.cache:
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
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
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
        while sum(1 for t in timings.values() if t != float("inf")) < self.n_trials:
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

        self.configs_timings = {configs[i]: t for i, t in timings.items()}
        return configs[min(timings, key=timings.get)]

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

        return [
            sorted(idxs, key=tile_order)[len(idxs) // 2] for idxs in groups.values()
        ]

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
            str(tuning_key),
        ] + [str(c) for c in configs]  # str(Config) is pre_hook-free and process-stable
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.bayes_autotune.json"

        # signature -> live Config (carries the pre_hook); used to re-match the cached winner
        by_sig = {tuple(sorted(c.all_kwargs().items())): c for c in configs}
        path = cache.get_file(file_name)
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                best = by_sig.get(tuple(sorted(data["best"].items())))
                if best is not None:
                    self.cache[tuning_key] = best
                    self.configs_timings = {
                        by_sig[s]: t
                        for kw, t in data["timings"]
                        for s in (tuple(sorted(kw.items())),)
                        if s in by_sig
                    }
                    return True
            except Exception:
                pass  # corrupt/stale cache file → fall through and re-tune

        bench_fn()
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


def bayesian_autotune(
    configs,
    key,
    *,
    n_trials: int = 80,
    n_startup_trials: int = 12,
    gamma: float = 0.25,
    refine: bool = True,
    max_refine_iters: int = 5,
    log_path: str | None = None,
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
            n_startup_trials=n_startup_trials,
            gamma=gamma,
            refine=refine,
            max_refine_iters=max_refine_iters,
            log_path=log_path,
            cache_results=cache_results,
            **kwargs,
        )

    return decorator
