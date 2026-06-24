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
models the good (top-``GAMMA`` by measured time) vs bad per-dimension value densities and
benches the unmeasured config that maximizes ``l(x)/g(x)`` — the Expected-Improvement proxy
— then a coordinate-descent pass polishes the best. Subclasses Triton's ``Autotuner`` and
uses only the Python stdlib (no external optimizer dependency).

Each new key's search is warm-started from the most recently tuned key's best config (nearby
workloads share tile-shape preferences). Grids smaller than ``n_trials`` defer to the stock
exhaustive bench-all.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

from triton.runtime.autotuner import Autotuner, Config

GAMMA = 0.25  # top fraction of measured configs treated as "good" by the TPE


class BayesianAutotuner(Autotuner):
    """Drop-in replacement for ``triton.runtime.autotuner.Autotuner`` that
    benches ~``n_trials`` configs per key via TPE Bayesian optimization +
    coordinate-descent refinement, instead of the full grid."""

    def __init__(
        self,
        *args,
        n_trials: int = 80,
        n_startup_trials: int = 12,
        refine: bool = True,
        max_refine_iters: int = 5,
        log_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.refine = refine
        self.max_refine_iters = max_refine_iters
        # JSONL log of every benched (key, config, ms) — set here or via the
        # FINEGRAINED_AUTOTUNE_LOG env var. Analyse offline to prune bad configs.
        self.log_path = log_path or os.environ.get("FINEGRAINED_AUTOTUNE_LOG")

    def run(self, *args, **kwargs):
        # Small grid → defer to parent (stock exhaustive bench-all).
        if len(self.configs) <= 1 or self.n_trials >= len(self.configs):
            return super().run(*args, **kwargs)

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
                self.bench_time = time.time() - t0
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

        # Seed phase: a few random configs (seeded → deterministic), warm-started from the
        # most recent cached key's best, to give the TPE an initial good/bad split.
        n_startup = max(2, min(self.n_startup_trials, self.n_trials))
        order = list(range(len(configs)))
        random.Random(0).shuffle(order)
        warm_idx = self._warm_start_index(configs)
        if warm_idx is not None:
            order = [warm_idx] + [i for i in order if i != warm_idx]
        for idx in order[:n_startup]:
            bench_idx(idx)

        # TPE: split measured configs into good (top-GAMMA) / bad, build per-dimension value
        # densities for each, and bench the unmeasured config maximizing log l(x) - log g(x)
        # (Expected-Improvement proxy), updating the model after each measurement.
        while len(timings) < self.n_trials:
            ranked = sorted(timings, key=timings.get)
            n_good = max(1, round(GAMMA * len(ranked)))
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
                    lp = (good_c[d][v] + 1.0) / (n_good + V)  # P(v | good), Laplace-smoothed
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
                "t": time.time(),
                "fn": getattr(self.fn, "__name__", str(self.fn)),
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


def bayesian_autotune(
    configs,
    key,
    *,
    n_trials: int = 80,
    n_startup_trials: int = 12,
    refine: bool = True,
    max_refine_iters: int = 5,
    log_path: str | None = None,
    reset_to_zero=None,
    restore_value=None,
    **kwargs,
):
    """Decorator mirroring ``@triton.autotune``. Extra kwargs:
    n_trials:                 total configs benched per key (TPE budget)
    n_startup_trials:         random seed configs before the TPE model kicks in
    refine, max_refine_iters: coordinate-descent refinement after the TPE
    log_path:                 JSONL of benched configs (or FINEGRAINED_AUTOTUNE_LOG)"""

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
            refine=refine,
            max_refine_iters=max_refine_iters,
            log_path=log_path,
            **kwargs,
        )

    return decorator
