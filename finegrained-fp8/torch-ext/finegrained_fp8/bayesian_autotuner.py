"""Bayesian alternative to ``@triton.autotune``. Subclasses Triton's
``Autotuner`` and swaps the exhaustive bench-all-configs inner loop for
Optuna TPE over the discrete config-index space, followed by coordinate-
descent local refinement from the TPE best.

Warm-starts each new key's first trial with the most recently cached key's
best config.

Fallbacks to stock exhaustive when Optuna is missing or the grid is
smaller than ``n_trials``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from triton.runtime.autotuner import Autotuner, Config

try:
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


class BayesianAutotuner(Autotuner):
    """Drop-in replacement for ``triton.runtime.autotuner.Autotuner`` that
    benches ~``n_trials`` configs per key via Optuna TPE + coordinate-descent
    refinement, instead of the full grid."""

    def __init__(
        self,
        *args,
        n_trials: int = 80,
        n_startup_trials: int = 10,
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
        # Degenerate grid → defer to parent (stock exhaustive).
        if (
            len(self.configs) <= 1
            or not _HAS_OPTUNA
            or self.n_trials >= len(self.configs)
        ):
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

        def objective(trial):
            return bench_idx(trial.suggest_int("config_idx", 0, len(configs) - 1))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=0,
            ),
        )

        # Warm-start: seed TPE's first trial with the most recent cached key's
        # best config. Nearby workloads tend to share tile-shape preferences.
        warm_idx = self._warm_start_index(configs)
        if warm_idx is not None:
            study.enqueue_trial({"config_idx": warm_idx})

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

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
    n_startup_trials: int = 10,
    refine: bool = True,
    max_refine_iters: int = 5,
    log_path: str | None = None,
    reset_to_zero=None,
    restore_value=None,
    **kwargs,
):
    """Decorator mirroring ``@triton.autotune``. Extra kwargs:
    n_trials, n_startup_trials: TPE budget
    refine, max_refine_iters:   coordinate-descent after TPE
    log_path:                   JSONL of benched configs (or FINEGRAINED_AUTOTUNE_LOG)"""

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
