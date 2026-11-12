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

"""Measured choice between alternative launch paths of a host function (``path_autotune``): what a fixed size
threshold would otherwise decide, tuned and cached per key like a kernel config."""

import contextvars
import functools
import hashlib
import inspect
import json
import logging
from pathlib import Path

import torch
from triton.runtime.autotuner import get_cache_invalidating_env_vars, get_cache_manager, triton_key

from .autotune_config import load_autotune_config
from .bayesian_autotuner import _TUNER_SOURCE_HASH, _tuning_lock
from .bench import do_bench
from .compat import launches_skipped

logger = logging.getLogger(__name__)

# The timing and lock implementations alongside this file's own.
_PATH_TUNER_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes() + _TUNER_SOURCE_HASH.encode()).hexdigest()


class PathAutotuner:
    """The path choices of one function (``path_autotune``): for each key, which of its alternative launch paths
    runs, measured the way ``BayesianAutotuner`` measures a kernel config. ``cache`` holds the decisions. A key seen
    for the first time re-runs the whole call once per offered path, with that path pinned, times each run as a
    CUDA-graph replay and keeps the fastest, also on disk under the cross-rank tuning lock."""

    def __init__(self, fn, paths):
        self.fn = fn
        self.name = fn.__name__
        self.paths = tuple(paths)
        self.cache: dict = {}
        self.forced = load_autotune_config().get("force", {}).get(self.name)
        assert self.forced is None or self.forced in self.paths, (
            f"force {self.name}={self.forced!r}: not one of {self.paths}"
        )
        self._call = contextvars.ContextVar(f"{self.name}_call", default=None)  # the arguments of the running call
        self._pinned = contextvars.ContextVar(f"{self.name}_pinned", default=None)  # the path being timed
        self._tuning = contextvars.ContextVar(f"{self.name}_tuning", default=False)  # inside a timing run
        # the path bodies decide the timings: their source keys the disk cache along with the tuner's
        self._source_hash = hashlib.sha256(
            (inspect.getsource(inspect.unwrap(fn)) + _PATH_TUNER_SOURCE_HASH).encode("utf-8")
        ).hexdigest()

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():  # dynamo traces the function itself; ``choose`` reads the cache only
            return self.fn(*args, **kwargs)
        token = self._call.set((args, kwargs))
        try:
            return self.fn(*args, **kwargs)
        finally:
            self._call.reset(token)

    def choose(self, key: tuple, offered: tuple | None = None) -> str:
        """The path the running call takes for ``key``, among ``offered`` (default: every path), the paths valid for
        it. A function with several decisions (e.g. a cuBLAS route, then the activation quantization) offers each its
        own paths. A single one needs no decision; where nothing can be timed (CUDA-graph capture, the fake-tensor
        pass of a compiled caller, another decision's timing run) an undecided key takes the first offered."""
        offered = self.paths if offered is None else tuple(offered)
        if len(offered) == 1:
            return offered[0]
        if self.forced in offered:
            return self.forced
        key = key + offered
        if torch.compiler.is_compiling():  # a traced caller (e.g. a MoE forward): the tuned decision or the default
            return self.cache.get(key, offered[0])
        pinned = self._pinned.get()
        if pinned in offered:  # the decision being timed; another decision of the call resolves as usual
            return pinned
        path = self.cache.get(key)
        if path is None:
            if launches_skipped() or torch.cuda.is_current_stream_capturing() or self._tuning.get():
                return offered[0]
            path = self.cache[key] = self._tune(key, offered)
        return path

    def _run(self, path, args, kwargs):
        token, tuning = self._pinned.set(path), self._tuning.set(True)
        try:
            return self.fn(*args, **kwargs)
        finally:
            self._pinned.reset(token)
            self._tuning.reset(tuning)

    def _tune(self, key, offered):
        args, kwargs = self._call.get()
        cache_key = hashlib.sha256("-".join([
            triton_key(), self._source_hash, str(sorted(get_cache_invalidating_env_vars().items())),
            torch.cuda.get_device_name(), str(key),
        ]).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{self.name[:150]}.path_autotune.json"

        def load():
            stored = cache.get_file(file_name)
            try:
                best = json.loads(Path(stored).read_text())["best"] if stored else None
            except Exception:
                return None  # corrupt or stale: tune again
            return best if best in offered else None

        best = load()
        if best is not None:
            return best
        with _tuning_lock(cache, file_name):
            best = load()  # another rank tuned this key while we waited
            if best is not None:
                return best
            timings = {}
            for path in offered:
                self._run(path, args, kwargs)  # tunes the path's own kernels before it is timed
                # graph replay: eager timing at decode sizes measures host dispatch, not the path (FP8 rowwise
                # N=7168 K=5120 M=1, B200: cuBLAS 62 vs Triton 102us eager, 11.8 vs 8.7us replayed)
                timings[path] = do_bench(
                    functools.partial(self._run, path, args, kwargs), quantiles=(0.5,), cudagraph=True, rep=100
                )
            best = min(timings, key=timings.get)
            logger.info("[autotune] %s %s: %s -> %s", self.name, key,
                        {p: f"{t * 1e3:.1f}us" for p, t in timings.items()}, best)
            try:
                cache.put(json.dumps({"best": best, "timings_ms": timings}), file_name, binary=False)
            except Exception:
                pass
        return best


def path_autotune(paths):
    """Decorator turning a size-threshold branch into a measured choice. At the branch the function asks
    ``fn.paths.choose(key, offered)`` for the path to take (``key``: what its kernels tune on, ``offered``: the paths
    valid for the call) and runs it, as with any ``if``. On an untuned key the choice re-runs the call once per
    offered path and keeps the fastest. ``paths`` lists every path in order; the first is the default where timing
    is impossible. FINEGRAINED_AUTOTUNE_CONFIG ``{"force": {"<function name>": "<path>"}}`` pins a path where it is
    offered."""

    def decorator(fn):
        tuner = PathAutotuner(fn, paths)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return tuner(*args, **kwargs)

        wrapper.paths = tuner
        return wrapper

    return decorator
