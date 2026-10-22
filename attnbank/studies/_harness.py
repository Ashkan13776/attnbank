"""Sweep runner, result files and summaries for the eight studies in ``STUDIES.md``.

A study is a grid of ``(method, config, data-slice)`` points, each producing a few
numbers. This module owns everything that is the same for all eight:

* :class:`Sweep` — runs the grid, prints progress, and keeps a **resumable** on-disk
  cache so a long sweep that dies continues where it stopped.
* :func:`save_results` / :func:`load_results` — ``lib/results/<name>.json``
  holding the *full* sweep plus the provenance STUDIES.md demands: git commit, seed,
  configs, model/data provenance, timestamp, device.
* :func:`summarize` — median and interquartile band, because every figure shows a
  band and no reported number is a bare mean.
* GPU plumbing — float64 error measurement by default, device selection, chunking
  helpers that keep the large-``n`` sweeps inside a memory budget, and OOM recovery.

Typical use::

    from attnbank.studies import _harness as H

    sweep = H.Sweep("S1_matrix_vs_output", seed=0)
    jobs = H.grid(method=["topk", "performer"], strength=[4, 16, 64])
    records = sweep.run(jobs, measure)        # measure(job, ctx) -> dict of metrics
    H.save_results("S1_matrix_vs_output", records, meta=sweep.meta())
    rows = H.summarize(records, by=("method", "strength"), value="output_error")
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "Sweep", "SweepContext", "grid", "zip_grid", "save_results", "load_results",
    "results_dir", "summarize", "to_rows", "provenance", "git_commit",
    "default_device", "device_info", "set_seed", "generator", "chunk_sizes",
    "iter_chunks", "memory_budget", "free_memory",
]

#: every error measurement in STUDIES.md is float64
DEFAULT_DTYPE = torch.float64


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    """The Python project directory (the one holding pyproject.toml).

    Derived from this file's location rather than hardcoded. It was spelled as the
    literal "attnbank" until the project moved to lib/, at which point the literal
    silently created a stray directory at the repository root -- which then shadowed
    the installed package, because it shared the package's name.
    """
    return Path(__file__).resolve().parents[2]


def git_commit(short: bool = False) -> str:
    """The commit the results were produced at, or ``"unknown"``. Read-only."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(args, cwd=_repo_root(), capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=_repo_root(),
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip())
    except Exception:
        return None


def device_info(device: torch.device | str | None = None) -> dict:
    device = torch.device(device) if device is not None else default_device()
    info = {"device": str(device), "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform()}
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        info.update({"gpu_name": p.name, "gpu_total_gb": round(p.total_memory / 2**30, 1),
                     "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
                     "capability": f"{p.major}.{p.minor}"})
    return info


def provenance(seed: int | None = None, device=None, **extra) -> dict:
    """The provenance block every ``results/*.json`` carries."""
    meta = {
        "git_commit": git_commit(), "git_short": git_commit(short=True),
        "git_dirty": git_dirty(), "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dtype": str(DEFAULT_DTYPE).replace("torch.", ""),
        **device_info(device),
    }
    meta.update(extra)
    return meta


# --------------------------------------------------------------------------------------
# devices, seeds, memory
# --------------------------------------------------------------------------------------


def default_device() -> torch.device:
    """The CUDA device with the most free memory, else CPU. ``ATTNBANK_DEVICE`` overrides."""
    env = os.environ.get("ATTNBANK_DEVICE")
    if env:
        return torch.device(env)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            free = 0
        if free > best_free:
            best, best_free = i, free
    return torch.device(f"cuda:{best}")


def free_memory(device: torch.device | str | None = None) -> int:
    """Free bytes on ``device`` (a large constant on CPU, where we do not police it)."""
    device = torch.device(device) if device is not None else default_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return 8 * 2**30
    free, _ = torch.cuda.mem_get_info(device.index or 0)
    return int(free)


def memory_budget(device=None, fraction: float = 0.4) -> int:
    """Bytes a single sweep step may use. Deliberately well under the card."""
    return int(free_memory(device) * fraction)


def chunk_sizes(total: int, bytes_per_item: int, device=None, fraction: float = 0.4,
                minimum: int = 1) -> int:
    """How many items of ``bytes_per_item`` fit in the budget (>= ``minimum``).

    The large-``n`` sweeps form ``[B,H,m,n]`` matrices; use this to pick how many
    (passage, head) units to process at once instead of guessing.
    """
    if bytes_per_item <= 0:
        return total
    k = int(memory_budget(device, fraction) // bytes_per_item)
    return min(max(minimum, k), total)


def iter_chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def set_seed(seed: int) -> None:
    """Seed the global RNGs. Prefer :func:`generator` — methods take a ``torch.Generator``."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


# --------------------------------------------------------------------------------------
# job grids
# --------------------------------------------------------------------------------------


def grid(**axes: Iterable) -> list[dict]:
    """Cartesian product of named axes, in a stable order.

    ``grid(method=["topk"], k=[4, 8])`` -> ``[{"method": "topk", "k": 4}, ...]``
    """
    keys = list(axes)
    vals = [list(axes[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def zip_grid(**axes: Iterable) -> list[dict]:
    """Like :func:`grid` but zips the axes (equal length) instead of crossing them."""
    keys = list(axes)
    vals = [list(axes[k]) for k in keys]
    if len({len(v) for v in vals}) != 1:
        raise ValueError("zip_grid needs axes of equal length")
    return [dict(zip(keys, combo)) for combo in zip(*vals)]


def _canon(obj: Any) -> Any:
    """JSON-safe canonical form of a job value (for hashing and for the results file)."""
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _canon(float(obj))
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.numel() <= 16 else f"<tensor{list(obj.shape)}>"
    if isinstance(obj, Mapping):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_canon(x) for x in obj]
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    return str(obj)


def job_key(job: Mapping) -> str:
    return hashlib.sha1(json.dumps(_canon(job), sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------


@dataclass
class SweepContext:
    """Passed to the measurement function as its second argument, if it takes one."""
    device: torch.device
    dtype: torch.dtype
    seed: int
    index: int = 0
    total: int = 0
    sweep: "Sweep | None" = None

    def generator(self, offset: int = 0) -> torch.Generator:
        return generator(self.seed + offset, device="cpu")

    def budget(self, fraction: float = 0.4) -> int:
        return memory_budget(self.device, fraction)


def _sweep_cache_dir() -> Path:
    env = os.environ.get("ATTNBANK_CACHE")
    root = Path(env).expanduser() if env else _project_root() / ".cache"
    return root / "sweeps"


class Sweep:
    """Run a grid of jobs, cache every finished job, resume after a crash.

    ``fn`` is called as ``fn(job)`` or ``fn(job, ctx)`` (whichever it accepts) and
    returns a flat ``dict`` of numbers. Everything it returns, plus the job itself, is
    written to the resume cache immediately, so interrupting the process loses at most
    one job.
    """

    def __init__(self, name: str, *, seed: int = 0, device=None,
                 dtype: torch.dtype = DEFAULT_DTYPE, resume: bool = True,
                 cache_dir: str | Path | None = None, progress: bool = True,
                 progress_every: int = 1, on_error: str = "record",
                 empty_cache_every: int = 32, extra_meta: Mapping | None = None):
        if on_error not in ("record", "raise"):
            raise ValueError("on_error must be 'record' or 'raise'")
        self.name = name
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self.resume = resume
        self.progress = progress
        self.progress_every = max(1, int(progress_every))
        self.on_error = on_error
        self.empty_cache_every = empty_cache_every
        self.extra_meta = dict(extra_meta or {})
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _sweep_cache_dir()
        self.cache_file = self.cache_dir / f"{name}.jsonl"
        self._records: dict[str, dict] = {}
        # Keys this Sweep actually submitted, in submission order. The resume cache may
        # hold records from earlier runs over a different grid; those must not end up in
        # the results file, or an analysis reads one population and annotates it with
        # another. See `records` below.
        self._submitted: dict[str, None] = {}
        self._loaded_from_cache = 0
        self._t_start: float | None = None
        if self.resume:
            self._load_cache()

    # -- resume cache ----------------------------------------------------------------
    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            return
        for line in self.cache_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn last line from a kill: drop it, recompute the job
            if rec.get("_key"):
                self._records[rec["_key"]] = rec
        self._loaded_from_cache = len(self._records)

    def _append_cache(self, rec: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def clear_cache(self) -> None:
        self._records.clear()
        self._loaded_from_cache = 0
        if self.cache_file.exists():
            self.cache_file.unlink()

    # -- running ---------------------------------------------------------------------
    def run(self, jobs: Sequence[Mapping], fn: Callable, *, label: str | None = None
            ) -> list[dict]:
        """Run every job not already in the cache; return records for *all* of them."""
        jobs = [dict(j) for j in jobs]
        takes_ctx = _takes_ctx(fn)
        total = len(jobs)
        done_at_start = sum(1 for j in jobs if job_key(j) in self._records)
        if self._t_start is None:
            self._t_start = time.time()
        if self.progress:
            self._say(f"{self.name}{'/' + label if label else ''}: {total} jobs, "
                      f"{done_at_start} cached, device={self.device}, dtype="
                      f"{str(self.dtype).replace('torch.', '')}")
        out: list[dict] = []
        for i, job in enumerate(jobs):
            key = job_key(job)
            self._submitted[key] = None
            if key in self._records:
                out.append(self._records[key])
                continue
            ctx = SweepContext(device=self.device, dtype=self.dtype, seed=self.seed,
                               index=i, total=total, sweep=self)
            rec = self._run_one(job, fn, ctx, key, takes_ctx)
            self._records[key] = rec
            self._append_cache(rec)
            out.append(rec)
            if self.progress and ((i + 1) % self.progress_every == 0 or i + 1 == total):
                self._report(i + 1, total, rec)
            if (self.empty_cache_every and self.device.type == "cuda"
                    and (i + 1) % self.empty_cache_every == 0):
                torch.cuda.empty_cache()
        return out

    def _run_one(self, job: dict, fn, ctx: SweepContext, key: str, takes_ctx: bool) -> dict:
        t0 = time.time()
        status, error, metrics = "ok", None, {}
        try:
            metrics = fn(job, ctx) if takes_ctx else fn(job)
        except torch.cuda.OutOfMemoryError as e:          # one retry with a clean cache
            torch.cuda.empty_cache()
            try:
                metrics = fn(job, ctx) if takes_ctx else fn(job)
            except Exception as e2:
                status, error, metrics = "oom", f"{type(e2).__name__}: {e2}", {}
                if self.on_error == "raise":
                    raise
        except Exception as e:
            status, error, metrics = "error", f"{type(e).__name__}: {e}", {}
            if self.on_error == "raise":
                raise
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, Mapping):
            raise TypeError(f"{self.name}: the measurement function must return a dict, "
                            f"got {type(metrics).__name__}")
        return {"job": _canon(job), "metrics": _canon(dict(metrics)), "_key": key,
                "_status": status, "_error": error, "_seconds": round(time.time() - t0, 4)}

    # -- reporting -------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def _report(self, done: int, total: int, rec: dict) -> None:
        el = time.time() - (self._t_start or time.time())
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("nan")
        tag = "" if rec["_status"] == "ok" else f"  [{rec['_status']}] {rec['_error']}"
        self._say(f"  [{done}/{total}] {_short(rec['job'])}  "
                  f"{rec['_seconds']:.2f}s  eta {eta/60:.1f}m{tag}")

    # -- results ---------------------------------------------------------------------
    @property
    def records(self) -> list[dict]:
        """Records for the jobs this Sweep submitted, in submission order.

        Deliberately *not* the whole resume cache. A cache accumulates across runs, so
        if the grid changes between runs the leftovers would silently join the results
        file and shift pooled statistics. Use `all_records` to inspect the cache itself.
        """
        if not self._submitted:
            return list(self._records.values())
        return [self._records[k] for k in self._submitted if k in self._records]

    @property
    def all_records(self) -> list[dict]:
        """Every record in the resume cache, including jobs this run did not submit."""
        return list(self._records.values())

    def failures(self) -> list[dict]:
        return [r for r in self.records if r["_status"] != "ok"]

    def meta(self, **extra) -> dict:
        m = provenance(seed=self.seed, device=self.device, sweep=self.name,
                       n_records=len(self._records),
                       n_failed=len(self.failures()),
                       resumed_from_cache=self._loaded_from_cache,
                       cache_file=str(self.cache_file),
                       elapsed_seconds=round(time.time() - (self._t_start or time.time()), 1))
        m.update(self.extra_meta)
        m.update(extra)
        return m

    def save(self, name: str | None = None, records: Sequence[Mapping] | None = None,
             **extra_meta) -> Path:
        return save_results(name or self.name,
                            self.records if records is None else records,
                            self.meta(**extra_meta))


def _takes_ctx(fn: Callable) -> bool:
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(params) >= 2
    except (TypeError, ValueError):
        return False


def _short(job: Mapping, n: int = 76) -> str:
    s = ", ".join(f"{k}={v}" for k, v in job.items())
    return s if len(s) <= n else s[:n - 1] + "…"


# --------------------------------------------------------------------------------------
# result files
# --------------------------------------------------------------------------------------


def results_dir() -> Path:
    env = os.environ.get("ATTNBANK_RESULTS")
    return Path(env).expanduser() if env else _project_root() / "results"


def save_results(name: str, records: Sequence[Mapping], meta: Mapping | None = None,
                 out_dir: str | Path | None = None) -> Path:
    """Write ``results/<name>.json``: the **full** sweep plus provenance.

    The file is the only place the paper's numbers may come from, so it carries the
    git commit, the seed, every config, the data provenance, the timestamp and the
    device — see STUDIES.md "Reporting rules".
    """
    out = Path(out_dir) if out_dir is not None else results_dir()
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    for k, v in provenance().items():
        meta.setdefault(k, v)
    meta.setdefault("name", name)
    meta["n_records"] = len(records)
    payload = {"name": name, "meta": _canon(meta), "records": [_canon(r) for r in records]}
    path = out / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False))
    tmp.replace(path)
    return path


def load_results(name: str, out_dir: str | Path | None = None) -> dict:
    """Load ``results/<name>.json`` -> ``{"name", "meta", "records"}``."""
    out = Path(out_dir) if out_dir is not None else results_dir()
    path = out / f"{name}.json" if not str(name).endswith(".json") else Path(name)
    if not path.exists():
        raise FileNotFoundError(f"no results file at {path}")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------
# summaries: median and IQR, never a bare mean
# --------------------------------------------------------------------------------------


def _lookup(record: Mapping, key: str, default=None):
    """Find ``key`` in a record: dotted path, then top level, then job, then metrics."""
    if "." in key:
        cur: Any = record
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
    if key in record and key not in ("job", "metrics"):
        return record[key]
    for section in ("job", "metrics"):
        sub = record.get(section)
        if isinstance(sub, Mapping) and key in sub:
            return sub[key]
    return default


def to_rows(records: Sequence[Mapping]) -> list[dict]:
    """Flatten records to ``{**job, **metrics, _status, _seconds}`` for plotting."""
    rows = []
    for r in records:
        row = {}
        row.update(r.get("job", {}) if isinstance(r.get("job"), Mapping) else {})
        row.update(r.get("metrics", {}) if isinstance(r.get("metrics"), Mapping) else {})
        for k in ("_status", "_seconds", "_error", "_key"):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    return rows


def _percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def summarize(records: Sequence[Mapping], by: Sequence[str] | str | None = None,
              value: str | Sequence[str] = "", *, ok_only: bool = True,
              quantiles: tuple[float, float] = (25.0, 75.0),
              min_count: int = 1) -> list[dict]:
    """Group records and report **median + interquartile band** per group.

    Returns one row per group: the grouping keys, then ``n``, ``median``, ``q1``,
    ``q3``, ``iqr``, ``mean``, ``std``, ``min``, ``max`` for each requested value
    (suffixed with the value name when more than one is requested).

    ``value`` may name a job key, a metric key, or a dotted path. Non-finite values
    are dropped and counted in ``n_dropped``.
    """
    if not value:
        raise ValueError("summarize(..., value=...) is required")
    values = [value] if isinstance(value, str) else list(value)
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    multi = len(values) > 1

    groups: dict[tuple, list[Mapping]] = {}
    order: list[tuple] = []
    for r in records:
        if ok_only and r.get("_status", "ok") != "ok":
            continue
        gk = tuple(_canon(_lookup(r, b)) for b in by)
        gk = tuple(json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x
                   for x in gk)
        if gk not in groups:
            groups[gk] = []
            order.append(gk)
        groups[gk].append(r)

    rows = []
    for gk in order:
        recs = groups[gk]
        row: dict[str, Any] = {b: g for b, g in zip(by, gk)}
        keep = True
        for v in values:
            raw = [_lookup(r, v) for r in recs]
            arr = np.asarray([float(x) for x in raw
                              if isinstance(x, (int, float, np.number))
                              and np.isfinite(float(x))], dtype=np.float64)
            sfx = f"_{v.split('.')[-1]}" if multi else ""
            if arr.size < min_count:
                keep = False
            stats = {f"n{sfx}": int(arr.size),
                     f"n_dropped{sfx}": int(len(raw) - arr.size)}
            if arr.size:
                stats.update({
                    f"median{sfx}": float(np.median(arr)),
                    f"q1{sfx}": _percentile(arr, quantiles[0]),
                    f"q3{sfx}": _percentile(arr, quantiles[1]),
                    f"iqr{sfx}": _percentile(arr, quantiles[1]) - _percentile(arr, quantiles[0]),
                    f"mean{sfx}": float(arr.mean()),
                    f"std{sfx}": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    f"min{sfx}": float(arr.min()), f"max{sfx}": float(arr.max()),
                })
            else:
                stats.update({f"{k}{sfx}": float("nan")
                              for k in ("median", "q1", "q3", "iqr", "mean", "std",
                                        "min", "max")})
            row.update(stats)
        if keep:
            rows.append(row)
    return rows
