"""Deployment-oriented resource measurements shared by the VM experiments."""
"""from __future__ import annotations"""

import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError as exc:  # fail clearly rather than silently reporting invalid memory data
    raise ImportError(
        "psutil is required for peak-memory measurements. Install it with: pip install psutil"
    ) from exc


class PeakRSSSampler:
    """Poll the current process RSS and report peak and incremental peak memory."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self.process.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRSSSampler":
        self.baseline_bytes = self.process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self.interval_s))
        rss = self.process.memory_info().rss
        self.peak_bytes = max(self.peak_bytes, rss)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 ** 2)

    @property
    def incremental_peak_mb(self) -> float:
        return max(0, self.peak_bytes - self.baseline_bytes) / (1024 ** 2)


def measure_call(function: Callable[[], Any], interval_s: float = 0.02) -> Tuple[Any, Dict[str, float]]:
    """Execute a callable and measure wall-clock time and process RSS."""
    with PeakRSSSampler(interval_s=interval_s) as memory:
        start = time.perf_counter()
        result = function()
        elapsed_s = time.perf_counter() - start
    return result, {
        "elapsed_s": elapsed_s,
        "peak_rss_mb": memory.peak_mb,
        "incremental_peak_rss_mb": memory.incremental_peak_mb,
    }


def benchmark_inference(
    predict_batch: Callable[[np.ndarray], Any],
    X: np.ndarray,
    repetitions: int = 20,
    warmup_runs: int = 3,
    single_sample_repetitions: int = 200,
) -> Dict[str, float]:
    """
    Measure end-to-end inference supplied by ``predict_batch``.

    ``predict_batch`` should include every transformation required at deployment
    (e.g., scaling and LDA) and the final prediction call.
    """
    if len(X) == 0:
        raise ValueError("Inference benchmarking requires at least one test sample.")

    for _ in range(warmup_runs):
        predict_batch(X[: min(len(X), 32)])

    batch_times = []
    with PeakRSSSampler() as batch_memory:
        for _ in range(repetitions):
            start = time.perf_counter()
            predict_batch(X)
            batch_times.append(time.perf_counter() - start)

    batch_times_arr = np.asarray(batch_times, dtype=float)
    per_sample_ms = batch_times_arr * 1000.0 / len(X)

    n_single = int(single_sample_repetitions)
    single_times_ms = []
    with PeakRSSSampler() as single_memory:
        for i in range(n_single):
            sample = X[i % len(X): (i % len(X)) + 1]
            start = time.perf_counter()
            predict_batch(sample)
            single_times_ms.append((time.perf_counter() - start) * 1000.0)

    single_arr = np.asarray(single_times_ms, dtype=float)
    mean_batch_s = float(batch_times_arr.mean())

    return {
        "inference_batch_samples": int(len(X)),
        "inference_batch_repetitions": int(repetitions),
        "inference_single_repetitions": int(n_single),
        "batch_time_mean_s": mean_batch_s,
        "batch_time_std_s": float(batch_times_arr.std(ddof=1)) if repetitions > 1 else 0.0,
        "batch_latency_per_sample_mean_ms": float(per_sample_ms.mean()),
        "batch_latency_per_sample_std_ms": float(per_sample_ms.std(ddof=1)) if repetitions > 1 else 0.0,
        "batch_throughput_samples_s": float(len(X) / mean_batch_s),
        "single_latency_mean_ms": float(single_arr.mean()),
        "single_latency_median_ms": float(np.median(single_arr)),
        "single_latency_std_ms": float(single_arr.std(ddof=1)) if n_single > 1 else 0.0,
        "single_latency_p95_ms": float(np.percentile(single_arr, 95)),
        "inference_peak_rss_mb": max(batch_memory.peak_mb, single_memory.peak_mb),
        "inference_incremental_peak_rss_mb": max(
            batch_memory.incremental_peak_mb,
            single_memory.incremental_peak_mb,
        ),
    }


def total_file_size_bytes(paths: Iterable[str]) -> int:
    total = 0
    for path in paths:
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            total += file_path.stat().st_size
    return total


def base_run_metadata(model_name: str, platform_name: str, random_state: int) -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "platform": platform_name,
        "random_state": random_state,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "system_ram_mb": vm.total / (1024 ** 2),
    }


def save_metrics(
    metrics: Dict[str, Any],
    csv_path: str = "deployment_metrics_vm_virtualized.csv",
    json_path: Optional[str] = None,
) -> None:
    """Append one run to a common CSV and save the same record as JSON."""
    row = pd.DataFrame([metrics])
    output = Path(csv_path)
    if output.exists():
        existing = pd.read_csv(output)
        columns = list(existing.columns)
        columns.extend(column for column in row.columns if column not in columns)
        existing = existing.reindex(columns=columns)
        row = row.reindex(columns=columns)
        pd.concat([existing, row], ignore_index=True).to_csv(output, index=False)
    else:
        row.to_csv(output, index=False)

    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True, default=str)

    print(f"Deployment metrics appended to '{csv_path}'")
    if json_path is not None:
        print(f"Deployment metrics saved to '{json_path}'")
