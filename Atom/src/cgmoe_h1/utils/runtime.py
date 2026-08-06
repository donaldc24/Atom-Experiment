"""Wall-clock and peak resident-memory measurement helpers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Self

import psutil


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    elapsed_seconds: float
    peak_rss_bytes: int


class RuntimeMonitor:
    """Poll process RSS in a lightweight background thread."""

    def __init__(self, poll_interval: float = 0.05) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.poll_interval = poll_interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._peak = 0
        self.measurement: RuntimeMeasurement | None = None

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.poll_interval):
            self._peak = max(self._peak, self._process.memory_info().rss)

    def __enter__(self) -> Self:
        self._started = time.perf_counter()
        self._peak = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval * 4))
        self._peak = max(self._peak, self._process.memory_info().rss)
        self.measurement = RuntimeMeasurement(
            elapsed_seconds=time.perf_counter() - self._started,
            peak_rss_bytes=self._peak,
        )

    def result(self) -> RuntimeMeasurement:
        if self.measurement is None:
            raise RuntimeError("runtime monitor has not completed")
        return self.measurement


__all__ = ["RuntimeMeasurement", "RuntimeMonitor"]
