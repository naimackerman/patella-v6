"""Timeout-safe grayscale image reader with a persistent worker process."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from typing import Optional

import numpy as np

from src.data.image_validation import read_grayscale_image


def _image_reader_loop(request_queue, response_queue) -> None:
    while True:
        item = request_queue.get()
        if item is None:
            break
        job_id, path = item
        try:
            image = read_grayscale_image(path)
        except Exception:
            image = None
        response_queue.put((job_id, image))


class TimedImageReader:
    """Read grayscale images with per-call timeouts and worker reuse.

    A single persistent subprocess handles image reads. If one read stalls
    beyond the configured timeout, the worker is restarted and the current
    image returns `None`.
    """

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._next_job_id = 0
        self._pending: dict[int, Optional[np.ndarray]] = {}

    def __enter__(self) -> "TimedImageReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._request_queue.put(None)
        except Exception:
            pass
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._pending.clear()

    def _start_worker(self) -> None:
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_image_reader_loop,
            args=(self._request_queue, self._response_queue),
        )
        self._process.daemon = True
        self._process.start()

    def _restart_worker(self) -> None:
        self.close()
        self._start_worker()

    def read(self, path: Path | str) -> Optional[np.ndarray]:
        if self.timeout_seconds <= 0:
            return read_grayscale_image(path)

        if self._process is None or not self._process.is_alive():
            self._start_worker()

        job_id = self._next_job_id
        self._next_job_id += 1
        self._request_queue.put((job_id, str(path)))
        deadline = time.monotonic() + self.timeout_seconds

        while True:
            if job_id in self._pending:
                return self._pending.pop(job_id)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._restart_worker()
                return None

            try:
                result_job_id, image = self._response_queue.get(timeout=remaining)
            except Empty:
                self._restart_worker()
                return None

            if result_job_id == job_id:
                return image
            self._pending[result_job_id] = image
