"""Timeout-safe grayscale image reader with a persistent worker process."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from typing import Optional

import numpy as np

from src.data.image_validation import read_grayscale_image

_READY_JOB_ID = "__timed_image_reader_ready__"


def _image_reader_loop(request_queue, response_queue) -> None:
    response_queue.put((_READY_JOB_ID, True))
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
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self._ctx = mp.get_context(start_method)
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._next_job_id = 0
        self._pending: dict[int, Optional[np.ndarray]] = {}
        self._worker_ready = False
        self._use_direct_reads = False
        self.last_status = "not_started"

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
        self._worker_ready = False

    def _start_worker(self) -> None:
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_image_reader_loop,
            args=(self._request_queue, self._response_queue),
        )
        self._process.daemon = True
        self._worker_ready = False
        self._process.start()

    def _restart_worker(self) -> None:
        self.close()
        self._start_worker()

    def read(self, path: Path | str) -> Optional[np.ndarray]:
        if self.timeout_seconds <= 0 or self._use_direct_reads:
            image = read_grayscale_image(path)
            self.last_status = "success" if image is not None else "decode_failed"
            return image

        if self._process is None or not self._process.is_alive():
            self._start_worker()

        job_id = self._next_job_id
        self._next_job_id += 1
        self._request_queue.put((job_id, str(path)))
        deadline = time.monotonic() + self.timeout_seconds

        while True:
            if job_id in self._pending:
                image = self._pending.pop(job_id)
                self.last_status = "success" if image is not None else "decode_failed"
                return image

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._restart_worker()
                self.last_status = "timeout"
                return None

            poll_timeout = min(remaining, 0.1)
            try:
                result_job_id, image = self._response_queue.get(timeout=poll_timeout)
            except Empty:
                if self._process is not None and not self._process.is_alive():
                    worker_was_ready = self._worker_ready
                    self.close()
                    if not worker_was_ready:
                        self._use_direct_reads = True
                        image = read_grayscale_image(path)
                        self.last_status = (
                            "worker_start_failed_direct_success"
                            if image is not None
                            else "worker_start_failed_decode_failed"
                        )
                        return image
                    self._start_worker()
                    self.last_status = "worker_died"
                    return None
                continue

            if result_job_id == _READY_JOB_ID:
                self._worker_ready = True
                continue
            if result_job_id == job_id:
                self.last_status = "success" if image is not None else "decode_failed"
                return image
            self._pending[result_job_id] = image
