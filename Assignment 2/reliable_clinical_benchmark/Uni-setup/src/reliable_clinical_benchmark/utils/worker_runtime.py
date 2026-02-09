"""Worker runtime helpers for generation pipelines."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

JobType = TypeVar("JobType")
ResultType = TypeVar("ResultType")


def is_lmstudio_runner(runner: object) -> bool:
    """Return True when a runner looks like an LM Studio-backed runner."""
    return hasattr(runner, "api_base")


def resolve_worker_count(
    requested_workers: Optional[int],
    runner: object,
    lmstudio_default: int = 4,
    non_lm_default: int = 1,
    log: Optional[logging.Logger] = None,
) -> int:
    """Resolve effective worker count with fail-closed gating."""
    target_log = log or logger
    if requested_workers is None:
        worker_count = lmstudio_default if is_lmstudio_runner(runner) else non_lm_default
    else:
        worker_count = max(1, int(requested_workers))

    if worker_count > 1 and not is_lmstudio_runner(runner):
        target_log.info(
            "Parallel workers >1 are only enabled for LM Studio runners. Falling back to 1 worker."
        )
        return 1

    return max(1, worker_count)


def append_jsonl_with_retry(
    cache_path: Path,
    entry: Dict,
    max_attempts: int = 3,
    base_sleep_seconds: float = 0.2,
    log: Optional[logging.Logger] = None,
) -> bool:
    """Append one JSONL entry with bounded retry for transient filesystem failures."""
    target_log = log or logger
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except OSError as error:
            target_log.warning(
                "JSONL write failed for %s (attempt %d/%d): %s",
                entry.get("id") or entry.get("case_id") or "<unknown>",
                attempt,
                max_attempts,
                error,
            )
            time.sleep(base_sleep_seconds * attempt)

    return False


def iter_threaded_results(
    jobs: Sequence[JobType],
    worker_count: int,
    worker_fn: Callable[[JobType], ResultType],
    progress_interval_seconds: int = 10,
    progress_label: str = "jobs",
    log: Optional[logging.Logger] = None,
) -> Iterator[Tuple[JobType, ResultType]]:
    """Yield completed job results using sequential or threaded execution."""
    target_log = log or logger
    total_jobs = len(jobs)
    if total_jobs == 0:
        return

    if worker_count <= 1:
        for job in jobs:
            yield job, worker_fn(job)
        return

    heartbeat_seconds = max(1, int(progress_interval_seconds))
    completed_count = 0
    pending_iter = iter(jobs)
    future_to_job: Dict = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for _ in range(min(worker_count, total_jobs)):
            next_job = next(pending_iter, None)
            if next_job is None:
                break
            submitted = executor.submit(worker_fn, next_job)
            future_to_job[submitted] = next_job

        while future_to_job:
            finished_now, _ = wait(
                set(future_to_job.keys()),
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not finished_now:
                in_flight = len(future_to_job)
                queued = max(0, total_jobs - completed_count - in_flight)
                target_log.info(
                    "[progress:%s] completed=%d/%d in_flight=%d queued=%d",
                    progress_label,
                    completed_count,
                    total_jobs,
                    in_flight,
                    queued,
                )
                continue

            for finished in finished_now:
                job = future_to_job.pop(finished)
                result = finished.result()
                completed_count += 1
                yield job, result

                next_job = next(pending_iter, None)
                if next_job is not None:
                    submitted = executor.submit(worker_fn, next_job)
                    future_to_job[submitted] = next_job
