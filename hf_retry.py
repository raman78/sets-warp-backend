"""
hf_retry.py — exponential backoff specifically for HF Hub 429 throttling
========================================================================

HF Cloudfront rate-limit windows are typically 60-120 s. Library defaults
(huggingface_hub's built-in retry: 1, 2, 4, 8, 8 s) and our previous
hf_clone retry (1, 2, 4, 8 s) cap out *inside* the rate-limit window, so
they always exhaust before the window clears.

This helper waits long enough to actually outlive the throttle:
    schedule = 60, 120, 240, 480, 480 s  (±15 s jitter)
    total budget = ~24 min across 5 attempts.

Only 429 is retried. Other exceptions propagate immediately — retrying a
bad token or a missing repo is pure latency.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Callable, TypeVar

T = TypeVar('T')

_DELAYS_S = (60.0, 120.0, 240.0, 480.0, 480.0)
_JITTER_S = 15.0


def _is_429(exc: BaseException) -> bool:
    """Best-effort: did this exception come from an HF 429 response?"""
    resp = getattr(exc, 'response', None)
    if resp is not None and getattr(resp, 'status_code', None) == 429:
        return True
    # Fallback for wrapped / chained exceptions and tools that surface 429
    # only in the error text (e.g. git stderr).
    return '429' in str(exc)


def retry_on_429(
    fn:    Callable[[], T],
    *,
    label: str = 'hf-call',
) -> T:
    """
    Run fn() with exponential backoff for HF 429 responses.

    Non-429 exceptions propagate on the first try.
    Returns fn()'s result on success; re-raises the last exception on failure.
    """
    last_exc: BaseException | None = None
    for attempt, base in enumerate(_DELAYS_S):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_429(e) or attempt == len(_DELAYS_S) - 1:
                raise
            delay = max(1.0, base + random.uniform(-_JITTER_S, _JITTER_S))
            print(
                f'  {label}: HF 429 (attempt {attempt + 1}/{len(_DELAYS_S)}) '
                f'— sleeping {delay:.0f}s',
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    # Defensive: loop guarantees one of return / raise.
    assert last_exc is not None
    raise last_exc
