"""Shared Retry-After handling and jittered backoff for LLM clients."""

from __future__ import annotations

import random
import time
from email.utils import parsedate_to_datetime

import httpx

# Retry these 5xx statuses with backoff; handle 429 separately using Retry-After.
RETRYABLE_SERVER_STATUS: tuple[int, ...] = (500, 502, 503, 504)
RETRY_BACKOFF_CAP_S = 60.0


def backoff_with_jitter(base: float, attempt: int, cap: float = RETRY_BACKOFF_CAP_S) -> float:
    """Return capped exponential backoff, base * 2^attempt, with 0..25% jitter. Jitter prevents
    concurrent callers from retrying in lockstep and repeatedly hitting rate limits.
    """
    delay = min(base * (2 ** attempt), cap)
    return delay + random.uniform(0.0, delay * 0.25)


def parse_retry_after(resp: httpx.Response) -> float | None:
    """Parse Retry-After as seconds, accepting either delta-seconds or an HTTP date. Return None
    when absent or invalid.
    """
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))  # delta-seconds
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)  # HTTP-date
        if dt is not None:
            return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        pass
    return None


def retry_after_delay(
    resp: httpx.Response, base: float, attempt: int, cap: float = RETRY_BACKOFF_CAP_S
) -> float:
    """Use capped server Retry-After for a 429 response, otherwise jittered exponential backoff."""
    ra = parse_retry_after(resp)
    if ra is not None:
        return min(ra, cap)
    return backoff_with_jitter(base, attempt, cap)
