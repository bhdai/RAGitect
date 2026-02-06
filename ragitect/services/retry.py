"""Retry utility with exponential backoff for URL fetching

Implements AC4 (NFR-R3) retry logic:
- Retry up to 3 times on transient failures
- Exponential backoff delays: 1s, 2s, 4s
- Retryable errors: timeout, network errors, HTTP 5xx
- Non-retryable errors: HTTP 4xx (immediate fail)

Usage:
    >>> from ragitect.services.retry import with_retry
    >>> result = await with_retry(fetch_url, "https://example.com")
"""

import asyncio
import logging
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

# Retry configuration constants (AC4)
MAX_RETRIES = 3
INITIAL_DELAY = 1.0  # seconds
BACKOFF_MULTIPLIER = 2

# Retryable exceptions (transient network issues)
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)

# HTTP status codes that should trigger retry
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


async def with_retry(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Execute async function with exponential backoff retry.

    Retries on:
    - Timeout exceptions (httpx.TimeoutException)
    - Network errors (httpx.NetworkError, httpx.ConnectError)
    - HTTP 5xx server errors (500, 502, 503, 504)

    Does NOT retry on:
    - HTTP 4xx client errors (400, 401, 403, 404)
    - Content extraction errors
    - Other application errors

    Args:
        func: Async function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func if successful

    Raises:
        Exception: The last exception if all retries exhausted,
            or immediately for non-retryable errors

    Example:
        >>> async def fetch_data(url: str) -> str:
        ...     async with httpx.AsyncClient() as client:
        ...         response = await client.get(url)
        ...         return response.text
        >>> result = await with_retry(fetch_data, "https://example.com")
    """
    delay = INITIAL_DELAY
    last_exception: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)

        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{MAX_RETRIES} after {delay:.1f}s: "
                    f"{type(e).__name__}: {e}"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_MULTIPLIER
            else:
                logger.error(f"All {MAX_RETRIES} retries exhausted: {e}")
                raise

        except httpx.HTTPStatusError as e:
            # Only retry on 5xx server errors
            if e.response.status_code in RETRYABLE_STATUS_CODES:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} after {delay:.1f}s: "
                        f"HTTP {e.response.status_code}"
                    )
                    await asyncio.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER
                else:
                    logger.error(
                        f"All {MAX_RETRIES} retries exhausted: "
                        f"HTTP {e.response.status_code}"
                    )
                    raise
            else:
                # 4xx errors - don't retry, fail immediately
                logger.warning(
                    f"Non-retryable HTTP error: {e.response.status_code}"
                )
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception

    raise RuntimeError("Unexpected state in retry logic")
