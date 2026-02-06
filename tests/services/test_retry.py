"""Unit tests for retry utility with exponential backoff

Tests retry logic (AC4) including:
- Retry on timeout exception (should retry 3 times)
- Retry on 5xx server error (should retry)
- No retry on 4xx client error (immediate fail)
- Exponential backoff delays: 1s, 2s, 4s
- Final failure after 3 retries
"""

from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import httpx
import pytest

from ragitect.services.retry import (
    with_retry,
    MAX_RETRIES,
    INITIAL_DELAY,
    BACKOFF_MULTIPLIER,
    RETRYABLE_EXCEPTIONS,
)


pytestmark = [pytest.mark.asyncio]


class TestRetryOnTimeoutException:
    """Test retry behavior for timeout exceptions (AC4)"""

    async def test_retries_on_timeout_exception(self):
        """Should retry up to 3 times on timeout"""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Connection timed out")

        with pytest.raises(httpx.TimeoutException):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(failing_func)

        assert call_count == MAX_RETRIES

    async def test_succeeds_after_retry(self):
        """Should succeed if function passes after retry"""
        call_count = 0

        async def sometimes_failing():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Temporary failure")
            return "success"

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await with_retry(sometimes_failing)

        assert result == "success"
        assert call_count == 3


class TestRetryOn5xxServerError:
    """Test retry behavior for HTTP 5xx server errors (AC4)"""

    async def test_retries_on_500_error(self):
        """Should retry on HTTP 500 Internal Server Error"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 500

        async def server_error():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(server_error)

        assert call_count == MAX_RETRIES

    async def test_retries_on_502_error(self):
        """Should retry on HTTP 502 Bad Gateway"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 502

        async def bad_gateway():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Bad Gateway",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(bad_gateway)

        assert call_count == MAX_RETRIES

    async def test_retries_on_503_error(self):
        """Should retry on HTTP 503 Service Unavailable"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 503

        async def service_unavailable():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Service Unavailable",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(service_unavailable)

        assert call_count == MAX_RETRIES

    async def test_retries_on_504_error(self):
        """Should retry on HTTP 504 Gateway Timeout"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 504

        async def gateway_timeout():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Gateway Timeout",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(gateway_timeout)

        assert call_count == MAX_RETRIES


class TestNoRetryOn4xxClientError:
    """Test no retry on HTTP 4xx client errors (AC4)"""

    async def test_no_retry_on_400_error(self):
        """Should NOT retry on HTTP 400 Bad Request - immediate fail"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 400

        async def bad_request():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Bad Request",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(bad_request)

        # Should only be called once (no retries)
        assert call_count == 1

    async def test_no_retry_on_401_error(self):
        """Should NOT retry on HTTP 401 Unauthorized"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 401

        async def unauthorized():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Unauthorized",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(unauthorized)

        assert call_count == 1

    async def test_no_retry_on_403_error(self):
        """Should NOT retry on HTTP 403 Forbidden"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 403

        async def forbidden():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Forbidden",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(forbidden)

        assert call_count == 1

    async def test_no_retry_on_404_error(self):
        """Should NOT retry on HTTP 404 Not Found"""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 404

        async def not_found():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Not Found",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(not_found)

        assert call_count == 1


class TestExponentialBackoff:
    """Test exponential backoff delays: 1s, 2s, 4s (AC4)"""

    async def test_exponential_backoff_delays(self):
        """Should use exponential backoff: 1s, 2s, 4s"""
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        call_count = 0

        async def always_timeout():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Timeout")

        with patch("ragitect.services.retry.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(httpx.TimeoutException):
                await with_retry(always_timeout)

        # Should have 2 sleep calls (before attempts 2 and 3)
        assert len(sleep_calls) == MAX_RETRIES - 1

        # Check exponential backoff delays
        assert sleep_calls[0] == INITIAL_DELAY  # 1s
        assert sleep_calls[1] == INITIAL_DELAY * BACKOFF_MULTIPLIER  # 2s


class TestRetryConstants:
    """Test retry configuration constants (AC4)"""

    def test_max_retries_is_3(self):
        """MAX_RETRIES should be 3"""
        assert MAX_RETRIES == 3

    def test_initial_delay_is_1_second(self):
        """Initial delay should be 1 second"""
        assert INITIAL_DELAY == 1.0

    def test_backoff_multiplier_is_2(self):
        """Backoff multiplier should be 2 (doubling)"""
        assert BACKOFF_MULTIPLIER == 2

    def test_timeout_is_retryable(self):
        """TimeoutException should be in retryable exceptions"""
        assert httpx.TimeoutException in RETRYABLE_EXCEPTIONS


class TestNetworkErrors:
    """Test retry behavior for network errors (AC4)"""

    async def test_retries_on_network_error(self):
        """Should retry on NetworkError"""
        call_count = 0

        async def network_error():
            nonlocal call_count
            call_count += 1
            raise httpx.NetworkError("Network unreachable")

        with pytest.raises(httpx.NetworkError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(network_error)

        assert call_count == MAX_RETRIES

    async def test_retries_on_connect_error(self):
        """Should retry on ConnectError"""
        call_count = 0

        async def connect_error():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            with patch("asyncio.sleep", new=AsyncMock()):
                await with_retry(connect_error)

        assert call_count == MAX_RETRIES
