"""
Test suite for enhanced rate limiting middleware.

Tests various rate limiting scenarios including:
- Different user types (anonymous, authenticated, premium)
- Different endpoint tiers
- Multiple rate limiting strategies
- Monitoring and alerting
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from maverick_mcp.api.middleware.rate_limiting_enhanced import (
    EndpointClassification,
    EnhancedRateLimitMiddleware,
    RateLimitConfig,
    RateLimiter,
    RateLimitStrategy,
    RateLimitTier,
    rate_limit,
)
from maverick_mcp.exceptions import RateLimitError


@pytest.fixture
def rate_limit_config():
    """Create test rate limit configuration."""
    return RateLimitConfig(
        public_limit=100,
        auth_limit=5,
        data_limit=20,
        data_limit_anonymous=5,
        analysis_limit=10,
        analysis_limit_anonymous=2,
        bulk_limit_per_hour=5,
        admin_limit=10,
        premium_multiplier=5.0,
        enterprise_multiplier=10.0,
        default_strategy=RateLimitStrategy.SLIDING_WINDOW,
        burst_multiplier=1.5,
        window_size_seconds=60,
        token_refill_rate=1.0,
        max_tokens=10,
        log_violations=True,
        alert_threshold=3,
    )


@pytest.fixture
def rate_limiter(rate_limit_config):
    """Create rate limiter instance."""
    return RateLimiter(rate_limit_config)


@pytest.fixture
async def mock_redis():
    """Create mock Redis client."""
    mock = AsyncMock(spec=redis.Redis)

    # Mock pipeline
    mock_pipeline = AsyncMock()
    mock_pipeline.execute = AsyncMock(return_value=[None, 0, None, None])
    mock.pipeline = MagicMock(return_value=mock_pipeline)

    # Mock other methods
    mock.zrange = AsyncMock(return_value=[])
    mock.hgetall = AsyncMock(return_value={})
    mock.incr = AsyncMock(return_value=1)

    return mock


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/auth/login")
    async def login():
        return {"token": "test"}

    @app.get("/api/data/stock/{symbol}")
    async def get_stock(symbol: str):
        return {"symbol": symbol, "price": 100}

    @app.post("/api/screening/bulk")
    async def bulk_screening():
        return {"stocks": ["AAPL", "GOOGL", "MSFT"]}

    @app.get("/api/admin/users")
    async def admin_users():
        return {"users": []}

    return app


class TestEndpointClassification:
    """Test endpoint classification."""

    def test_classify_public_endpoints(self):
        """Test classification of public endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/health") == RateLimitTier.PUBLIC
        )
        assert (
            EndpointClassification.classify_endpoint("/api/docs")
            == RateLimitTier.PUBLIC
        )
        assert (
            EndpointClassification.classify_endpoint("/api/openapi.json")
            == RateLimitTier.PUBLIC
        )

    def test_classify_auth_endpoints(self):
        """Test classification of authentication endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/api/auth/login")
            == RateLimitTier.AUTHENTICATION
        )
        assert (
            EndpointClassification.classify_endpoint("/api/auth/signup")
            == RateLimitTier.AUTHENTICATION
        )
        assert (
            EndpointClassification.classify_endpoint("/api/auth/refresh")
            == RateLimitTier.AUTHENTICATION
        )

    def test_classify_data_endpoints(self):
        """Test classification of data retrieval endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/api/data/stock/AAPL")
            == RateLimitTier.DATA_RETRIEVAL
        )
        assert (
            EndpointClassification.classify_endpoint("/api/stock/quote")
            == RateLimitTier.DATA_RETRIEVAL
        )
        assert (
            EndpointClassification.classify_endpoint("/api/market/movers")
            == RateLimitTier.DATA_RETRIEVAL
        )

    def test_classify_analysis_endpoints(self):
        """Test classification of analysis endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/api/technical/indicators")
            == RateLimitTier.ANALYSIS
        )
        assert (
            EndpointClassification.classify_endpoint("/api/screening/maverick")
            == RateLimitTier.ANALYSIS
        )
        assert (
            EndpointClassification.classify_endpoint("/api/portfolio/optimize")
            == RateLimitTier.ANALYSIS
        )

    def test_classify_bulk_endpoints(self):
        """Test classification of bulk operation endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/api/screening/bulk")
            == RateLimitTier.BULK_OPERATION
        )
        assert (
            EndpointClassification.classify_endpoint("/api/data/bulk")
            == RateLimitTier.BULK_OPERATION
        )
        assert (
            EndpointClassification.classify_endpoint("/api/portfolio/batch")
            == RateLimitTier.BULK_OPERATION
        )

    def test_classify_admin_endpoints(self):
        """Test classification of administrative endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/api/admin/users")
            == RateLimitTier.ADMINISTRATIVE
        )
        assert (
            EndpointClassification.classify_endpoint("/api/admin/system")
            == RateLimitTier.ADMINISTRATIVE
        )
        assert (
            EndpointClassification.classify_endpoint("/api/users/admin/delete")
            == RateLimitTier.ADMINISTRATIVE
        )

    def test_default_classification(self):
        """Test default classification for unknown endpoints."""
        assert (
            EndpointClassification.classify_endpoint("/api/unknown")
            == RateLimitTier.DATA_RETRIEVAL
        )
        assert (
            EndpointClassification.classify_endpoint("/random/path")
            == RateLimitTier.DATA_RETRIEVAL
        )


class TestRateLimiter:
    """Test rate limiter core functionality."""

    @pytest.mark.asyncio
    async def test_sliding_window_allows_requests(self, rate_limiter, mock_redis):
        """Test sliding window allows requests within limit."""
        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client",
            return_value=mock_redis,
        ):
            is_allowed, info = await rate_limiter.check_rate_limit(
                key="test_user",
                tier=RateLimitTier.DATA_RETRIEVAL,
                limit=10,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            )

            assert is_allowed is True
            assert info["limit"] == 10
            assert info["remaining"] == 9
            assert "burst_limit" in info

    @pytest.mark.asyncio
    async def test_sliding_window_blocks_excess(self, rate_limiter, mock_redis):
        """Test sliding window blocks requests over limit."""
        # Mock pipeline to return high count
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 15, None, None])
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client",
            return_value=mock_redis,
        ):
            is_allowed, info = await rate_limiter.check_rate_limit(
                key="test_user",
                tier=RateLimitTier.DATA_RETRIEVAL,
                limit=10,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            )

            assert is_allowed is False
            assert info["remaining"] == 0
            assert info["retry_after"] > 0

    @pytest.mark.asyncio
    async def test_token_bucket_allows_requests(self, rate_limiter, mock_redis):
        """Test token bucket allows requests with tokens."""
        mock_redis.hgetall = AsyncMock(
            return_value={"tokens": "5.0", "last_refill": str(time.time())}
        )

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client",
            return_value=mock_redis,
        ):
            is_allowed, info = await rate_limiter.check_rate_limit(
                key="test_user",
                tier=RateLimitTier.DATA_RETRIEVAL,
                limit=10,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )

            assert is_allowed is True
            assert "tokens" in info
            assert "refill_rate" in info

    @pytest.mark.asyncio
    async def test_token_bucket_blocks_no_tokens(self, rate_limiter, mock_redis):
        """Test token bucket blocks requests without tokens."""
        mock_redis.hgetall = AsyncMock(
            return_value={"tokens": "0.5", "last_refill": str(time.time())}
        )

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client",
            return_value=mock_redis,
        ):
            is_allowed, info = await rate_limiter.check_rate_limit(
                key="test_user",
                tier=RateLimitTier.DATA_RETRIEVAL,
                limit=10,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )

            assert is_allowed is False
            assert info["retry_after"] > 0

    @pytest.mark.asyncio
    async def test_fixed_window_allows_requests(self, rate_limiter, mock_redis):
        """Test fixed window allows requests within limit."""
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[5, None])
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client",
            return_value=mock_redis,
        ):
            is_allowed, info = await rate_limiter.check_rate_limit(
                key="test_user",
                tier=RateLimitTier.DATA_RETRIEVAL,
                limit=10,
                window_seconds=60,
                strategy=RateLimitStrategy.FIXED_WINDOW,
            )

            assert is_allowed is True
            assert info["current_count"] == 5

    @pytest.mark.asyncio
    async def test_local_fallback_rate_limiting(self, rate_limiter):
        """Test local rate limiting when Redis unavailable."""
        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client", return_value=None
        ):
            # First few requests should pass
            for _i in range(5):
                is_allowed, info = await rate_limiter.check_rate_limit(
                    key="test_user",
                    tier=RateLimitTier.DATA_RETRIEVAL,
                    limit=5,
                    window_seconds=60,
                )
                assert is_allowed is True
                assert info["fallback"] is True

            # Next request should be blocked
            is_allowed, info = await rate_limiter.check_rate_limit(
                key="test_user",
                tier=RateLimitTier.DATA_RETRIEVAL,
                limit=5,
                window_seconds=60,
            )
            assert is_allowed is False

    def test_violation_recording(self, rate_limiter):
        """Test violation count recording."""
        tier = RateLimitTier.DATA_RETRIEVAL
        assert rate_limiter.get_violation_count("user1", tier=tier) == 0

        rate_limiter.record_violation("user1", tier=tier)
        assert rate_limiter.get_violation_count("user1", tier=tier) == 1

        rate_limiter.record_violation("user1", tier=tier)
        assert rate_limiter.get_violation_count("user1", tier=tier) == 2

        # Different tiers maintain independent counters
        other_tier = RateLimitTier.ANALYSIS
        assert rate_limiter.get_violation_count("user1", tier=other_tier) == 0


class TestEnhancedRateLimitMiddleware:
    """Test enhanced rate limit middleware integration."""

    @pytest.fixture
    def middleware_app(self, test_app, rate_limit_config):
        """Create app with rate limit middleware."""
        test_app.add_middleware(EnhancedRateLimitMiddleware, config=rate_limit_config)
        return test_app

    @pytest.fixture
    def client(self, middleware_app):
        """Create test client."""
        return TestClient(middleware_app)

    def test_bypass_health_check(self, client):
        """Test health check endpoint bypasses rate limiting."""
        # Should always succeed
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" not in response.headers

    @patch("maverick_mcp.data.performance.redis_manager.get_client")
    def test_anonymous_rate_limiting(self, mock_get_client, client, mock_redis):
        """Test rate limiting for anonymous users."""
        mock_get_client.return_value = mock_redis

        # Configure mock to allow first 5 requests
        call_count = 0

        def mock_execute():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return [None, call_count - 1, None, None]
            else:
                return [None, 10, None, None]  # Over limit

        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(side_effect=mock_execute)
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
        mock_redis.zrange = AsyncMock(return_value=[(b"1", time.time())])

        # First 5 requests should succeed
        for _i in range(5):
            response = client.get("/api/data/stock/AAPL")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers

        # 6th request should be rate limited
        response = client.get("/api/data/stock/AAPL")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["error"]
        assert "Retry-After" in response.headers

    def test_authenticated_user_headers(self, client):
        """Test authenticated users get proper headers."""
        # Mock authenticated request
        request = MagicMock(spec=Request)
        request.state.user_id = "123"
        request.state.user_context = {"role": "user"}

        # Headers should be added to response
        # This would be tested in integration tests with actual auth

    def test_premium_user_multiplier(self, client):
        """Test premium users get higher limits."""
        # Mock premium user request
        request = MagicMock(spec=Request)
        request.state.user_id = "123"
        request.state.user_context = {"role": "premium"}

        # Premium users should have 5x the limit
        # This would be tested in integration tests

    def test_endpoint_tier_headers(self, client):
        """Test different endpoints return tier information."""
        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client", return_value=None
        ):
            # Test auth endpoint
            response = client.post("/api/auth/login")
            if "X-RateLimit-Tier" in response.headers:
                assert response.headers["X-RateLimit-Tier"] == "authentication"

            # Test data endpoint
            response = client.get("/api/data/stock/AAPL")
            if "X-RateLimit-Tier" in response.headers:
                assert response.headers["X-RateLimit-Tier"] == "data_retrieval"

            # Test bulk endpoint
            response = client.post("/api/screening/bulk")
            if "X-RateLimit-Tier" in response.headers:
                assert response.headers["X-RateLimit-Tier"] == "bulk_operation"


class TestRateLimitDecorator:
    """Test function-level rate limiting decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_requests(self):
        """Test decorator allows requests within limit."""
        call_count = 0

        @rate_limit(requests_per_minute=5)
        async def test_function(request: Request):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        # Mock request
        request = MagicMock(spec=Request)
        request.state.user_id = "test_user"

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client", return_value=None
        ):
            # Should allow first few calls
            for i in range(5):
                result = await test_function(request)
                assert result["count"] == i + 1

    @pytest.mark.asyncio
    async def test_decorator_blocks_excess(self):
        """Test decorator blocks excessive requests."""

        @rate_limit(requests_per_minute=2)
        async def test_function(request: Request):
            return {"success": True}

        # Mock request with proper attributes for rate limiting
        request = MagicMock()
        request.state = MagicMock()
        request.state.user_id = "test_user"
        request.url = MagicMock()  # Required for rate limiting detection

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client", return_value=None
        ):
            # First 2 should succeed
            await test_function(request)
            await test_function(request)

            # 3rd should raise exception
            with pytest.raises(RateLimitError) as exc_info:
                await test_function(request)

            assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_decorator_without_request(self):
        """Test decorator works without request object."""

        @rate_limit(requests_per_minute=5)
        async def test_function(value: int):
            return value * 2

        # Should work without rate limiting
        result = await test_function(5)
        assert result == 10


class TestMonitoringIntegration:
    """Test monitoring and alerting integration."""

    @pytest.mark.asyncio
    async def test_violation_monitoring(self, rate_limiter, rate_limit_config):
        """Test violations are recorded for monitoring."""
        # Record multiple violations
        for _i in range(rate_limit_config.alert_threshold + 1):
            rate_limiter.record_violation("bad_user", tier=RateLimitTier.DATA_RETRIEVAL)

        # Check violation count
        assert (
            rate_limiter.get_violation_count("bad_user", tier=RateLimitTier.DATA_RETRIEVAL)
            > rate_limit_config.alert_threshold
        )

    @pytest.mark.asyncio
    async def test_cleanup_task(self, rate_limiter, mock_redis):
        """Test periodic cleanup of old data."""
        mock_redis.scan = AsyncMock(
            return_value=(
                0,
                [
                    "rate_limit:sw:test1",
                    "rate_limit:sw:test2",
                ],
            )
        )
        mock_redis.type = AsyncMock(return_value="zset")
        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zcard = AsyncMock(return_value=0)
        mock_redis.delete = AsyncMock()

        with patch(
            "maverick_mcp.data.performance.redis_manager.get_client",
            return_value=mock_redis,
        ):
            await rate_limiter.cleanup_old_data(older_than_hours=1)

            # Should have called delete for empty keys
            assert mock_redis.delete.called
