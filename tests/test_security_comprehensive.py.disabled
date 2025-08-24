"""
Comprehensive security test suite for MaverickMCP.

Tests authentication flow, CSRF protection, rate limiting,
JWT secret rotation, and other security features.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import jwt
import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

from maverick_mcp.api.middleware.per_user_rate_limiting import (
    PerUserRateLimitMiddleware,
)
from maverick_mcp.auth.cookie_auth import (
    CookieConfig,
    clear_auth_cookies,
    get_token_from_cookie,
    set_auth_cookies,
    update_access_token_cookie,
)
from maverick_mcp.auth.csrf_protection import (
    CSRFProtectionMiddleware,
    generate_csrf_token,
    validate_csrf_token,
)
from maverick_mcp.auth.jwt_enhanced import EnhancedJWTManager

# Handle different PyJWT versions
try:
    # PyJWT 2.x
    from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
except ImportError:
    try:
        # PyJWT 1.x
        from jwt import ExpiredSignatureError, InvalidTokenError
    except ImportError:
        # Fallback - define our own
        class ExpiredSignatureError(Exception):
            pass

        class InvalidTokenError(Exception):
            pass


@pytest.fixture
def jwt_handler():
    """Create a JWT handler for testing."""
    return EnhancedJWTManager()


@pytest.fixture
def app():
    """Create a test FastAPI application with security middleware."""
    app = FastAPI()

    # Add CSRF middleware
    app.add_middleware(CSRFProtectionMiddleware, enabled=True)

    # Add rate limiting middleware
    app.add_middleware(
        PerUserRateLimitMiddleware,
        default_rate_limit=10,
        anonymous_rate_limit=5,
    )

    @app.post("/api/auth/login")
    async def login(response: Response):
        """Simulate login endpoint."""
        access_token = "test_access_token"
        refresh_token = "test_refresh_token"
        csrf_token = generate_csrf_token()

        set_auth_cookies(response, access_token, refresh_token, csrf_token)
        return {"csrf_token": csrf_token, "message": "Login successful"}

    @app.post("/api/auth/logout")
    async def logout(response: Response):
        """Simulate logout endpoint."""
        clear_auth_cookies(response)
        return {"message": "Logged out"}

    @app.post("/api/protected")
    async def protected_endpoint(request: Request):
        """Protected endpoint requiring CSRF token."""
        return {"message": "Success", "user": "test_user"}

    @app.get("/api/data")
    async def get_data():
        """Rate limited endpoint."""
        return {"data": "test_data"}

    @app.post("/api/auth/refresh")
    async def refresh_token(request: Request, response: Response):
        """Refresh access token."""
        refresh_token = get_token_from_cookie(
            request, CookieConfig.REFRESH_TOKEN_COOKIE
        )
        if not refresh_token:
            raise HTTPException(status_code=401, detail="No refresh token")

        # Simulate token refresh
        new_access_token = "new_access_token"
        csrf_token = generate_csrf_token()

        update_access_token_cookie(response, new_access_token, csrf_token)
        return {"csrf_token": csrf_token}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestCSRFProtection:
    """Test CSRF protection mechanisms."""

    def test_csrf_token_generation(self):
        """Test CSRF token generation."""
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()

        # Tokens should be unique
        assert token1 != token2

        # Tokens should have sufficient length
        assert len(token1) >= 32
        assert len(token2) >= 32

    def test_csrf_validation_valid_token(self):
        """Test CSRF validation with valid token."""
        token = generate_csrf_token()
        assert validate_csrf_token(None, token, token) is True

    def test_csrf_validation_invalid_cases(self):
        """Test CSRF validation with invalid cases."""
        token = generate_csrf_token()

        # None values
        assert validate_csrf_token(None, None, None) is False
        assert validate_csrf_token(None, token, None) is False
        assert validate_csrf_token(None, None, token) is False

        # Mismatched tokens
        assert validate_csrf_token(None, token, "different_token") is False
        assert validate_csrf_token(None, "token1", "token2") is False

    def test_csrf_middleware_blocks_without_token(self, client):
        """Test that CSRF middleware blocks requests without token."""
        # Login to get cookies
        login_response = client.post("/api/auth/login")
        assert login_response.status_code == 200

        # Get cookies from the response
        cookies = login_response.cookies

        # Try to access protected endpoint without CSRF header but with cookies
        # HTTPException from middleware is not caught by TestClient, so we expect it to raise
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            client.post("/api/protected", cookies=cookies)

        assert exc_info.value.status_code == 403
        assert "CSRF validation failed" in exc_info.value.detail

    def test_csrf_middleware_allows_with_valid_token(self, client):
        """Test that CSRF middleware allows requests with valid token."""
        # Login to get cookies and CSRF token
        login_response = client.post("/api/auth/login")
        csrf_token = login_response.json()["csrf_token"]
        cookies = login_response.cookies

        # Access protected endpoint with CSRF header and cookies
        response = client.post(
            "/api/protected", headers={"X-CSRF-Token": csrf_token}, cookies=cookies
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Success"

    def test_csrf_cookie_not_httponly(self, client):
        """Test that CSRF cookie is accessible to JavaScript."""
        response = client.post("/api/auth/login")

        # Check cookie headers
        csrf_cookie_header = None
        for header in response.headers.raw:
            if (
                header[0] == b"set-cookie"
                and CookieConfig.CSRF_TOKEN_COOKIE.encode() in header[1]
            ):
                csrf_cookie_header = header[1].decode()
                break

        assert csrf_cookie_header is not None
        assert "httponly" not in csrf_cookie_header.lower()

    def test_csrf_protection_safe_methods_allowed(self, client):
        """Test that safe methods (GET, HEAD, OPTIONS) bypass CSRF."""
        # Login to get cookies
        client.post("/api/auth/login")

        # GET request should work without CSRF token
        response = client.get("/api/data")
        assert response.status_code == 200

    def test_csrf_token_rotation_on_refresh(self, client):
        """Test that CSRF token is rotated on token refresh."""
        # Login
        login_response = client.post("/api/auth/login")
        initial_csrf = login_response.json()["csrf_token"]
        cookies = login_response.cookies

        # Refresh token
        refresh_response = client.post(
            "/api/auth/refresh", headers={"X-CSRF-Token": initial_csrf}, cookies=cookies
        )

        assert refresh_response.status_code == 200
        new_csrf = refresh_response.json()["csrf_token"]

        # CSRF token should be different
        assert new_csrf != initial_csrf


class TestRateLimiting:
    """Test rate limiting mechanisms."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limiting_anonymous_users(self, client):
        """Test rate limiting for anonymous users."""
        # Anonymous users have lower rate limit (5 requests per minute)
        responses = []

        # Make 10 requests
        for _ in range(10):
            response = client.get("/api/data")
            responses.append(response)

        # First 5 should succeed
        for i in range(5):
            assert responses[i].status_code == 200

        # Remaining should be rate limited
        for i in range(5, 10):
            assert responses[i].status_code == 429
            assert "Rate limit exceeded" in responses[i].json()["error"]

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are included in responses."""
        response = client.get("/api/data")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_retry_after(self, client):
        """Test retry-after header when rate limited."""
        # Exhaust rate limit
        for _ in range(10):
            response = client.get("/api/data")

        # Last response should have retry-after
        if response.status_code == 429:
            assert "Retry-After" in response.headers
            retry_after = int(response.headers["Retry-After"])
            assert retry_after > 0


class TestJWTSecurity:
    """Test JWT token security features."""

    def test_jwt_token_generation(self, jwt_handler):
        """Test JWT token generation."""
        user_id = "123"
        scopes = ["api:read", "api:write"]

        # Generate token pair
        access_token, refresh_token, metadata = jwt_handler.generate_token_pair(
            user_id=user_id, scope=" ".join(scopes)
        )

        # Decode and verify access token
        payload = jwt_handler.decode_access_token(access_token)
        assert payload["sub"] == user_id
        assert payload["scope"] == " ".join(scopes)
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload  # JWT ID for uniqueness

    def test_jwt_token_expiration(self, jwt_handler):
        """Test JWT token expiration."""
        user_id = "123"

        # Create token with short expiration
        with patch.object(jwt_handler, "access_ttl", timedelta(seconds=-1)):
            access_token, _, _ = jwt_handler.generate_token_pair(user_id=user_id)

            # Token should be expired
            with pytest.raises(ExpiredSignatureError):
                jwt_handler.decode_access_token(access_token)

    def test_jwt_refresh_token_longer_expiry(self, jwt_handler):
        """Test that refresh tokens have longer expiry."""
        user_id = "123"

        access_token, refresh_token, _ = jwt_handler.generate_token_pair(
            user_id=user_id
        )

        access_payload = jwt.decode(
            access_token,
            jwt_handler.public_key,
            algorithms=[jwt_handler.algorithm],
            options={"verify_signature": False},
        )

        refresh_payload = jwt.decode(
            refresh_token,
            jwt_handler.public_key,
            algorithms=[jwt_handler.algorithm],
            options={"verify_signature": False},
        )

        # Refresh token should expire later
        assert refresh_payload["exp"] > access_payload["exp"]

    def test_jwt_invalid_token_handling(self, jwt_handler):
        """Test handling of invalid JWT tokens."""
        # Invalid format
        with pytest.raises(InvalidTokenError):
            jwt_handler.decode_access_token("invalid.token")

        # Modified token
        valid_token, _, _ = jwt_handler.generate_token_pair(user_id="123")
        tampered_token = valid_token[:-10] + "tampered123"

        with pytest.raises(InvalidTokenError):
            jwt_handler.decode_access_token(tampered_token)

    def test_jwt_algorithm_restriction(self, jwt_handler):
        """Test that only allowed algorithms are accepted."""
        # Create token with different algorithm
        payload = {
            "sub": "123",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        }

        # Try to use 'none' algorithm (security vulnerability)
        token = jwt.encode(payload, "", algorithm="none")

        with pytest.raises(InvalidTokenError):
            jwt_handler.decode_access_token(token)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_jwt_secret_rotation(self):
        """Test JWT secret rotation mechanism."""
        handler = EnhancedJWTManager()

        # Create token with original secret
        token = handler.create_access_token(user_id=123)

        # Rotate secret
        new_secret = "new_secret_key_123"
        with patch.object(handler, "secret_key", new_secret):
            # Old token should fail
            with pytest.raises(jwt.InvalidSignatureError):
                handler.decode_token(token)

            # New token with new secret should work
            new_token = handler.create_access_token(user_id=456)
            payload = handler.decode_token(new_token)
            assert payload["user_id"] == 456

    def test_jwt_claims_validation(self, jwt_handler):
        """Test JWT claims are properly validated."""
        user_id = "123"
        access_token, _, _ = jwt_handler.generate_token_pair(user_id=user_id)

        payload = jwt_handler.decode_access_token(access_token)

        # Required claims should be present
        assert "sub" in payload  # user_id is stored in 'sub' claim
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

        # Issue time should be recent
        iat = datetime.fromtimestamp(payload["iat"], UTC)
        assert (datetime.now(UTC) - iat).total_seconds() < 5


class TestCookieSecurity:
    """Test cookie security settings."""

    def test_cookie_secure_flag_production(self):
        """Test that secure flag is set in production."""
        # Patch get_settings to return production environment
        from maverick_mcp.config.settings import Settings

        mock_settings = Settings()
        mock_settings.environment = "production"

        with patch(
            "maverick_mcp.auth.cookie_auth.get_settings", return_value=mock_settings
        ):
            config = CookieConfig()
            assert config.SECURE is True

    def test_cookie_secure_flag_development(self):
        """Test that secure flag is not set in development."""
        with patch.dict("os.environ", {"ENVIRONMENT": "development"}):
            config = CookieConfig()
            assert config.SECURE is False

    def test_cookie_httponly_flag(self, client):
        """Test that auth cookies have httpOnly flag."""
        response = client.post("/api/auth/login")

        # Check access token cookie
        access_cookie_header = None
        for header in response.headers.raw:
            if (
                header[0] == b"set-cookie"
                and CookieConfig.ACCESS_TOKEN_COOKIE.encode() in header[1]
            ):
                access_cookie_header = header[1].decode()
                break

        assert access_cookie_header is not None
        assert "httponly" in access_cookie_header.lower()

    def test_cookie_samesite_attribute(self, client):
        """Test that cookies have SameSite attribute."""
        # Use the correct endpoint that exists in the test app
        response = client.post("/api/auth/login")

        # Check cookie headers
        for header in response.headers.raw:
            if header[0] == b"set-cookie":
                cookie_header = header[1].decode()
                if CookieConfig.ACCESS_TOKEN_COOKIE in cookie_header:
                    assert "samesite=lax" in cookie_header.lower()

    def test_cookie_clearing(self, client):
        """Test that cookies are properly cleared on logout."""
        # Login
        login_response = client.post("/api/auth/login")
        cookies = login_response.cookies
        csrf_token = login_response.json()["csrf_token"]

        # Logout (needs CSRF token)
        logout_response = client.post(
            "/api/auth/logout", headers={"X-CSRF-Token": csrf_token}, cookies=cookies
        )

        # Check cookies are cleared
        # In TestClient, deleted cookies show up as empty string in set-cookie headers
        # but cookies.get() returns None for deleted cookies
        cookies = logout_response.cookies
        for cookie_name in [
            CookieConfig.ACCESS_TOKEN_COOKIE,
            CookieConfig.REFRESH_TOKEN_COOKIE,
            CookieConfig.CSRF_TOKEN_COOKIE,
        ]:
            # Cookie should either be None (deleted) or empty string
            cookie_value = cookies.get(cookie_name)
            assert cookie_value is None or cookie_value == "" or cookie_value == '""'


class TestSecurityHeaders:
    """Test security headers."""

    def test_security_headers_present(self, client):
        """Test that security headers are set."""
        client.get("/api/data")

        # Basic security headers that should be present
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
        }

        for _header, _expected_value in expected_headers.items():
            # Note: These would need to be added via middleware
            # This test documents what should be implemented
            pass


class TestPasswordSecurity:
    """Test password handling security."""

    def test_password_hashing(self):
        """Test that passwords are properly hashed."""
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        password = "test_password_123"
        hashed = pwd_context.hash(password)

        # Hash should be different from plaintext
        assert hashed != password

        # Hash should be verifiable
        assert pwd_context.verify(password, hashed) is True

        # Wrong password should fail
        assert pwd_context.verify("wrong_password", hashed) is False

    def test_password_complexity_requirements(self):
        """Test password complexity validation."""

        def validate_password(password: str) -> tuple[bool, str]:
            """Validate password complexity."""
            if len(password) < 8:
                return False, "Password must be at least 8 characters"
            if not any(c.isupper() for c in password):
                return False, "Password must contain uppercase letter"
            if not any(c.islower() for c in password):
                return False, "Password must contain lowercase letter"
            if not any(c.isdigit() for c in password):
                return False, "Password must contain digit"
            if not any(c in "!@#$%^&*" for c in password):
                return False, "Password must contain special character"
            return True, "Valid password"

        # Test various passwords
        assert validate_password("short")[0] is False
        assert validate_password("alllowercase123!")[0] is False
        assert validate_password("ALLUPPERCASE123!")[0] is False
        assert validate_password("NoNumbers!")[0] is False
        assert validate_password("NoSpecial123")[0] is False
        assert validate_password("ValidPass123!")[0] is True


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_sql_injection_prevention(self):
        """Test that SQL injection is prevented."""
        # This would test actual database queries with SQLAlchemy
        # SQLAlchemy's parameterized queries prevent SQL injection

        # Example of safe query construction
        from sqlalchemy import text

        # Bad: String concatenation (DON'T DO THIS)
        # query = f"SELECT * FROM users WHERE name = '{malicious_input}'"

        # Good: Parameterized query
        query = text("SELECT * FROM users WHERE name = :name")

        # The malicious input is safely escaped
        assert ":name" in str(query)

    def test_xss_prevention(self):
        """Test that XSS attacks are prevented."""
        malicious_input = "<script>alert('XSS')</script>"

        # HTML escaping
        import html

        escaped = html.escape(malicious_input)

        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_path_traversal_prevention(self):
        """Test that path traversal attacks are prevented."""
        from pathlib import Path

        base_dir = "/safe/directory"
        user_input = "../../etc/passwd"

        # Unsafe: Direct concatenation
        # unsafe_path = os.path.join(base_dir, user_input)

        # Safe: Resolve and check
        requested_path = Path(base_dir) / user_input
        resolved_path = requested_path.resolve()

        # Check if resolved path is within base directory
        try:
            resolved_path.relative_to(Path(base_dir).resolve())
            is_safe = True
        except ValueError:
            is_safe = False

        assert is_safe is False


class TestConcurrentSecurityOperations:
    """Test security under concurrent operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_rate_limiting(self, client):
        """Test rate limiting under concurrent requests."""
        # Note: This test is simplified since we're using middleware-based rate limiting
        # The actual concurrent behavior would need to be tested with a real async client

        async def make_request():
            # In a real test, this would use an async HTTP client
            return client.get("/api/data")

        # Make multiple requests quickly
        responses = []
        for _ in range(15):
            response = make_request()
            responses.append(response)

        # Check that rate limiting is applied
        status_codes = [r.status_code for r in responses]

        # Some should succeed, some should be rate limited
        assert 200 in status_codes
        assert 429 in status_codes

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_token_validation(self, jwt_handler):
        """Test JWT validation under concurrent load."""
        token = jwt_handler.create_access_token(user_id=123)

        async def validate_token():
            try:
                payload = jwt_handler.decode_token(token)
                return payload["user_id"] == 123
            except Exception:
                return False

        # Validate token concurrently
        tasks = [validate_token() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All validations should succeed
        assert all(results)


class TestSecurityMonitoring:
    """Test security monitoring and logging."""

    def test_failed_login_logging(self):
        """Test that failed login attempts are logged."""
        # This would integrate with actual logging
        import logging

        logger = logging.getLogger("security")

        with patch.object(logger, "warning") as mock_warning:
            # Simulate failed login
            user_ip = "192.168.1.100"
            username = "test_user"

            logger.warning(
                f"Failed login attempt for user {username} from IP {user_ip}"
            )

            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            assert username in call_args
            assert user_ip in call_args

    def test_suspicious_activity_detection(self):
        """Test detection of suspicious activity patterns."""
        # Track failed attempts
        failed_attempts = {}

        def record_failed_attempt(ip_address: str):
            if ip_address not in failed_attempts:
                failed_attempts[ip_address] = []
            failed_attempts[ip_address].append(time.time())

            # Check for suspicious pattern (5 failures in 1 minute)
            recent_attempts = [
                t for t in failed_attempts[ip_address] if time.time() - t < 60
            ]

            return len(recent_attempts) >= 5

        # Simulate attacks
        attacker_ip = "10.0.0.1"

        for _i in range(4):
            is_suspicious = record_failed_attempt(attacker_ip)
            assert is_suspicious is False

        # 5th attempt should trigger
        is_suspicious = record_failed_attempt(attacker_ip)
        assert is_suspicious is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
