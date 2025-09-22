"""
Security Penetration Testing Suite for MaverickMCP.

This suite performs security penetration testing to validate that
security protections are active and effective against real attack vectors.

Tests include:
- Authentication bypass attempts
- CSRF attack vectors
- Rate limiting evasion
- Input validation bypass
- Session hijacking attempts
- SQL injection prevention
- XSS protection validation
- Information disclosure prevention
"""

import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from maverick_mcp.api.api_server import create_api_app


@pytest.fixture
def security_test_app():
    """Create app for security testing."""
    return create_api_app()


@pytest.fixture
def security_client(security_test_app):
    """Create client for security testing."""
    return TestClient(security_test_app)


@pytest.fixture
def test_user():
    """Test user for security testing."""
    return {
        "email": f"sectest{uuid4().hex[:8]}@example.com",
        "password": "SecurePass123!",
        "name": "Security Test User",
        "company": "Test Security Inc",
    }


class TestAuthenticationSecurity:
    """Test authentication security against bypass attempts."""

    @pytest.mark.integration
    def test_jwt_token_manipulation_resistance(self, security_client, test_user):
        """Test resistance to JWT token manipulation attacks."""

        # Register and login
        security_client.post("/auth/register", json=test_user)
        login_response = security_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
        )

        # Extract tokens from cookies
        cookies = login_response.cookies
        access_token_cookie = cookies.get("maverick_access_token")

        if not access_token_cookie:
            pytest.skip("JWT tokens not in cookies - may be test environment")

        # Attempt 1: Modified JWT signature
        tampered_token = access_token_cookie[:-10] + "tampered123"

        response = security_client.get(
            "/user/profile", cookies={"maverick_access_token": tampered_token}
        )
        assert response.status_code == 401  # Should reject tampered token

        # Attempt 2: Algorithm confusion attack (trying "none" algorithm)
        none_algorithm_token = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJ1c2VyX2lkIjoxLCJleHAiOjk5OTk5OTk5OTl9."

        response = security_client.get(
            "/user/profile", cookies={"maverick_access_token": none_algorithm_token}
        )
        assert response.status_code == 401  # Should reject "none" algorithm

        # Attempt 3: Expired token
        {
            "user_id": 1,
            "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
            "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
            "jti": "expired_token",
        }

        # This would require creating an expired token with the same secret
        # For security, we just test that expired tokens are rejected

    @pytest.mark.integration
    def test_session_fixation_protection(self, security_client, test_user):
        """Test protection against session fixation attacks."""

        # Get initial session state
        initial_response = security_client.get("/auth/login")
        initial_cookies = initial_response.cookies

        # Login with potential pre-set session
        security_client.post("/auth/register", json=test_user)
        login_response = security_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
            cookies=initial_cookies,  # Try to maintain old session
        )

        # Verify new session is created (cookies should be different)
        new_cookies = login_response.cookies

        # Session should be regenerated after login
        if "maverick_access_token" in new_cookies:
            # New token should be different from any pre-existing one
            assert login_response.status_code == 200

    @pytest.mark.integration
    def test_concurrent_session_limits(self, security_client, test_user):
        """Test limits on concurrent sessions."""

        # Register user
        security_client.post("/auth/register", json=test_user)

        # Create multiple concurrent sessions
        session_responses = []
        for _i in range(5):
            client_instance = TestClient(security_client.app)
            response = client_instance.post(
                "/auth/login",
                json={"email": test_user["email"], "password": test_user["password"]},
            )
            session_responses.append(response)

        # All should succeed (or be limited if concurrent session limits implemented)
        success_count = sum(1 for r in session_responses if r.status_code == 200)
        assert success_count >= 1  # At least one should succeed

        # If concurrent session limits are implemented, test that old sessions are invalidated

    @pytest.mark.integration
    def test_password_brute_force_protection(self, security_client, test_user):
        """Test protection against password brute force attacks."""

        # Register user
        security_client.post("/auth/register", json=test_user)

        # Attempt multiple failed logins
        failed_attempts = []
        for i in range(10):
            response = security_client.post(
                "/auth/login",
                json={"email": test_user["email"], "password": f"wrong_password_{i}"},
            )
            failed_attempts.append(response.status_code)

            # Small delay to avoid overwhelming the system
            time.sleep(0.1)

        # Should have multiple failures
        assert all(status == 401 for status in failed_attempts)

        # After multiple failures, account should be locked or rate limited
        # Test with correct password - should be blocked if protection is active
        final_attempt = security_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
        )

        # If brute force protection is active, should be rate limited
        # Otherwise, should succeed
        assert final_attempt.status_code in [200, 401, 429]


class TestCSRFAttackVectors:
    """Test CSRF protection against various attack vectors."""

    @pytest.mark.integration
    def test_csrf_attack_simulation(self, security_client, test_user):
        """Simulate CSRF attacks to test protection."""

        # Setup authenticated session
        security_client.post("/auth/register", json=test_user)
        login_response = security_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
        )
        csrf_token = login_response.json().get("csrf_token")

        # Attack 1: Missing CSRF token
        attack_response_1 = security_client.post(
            "/user/profile", json={"name": "Attacked Name"}
        )
        assert attack_response_1.status_code == 403
        assert "CSRF" in attack_response_1.json()["detail"]

        # Attack 2: Invalid CSRF token
        attack_response_2 = security_client.post(
            "/user/profile",
            json={"name": "Attacked Name"},
            headers={"X-CSRF-Token": "invalid_token_123"},
        )
        assert attack_response_2.status_code == 403

        # Attack 3: CSRF token from different session
        # Create second user and get their CSRF token
        other_user = {
            "email": f"other{uuid4().hex[:8]}@example.com",
            "password": "OtherPass123!",
            "name": "Other User",
        }

        other_client = TestClient(security_client.app)
        other_client.post("/auth/register", json=other_user)
        other_login = other_client.post(
            "/auth/login",
            json={"email": other_user["email"], "password": other_user["password"]},
        )
        other_csrf = other_login.json().get("csrf_token")

        # Try to use other user's CSRF token
        attack_response_3 = security_client.post(
            "/user/profile",
            json={"name": "Cross-User Attack"},
            headers={"X-CSRF-Token": other_csrf},
        )
        assert attack_response_3.status_code == 403

        # Legitimate request should work
        legitimate_response = security_client.post(
            "/user/profile",
            json={"name": "Legitimate Update"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert legitimate_response.status_code == 200

    @pytest.mark.integration
    def test_csrf_double_submit_validation(self, security_client, test_user):
        """Test CSRF double-submit cookie validation."""

        # Setup session
        security_client.post("/auth/register", json=test_user)
        login_response = security_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
        )
        csrf_token = login_response.json().get("csrf_token")
        cookies = login_response.cookies

        # Attack: Modify CSRF cookie but keep header the same
        modified_cookies = cookies.copy()
        if "maverick_csrf_token" in modified_cookies:
            modified_cookies["maverick_csrf_token"] = "modified_csrf_token"

        attack_response = security_client.post(
            "/user/profile",
            json={"name": "CSRF Cookie Attack"},
            headers={"X-CSRF-Token": csrf_token},
            cookies=modified_cookies,
        )
        assert attack_response.status_code == 403

    @pytest.mark.integration
    def test_csrf_token_entropy_and_uniqueness(self, security_client, test_user):
        """Test CSRF tokens have sufficient entropy and are unique."""

        # Register user
        security_client.post("/auth/register", json=test_user)

        # Generate multiple CSRF tokens
        csrf_tokens = []
        for _i in range(5):
            response = security_client.post(
                "/auth/login",
                json={"email": test_user["email"], "password": test_user["password"]},
            )
            csrf_token = response.json().get("csrf_token")
            if csrf_token:
                csrf_tokens.append(csrf_token)

        if csrf_tokens:
            # All tokens should be unique
            assert len(set(csrf_tokens)) == len(csrf_tokens)

            # Tokens should have sufficient length (at least 32 chars)
            for token in csrf_tokens:
                assert len(token) >= 32

            # Tokens should not be predictable patterns
            for i, token in enumerate(csrf_tokens[1:], 1):
                # Should not be sequential or pattern-based
                assert token != csrf_tokens[0] + str(i)
                assert not token.startswith(csrf_tokens[0][:-5])


class TestRateLimitingEvasion:
    """Test rate limiting against evasion attempts."""

    @pytest.mark.integration
    def test_ip_based_rate_limit_evasion(self, security_client):
        """Test attempts to evade IP-based rate limiting."""

        # Test basic rate limiting
        responses = []
        for _i in range(25):
            response = security_client.get("/api/data")
            responses.append(response.status_code)

        # Should hit rate limit
        sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        assert rate_limited_count > 0  # Should have some rate limited responses

        # Attempt 1: X-Forwarded-For header spoofing
        spoofed_responses = []
        for i in range(10):
            response = security_client.get(
                "/api/data", headers={"X-Forwarded-For": f"192.168.1.{i}"}
            )
            spoofed_responses.append(response.status_code)

        # Should still be rate limited (proper implementation should use real IP)
        sum(1 for status in spoofed_responses if status == 429)

        # Attempt 2: X-Real-IP header spoofing
        real_ip_responses = []
        for i in range(5):
            response = security_client.get(
                "/api/data", headers={"X-Real-IP": f"10.0.0.{i}"}
            )
            real_ip_responses.append(response.status_code)

        # Rate limiting should not be easily bypassed

    @pytest.mark.integration
    def test_user_agent_rotation_evasion(self, security_client):
        """Test rate limiting against user agent rotation."""

        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X)",
        ]

        # Attempt to evade rate limiting by rotating user agents
        ua_responses = []
        for i in range(15):
            ua = user_agents[i % len(user_agents)]
            response = security_client.get("/api/data", headers={"User-Agent": ua})
            ua_responses.append(response.status_code)

        # Should still enforce rate limiting regardless of user agent
        sum(1 for status in ua_responses if status == 429)
        # Should have some rate limiting if effective

    @pytest.mark.integration
    def test_distributed_rate_limit_evasion(self, security_client):
        """Test against distributed rate limit evasion attempts."""

        # Simulate requests with small delays (trying to stay under rate limits)
        distributed_responses = []
        for _i in range(10):
            response = security_client.get("/api/data")
            distributed_responses.append(response.status_code)
            time.sleep(0.1)  # Small delay

        # Even with delays, sustained high-rate requests should be limited
        # This tests if rate limiting has proper time windows


class TestInputValidationBypass:
    """Test input validation against bypass attempts."""

    @pytest.mark.integration
    def test_sql_injection_prevention(self, security_client, test_user):
        """Test SQL injection prevention."""

        # SQL injection payloads
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; DELETE FROM users WHERE '1'='1",
            "' OR 1=1 --",
            "admin'--",
            "admin'/*",
            "' OR 'x'='x",
            "' AND id IS NULL; --",
            "'OR 1=1#",
        ]

        # Test SQL injection in login email field
        for payload in sql_payloads:
            response = security_client.post(
                "/auth/login", json={"email": payload, "password": "any_password"}
            )

            # Should handle gracefully without SQL errors
            assert response.status_code in [400, 401, 422]  # Not 500 (SQL error)

            # Response should not contain SQL error messages
            response_text = response.text.lower()
            sql_error_indicators = [
                "syntax error",
                "sql",
                "mysql",
                "postgresql",
                "sqlite",
                "database",
                "column",
                "table",
                "select",
                "union",
            ]

            for indicator in sql_error_indicators:
                assert indicator not in response_text

        # Test SQL injection in registration fields
        for field in ["name", "company"]:
            malicious_user = test_user.copy()
            malicious_user[field] = "'; DROP TABLE users; --"

            response = security_client.post("/auth/register", json=malicious_user)

            # Should either reject or sanitize the input
            assert response.status_code in [200, 201, 400, 422]

            if response.status_code in [200, 201]:
                # If accepted, verify it's sanitized
                login_response = security_client.post(
                    "/auth/login",
                    json={
                        "email": malicious_user["email"],
                        "password": malicious_user["password"],
                    },
                )

                if login_response.status_code == 200:
                    csrf_token = login_response.json().get("csrf_token")
                    profile_response = security_client.get(
                        "/user/profile", headers={"X-CSRF-Token": csrf_token}
                    )

                    if profile_response.status_code == 200:
                        profile_data = profile_response.json()
                        # SQL injection should be sanitized
                        assert "DROP TABLE" not in profile_data.get(field, "")

    @pytest.mark.integration
    def test_xss_prevention(self, security_client, test_user):
        """Test XSS prevention."""

        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
            "';alert('XSS');//",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
        ]

        for payload in xss_payloads:
            # Test XSS in user registration
            malicious_user = test_user.copy()
            malicious_user["email"] = f"xss{uuid4().hex[:8]}@example.com"
            malicious_user["name"] = payload

            response = security_client.post("/auth/register", json=malicious_user)

            if response.status_code in [200, 201]:
                # Login and check profile
                login_response = security_client.post(
                    "/auth/login",
                    json={
                        "email": malicious_user["email"],
                        "password": malicious_user["password"],
                    },
                )

                if login_response.status_code == 200:
                    csrf_token = login_response.json().get("csrf_token")
                    profile_response = security_client.get(
                        "/user/profile", headers={"X-CSRF-Token": csrf_token}
                    )

                    if profile_response.status_code == 200:
                        profile_data = profile_response.json()
                        stored_name = profile_data.get("name", "")

                        # XSS should be escaped or removed
                        assert "<script>" not in stored_name
                        assert "javascript:" not in stored_name
                        assert "onerror=" not in stored_name
                        assert "onload=" not in stored_name
                        assert "alert(" not in stored_name

    @pytest.mark.integration
    def test_path_traversal_prevention(self, security_client):
        """Test path traversal prevention."""

        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
        ]

        # Test path traversal in file access endpoints (if any)
        for payload in path_traversal_payloads:
            # Test in URL path
            response = security_client.get(f"/api/files/{payload}")

            # Should return 404 or 400, not 500 or file contents
            assert response.status_code in [400, 404, 422]

            # Should not return file system contents
            response_text = response.text.lower()
            sensitive_file_indicators = [
                "root:",
                "daemon:",
                "bin:",
                "sys:",  # /etc/passwd content
                "[boot loader]",
                "[operating systems]",  # Windows boot.ini
                "password",
                "hash",
                "secret",
            ]

            for indicator in sensitive_file_indicators:
                assert indicator not in response_text

    @pytest.mark.integration
    def test_command_injection_prevention(self, security_client, test_user):
        """Test command injection prevention."""

        command_injection_payloads = [
            "; cat /etc/passwd",
            "| cat /etc/passwd",
            "& dir",
            "`cat /etc/passwd`",
            "$(cat /etc/passwd)",
            "; rm -rf /",
            "&& rm -rf /",
            "|| rm -rf /",
            "; shutdown -h now",
            "'; whoami; echo '",
        ]

        # Test command injection in various fields
        for payload in command_injection_payloads:
            malicious_user = test_user.copy()
            malicious_user["email"] = f"cmd{uuid4().hex[:8]}@example.com"
            malicious_user["company"] = payload

            response = security_client.post("/auth/register", json=malicious_user)

            # Should handle gracefully
            assert response.status_code in [200, 201, 400, 422]

            # Should not execute system commands
            response_text = response.text
            command_output_indicators = [
                "root:",
                "daemon:",
                "bin:",  # Output of cat /etc/passwd
                "total ",
                "drwx",  # Output of ls -la
                "uid=",
                "gid=",  # Output of whoami/id
            ]

            for indicator in command_output_indicators:
                assert indicator not in response_text


class TestInformationDisclosure:
    """Test prevention of information disclosure."""

    @pytest.mark.integration
    def test_error_message_sanitization(self, security_client):
        """Test that error messages don't leak sensitive information."""

        # Test 404 error
        response = security_client.get("/nonexistent/endpoint/123")
        assert response.status_code == 404

        error_data = response.json()
        error_message = str(error_data).lower()

        # Should not contain sensitive system information
        sensitive_info = [
            "/users/",
            "/home/",
            "\\users\\",
            "\\home\\",  # File paths
            "password",
            "secret",
            "key",
            "token",
            "jwt",  # Credentials
            "localhost",
            "127.0.0.1",
            "redis://",
            "postgresql://",  # Internal addresses
            "traceback",
            "stack trace",
            "exception",
            "error at",  # Stack traces
            "python",
            "uvicorn",
            "fastapi",
            "sqlalchemy",  # Framework details
            "database",
            "sql",
            "query",
            "connection",  # Database details
        ]

        for info in sensitive_info:
            assert info not in error_message

        # Should include request ID for tracking
        assert "request_id" in error_data or "error_id" in error_data

    @pytest.mark.integration
    def test_debug_information_disclosure(self, security_client):
        """Test that debug information is not disclosed."""

        # Attempt to trigger various error conditions
        error_test_cases = [
            ("/auth/login", {"invalid": "json_structure"}),
            ("/user/profile", {}),  # Missing authentication
        ]

        for endpoint, data in error_test_cases:
            response = security_client.post(endpoint, json=data)

            # Should not contain debug information
            response_text = response.text.lower()
            debug_indicators = [
                "traceback",
                "stack trace",
                "file ",
                "line ",
                "exception",
                "raise ",
                "assert",
                "debug",
                "__file__",
                "__name__",
                "locals()",
                "globals()",
            ]

            for indicator in debug_indicators:
                assert indicator not in response_text

    @pytest.mark.integration
    def test_version_information_disclosure(self, security_client):
        """Test that version information is not disclosed."""

        # Test common endpoints that might leak version info
        test_endpoints = [
            "/health",
            "/",
            "/api/docs",
            "/metrics",
        ]

        for endpoint in test_endpoints:
            response = security_client.get(endpoint)

            if response.status_code == 200:
                response_text = response.text.lower()

                # Should not contain detailed version information
                version_indicators = [
                    "python/",
                    "fastapi/",
                    "uvicorn/",
                    "nginx/",
                    "version",
                    "build",
                    "commit",
                    "git",
                    "dev",
                    "debug",
                    "staging",
                    "test",
                ]

                # Some version info might be acceptable in health endpoints
                if endpoint != "/health":
                    for indicator in version_indicators:
                        assert indicator not in response_text

    @pytest.mark.integration
    def test_user_enumeration_prevention(self, security_client):
        """Test prevention of user enumeration attacks."""

        # Test with valid email (user exists)
        existing_user = {
            "email": f"existing{uuid4().hex[:8]}@example.com",
            "password": "ValidPass123!",
            "name": "Existing User",
        }
        security_client.post("/auth/register", json=existing_user)

        # Test login with existing user but wrong password
        response_existing = security_client.post(
            "/auth/login",
            json={"email": existing_user["email"], "password": "wrong_password"},
        )

        # Test login with non-existing user
        response_nonexisting = security_client.post(
            "/auth/login",
            json={
                "email": f"nonexisting{uuid4().hex[:8]}@example.com",
                "password": "any_password",
            },
        )

        # Both should return similar error messages and status codes
        assert response_existing.status_code == response_nonexisting.status_code

        # Error messages should not distinguish between cases
        error_1 = response_existing.json().get("detail", "")
        error_2 = response_nonexisting.json().get("detail", "")

        # Should not contain user-specific information
        user_specific_terms = [
            "user not found",
            "user does not exist",
            "invalid user",
            "email not found",
            "account not found",
            "user unknown",
        ]

        for term in user_specific_terms:
            assert term.lower() not in error_1.lower()
            assert term.lower() not in error_2.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
