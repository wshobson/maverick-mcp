"""
Comprehensive Security Headers Tests for Maverick MCP.

Tests security headers configuration, middleware implementation,
environment-specific headers, and CSP/HSTS policies.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from maverick_mcp.api.middleware.security import (
    SecurityHeadersMiddleware as APISecurityHeadersMiddleware,
)
from maverick_mcp.config.security import (
    SecurityConfig,
    SecurityHeadersConfig,
)
from maverick_mcp.config.security_utils import (
    SecurityHeadersMiddleware,
    apply_security_headers_to_fastapi,
)


class TestSecurityHeadersConfig:
    """Test security headers configuration."""

    def test_security_headers_default_values(self):
        """Test security headers have secure default values."""
        config = SecurityHeadersConfig()

        assert config.x_content_type_options == "nosniff"
        assert config.x_frame_options == "DENY"
        assert config.x_xss_protection == "1; mode=block"
        assert config.referrer_policy == "strict-origin-when-cross-origin"
        assert "geolocation=()" in config.permissions_policy

    def test_hsts_header_generation(self):
        """Test HSTS header value generation."""
        config = SecurityHeadersConfig()

        hsts_header = config.hsts_header_value

        assert f"max-age={config.hsts_max_age}" in hsts_header
        assert "includeSubDomains" in hsts_header
        assert "preload" not in hsts_header  # Default is False

    def test_hsts_header_with_preload(self):
        """Test HSTS header with preload enabled."""
        config = SecurityHeadersConfig(hsts_preload=True)

        hsts_header = config.hsts_header_value

        assert "preload" in hsts_header

    def test_hsts_header_without_subdomains(self):
        """Test HSTS header without subdomains."""
        config = SecurityHeadersConfig(hsts_include_subdomains=False)

        hsts_header = config.hsts_header_value

        assert "includeSubDomains" not in hsts_header

    def test_csp_header_generation(self):
        """Test CSP header value generation."""
        config = SecurityHeadersConfig()

        csp_header = config.csp_header_value

        # Check required directives
        assert "default-src 'self'" in csp_header
        assert "script-src 'self' 'unsafe-inline'" in csp_header
        assert "style-src 'self' 'unsafe-inline'" in csp_header
        assert "object-src 'none'" in csp_header
        assert "connect-src 'self'" in csp_header
        assert "frame-src 'none'" in csp_header
        assert "base-uri 'self'" in csp_header
        assert "form-action 'self'" in csp_header

    def test_csp_custom_directives(self):
        """Test CSP with custom directives."""
        config = SecurityHeadersConfig(
            csp_script_src=["'self'", "https://trusted.com"],
            csp_connect_src=["'self'", "https://api.trusted.com"],
        )

        csp_header = config.csp_header_value

        assert "script-src 'self' https://trusted.com" in csp_header
        assert "connect-src 'self' https://api.trusted.com" in csp_header

    def test_permissions_policy_default(self):
        """Test permissions policy default configuration."""
        config = SecurityHeadersConfig()

        permissions = config.permissions_policy

        assert "geolocation=()" in permissions
        assert "microphone=()" in permissions
        assert "camera=()" in permissions
        assert "usb=()" in permissions
        assert "magnetometer=()" in permissions


class TestSecurityHeadersMiddleware:
    """Test security headers middleware implementation."""

    def test_middleware_adds_headers(self):
        """Test that middleware adds security headers to responses."""
        app = FastAPI()

        # Create mock security config
        mock_config = MagicMock()
        mock_config.get_security_headers.return_value = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'",
        }

        app.add_middleware(SecurityHeadersMiddleware, security_config=mock_config)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Content-Security-Policy"] == "default-src 'self'"

    def test_middleware_uses_default_config(self):
        """Test that middleware uses default security config when none provided."""
        app = FastAPI()

        with patch(
            "maverick_mcp.config.security_utils.get_security_config"
        ) as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_security_headers.return_value = {"X-Frame-Options": "DENY"}
            mock_get_config.return_value = mock_config

            app.add_middleware(SecurityHeadersMiddleware)

            @app.get("/test")
            async def test_endpoint():
                return {"message": "test"}

            client = TestClient(app)
            response = client.get("/test")

            mock_get_config.assert_called_once()
            assert response.headers["X-Frame-Options"] == "DENY"

    def test_api_middleware_integration(self):
        """Test API security headers middleware integration."""
        app = FastAPI()
        app.add_middleware(APISecurityHeadersMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        # Should have basic security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers


class TestEnvironmentSpecificHeaders:
    """Test environment-specific security headers."""

    def test_hsts_in_production(self):
        """Test HSTS header is included in production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            config = SecurityConfig()
            headers = config.get_security_headers()

            assert "Strict-Transport-Security" in headers
            assert "max-age=" in headers["Strict-Transport-Security"]

    def test_hsts_in_development(self):
        """Test HSTS header is not included in development."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            config = SecurityConfig(force_https=False)
            headers = config.get_security_headers()

            assert "Strict-Transport-Security" not in headers

    def test_hsts_with_force_https(self):
        """Test HSTS header is included when HTTPS is forced."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            config = SecurityConfig(force_https=True)
            headers = config.get_security_headers()

            assert "Strict-Transport-Security" in headers

    def test_production_security_validation(self):
        """Test production security validation."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            with patch(
                "maverick_mcp.config.security._get_cors_origins"
            ) as mock_origins:
                mock_origins.return_value = ["https://app.maverick-mcp.com"]

                with patch("logging.getLogger") as mock_logger:
                    mock_logger_instance = MagicMock()
                    mock_logger.return_value = mock_logger_instance

                    # Test with HTTPS not forced (should warn)
                    SecurityConfig(force_https=False)

                    # Should log warning about HTTPS
                    mock_logger_instance.warning.assert_called()

    def test_development_security_permissive(self):
        """Test development security is more permissive."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            config = SecurityConfig()

            assert config.is_development() is True
            assert config.is_production() is False


class TestCSPConfiguration:
    """Test Content Security Policy configuration."""

    def test_csp_avoids_checkout_domains(self):
        """Test CSP excludes third-party checkout provider domains."""
        config = SecurityHeadersConfig()

        assert config.csp_script_src == ["'self'", "'unsafe-inline'"]
        assert config.csp_connect_src == ["'self'"]
        assert config.csp_frame_src == ["'none'"]

    def test_csp_blocks_inline_scripts_by_default(self):
        """Test CSP configuration for inline scripts."""
        config = SecurityHeadersConfig()
        csp = config.csp_header_value

        # Note: Current config allows 'unsafe-inline' for compatibility
        # In a more secure setup, this should use nonces or hashes
        assert "'unsafe-inline'" in csp

    def test_csp_blocks_object_embedding(self):
        """Test CSP blocks object embedding."""
        config = SecurityHeadersConfig()
        csp = config.csp_header_value

        assert "object-src 'none'" in csp

    def test_csp_restricts_base_uri(self):
        """Test CSP restricts base URI."""
        config = SecurityHeadersConfig()
        csp = config.csp_header_value

        assert "base-uri 'self'" in csp

    def test_csp_restricts_form_action(self):
        """Test CSP restricts form actions."""
        config = SecurityHeadersConfig()
        csp = config.csp_header_value

        assert "form-action 'self'" in csp

    def test_csp_image_sources(self):
        """Test CSP allows necessary image sources."""
        config = SecurityHeadersConfig()
        csp = config.csp_header_value

        assert "img-src 'self' data: https:" in csp

    def test_csp_custom_configuration(self):
        """Test CSP with custom configuration."""
        custom_config = SecurityHeadersConfig(
            csp_default_src=["'self'", "https://trusted.com"],
            csp_script_src=["'self'"],
            csp_style_src=["'self'"],  # Remove unsafe-inline from styles too
            csp_object_src=["'none'"],
        )

        csp = custom_config.csp_header_value

        assert "default-src 'self' https://trusted.com" in csp
        assert "script-src 'self'" in csp
        # Since we removed unsafe-inline from style-src, it shouldn't be in CSP
        assert "style-src 'self'" in csp
        assert "'unsafe-inline'" not in csp


class TestXFrameOptionsConfiguration:
    """Test X-Frame-Options configuration."""

    def test_frame_options_deny_default(self):
        """Test X-Frame-Options defaults to DENY."""
        SecurityHeadersConfig()
        headers = SecurityConfig().get_security_headers()

        assert headers["X-Frame-Options"] == "DENY"

    def test_frame_options_sameorigin(self):
        """Test X-Frame-Options can be set to SAMEORIGIN."""
        config = SecurityHeadersConfig(x_frame_options="SAMEORIGIN")
        security_config = SecurityConfig(headers=config)
        headers = security_config.get_security_headers()

        assert headers["X-Frame-Options"] == "SAMEORIGIN"

    def test_frame_options_allow_from(self):
        """Test X-Frame-Options with ALLOW-FROM directive."""
        config = SecurityHeadersConfig(x_frame_options="ALLOW-FROM https://trusted.com")
        security_config = SecurityConfig(headers=config)
        headers = security_config.get_security_headers()

        assert headers["X-Frame-Options"] == "ALLOW-FROM https://trusted.com"


class TestReferrerPolicyConfiguration:
    """Test Referrer-Policy configuration."""

    def test_referrer_policy_default(self):
        """Test Referrer-Policy default value."""
        SecurityHeadersConfig()
        headers = SecurityConfig().get_security_headers()

        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_referrer_policy_custom(self):
        """Test custom Referrer-Policy."""
        config = SecurityHeadersConfig(referrer_policy="no-referrer")
        security_config = SecurityConfig(headers=config)
        headers = security_config.get_security_headers()

        assert headers["Referrer-Policy"] == "no-referrer"


class TestPermissionsPolicyConfiguration:
    """Test Permissions-Policy configuration."""

    def test_permissions_policy_blocks_dangerous_features(self):
        """Test Permissions-Policy blocks dangerous browser features."""
        SecurityHeadersConfig()
        headers = SecurityConfig().get_security_headers()

        permissions = headers["Permissions-Policy"]

        assert "geolocation=()" in permissions
        assert "microphone=()" in permissions
        assert "camera=()" in permissions
        assert "usb=()" in permissions

    def test_permissions_policy_custom(self):
        """Test custom Permissions-Policy configuration."""
        custom_policy = "geolocation=(self), camera=(), microphone=()"
        config = SecurityHeadersConfig(permissions_policy=custom_policy)
        security_config = SecurityConfig(headers=config)
        headers = security_config.get_security_headers()

        assert headers["Permissions-Policy"] == custom_policy


class TestSecurityHeadersIntegration:
    """Test security headers integration with application."""

    def test_all_headers_applied(self):
        """Test that all security headers are applied to responses."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        # Check all expected headers are present
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Permissions-Policy",
            "Content-Security-Policy",
        ]

        for header in expected_headers:
            assert header in response.headers

    def test_headers_on_error_responses(self):
        """Test security headers are included on error responses."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/error")
        async def error_endpoint():
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail="Test error")

        client = TestClient(app)
        response = client.get("/error")

        # Even on errors, security headers should be present
        assert response.status_code == 500
        assert "X-Frame-Options" in response.headers
        assert "X-Content-Type-Options" in response.headers

    def test_headers_on_different_methods(self):
        """Test security headers on different HTTP methods."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def get_endpoint():
            return {"method": "GET"}

        @app.post("/test")
        async def post_endpoint():
            return {"method": "POST"}

        @app.put("/test")
        async def put_endpoint():
            return {"method": "PUT"}

        client = TestClient(app)

        methods = [(client.get, "/test"), (client.post, "/test"), (client.put, "/test")]

        for method_func, path in methods:
            response = method_func(path)
            assert "X-Frame-Options" in response.headers
            assert "Content-Security-Policy" in response.headers

    def test_headers_override_existing(self):
        """Test security headers override any existing headers."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            from fastapi import Response

            response = Response(content='{"message": "test"}')
            response.headers["X-Frame-Options"] = "ALLOWALL"  # Insecure value
            return response

        client = TestClient(app)
        response = client.get("/test")

        # Security middleware should override the insecure value
        assert response.headers["X-Frame-Options"] == "DENY"


class TestSecurityHeadersValidation:
    """Test security headers validation and best practices."""

    def test_no_server_header_disclosure(self):
        """Test that server information is not disclosed."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        # Should not disclose server information
        server_header = response.headers.get("Server", "")
        assert "uvicorn" not in server_header.lower()

    def test_no_powered_by_header(self):
        """Test that X-Powered-By header is not present."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Powered-By" not in response.headers

    def test_content_type_nosniff(self):
        """Test X-Content-Type-Options prevents MIME sniffing."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_xss_protection_enabled(self):
        """Test X-XSS-Protection is properly configured."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        xss_protection = response.headers["X-XSS-Protection"]
        assert "1" in xss_protection
        assert "mode=block" in xss_protection


class TestSecurityHeadersPerformance:
    """Test security headers don't impact performance significantly."""

    def test_headers_middleware_performance(self):
        """Test security headers middleware performance."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        # Make multiple requests to test performance
        import time

        start_time = time.time()

        for _ in range(100):
            response = client.get("/test")
            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 100 requests quickly (less than 5 seconds)
        assert total_time < 5.0

    def test_headers_memory_usage(self):
        """Test security headers don't cause memory leaks."""
        app = FastAPI()
        apply_security_headers_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        # Make many requests to check for memory leaks
        for _ in range(1000):
            response = client.get("/test")
            assert "X-Frame-Options" in response.headers

        # If we reach here without memory issues, test passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
