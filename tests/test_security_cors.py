"""
Comprehensive CORS Security Tests for Maverick MCP.

Tests CORS configuration, validation, origin blocking, wildcard security,
and environment-specific behaviors.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from maverick_mcp.config.security import (
    CORSConfig,
    SecurityConfig,
    validate_security_config,
)
from maverick_mcp.config.security_utils import (
    apply_cors_to_fastapi,
    check_security_config,
    get_safe_cors_config,
)


class TestCORSConfiguration:
    """Test CORS configuration validation and creation."""

    def test_cors_config_valid_origins(self):
        """Test CORS config creation with valid origins."""
        config = CORSConfig(
            allowed_origins=["https://example.com", "https://app.example.com"],
            allow_credentials=True,
        )

        assert config.allowed_origins == [
            "https://example.com",
            "https://app.example.com",
        ]
        assert config.allow_credentials is True

    def test_cors_config_wildcard_with_credentials_raises_error(self):
        """Test that wildcard origins with credentials raises validation error."""
        with pytest.raises(
            ValueError,
            match="CORS Security Error.*wildcard origin.*serious security vulnerability",
        ):
            CORSConfig(allowed_origins=["*"], allow_credentials=True)

    def test_cors_config_wildcard_without_credentials_warns(self):
        """Test that wildcard origins without credentials logs warning."""
        with patch("logging.getLogger") as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            config = CORSConfig(allowed_origins=["*"], allow_credentials=False)

            assert config.allowed_origins == ["*"]
            assert config.allow_credentials is False
            mock_logger_instance.warning.assert_called_once()

    def test_cors_config_multiple_origins_with_wildcard_fails(self):
        """Test that mixed origins including wildcard with credentials fails."""
        with pytest.raises(ValueError, match="CORS Security Error"):
            CORSConfig(
                allowed_origins=["https://example.com", "*"], allow_credentials=True
            )

    def test_cors_config_default_values(self):
        """Test CORS config default values are secure."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            with patch(
                "maverick_mcp.config.security._get_cors_origins"
            ) as mock_origins:
                mock_origins.return_value = ["http://localhost:3000"]

                config = CORSConfig()

                assert config.allow_credentials is True
                assert "GET" in config.allowed_methods
                assert "POST" in config.allowed_methods
                assert "Authorization" in config.allowed_headers
                assert "Content-Type" in config.allowed_headers
                assert config.max_age == 86400

    def test_cors_config_expose_headers(self):
        """Test that proper headers are exposed to clients."""
        config = CORSConfig()

        expected_exposed = [
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID",
        ]

        for header in expected_exposed:
            assert header in config.exposed_headers


class TestCORSEnvironmentConfiguration:
    """Test environment-specific CORS configuration."""

    def test_production_cors_origins(self):
        """Test production CORS origins are restrictive."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            with patch(
                "maverick_mcp.config.security._get_cors_origins"
            ) as mock_origins:
                mock_origins.return_value = [
                    "https://app.maverick-mcp.com",
                    "https://maverick-mcp.com",
                ]

                config = SecurityConfig()

                assert "localhost" not in str(config.cors.allowed_origins).lower()
                assert "127.0.0.1" not in str(config.cors.allowed_origins).lower()
                assert all(
                    origin.startswith("https://")
                    for origin in config.cors.allowed_origins
                )

    def test_development_cors_origins(self):
        """Test development CORS origins include localhost."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            with patch(
                "maverick_mcp.config.security._get_cors_origins"
            ) as mock_origins:
                mock_origins.return_value = [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                ]

                config = SecurityConfig()

                localhost_found = any(
                    "localhost" in origin for origin in config.cors.allowed_origins
                )
                assert localhost_found

    def test_staging_cors_origins(self):
        """Test staging CORS origins are appropriate."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            with patch(
                "maverick_mcp.config.security._get_cors_origins"
            ) as mock_origins:
                mock_origins.return_value = [
                    "https://staging.maverick-mcp.com",
                    "http://localhost:3000",
                ]

                config = SecurityConfig()

                staging_found = any(
                    "staging" in origin for origin in config.cors.allowed_origins
                )
                assert staging_found

    def test_custom_cors_origins_from_env(self):
        """Test custom CORS origins from environment variable."""
        custom_origins = "https://custom1.com,https://custom2.com"

        with patch.dict(os.environ, {"CORS_ORIGINS": custom_origins}, clear=False):
            with patch(
                "maverick_mcp.config.security._get_cors_origins"
            ) as mock_origins:
                mock_origins.return_value = [
                    "https://custom1.com",
                    "https://custom2.com",
                ]

                config = SecurityConfig()

                assert "https://custom1.com" in config.cors.allowed_origins
                assert "https://custom2.com" in config.cors.allowed_origins


class TestCORSValidation:
    """Test CORS security validation."""

    def test_validate_security_config_valid_cors(self):
        """Test security validation passes with valid CORS config."""
        with patch("maverick_mcp.config.security.get_security_config") as mock_config:
            mock_security_config = MagicMock()
            mock_security_config.cors.allowed_origins = ["https://example.com"]
            mock_security_config.cors.allow_credentials = True
            mock_security_config.is_production.return_value = False
            mock_security_config.force_https = True
            mock_security_config.headers.x_frame_options = "DENY"
            mock_config.return_value = mock_security_config

            result = validate_security_config()

            assert result["valid"] is True
            assert len(result["issues"]) == 0

    def test_validate_security_config_wildcard_with_credentials(self):
        """Test security validation fails with wildcard + credentials."""
        with patch("maverick_mcp.config.security.get_security_config") as mock_config:
            mock_security_config = MagicMock()
            mock_security_config.cors.allowed_origins = ["*"]
            mock_security_config.cors.allow_credentials = True
            mock_security_config.is_production.return_value = False
            mock_security_config.force_https = True
            mock_security_config.headers.x_frame_options = "DENY"
            mock_config.return_value = mock_security_config

            result = validate_security_config()

            assert result["valid"] is False
            assert any(
                "Wildcard CORS origins with credentials enabled" in issue
                for issue in result["issues"]
            )

    def test_validate_security_config_production_wildcards(self):
        """Test security validation fails with wildcards in production."""
        with patch("maverick_mcp.config.security.get_security_config") as mock_config:
            mock_security_config = MagicMock()
            mock_security_config.cors.allowed_origins = ["*"]
            mock_security_config.cors.allow_credentials = False
            mock_security_config.is_production.return_value = True
            mock_security_config.force_https = True
            mock_security_config.headers.x_frame_options = "DENY"
            mock_config.return_value = mock_security_config

            result = validate_security_config()

            assert result["valid"] is False
            assert any(
                "Wildcard CORS origins in production" in issue
                for issue in result["issues"]
            )

    def test_validate_security_config_production_localhost_warning(self):
        """Test security validation warns about localhost in production."""
        with patch("maverick_mcp.config.security.get_security_config") as mock_config:
            mock_security_config = MagicMock()
            mock_security_config.cors.allowed_origins = [
                "https://app.com",
                "http://localhost:3000",
            ]
            mock_security_config.cors.allow_credentials = True
            mock_security_config.is_production.return_value = True
            mock_security_config.force_https = True
            mock_security_config.headers.x_frame_options = "DENY"
            mock_config.return_value = mock_security_config

            result = validate_security_config()

            assert result["valid"] is True  # Warning, not error
            assert any("localhost" in warning.lower() for warning in result["warnings"])


class TestCORSMiddlewareIntegration:
    """Test CORS middleware integration with FastAPI."""

    def create_test_app(self, security_config=None):
        """Create a test FastAPI app with CORS applied."""
        app = FastAPI()

        if security_config:
            with patch(
                "maverick_mcp.config.security_utils.get_security_config",
                return_value=security_config,
            ):
                apply_cors_to_fastapi(app)
        else:
            apply_cors_to_fastapi(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        @app.post("/test")
        async def test_post_endpoint():
            return {"message": "post test"}

        return app

    def test_cors_middleware_allows_configured_origins(self):
        """Test that CORS middleware allows configured origins."""
        # Create mock security config
        mock_config = MagicMock()
        mock_config.get_cors_middleware_config.return_value = {
            "allow_origins": ["https://allowed.com"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": [],
            "max_age": 86400,
        }

        # Mock validation to pass
        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": [], "warnings": []}

            app = self.create_test_app(mock_config)
            client = TestClient(app)

            # Test preflight request
            response = client.options(
                "/test",
                headers={
                    "Origin": "https://allowed.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )

            assert response.status_code == 200
            assert (
                response.headers.get("Access-Control-Allow-Origin")
                == "https://allowed.com"
            )
            assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")

    def test_cors_middleware_blocks_unauthorized_origins(self):
        """Test that CORS middleware blocks unauthorized origins."""
        mock_config = MagicMock()
        mock_config.get_cors_middleware_config.return_value = {
            "allow_origins": ["https://allowed.com"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["Content-Type"],
            "expose_headers": [],
            "max_age": 86400,
        }

        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": [], "warnings": []}

            app = self.create_test_app(mock_config)
            client = TestClient(app)

            # Test request from unauthorized origin
            response = client.get(
                "/test", headers={"Origin": "https://unauthorized.com"}
            )

            # The request should succeed (CORS is browser-enforced)
            # but the CORS headers should not allow the unauthorized origin
            assert response.status_code == 200
            cors_origin = response.headers.get("Access-Control-Allow-Origin")
            assert cors_origin != "https://unauthorized.com"

    def test_cors_middleware_credentials_handling(self):
        """Test CORS middleware credentials handling."""
        mock_config = MagicMock()
        mock_config.get_cors_middleware_config.return_value = {
            "allow_origins": ["https://allowed.com"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["Content-Type"],
            "expose_headers": [],
            "max_age": 86400,
        }

        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": [], "warnings": []}

            app = self.create_test_app(mock_config)
            client = TestClient(app)

            response = client.options(
                "/test",
                headers={
                    "Origin": "https://allowed.com",
                    "Access-Control-Request-Method": "POST",
                },
            )

            assert response.headers.get("Access-Control-Allow-Credentials") == "true"

    def test_cors_middleware_exposed_headers(self):
        """Test that CORS middleware exposes configured headers."""
        mock_config = MagicMock()
        mock_config.get_cors_middleware_config.return_value = {
            "allow_origins": ["https://allowed.com"],
            "allow_credentials": True,
            "allow_methods": ["GET"],
            "allow_headers": ["Content-Type"],
            "expose_headers": ["X-Custom-Header", "X-Rate-Limit"],
            "max_age": 86400,
        }

        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": [], "warnings": []}

            app = self.create_test_app(mock_config)
            client = TestClient(app)

            response = client.get("/test", headers={"Origin": "https://allowed.com"})

            exposed_headers = response.headers.get("Access-Control-Expose-Headers", "")
            assert "X-Custom-Header" in exposed_headers
            assert "X-Rate-Limit" in exposed_headers


class TestCORSSecurityValidation:
    """Test CORS security validation and safety measures."""

    def test_apply_cors_fails_with_invalid_config(self):
        """Test that applying CORS fails with invalid configuration."""
        app = FastAPI()

        # Mock invalid configuration
        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "issues": ["Wildcard CORS origins with credentials"],
                "warnings": [],
            }

            with pytest.raises(ValueError, match="Security configuration is invalid"):
                apply_cors_to_fastapi(app)

    def test_get_safe_cors_config_production_fallback(self):
        """Test safe CORS config fallback for production."""
        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "issues": ["Invalid config"],
                "warnings": [],
            }

            with patch(
                "maverick_mcp.config.security_utils.get_security_config"
            ) as mock_config:
                mock_security_config = MagicMock()
                mock_security_config.is_production.return_value = True
                mock_config.return_value = mock_security_config

                safe_config = get_safe_cors_config()

                assert safe_config["allow_origins"] == ["https://maverick-mcp.com"]
                assert safe_config["allow_credentials"] is True
                assert "localhost" not in str(safe_config["allow_origins"])

    def test_get_safe_cors_config_development_fallback(self):
        """Test safe CORS config fallback for development."""
        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "issues": ["Invalid config"],
                "warnings": [],
            }

            with patch(
                "maverick_mcp.config.security_utils.get_security_config"
            ) as mock_config:
                mock_security_config = MagicMock()
                mock_security_config.is_production.return_value = False
                mock_config.return_value = mock_security_config

                safe_config = get_safe_cors_config()

                assert safe_config["allow_origins"] == ["http://localhost:3000"]
                assert safe_config["allow_credentials"] is True

    def test_check_security_config_function(self):
        """Test security config check function."""
        with patch(
            "maverick_mcp.config.security_utils.validate_security_config"
        ) as mock_validate:
            # Test valid config
            mock_validate.return_value = {"valid": True, "issues": [], "warnings": []}
            assert check_security_config() is True

            # Test invalid config
            mock_validate.return_value = {
                "valid": False,
                "issues": ["Error"],
                "warnings": [],
            }
            assert check_security_config() is False


class TestCORSPreflightRequests:
    """Test CORS preflight request handling."""

    def test_preflight_request_max_age(self):
        """Test CORS preflight max-age header."""
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://example.com"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            max_age=3600,
        )

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        response = client.options(
            "/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.headers.get("Access-Control-Max-Age") == "3600"

    def test_preflight_request_methods(self):
        """Test CORS preflight allowed methods."""
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://example.com"],
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["Content-Type"],
        )

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        response = client.options(
            "/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "PUT",
            },
        )

        allowed_methods = response.headers.get("Access-Control-Allow-Methods", "")
        assert "PUT" in allowed_methods
        assert "GET" in allowed_methods
        assert "POST" in allowed_methods

    def test_preflight_request_headers(self):
        """Test CORS preflight allowed headers."""
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://example.com"],
            allow_methods=["POST"],
            allow_headers=["Content-Type", "Authorization", "X-Custom"],
        )

        @app.post("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        response = client.options(
            "/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, Authorization",
            },
        )

        allowed_headers = response.headers.get("Access-Control-Allow-Headers", "")
        assert "Content-Type" in allowed_headers
        assert "Authorization" in allowed_headers


class TestCORSEdgeCases:
    """Test CORS edge cases and security scenarios."""

    def test_cors_with_vary_header(self):
        """Test that CORS responses include Vary header."""
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://example.com"],
            allow_methods=["GET"],
            allow_headers=["Content-Type"],
        )

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        response = client.get("/test", headers={"Origin": "https://example.com"})

        vary_header = response.headers.get("Vary", "")
        assert "Origin" in vary_header

    def test_cors_null_origin_handling(self):
        """Test CORS handling of null origin (file:// protocol)."""
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["null"],  # Sometimes needed for file:// protocol
            allow_methods=["GET"],
            allow_headers=["Content-Type"],
        )

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        response = client.get("/test", headers={"Origin": "null"})

        # Should handle null origin appropriately
        assert response.status_code == 200

    def test_cors_case_insensitive_origin(self):
        """Test CORS origin matching is case-sensitive (as it should be)."""
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://Example.com"],  # Capital E
            allow_methods=["GET"],
            allow_headers=["Content-Type"],
        )

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        # Test with different case
        response = client.get(
            "/test",
            headers={"Origin": "https://example.com"},  # lowercase e
        )

        # Should not match due to case sensitivity
        cors_origin = response.headers.get("Access-Control-Allow-Origin")
        assert cors_origin != "https://example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
