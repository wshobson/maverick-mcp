"""Tests for SQL safety, credential masking, security headers, and ticker validation fixes."""

import ast
import importlib
import re

import pytest

# --------------------------------------------------------------------------- #
# 1. Credential masking — mask_url()
# --------------------------------------------------------------------------- #


class TestMaskUrl:
    """Tests for the mask_url() utility in settings."""

    @pytest.fixture(autouse=True)
    def _import_mask_url(self):
        from maverick_mcp.config.settings import mask_url

        self.mask_url = mask_url

    def test_masks_password_only(self):
        url = "redis://:mysecretpassword@localhost:6379/0"
        masked = self.mask_url(url)
        assert "mysecretpassword" not in masked
        assert "***" in masked
        assert masked == "redis://:***@localhost:6379/0"

    def test_masks_user_and_password(self):
        url = "postgresql://admin:s3cret@db.example.com:5432/mydb"
        masked = self.mask_url(url)
        assert "s3cret" not in masked
        assert "admin" in masked  # username is preserved
        assert "***" in masked

    def test_no_credentials_unchanged(self):
        url = "sqlite:///maverick_mcp.db"
        masked = self.mask_url(url)
        assert masked == url

    def test_redis_ssl_with_password(self):
        url = "rediss://:p@ssw0rd!@redis.cloud:6380/1"
        masked = self.mask_url(url)
        assert "p@ssw0rd!" not in masked
        assert "***" in masked
        assert "rediss://" in masked

    def test_database_url_with_credentials(self):
        url = "postgresql://user:hunter2@localhost:5432/maverick"
        masked = self.mask_url(url)
        assert "hunter2" not in masked
        assert masked == "postgresql://user:***@localhost:5432/maverick"


# --------------------------------------------------------------------------- #
# 2. DatabaseSettings and RedisSettings masked_url properties
# --------------------------------------------------------------------------- #


class TestSettingsMaskedUrl:
    """Tests for masked_url properties on settings classes."""

    def test_database_settings_masked_url(self):
        from maverick_mcp.config.settings import DatabaseSettings

        db = DatabaseSettings()
        # Default is sqlite with no credentials — should pass through
        assert db.masked_url == db.url

    def test_redis_settings_masked_url_no_password(self):
        from maverick_mcp.config.settings import RedisSettings

        redis = RedisSettings(password=None)
        assert redis.masked_url == redis.url
        assert "***" not in redis.masked_url

    def test_redis_settings_masked_url_with_password(self):
        from maverick_mcp.config.settings import RedisSettings

        redis = RedisSettings(password="topsecret", host="localhost", port=6379, db=0)
        assert "topsecret" not in redis.masked_url
        assert "***" in redis.masked_url
        # Real url should still have the password for connections
        assert "topsecret" in redis.url


# --------------------------------------------------------------------------- #
# 3. Ticker validation — portfolio router
# --------------------------------------------------------------------------- #


class TestTickerValidation:
    """Tests for the fixed _validate_ticker in portfolio router."""

    @pytest.fixture(autouse=True)
    def _import_validate(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker

        self.validate = _validate_ticker

    def test_simple_ticker_valid(self):
        valid, err = self.validate("AAPL")
        assert valid is True
        assert err is None

    def test_dot_ticker_valid(self):
        """BRK.B should be accepted (was rejected by isalnum)."""
        valid, err = self.validate("BRK.B")
        assert valid is True
        assert err is None

    def test_hyphen_ticker_valid(self):
        """BF-A should be accepted (was rejected by isalnum)."""
        valid, err = self.validate("BF-A")
        assert valid is True
        assert err is None

    def test_empty_ticker_rejected(self):
        valid, err = self.validate("")
        assert valid is False
        assert err is not None

    def test_whitespace_only_rejected(self):
        valid, err = self.validate("   ")
        assert valid is False

    def test_special_chars_rejected(self):
        valid, err = self.validate("AA$PL")
        assert valid is False

    def test_too_long_rejected(self):
        valid, err = self.validate("A" * 11)
        assert valid is False

    def test_lowercase_normalised(self):
        """Lowercase input should be normalised to uppercase before regex check."""
        valid, err = self.validate("aapl")
        assert valid is True


# --------------------------------------------------------------------------- #
# 4. Security headers — middleware
# --------------------------------------------------------------------------- #


class TestSecurityHeaders:
    """Tests for CSP and Permissions-Policy in security middleware."""

    def test_csp_header_present(self):
        """Verify Content-Security-Policy is set in middleware source."""
        from maverick_mcp.api.middleware import security

        src = importlib.util.find_spec(security.__name__)
        assert src is not None
        source = open(src.origin).read()
        assert "Content-Security-Policy" in source

    def test_permissions_policy_present(self):
        """Verify Permissions-Policy is set in middleware source."""
        from maverick_mcp.api.middleware import security

        source = open(importlib.util.find_spec(security.__name__).origin).read()
        assert "Permissions-Policy" in source

    def test_existing_headers_preserved(self):
        """Verify legacy headers are still present."""
        from maverick_mcp.api.middleware import security

        source = open(importlib.util.find_spec(security.__name__).origin).read()
        assert "X-Content-Type-Options" in source
        assert "X-Frame-Options" in source
        assert "Referrer-Policy" in source

    def test_csp_includes_key_directives(self):
        """Verify CSP value contains essential directives."""
        from maverick_mcp.api.middleware import security

        source = open(importlib.util.find_spec(security.__name__).origin).read()
        assert "default-src" in source
        assert "script-src" in source
        assert "frame-src" in source


# --------------------------------------------------------------------------- #
# 5. SQL text() wrapper — static import checks
# --------------------------------------------------------------------------- #


class TestSqlTextImports:
    """Verify that sqlalchemy.text is imported in files that were fixed."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "maverick_mcp.api.routers.health_enhanced",
            "maverick_mcp.infrastructure.caching.cache_management_service",
            "maverick_mcp.infrastructure.health.health_checker",
        ],
    )
    def test_text_imported(self, module_path):
        """Check that 'from sqlalchemy import text' exists in the module source."""
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"Module {module_path} not found"
        source = open(spec.origin).read()
        tree = ast.parse(source)

        text_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "sqlalchemy" and any(
                    alias.name == "text" for alias in node.names
                ):
                    text_imported = True
                    break

        assert text_imported, (
            f"'from sqlalchemy import text' not found in {module_path}"
        )

    @pytest.mark.parametrize(
        "module_path",
        [
            "maverick_mcp.api.routers.health_enhanced",
            "maverick_mcp.infrastructure.caching.cache_management_service",
            "maverick_mcp.infrastructure.health.health_checker",
        ],
    )
    def test_no_raw_sql_strings(self, module_path):
        """Ensure execute() calls use text() wrapper, not raw strings."""
        spec = importlib.util.find_spec(module_path)
        source = open(spec.origin).read()
        # Match execute("...") but NOT execute(text("..."))
        raw_sql = re.findall(r'\.execute\(\s*"', source)
        assert len(raw_sql) == 0, (
            f"Found raw SQL string in execute() call in {module_path}. "
            "Use text() wrapper instead."
        )
