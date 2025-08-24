"""
Tests for enhanced database session management.

Tests the new context managers and connection pool monitoring
introduced to fix Issue #55: Database Session Management.
"""

from unittest.mock import Mock, patch

import pytest

from maverick_mcp.data.session_management import (
    check_connection_pool_health,
    get_connection_pool_status,
    get_db_session,
    get_db_session_read_only,
)


class TestSessionManagement:
    """Test suite for database session management context managers."""

    @patch("maverick_mcp.data.session_management.SessionLocal")
    def test_get_db_session_success(self, mock_session_local):
        """Test successful database session with automatic commit."""
        mock_session = Mock()
        mock_session_local.return_value = mock_session

        with get_db_session() as session:
            assert session == mock_session
            # Simulate some database operation

        # Verify session lifecycle
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @patch("maverick_mcp.data.session_management.SessionLocal")
    def test_get_db_session_exception_rollback(self, mock_session_local):
        """Test database session rollback on exception."""
        mock_session = Mock()
        mock_session_local.return_value = mock_session

        with pytest.raises(ValueError):
            with get_db_session() as session:
                assert session == mock_session
                raise ValueError("Test exception")

        # Verify rollback was called, but not commit
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()

    @patch("maverick_mcp.data.session_management.SessionLocal")
    def test_get_db_session_read_only_success(self, mock_session_local):
        """Test read-only database session (no commit)."""
        mock_session = Mock()
        mock_session_local.return_value = mock_session

        with get_db_session_read_only() as session:
            assert session == mock_session
            # Simulate some read-only operation

        # Verify no commit for read-only operations
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @patch("maverick_mcp.data.session_management.SessionLocal")
    def test_get_db_session_read_only_exception_rollback(self, mock_session_local):
        """Test read-only database session rollback on exception."""
        mock_session = Mock()
        mock_session_local.return_value = mock_session

        with pytest.raises(RuntimeError):
            with get_db_session_read_only() as session:
                assert session == mock_session
                raise RuntimeError("Read operation failed")

        # Verify rollback was called, but not commit
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()


class TestConnectionPoolMonitoring:
    """Test suite for connection pool monitoring functionality."""

    @patch("maverick_mcp.data.models.engine")
    def test_get_connection_pool_status(self, mock_engine):
        """Test connection pool status reporting."""
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 5
        mock_pool.checkedout.return_value = 3
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_engine.pool = mock_pool

        status = get_connection_pool_status()

        expected = {
            "pool_size": 10,
            "checked_in": 5,
            "checked_out": 3,
            "overflow": 0,
            "invalid": 0,
            "pool_status": "healthy",  # 3/10 = 30% < 80%
        }
        assert status == expected

    @patch("maverick_mcp.data.models.engine")
    def test_get_connection_pool_status_warning(self, mock_engine):
        """Test connection pool status with high utilization warning."""
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 1
        mock_pool.checkedout.return_value = 9  # 90% utilization
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_engine.pool = mock_pool

        status = get_connection_pool_status()

        assert status["pool_status"] == "warning"
        assert status["checked_out"] == 9

    @patch("maverick_mcp.data.session_management.get_connection_pool_status")
    def test_check_connection_pool_health_healthy(self, mock_get_status):
        """Test connection pool health check - healthy scenario."""
        mock_get_status.return_value = {
            "pool_size": 10,
            "checked_out": 5,  # 50% utilization
            "invalid": 0,
        }

        assert check_connection_pool_health() is True

    @patch("maverick_mcp.data.session_management.get_connection_pool_status")
    def test_check_connection_pool_health_high_utilization(self, mock_get_status):
        """Test connection pool health check - high utilization."""
        mock_get_status.return_value = {
            "pool_size": 10,
            "checked_out": 9,  # 90% utilization > 80% threshold
            "invalid": 0,
        }

        assert check_connection_pool_health() is False

    @patch("maverick_mcp.data.session_management.get_connection_pool_status")
    def test_check_connection_pool_health_invalid_connections(self, mock_get_status):
        """Test connection pool health check - invalid connections detected."""
        mock_get_status.return_value = {
            "pool_size": 10,
            "checked_out": 3,  # Low utilization
            "invalid": 2,  # But has invalid connections
        }

        assert check_connection_pool_health() is False

    @patch("maverick_mcp.data.session_management.get_connection_pool_status")
    def test_check_connection_pool_health_exception(self, mock_get_status):
        """Test connection pool health check with exception handling."""
        mock_get_status.side_effect = Exception("Pool access failed")

        assert check_connection_pool_health() is False


class TestSessionManagementIntegration:
    """Integration tests for session management with real database."""

    @pytest.mark.integration
    def test_session_context_manager_real_db(self):
        """Test session context manager with real database connection."""
        try:
            with get_db_session_read_only() as session:
                # Simple test query that should work on any PostgreSQL database
                result = session.execute("SELECT 1 as test_value")
                row = result.fetchone()
                assert row[0] == 1
        except Exception as e:
            # If database is not available, skip this test
            pytest.skip(f"Database not available for integration test: {e}")

    @pytest.mark.integration
    def test_connection_pool_status_real(self):
        """Test connection pool status with real database."""
        try:
            status = get_connection_pool_status()

            # Verify the status has expected keys
            required_keys = [
                "pool_size",
                "checked_in",
                "checked_out",
                "overflow",
                "invalid",
                "pool_status",
            ]
            for key in required_keys:
                assert key in status

            # Verify status values are reasonable
            assert isinstance(status["pool_size"], int)
            assert status["pool_size"] > 0
            assert status["pool_status"] in ["healthy", "warning"]

        except Exception as e:
            # If database is not available, skip this test
            pytest.skip(f"Database not available for integration test: {e}")
