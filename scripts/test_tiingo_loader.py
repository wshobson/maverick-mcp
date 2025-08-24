#!/usr/bin/env python3
"""
Test script for the Tiingo data loader.

This script performs basic validation that the loader components work correctly
without requiring an actual API call or database connection.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_tiingo_data import (
    SP500_SYMBOLS,
    ProgressTracker,
    RateLimiter,
    TiingoDataLoader,
)
from scripts.tiingo_config import (
    SCREENING_CONFIGS,
    SYMBOL_LISTS,
    TiingoConfig,
    get_config_for_environment,
)


class TestProgressTracker(unittest.TestCase):
    """Test the progress tracking functionality."""

    def setUp(self):
        self.tracker = ProgressTracker("test_progress.json")

    def test_initialization(self):
        """Test that progress tracker initializes correctly."""
        self.assertEqual(self.tracker.processed_symbols, 0)
        self.assertEqual(self.tracker.successful_symbols, 0)
        self.assertEqual(len(self.tracker.failed_symbols), 0)
        self.assertEqual(len(self.tracker.completed_symbols), 0)

    def test_update_progress_success(self):
        """Test updating progress for successful symbol."""
        self.tracker.total_symbols = 5
        self.tracker.update_progress("AAPL", True)

        self.assertEqual(self.tracker.processed_symbols, 1)
        self.assertEqual(self.tracker.successful_symbols, 1)
        self.assertIn("AAPL", self.tracker.completed_symbols)
        self.assertEqual(len(self.tracker.failed_symbols), 0)

    def test_update_progress_failure(self):
        """Test updating progress for failed symbol."""
        self.tracker.total_symbols = 5
        self.tracker.update_progress("BADSTOCK", False, "Not found")

        self.assertEqual(self.tracker.processed_symbols, 1)
        self.assertEqual(self.tracker.successful_symbols, 0)
        self.assertIn("BADSTOCK", self.tracker.failed_symbols)
        self.assertEqual(len(self.tracker.errors), 1)


class TestRateLimiter(unittest.TestCase):
    """Test the rate limiting functionality."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(3600)  # 1 request per second
        self.assertEqual(limiter.max_requests, 3600)
        self.assertEqual(limiter.min_interval, 1.0)

    def test_tiingo_rate_limit(self):
        """Test Tiingo-specific rate limit calculation."""
        limiter = RateLimiter(2400)  # Tiingo free tier
        expected_interval = 3600.0 / 2400  # 1.5 seconds
        self.assertEqual(limiter.min_interval, expected_interval)


class TestTiingoConfig(unittest.TestCase):
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TiingoConfig()

        self.assertEqual(config.rate_limit_per_hour, 2400)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.default_batch_size, 50)
        self.assertEqual(config.rsi_period, 14)
        self.assertIsInstance(config.sma_periods, list)
        self.assertIn(50, config.sma_periods)
        self.assertIn(200, config.sma_periods)

    def test_environment_configs(self):
        """Test environment-specific configurations."""
        dev_config = get_config_for_environment("development")
        prod_config = get_config_for_environment("production")
        test_config = get_config_for_environment("testing")

        # Production should have higher limits
        self.assertGreaterEqual(
            prod_config.max_concurrent_requests, dev_config.max_concurrent_requests
        )
        self.assertGreaterEqual(
            prod_config.default_batch_size, dev_config.default_batch_size
        )

        # Test should have lower limits
        self.assertLessEqual(
            test_config.max_concurrent_requests, dev_config.max_concurrent_requests
        )
        self.assertLessEqual(
            test_config.default_batch_size, dev_config.default_batch_size
        )

    def test_symbol_lists(self):
        """Test that symbol lists are properly configured."""
        self.assertIn("sp500_top_100", SYMBOL_LISTS)
        self.assertIn("nasdaq_100", SYMBOL_LISTS)
        self.assertIn("dow_30", SYMBOL_LISTS)

        # Check that lists have reasonable sizes
        self.assertGreater(len(SYMBOL_LISTS["sp500_top_100"]), 50)
        self.assertLess(len(SYMBOL_LISTS["dow_30"]), 35)

    def test_screening_configs(self):
        """Test screening algorithm configurations."""
        maverick_config = SCREENING_CONFIGS["maverick_momentum"]

        self.assertIn("min_momentum_score", maverick_config)
        self.assertIn("scoring_weights", maverick_config)
        self.assertIsInstance(maverick_config["scoring_weights"], dict)


class TestTiingoDataLoader(unittest.TestCase):
    """Test the main TiingoDataLoader class."""

    @patch.dict("os.environ", {"TIINGO_API_TOKEN": "test_token"})
    def test_initialization(self):
        """Test loader initialization."""
        loader = TiingoDataLoader(batch_size=25, max_concurrent=3)

        self.assertEqual(loader.batch_size, 25)
        self.assertEqual(loader.max_concurrent, 3)
        self.assertEqual(loader.api_token, "test_token")
        self.assertIsNotNone(loader.rate_limiter)

    def test_initialization_without_token(self):
        """Test that loader fails without API token."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                TiingoDataLoader()

    @patch("aiohttp.ClientSession")
    async def test_context_manager(self, mock_session_class):
        """Test async context manager functionality."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        with patch.dict("os.environ", {"TIINGO_API_TOKEN": "test_token"}):
            async with TiingoDataLoader() as loader:
                self.assertIsNotNone(loader.session)

            # Session should be closed after context exit
            mock_session.close.assert_called_once()


class TestSymbolValidation(unittest.TestCase):
    """Test symbol validation and processing."""

    def test_sp500_symbols(self):
        """Test that S&P 500 symbols are valid."""
        self.assertIsInstance(SP500_SYMBOLS, list)
        self.assertGreater(len(SP500_SYMBOLS), 90)  # Should have at least 90 symbols

        # Check that symbols are uppercase strings
        for symbol in SP500_SYMBOLS[:10]:  # Check first 10
            self.assertIsInstance(symbol, str)
            self.assertEqual(symbol, symbol.upper())
            self.assertGreater(len(symbol), 0)
            self.assertLess(len(symbol), 10)  # Reasonable symbol length


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_symbol_file_content(self):
        """Test the format that would be expected in symbol files."""
        # Test comma-separated format
        test_content = "AAPL,MSFT,GOOGL\nTSLA,NVDA\n# Comment\nAMZN"
        lines = test_content.split("\n")

        symbols = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                line_symbols = [s.strip().upper() for s in line.split(",")]
                symbols.extend(line_symbols)

        expected = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN"]
        self.assertEqual(symbols, expected)


def run_basic_validation():
    """Run basic validation without external dependencies."""
    print("🧪 Running basic validation tests...")

    # Test imports
    try:
        from scripts.load_tiingo_data import ProgressTracker
        from scripts.tiingo_config import SYMBOL_LISTS, TiingoConfig

        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test configuration
    try:
        config = TiingoConfig()
        assert config.rate_limit_per_hour == 2400
        assert len(config.sma_periods) > 0
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

    # Test symbol lists
    try:
        assert len(SP500_SYMBOLS) > 90
        assert len(SYMBOL_LISTS["sp500_top_100"]) > 90
        assert all(isinstance(s, str) for s in SP500_SYMBOLS[:10])
        print("✅ Symbol list validation passed")
    except Exception as e:
        print(f"❌ Symbol list error: {e}")
        return False

    # Test progress tracker
    try:
        tracker = ProgressTracker("test.json")
        tracker.update_progress("TEST", True)
        assert tracker.successful_symbols == 1
        assert "TEST" in tracker.completed_symbols
        print("✅ Progress tracker validation passed")
    except Exception as e:
        print(f"❌ Progress tracker error: {e}")
        return False

    print("🎉 All basic validations passed!")
    return True


if __name__ == "__main__":
    print("Tiingo Data Loader Test Suite")
    print("=" * 40)

    # Run basic validation first
    if not run_basic_validation():
        sys.exit(1)

    # Run unit tests
    print("\n🧪 Running unit tests...")
    unittest.main(verbosity=2, exit=False)

    print("\n✅ Test suite completed!")
