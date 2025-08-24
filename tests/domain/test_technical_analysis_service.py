"""
Unit tests for the TechnicalAnalysisService domain service.

These tests demonstrate that the domain service can be tested
without any infrastructure dependencies (no mocks needed).
"""

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.domain.services.technical_analysis_service import (
    TechnicalAnalysisService,
)
from maverick_mcp.domain.value_objects.technical_indicators import (
    Signal,
    TrendDirection,
)


class TestTechnicalAnalysisService:
    """Test the technical analysis domain service."""

    @pytest.fixture
    def service(self):
        """Create a technical analysis service instance."""
        return TechnicalAnalysisService()

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        # Generate synthetic price data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        return pd.Series(prices, index=dates)

    @pytest.fixture
    def sample_ohlc(self):
        """Create sample OHLC data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        close = 100 + np.cumsum(np.random.randn(100) * 2)

        # Generate high/low based on close
        high = close + np.abs(np.random.randn(100))
        low = close - np.abs(np.random.randn(100))

        return pd.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
            },
            index=dates,
        )

    def test_calculate_rsi(self, service, sample_prices):
        """Test RSI calculation."""
        rsi = service.calculate_rsi(sample_prices, period=14)

        # RSI should be between 0 and 100
        assert 0 <= rsi.value <= 100
        assert rsi.period == 14

        # Check signal logic
        if rsi.value >= 70:
            assert rsi.is_overbought
        if rsi.value <= 30:
            assert rsi.is_oversold

    def test_calculate_rsi_insufficient_data(self, service):
        """Test RSI with insufficient data."""
        prices = pd.Series([100, 101, 102])  # Only 3 prices

        with pytest.raises(ValueError, match="Need at least 14 prices"):
            service.calculate_rsi(prices, period=14)

    def test_calculate_macd(self, service, sample_prices):
        """Test MACD calculation."""
        macd = service.calculate_macd(sample_prices)

        # Check structure
        assert hasattr(macd, "macd_line")
        assert hasattr(macd, "signal_line")
        assert hasattr(macd, "histogram")

        # Histogram should be difference between MACD and signal
        assert abs(macd.histogram - (macd.macd_line - macd.signal_line)) < 0.01

        # Check signal logic
        if macd.macd_line > macd.signal_line and macd.histogram > 0:
            assert macd.is_bullish_crossover
        if macd.macd_line < macd.signal_line and macd.histogram < 0:
            assert macd.is_bearish_crossover

    def test_calculate_bollinger_bands(self, service, sample_prices):
        """Test Bollinger Bands calculation."""
        bb = service.calculate_bollinger_bands(sample_prices)

        # Check structure
        assert bb.upper_band > bb.middle_band
        assert bb.middle_band > bb.lower_band
        assert bb.period == 20
        assert bb.std_dev == 2

        # Check bandwidth calculation
        expected_bandwidth = (bb.upper_band - bb.lower_band) / bb.middle_band
        assert abs(bb.bandwidth - expected_bandwidth) < 0.01

        # Check %B calculation
        expected_percent_b = (bb.current_price - bb.lower_band) / (
            bb.upper_band - bb.lower_band
        )
        assert abs(bb.percent_b - expected_percent_b) < 0.01

    def test_calculate_stochastic(self, service, sample_ohlc):
        """Test Stochastic Oscillator calculation."""
        stoch = service.calculate_stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            period=14,
        )

        # Values should be between 0 and 100
        assert 0 <= stoch.k_value <= 100
        assert 0 <= stoch.d_value <= 100
        assert stoch.period == 14

        # Check overbought/oversold logic
        if stoch.k_value >= 80:
            assert stoch.is_overbought
        if stoch.k_value <= 20:
            assert stoch.is_oversold

    def test_identify_trend_uptrend(self, service):
        """Test trend identification for uptrend."""
        # Create clear uptrend data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = pd.Series(range(100, 200), index=dates)  # Linear uptrend

        trend = service.identify_trend(prices, period=50)
        assert trend in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND]

    def test_identify_trend_downtrend(self, service):
        """Test trend identification for downtrend."""
        # Create clear downtrend data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = pd.Series(range(200, 100, -1), index=dates)  # Linear downtrend

        trend = service.identify_trend(prices, period=50)
        assert trend in [TrendDirection.DOWNTREND, TrendDirection.STRONG_DOWNTREND]

    def test_analyze_volume(self, service):
        """Test volume analysis."""
        # Create volume data with spike
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        volume = pd.Series([1000000] * 29 + [3000000], index=dates)  # Spike at end

        volume_profile = service.analyze_volume(volume, period=20)

        assert volume_profile.current_volume == 3000000
        assert volume_profile.average_volume < 1500000
        assert volume_profile.relative_volume > 2.0
        assert volume_profile.unusual_activity  # 3x average is unusual

    def test_calculate_composite_signal_bullish(self, service):
        """Test composite signal calculation with bullish indicators."""
        # Manually create bullish indicators for testing
        from maverick_mcp.domain.value_objects.technical_indicators import (
            MACDIndicator,
            RSIIndicator,
        )

        bullish_rsi = RSIIndicator(value=25, period=14)  # Oversold
        bullish_macd = MACDIndicator(
            macd_line=1.0,
            signal_line=0.5,
            histogram=0.5,
        )  # Bullish crossover

        signal = service.calculate_composite_signal(
            rsi=bullish_rsi,
            macd=bullish_macd,
        )

        assert signal in [Signal.BUY, Signal.STRONG_BUY]

    def test_calculate_composite_signal_mixed(self, service):
        """Test composite signal with mixed indicators."""
        from maverick_mcp.domain.value_objects.technical_indicators import (
            BollingerBands,
            MACDIndicator,
            RSIIndicator,
        )

        # Create mixed signals
        neutral_rsi = RSIIndicator(value=50, period=14)  # Neutral
        bearish_macd = MACDIndicator(
            macd_line=-0.5,
            signal_line=0.0,
            histogram=-0.5,
        )  # Bearish
        neutral_bb = BollingerBands(
            upper_band=110,
            middle_band=100,
            lower_band=90,
            current_price=100,
        )  # Neutral

        signal = service.calculate_composite_signal(
            rsi=neutral_rsi,
            macd=bearish_macd,
            bollinger=neutral_bb,
        )

        # With mixed signals, should be neutral or slightly bearish
        assert signal in [Signal.NEUTRAL, Signal.SELL]

    def test_domain_service_has_no_infrastructure_dependencies(self, service):
        """Verify the domain service has no infrastructure dependencies."""
        # Check that the service has no database, API, or cache attributes
        assert not hasattr(service, "db")
        assert not hasattr(service, "session")
        assert not hasattr(service, "cache")
        assert not hasattr(service, "api_client")
        assert not hasattr(service, "http_client")

        # Check that all methods are pure functions (no side effects)
        # This is verified by the fact that all tests above work without mocks
