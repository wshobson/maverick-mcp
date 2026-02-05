from unittest.mock import AsyncMock

import pandas as pd
import pytest

from maverick_mcp.infrastructure.persistence.stock_repository import (
    StockDataProviderAdapter,
)


@pytest.fixture
def mock_stock_provider():
    return AsyncMock()


@pytest.fixture
def adapter(mock_stock_provider):
    return StockDataProviderAdapter(stock_provider=mock_stock_provider)


@pytest.mark.asyncio
async def test_get_price_data_async_delegates_to_async_provider(
    adapter, mock_stock_provider
):
    # Setup
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [110.0],
            "Low": [90.0],
            "Close": [105.0],
            "Volume": [1000],
        }
    )
    mock_stock_provider.get_stock_data_async.return_value = df

    # Execute
    result = await adapter.get_price_data_async("AAPL", "2023-01-01", "2023-01-31")

    # Verify
    mock_stock_provider.get_stock_data_async.assert_called_once_with(
        "AAPL", "2023-01-01", "2023-01-31"
    )
    assert result is df
    assert "open" in result.columns
    assert "Open" not in result.columns
