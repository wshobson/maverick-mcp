"""Watchlist mutations must not create or silently operate on missing lists."""

from unittest.mock import Mock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from maverick.market_data.service import MarketDataService
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings
from maverick.portfolio.service import PortfolioService
from maverick.portfolio.watchlist import WATCHLIST_ITEMS


@pytest.mark.parametrize("operation", ["add", "remove"])
async def test_watchlist_mutation_rejects_missing_id_without_writing(
    tmp_path, operation
):
    engine = create_engine_from_settings(
        DatabaseSettings(url=f"sqlite:///{tmp_path / 'watchlists.db'}")
    )
    market_data = Mock(spec=MarketDataService)
    service = PortfolioService(engine, market_data)
    try:
        with pytest.raises(ValueError, match="Watchlist 999999 not found"):
            if operation == "add":
                await service.add_watchlist_item(999999, "AAPL")
            else:
                await service.remove_watchlist_item(999999, "AAPL")
        with sessionmaker(bind=engine)() as session:
            assert session.execute(select(WATCHLIST_ITEMS)).all() == []
        market_data.get_quote.assert_not_called()
    finally:
        engine.dispose()
