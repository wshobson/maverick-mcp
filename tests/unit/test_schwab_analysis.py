from unittest.mock import MagicMock

from maverick_mcp.providers.schwab.analysis import refresh_and_analyze_portfolio


def test_refresh_and_analyze_portfolio_summarizes_live_positions(monkeypatch):
    client = MagicMock()
    client.accounts.return_value = [
        {
            "securitiesAccount": {
                "type": "CASH",
                "currentBalances": {
                    "cashBalance": 1000,
                    "liquidationValue": 2600,
                },
                "positions": [
                    {
                        "longQuantity": 10,
                        "averagePrice": 100,
                        "marketValue": 1600,
                        "instrument": {"symbol": "AAPL", "assetType": "EQUITY"},
                    }
                ],
            }
        }
    ]
    monkeypatch.setattr(
        "maverick_mcp.providers.schwab.analysis.sync_schwab_portfolio",
        lambda *args, **kwargs: {
            "status": "ok",
            "positions_synced": len(kwargs["positions"]),
        },
    )

    result = refresh_and_analyze_portfolio(client)

    assert result["status"] == "ok"
    assert result["summary"]["position_count"] == 1
    assert result["summary"]["cash_balance"] == 1000.0
    assert result["summary"]["positions_market_value"] == 1600.0
    assert result["summary"]["total_cost_basis"] == 1000.0
    assert result["summary"]["unrealized_pnl"] == 600.0
    assert result["top_positions"][0]["ticker"] == "AAPL"
    assert result["top_positions"][0]["allocation_pct"] == 100.0
    assert result["concentration_warnings"][0]["ticker"] == "AAPL"
