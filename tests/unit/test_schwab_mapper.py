from decimal import Decimal

from maverick_mcp.providers.schwab.mapper import normalize_positions


def test_normalize_positions_maps_valid_long_positions():
    accounts = [
        {
            "securitiesAccount": {
                "positions": [
                    {
                        "longQuantity": 10.0,
                        "averagePrice": 150.25,
                        "marketValue": 1600.0,
                        "instrument": {"symbol": "AAPL", "assetType": "EQUITY"},
                    }
                ]
            }
        }
    ]

    positions = normalize_positions(accounts)

    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
    assert positions[0].shares == Decimal("10.0")
    assert positions[0].average_price == Decimal("150.25")
    assert positions[0].total_cost == Decimal("1502.500")


def test_normalize_positions_skips_positions_without_cost_basis():
    accounts = [
        {
            "securitiesAccount": {
                "positions": [
                    {
                        "longQuantity": 2.0,
                        "instrument": {"symbol": "MSFT", "assetType": "EQUITY"},
                    }
                ]
            }
        }
    ]

    assert normalize_positions(accounts) == []
