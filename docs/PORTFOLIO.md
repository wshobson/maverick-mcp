# Portfolio Management Guide

Complete guide to using MaverickMCP's portfolio personalization features for intelligent, context-aware stock analysis.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Portfolio Management](#portfolio-management)
- [Intelligent Analysis](#intelligent-analysis)
- [MCP Resource](#mcp-resource)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

## Overview

MaverickMCP's portfolio features transform your AI financial assistant from stateless analysis to personalized, context-aware recommendations. The system:

- **Tracks your holdings** with automatic cost basis averaging
- **Calculates live P&L** using real-time market data
- **Enhances analysis tools** to auto-detect your positions
- **Provides AI context** through MCP resources

**DISCLAIMER**: All portfolio features are for educational purposes only. This is not investment advice. Always consult qualified financial professionals before making investment decisions.

## Quick Start

### 1. Add Your First Position

```
Add 10 shares of Apple stock I bought at $150.50 on January 15, 2024
```

Behind the scenes, this uses:
```python
portfolio_add_position(
    ticker="AAPL",
    shares=10,
    purchase_price=150.50,
    purchase_date="2024-01-15"
)
```

### 2. View Your Portfolio

```
Show me my portfolio
```

Response includes:
- All positions with current prices
- Unrealized P&L for each position
- Total portfolio value and performance
- Diversification metrics

### 3. Smart Analysis

```
Compare my portfolio holdings
```

Automatically compares all your positions without manual ticker entry!

## Portfolio Management

### Adding Positions

**Add a new position:**
```
Add 50 shares of Microsoft at $380.25
```

**Add to existing position (automatic cost averaging):**
```
Add 25 more shares of Apple at $165.00
```

The system automatically:
- Averages your cost basis
- Updates total investment
- Preserves earliest purchase date

**Example:**
- Initial: 10 shares @ $150 = $1,500 total cost, $150 avg cost
- Add: 10 shares @ $170 = $1,700 total cost
- Result: 20 shares, $160 avg cost, $3,200 total invested

### Viewing Positions

**Get complete portfolio:**
```
Show my portfolio with current prices
```

Returns:
```json
{
  "portfolio_name": "My Portfolio",
  "positions": [
    {
      "ticker": "AAPL",
      "shares": 20.0,
      "average_cost_basis": 160.00,
      "current_price": 175.50,
      "unrealized_pnl": 310.00,
      "unrealized_pnl_pct": 9.69
    }
  ],
  "total_value": 3510.00,
  "total_invested": 3200.00,
  "total_pnl": 310.00,
  "total_pnl_pct": 9.69
}
```

### Removing Positions

**Partial sale:**
```
Sell 10 shares of Apple
```

Maintains average cost basis on remaining shares.

**Full position exit:**
```
Remove all my Tesla shares
```

or simply:
```
Remove TSLA
```

### Clearing Portfolio

**Remove all positions:**
```
Clear my entire portfolio
```

Requires confirmation for safety.

## Intelligent Analysis

### Auto-Compare Holdings

Instead of:
```
Compare AAPL, MSFT, GOOGL, TSLA
```

Simply use:
```
Compare my holdings
```

The tool automatically:
- Pulls all tickers from your portfolio
- Analyzes relative performance
- Ranks by metrics
- Shows best/worst performers

### Auto-Correlation Analysis

```
Analyze correlation in my portfolio
```

Automatically:
- Calculates correlation matrix for all holdings
- Identifies highly correlated pairs (diversification issues)
- Finds negative correlations (natural hedges)
- Provides diversification score

Example output:
```json
{
  "average_portfolio_correlation": 0.612,
  "diversification_score": 38.8,
  "high_correlation_pairs": [
    {
      "pair": ["AAPL", "MSFT"],
      "correlation": 0.823,
      "interpretation": "High positive correlation"
    }
  ],
  "recommendation": "Consider adding uncorrelated assets"
}
```

### Position-Aware Risk Analysis

```
Analyze AAPL with risk analysis
```

If you own AAPL, automatically shows:
- Your current position (shares, cost basis)
- Unrealized P&L
- Position sizing recommendations
- Averaging down/up suggestions

Example with existing position:
```json
{
  "ticker": "AAPL",
  "current_price": 175.50,
  "existing_position": {
    "shares_owned": 20.0,
    "average_cost_basis": 160.00,
    "total_invested": 3200.00,
    "current_value": 3510.00,
    "unrealized_pnl": 310.00,
    "unrealized_pnl_pct": 9.69,
    "position_recommendation": "Hold current position"
  }
}
```

## MCP Resource

### portfolio://my-holdings Resource

The portfolio resource provides automatic context to AI agents during conversations.

**Accessed via:**
```
What's in my portfolio?
```

The AI automatically sees:
- All current positions
- Live prices and P&L
- Portfolio composition
- Diversification status

This enables natural conversations:
```
Should I add more tech exposure?
```

The AI knows you already own AAPL, MSFT, GOOGL and can provide personalized advice.

## Best Practices

### Cost Basis Tracking

- **Always specify purchase date** for accurate records
- **Add notes** for important context (e.g., "RSU vest", "DCA purchase #3")
- **Review regularly** to ensure accuracy

### Diversification

- Use `portfolio_correlation_analysis()` monthly
- Watch for correlations above 0.7 (concentration risk)
- Consider uncorrelated assets when diversification score < 50

### Position Sizing

- Use `risk_adjusted_analysis()` before adding to positions
- Follow position sizing recommendations
- Respect stop-loss suggestions

### Maintenance

- **Weekly**: Review portfolio performance
- **Monthly**: Analyze correlations
- **Quarterly**: Rebalance based on analysis tools

## Troubleshooting

### "No portfolio found"

**Problem**: Trying to use auto-detection features without any positions.

**Solution**: Add at least one position:
```
Add 1 share of SPY at current price
```

### "Insufficient positions for comparison"

**Problem**: Need minimum 2 positions for comparison/correlation.

**Solution**: Add another position or specify tickers manually:
```
Compare AAPL, MSFT
```

### "Invalid ticker symbol"

**Problem**: Ticker doesn't exist or is incorrectly formatted.

**Solution**:
- Check ticker spelling
- Verify symbol on financial websites
- Use standard format (e.g., "BRK.B" not "BRKB")

### Stale Price Data

**Problem**: Portfolio shows old prices.

**Solution**: Refresh by calling `get_my_portfolio(include_current_prices=True)`

### Position Not Found

**Problem**: Trying to remove shares from position you don't own.

**Solution**: Check your portfolio first:
```
Show my portfolio
```

## Technical Details

### Cost Basis Method

**Average Cost Method**: Simplest and most appropriate for educational use.

Formula:
```
New Avg Cost = (Existing Total Cost + New Purchase Cost) / Total Shares
```

Example:
- Buy 10 @ $100 = $1,000 total, $100 avg
- Buy 10 @ $120 = $1,200 additional
- Result: 20 shares, $110 avg cost ($2,200 / 20)

### Database Schema

**Tables:**
- `mcp_portfolios`: User portfolio metadata
- `mcp_portfolio_positions`: Individual positions

**Precision:**
- Shares: Numeric(20,8) - supports fractional shares
- Prices: Numeric(12,4) - 4 decimal precision
- Total Cost: Numeric(20,4) - high precision for large positions

### Supported Features

✅ Fractional shares (0.001 minimum)
✅ Multiple portfolios per user
✅ Automatic cost averaging
✅ Live P&L calculations
✅ Position notes/annotations
✅ Timezone-aware timestamps
✅ Cascade deletion (portfolio → positions)

### Limitations

- Single currency (USD)
- Stock equities only (no options, futures, crypto)
- Average cost method only (no FIFO/LIFO)
- No tax lot tracking
- No dividend tracking (planned for future)
- No transaction history (planned for future)

### Data Sources

- **Historical Prices**: Tiingo API (free tier: 500 req/day)
- **Live Prices**: Same as historical (delayed 15 minutes on free tier)
- **Company Info**: Pre-seeded S&P 500 database

### Performance

- **Database**: SQLite default (PostgreSQL optional for better performance)
- **Caching**: In-memory by default (Redis optional)
- **Price Fetching**: Sequential (batch optimization in Phase 3)
- **Query Optimization**: selectin loading for relationships

### Privacy & Security

- **Local-first**: All data stored locally in your database
- **No cloud sync**: Portfolio data never leaves your machine
- **No authentication**: Personal use only (no multi-user)
- **No external sharing**: Data accessible only to you

## Migration Guide

### Upgrading from No Portfolio

1. Start MaverickMCP server
2. Migration runs automatically on first startup
3. Add your first position
4. Verify with `get_my_portfolio()`

### Downgrading (Rollback)

```bash
# Backup first
cp maverick_mcp.db maverick_mcp.db.backup

# Rollback migration
alembic downgrade -1

# Verify
alembic current
```

### Exporting Portfolio

Currently manual:
```bash
sqlite3 maverick_mcp.db "SELECT * FROM mcp_portfolio_positions;" > portfolio_export.csv
```

Future: Built-in export tool planned.

## FAQs

**Q: Can I track multiple portfolios?**
A: Yes! Use the `portfolio_name` parameter:
```python
add_portfolio_position("AAPL", 10, 150, portfolio_name="IRA")
add_portfolio_position("VOO", 5, 400, portfolio_name="401k")
```

**Q: What happens if I add wrong data?**
A: Simply remove the position and re-add:
```
Remove AAPL
Add 10 shares of AAPL at $150.50
```

**Q: Can I track realized gains?**
A: Not yet. Currently tracks unrealized P&L only. Transaction history is planned for future release.

**Q: Is my data backed up?**
A: No automatic backups. Manually copy `maverick_mcp.db` regularly:
```bash
cp maverick_mcp.db ~/backups/maverick_mcp_$(date +%Y%m%d).db
```

**Q: Can I use this for tax purposes?**
A: **NO**. This is educational software only. Use professional tax software for tax reporting.

---

## Getting Help

- **Issues**: https://github.com/wshobson/maverick-mcp/issues
- **Discussions**: https://github.com/wshobson/maverick-mcp/discussions
- **Documentation**: https://github.com/wshobson/maverick-mcp/tree/main/docs

---

**Remember**: This software is for educational purposes only. Always consult qualified financial professionals before making investment decisions.
