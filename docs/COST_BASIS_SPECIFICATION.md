# Cost Basis Specification for Portfolio Management

## 1. Overview

This document specifies the cost basis tracking algorithm for MaverickMCP's portfolio management system. The system uses the **Average Cost Method** for educational simplicity.

## 2. Cost Basis Method: Average Cost

### Definition
The average cost method calculates the cost basis by taking the total cost of all shares purchased and dividing by the total number of shares owned.

### Formula
```
Average Cost Basis = Total Cost of All Shares / Total Number of Shares
```

### Why Average Cost?
1. **Simplicity**: Easiest to understand for educational purposes
2. **Consistency**: Matches existing `PortfolioManager` implementation
3. **No Tax Complexity**: Avoids FIFO/LIFO tax accounting rules
4. **Educational Focus**: Appropriate for learning, not tax optimization

## 3. Edge Cases and Handling

### 3.1 Multiple Purchases at Different Prices

**Scenario**: User buys same stock multiple times at different prices

**Example**:
```
Purchase 1: 10 shares @ $150.00 = $1,500.00
Purchase 2: 10 shares @ $170.00 = $1,700.00
Result: 20 shares @ $160.00 average cost = $3,200.00 total
```

**Algorithm**:
```python
new_total_shares = existing_shares + new_shares
new_total_cost = existing_total_cost + (new_shares * new_price)
new_average_cost = new_total_cost / new_total_shares
```

**Precision**: Use Decimal type throughout, round final result to 4 decimal places

### 3.2 Partial Position Sales

**Scenario**: User sells portion of position

**Example**:
```
Holding: 20 shares @ $160.00 average cost = $3,200.00 total
Sell: 10 shares
Result: 10 shares @ $160.00 average cost = $1,600.00 total
```

**Algorithm**:
```python
new_shares = existing_shares - sold_shares
new_total_cost = new_shares * average_cost_basis
# Average cost basis remains unchanged
```

**Important**: Average cost basis does NOT change on partial sales

### 3.3 Full Position Close

**Scenario**: User sells all shares

**Algorithm**:
```python
if sold_shares >= existing_shares:
    # Remove position entirely from portfolio
    position = None
```

**Database**: Delete PortfolioPosition row

### 3.4 Zero or Negative Shares

**Validation Rules**:
- Shares to add: Must be > 0
- Shares to remove: Must be > 0
- Result after removal: Must be >= 0

**Error Handling**:
```python
if new_shares <= 0:
    raise ValueError("Invalid share quantity")
```

### 3.5 Zero or Negative Prices

**Validation Rules**:
- Purchase price: Must be > 0
- Sell price: Optional (not used in cost basis calculation)

### 3.6 Fractional Shares

**Support**: YES - Use Numeric(20, 8) for up to 8 decimal places

**Example**:
```
Purchase: 10.5 shares @ $150.25 = $1,577.625
Valid and supported
```

### 3.7 Rounding and Precision

**Database Storage**:
- Shares: `Numeric(20, 8)` - 8 decimal places
- Prices: `Numeric(12, 4)` - 4 decimal places (cents precision)
- Total Cost: `Numeric(20, 4)` - 4 decimal places

**Calculation Precision**:
- Use Python `Decimal` type throughout calculations
- Only round when storing to database or displaying to user
- Never use float for financial calculations

**Rounding Rules**:
```python
from decimal import Decimal, ROUND_HALF_UP

# For display (2 decimal places)
display_value = value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

# For database storage (4 decimal places for prices)
db_price = price.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

# For database storage (8 decimal places for shares)
db_shares = shares.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
```

### 3.8 Division by Zero

**Scenario**: Calculating average when shares = 0

**Prevention**:
```python
if total_shares == 0:
    raise ValueError("Cannot calculate average cost with zero shares")
```

**Should never occur** due to validation preventing zero-share positions

## 4. P&L Calculation

### Unrealized P&L Formula
```
Current Value = Shares × Current Price
Unrealized P&L = Current Value - Total Cost
P&L Percentage = (Unrealized P&L / Total Cost) × 100
```

### Example
```
Position: 20 shares @ $160.00 cost basis = $3,200.00 total cost
Current Price: $175.50
Current Value: 20 × $175.50 = $3,510.00
Unrealized P&L: $3,510.00 - $3,200.00 = $310.00
P&L %: ($310.00 / $3,200.00) × 100 = 9.69%
```

### Edge Cases
- **Current price unavailable**: Use cost basis as fallback
- **Zero cost basis**: Return 0% (should never occur with validation)

## 5. Database Constraints

### Unique Constraint
```sql
UNIQUE (portfolio_id, ticker)
```
**Rationale**: One position per ticker per portfolio

### Check Constraints (Optional - Enforce in Application Layer)
```python
# Application-level validation (preferred)
assert shares > 0, "Shares must be positive"
assert average_cost_basis > 0, "Cost basis must be positive"
assert total_cost > 0, "Total cost must be positive"
```

## 6. Concurrency Considerations

### Single-User System
- No concurrent writes expected (personal use)
- Database-level unique constraints prevent duplicates
- SQLAlchemy sessions with auto-rollback handle errors

### Future Multi-User Support
- Would require row-level locking: `SELECT FOR UPDATE`
- Optimistic concurrency with version column
- Currently not needed for personal use

## 7. Performance Benchmarks

### Expected Performance (100 Positions, 1000 Transactions)
- Add position: < 10ms (with database write)
- Calculate portfolio value: < 50ms (without live prices)
- Calculate portfolio value with live prices: < 2s (network bound)

### Optimization Strategies
- Batch price fetches for portfolio valuation
- Cache live prices (5-minute expiry)
- Use database indexes for ticker lookups
- Lazy-load positions only when needed

## 8. Migration Strategy

### Initial Migration (014_add_portfolio_models)
```sql
CREATE TABLE mcp_portfolios (
    id UUID PRIMARY KEY,
    user_id VARCHAR(50) DEFAULT 'default',
    name VARCHAR(200) DEFAULT 'My Portfolio',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE mcp_portfolio_positions (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES mcp_portfolios(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    shares NUMERIC(20, 8) NOT NULL,
    average_cost_basis NUMERIC(12, 4) NOT NULL,
    total_cost NUMERIC(20, 4) NOT NULL,
    purchase_date TIMESTAMP WITH TIME ZONE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(portfolio_id, ticker)
);
```

### Data Migration from PortfolioManager (Future)
If users have existing portfolio JSON files:
```python
def migrate_from_json(json_file: str) -> None:
    """Migrate existing portfolio from JSON to database."""
    # Load JSON portfolio
    # Create UserPortfolio
    # Create PortfolioPositions for each holding
    # Verify cost basis calculations match
```

## 9. Testing Requirements

### Unit Tests (Domain Layer)
- ✅ Add shares: Multiple purchases, average calculation
- ✅ Remove shares: Partial removal, full removal
- ✅ P&L calculation: Various price scenarios
- ✅ Edge cases: Zero shares, negative values, division by zero
- ✅ Precision: Decimal arithmetic accuracy

### Integration Tests (Database Layer)
- ✅ CRUD operations: Create, read, update, delete positions
- ✅ Unique constraint: Prevent duplicate tickers
- ✅ Cascade delete: Portfolio deletion removes positions
- ✅ Transaction rollback: Error handling

### Property-Based Tests
- ✅ Adding and removing shares always maintains valid state
- ✅ Average cost formula always correct
- ✅ P&L calculations always sum correctly

## 10. Example Scenarios

### Scenario 1: Build Position Over Time
```
Day 1: Buy 10 AAPL @ $150.00
  - Shares: 10, Avg Cost: $150.00, Total: $1,500.00

Day 30: Buy 5 AAPL @ $160.00
  - Shares: 15, Avg Cost: $153.33, Total: $2,300.00

Day 60: Buy 10 AAPL @ $145.00
  - Shares: 25, Avg Cost: $150.80, Total: $3,770.00
```

### Scenario 2: Take Profits
```
Start: 25 AAPL @ $150.80 = $3,770.00
Current Price: $175.50
Unrealized P&L: +$617.50 (+16.38%)

Sell 10 shares @ $175.50 (realized gain: $247.00)
Remaining: 15 AAPL @ $150.80 = $2,262.00
Current Value @ $175.50: $2,632.50
Unrealized P&L: +$370.50 (+16.38% - same percentage)
```

### Scenario 3: Dollar-Cost Averaging
```
Monthly purchases of $1,000:
Month 1: 6.67 shares @ $150.00 = $1,000.00
Month 2: 6.25 shares @ $160.00 = $1,000.00
Month 3: 6.90 shares @ $145.00 = $1,000.00
Total: 19.82 shares @ $151.26 avg = $3,000.00
```

## 11. Compliance and Disclaimers

### Educational Purpose
This cost basis tracking is for **educational purposes only** and should not be used for tax reporting.

### Tax Reporting
Users should consult tax professionals and use official brokerage cost basis reporting for tax purposes.

### Disclaimers in Tools
All portfolio tools include:
```
DISCLAIMER: This portfolio tracking is for educational purposes only and does not
constitute investment advice. All investments carry risk of loss. Consult qualified
financial and tax professionals for investment and tax advice.
```

## 12. References

- **IRS Publication 550**: Investment Income and Expenses
- **Existing Code**: `maverick_mcp/tools/portfolio_manager.py` (average cost implementation)
- **Financial Precision**: IEEE 754 vs Decimal arithmetic
- **SQLAlchemy Numeric**: Column type documentation

---

**Document Version**: 1.0
**Last Updated**: 2025-11-01
**Author**: Portfolio Personalization Feature Team
