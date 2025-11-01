# PORTFOLIO PERSONALIZATION - EXECUTION PLAN

## 1. Big Picture / Goal

**Objective:** Transform MaverickMCP's portfolio analysis tools from stateless, repetitive-input operations into an intelligent, personalized AI financial assistant through persistent portfolio storage and context-aware tool integration.

**Architectural Goal:** Implement a two-phase system that (1) adds persistent portfolio storage with cost basis tracking using established DDD patterns, and (2) intelligently enhances existing tools to auto-detect user holdings and provide personalized analysis without breaking the stateless MCP tool contract.

**Success Criteria (Mandatory):**
- **Phase 1 Complete:** 4 new MCP tools (`add_portfolio_position`, `get_my_portfolio`, `remove_portfolio_position`, `clear_my_portfolio`) and 1 MCP resource (`portfolio://my-holdings`) fully functional
- **Database Integration:** SQLAlchemy models with proper cost basis averaging, Alembic migration creating tables without conflicts
- **Phase 2 Integration:** 3 existing tools enhanced (`risk_adjusted_analysis`, `portfolio_correlation_analysis`, `compare_tickers`) with automatic portfolio detection
- **AI Context Injection:** Portfolio resource provides live P&L, diversification metrics, and position details to AI agents automatically
- **Test Coverage:** 85%+ test coverage with unit, integration, and domain tests passing
- **Code Quality:** Zero linting errors (ruff), full type annotations (ty), all hooks passing
- **Documentation:** PORTFOLIO.md guide, updated tool docstrings, usage examples in Claude Desktop

**Financial Disclaimer:** All portfolio features include educational disclaimers. No investment recommendations. Local-first storage only. No tax advice provided.

## 2. To-Do List (High Level)

### Phase 1: Persistent Portfolio Storage Foundation (4-5 days)
- [ ] **Spike 1:** Research cost basis averaging algorithms and edge cases (FIFO, average cost)
- [ ] **Domain Entities:** Create `Portfolio` and `Position` domain entities with business logic
- [ ] **Database Models:** Implement `UserPortfolio` and `PortfolioPosition` SQLAlchemy models
- [ ] **Migration:** Create Alembic migration with proper indexes and constraints
- [ ] **MCP Tools:** Implement 4 portfolio management tools with validation
- [ ] **MCP Resource:** Implement `portfolio://my-holdings` with live P&L calculations
- [ ] **Unit Tests:** Comprehensive domain entity and cost basis tests
- [ ] **Integration Tests:** Database operation and transaction tests

### Phase 2: Intelligent Tool Integration (2-3 days)
- [ ] **Risk Analysis Enhancement:** Add position awareness to `risk_adjusted_analysis`
- [ ] **Correlation Enhancement:** Enable `portfolio_correlation_analysis` with no arguments
- [ ] **Comparison Enhancement:** Enable `compare_tickers` with optional portfolio auto-fill
- [ ] **Resource Enhancement:** Add live market data to portfolio resource
- [ ] **Integration Tests:** Cross-tool functionality validation
- [ ] **Documentation:** Update existing tool docstrings with new capabilities

### Phase 3: Polish & Documentation (1-2 days)
- [ ] **Manual Testing:** Claude Desktop end-to-end workflow validation
- [ ] **Error Handling:** Edge case coverage (partial sells, zero shares, invalid tickers)
- [ ] **Performance:** Query optimization, batch operations, caching strategy
- [ ] **Documentation:** Complete PORTFOLIO.md with examples and screenshots
- [ ] **Migration Testing:** Test upgrade/downgrade paths

## 3. Plan Details (Spikes & Features)

### Spike 1: Cost Basis Averaging Research

**Action:** Investigate cost basis calculation methods (FIFO, LIFO, average cost) and determine optimal approach for educational portfolio tracking.

**Steps:**
1. Research IRS cost basis methods and educational best practices
2. Analyze existing `PortfolioManager` tool (JSON-based, average cost) for patterns
3. Design algorithm for averaging purchases and handling partial sells
4. Create specification document for edge cases:
   - Multiple purchases at different prices
   - Partial position sales
   - Zero/negative share handling
   - Rounding and precision (financial data uses Numeric(12,4))
5. Benchmark performance for 100+ positions with 1000+ transactions

**Expected Outcome:** Clear specification for cost basis implementation using **average cost method** (simplest for educational use, matches existing PortfolioManager), with edge case handling documented.

**Decision Rationale:** Average cost is simpler than FIFO/LIFO, appropriate for educational context, and avoids tax accounting complexity.

---

### Feature A: Domain Entities (DDD Pattern)

**Goal:** Create pure business logic entities following MaverickMCP's established DDD patterns (similar to backtesting domain entities).

**Files to Create:**
- `maverick_mcp/domain/portfolio.py` - Core domain entities
- `maverick_mcp/domain/position.py` - Position value objects

**Domain Entity Design:**

```python
# maverick_mcp/domain/portfolio.py
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

@dataclass
class Position:
    """Value object representing a single portfolio position."""
    ticker: str
    shares: Decimal  # Use Decimal for precision
    average_cost_basis: Decimal
    total_cost: Decimal
    purchase_date: datetime  # Earliest purchase
    notes: Optional[str] = None

    def add_shares(self, shares: Decimal, price: Decimal, date: datetime) -> "Position":
        """Add shares with automatic cost basis averaging."""
        new_total_shares = self.shares + shares
        new_total_cost = self.total_cost + (shares * price)
        new_avg_cost = new_total_cost / new_total_shares

        return Position(
            ticker=self.ticker,
            shares=new_total_shares,
            average_cost_basis=new_avg_cost,
            total_cost=new_total_cost,
            purchase_date=min(self.purchase_date, date),
            notes=self.notes
        )

    def remove_shares(self, shares: Decimal) -> Optional["Position"]:
        """Remove shares, return None if position fully closed."""
        if shares >= self.shares:
            return None  # Full position close

        new_shares = self.shares - shares
        new_total_cost = new_shares * self.average_cost_basis

        return Position(
            ticker=self.ticker,
            shares=new_shares,
            average_cost_basis=self.average_cost_basis,
            total_cost=new_total_cost,
            purchase_date=self.purchase_date,
            notes=self.notes
        )

    def calculate_current_value(self, current_price: Decimal) -> dict:
        """Calculate live P&L metrics."""
        current_value = self.shares * current_price
        unrealized_pnl = current_value - self.total_cost
        pnl_percentage = (unrealized_pnl / self.total_cost * 100) if self.total_cost else Decimal(0)

        return {
            "current_value": current_value,
            "unrealized_pnl": unrealized_pnl,
            "pnl_percentage": pnl_percentage
        }

@dataclass
class Portfolio:
    """Aggregate root for user portfolio."""
    portfolio_id: str  # UUID
    user_id: str  # "default" for single-user
    name: str
    positions: List[Position]
    created_at: datetime
    updated_at: datetime

    def add_position(self, ticker: str, shares: Decimal, price: Decimal,
                    date: datetime, notes: Optional[str] = None) -> None:
        """Add or update position with automatic averaging."""
        # Find existing position
        for i, pos in enumerate(self.positions):
            if pos.ticker == ticker:
                self.positions[i] = pos.add_shares(shares, price, date)
                self.updated_at = datetime.now(UTC)
                return

        # Create new position
        new_position = Position(
            ticker=ticker,
            shares=shares,
            average_cost_basis=price,
            total_cost=shares * price,
            purchase_date=date,
            notes=notes
        )
        self.positions.append(new_position)
        self.updated_at = datetime.now(UTC)

    def remove_position(self, ticker: str, shares: Optional[Decimal] = None) -> bool:
        """Remove position or partial shares."""
        for i, pos in enumerate(self.positions):
            if pos.ticker == ticker:
                if shares is None or shares >= pos.shares:
                    # Full position removal
                    self.positions.pop(i)
                else:
                    # Partial removal
                    updated_pos = pos.remove_shares(shares)
                    if updated_pos:
                        self.positions[i] = updated_pos
                    else:
                        self.positions.pop(i)

                self.updated_at = datetime.now(UTC)
                return True
        return False

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position by ticker."""
        return next((pos for pos in self.positions if pos.ticker == ticker), None)

    def get_total_invested(self) -> Decimal:
        """Calculate total capital invested."""
        return sum(pos.total_cost for pos in self.positions)

    def calculate_portfolio_metrics(self, current_prices: dict[str, Decimal]) -> dict:
        """Calculate comprehensive portfolio metrics."""
        total_value = Decimal(0)
        total_cost = Decimal(0)
        position_details = []

        for pos in self.positions:
            current_price = current_prices.get(pos.ticker, pos.average_cost_basis)
            metrics = pos.calculate_current_value(current_price)

            total_value += metrics["current_value"]
            total_cost += pos.total_cost

            position_details.append({
                "ticker": pos.ticker,
                "shares": float(pos.shares),
                "cost_basis": float(pos.average_cost_basis),
                "current_price": float(current_price),
                **{k: float(v) for k, v in metrics.items()}
            })

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else Decimal(0)

        return {
            "total_value": float(total_value),
            "total_invested": float(total_cost),
            "total_pnl": float(total_pnl),
            "total_pnl_percentage": float(total_pnl_pct),
            "position_count": len(self.positions),
            "positions": position_details
        }
```

**Testing Strategy:**
- Unit tests for cost basis averaging edge cases
- Property-based tests for arithmetic precision
- Edge case tests: zero shares, negative P&L, division by zero

---

### Feature B: Database Models (SQLAlchemy ORM)

**Goal:** Create persistent storage models following established patterns in `maverick_mcp/data/models.py`.

**Files to Modify:**
- `maverick_mcp/data/models.py` - Add new models (lines ~1700+)

**Model Design:**

```python
# Add to maverick_mcp/data/models.py

class UserPortfolio(TimestampMixin, Base):
    """
    User portfolio for tracking investment holdings.

    Follows personal-use design: single user_id="default"
    """
    __tablename__ = "mcp_portfolios"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(String(50), nullable=False, default="default", index=True)
    name = Column(String(200), nullable=False, default="My Portfolio")

    # Relationships
    positions = relationship(
        "PortfolioPosition",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin"  # Efficient loading
    )

    # Indexes for queries
    __table_args__ = (
        Index("idx_portfolio_user", "user_id"),
        UniqueConstraint("user_id", "name", name="uq_user_portfolio_name"),
    )

    def __repr__(self):
        return f"<UserPortfolio(id={self.id}, name='{self.name}', positions={len(self.positions)})>"


class PortfolioPosition(TimestampMixin, Base):
    """
    Individual position within a portfolio with cost basis tracking.
    """
    __tablename__ = "mcp_portfolio_positions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(Uuid, ForeignKey("mcp_portfolios.id", ondelete="CASCADE"), nullable=False)

    # Position details
    ticker = Column(String(20), nullable=False, index=True)
    shares = Column(Numeric(20, 8), nullable=False)  # High precision for fractional shares
    average_cost_basis = Column(Numeric(12, 4), nullable=False)  # Financial precision
    total_cost = Column(Numeric(20, 4), nullable=False)  # Total capital invested
    purchase_date = Column(DateTime(timezone=True), nullable=False)  # Earliest purchase
    notes = Column(Text, nullable=True)  # Optional user notes

    # Relationships
    portfolio = relationship("UserPortfolio", back_populates="positions")

    # Indexes for efficient queries
    __table_args__ = (
        Index("idx_position_portfolio", "portfolio_id"),
        Index("idx_position_ticker", "ticker"),
        Index("idx_position_portfolio_ticker", "portfolio_id", "ticker"),
        UniqueConstraint("portfolio_id", "ticker", name="uq_portfolio_position_ticker"),
    )

    def __repr__(self):
        return f"<PortfolioPosition(ticker='{self.ticker}', shares={self.shares}, cost_basis={self.average_cost_basis})>"
```

**Key Design Decisions:**
1. **Table Names:** `mcp_portfolios` and `mcp_portfolio_positions` (consistent with `mcp_*` pattern)
2. **user_id:** Default "default" for single-user personal use
3. **Numeric Precision:** Matches existing financial data patterns (12,4 for prices, 20,8 for shares)
4. **Cascade Delete:** Portfolio deletion removes all positions automatically
5. **Unique Constraint:** One position per ticker per portfolio
6. **Indexes:** Optimized for common queries (user lookup, ticker filtering)

---

### Feature C: Alembic Migration

**Goal:** Create database migration following established patterns without conflicts.

**File to Create:**
- `alembic/versions/014_add_portfolio_models.py`

**Migration Pattern:**

```python
"""Add portfolio and position models

Revision ID: 014_add_portfolio_models
Revises: 013_add_backtest_persistence_models
Create Date: 2025-11-01 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '014_add_portfolio_models'
down_revision = '013_add_backtest_persistence_models'
branch_labels = None
depends_on = None


def upgrade():
    """Create portfolio management tables."""

    # Create portfolios table
    op.create_table(
        'mcp_portfolios',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(50), nullable=False, server_default='default'),
        sa.Column('name', sa.String(200), nullable=False, server_default='My Portfolio'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create indexes on portfolios
    op.create_index('idx_portfolio_user', 'mcp_portfolios', ['user_id'])
    op.create_unique_constraint('uq_user_portfolio_name', 'mcp_portfolios', ['user_id', 'name'])

    # Create positions table
    op.create_table(
        'mcp_portfolio_positions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('portfolio_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('ticker', sa.String(20), nullable=False),
        sa.Column('shares', sa.Numeric(20, 8), nullable=False),
        sa.Column('average_cost_basis', sa.Numeric(12, 4), nullable=False),
        sa.Column('total_cost', sa.Numeric(20, 4), nullable=False),
        sa.Column('purchase_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['portfolio_id'], ['mcp_portfolios.id'], ondelete='CASCADE'),
    )

    # Create indexes on positions
    op.create_index('idx_position_portfolio', 'mcp_portfolio_positions', ['portfolio_id'])
    op.create_index('idx_position_ticker', 'mcp_portfolio_positions', ['ticker'])
    op.create_index('idx_position_portfolio_ticker', 'mcp_portfolio_positions', ['portfolio_id', 'ticker'])
    op.create_unique_constraint('uq_portfolio_position_ticker', 'mcp_portfolio_positions', ['portfolio_id', 'ticker'])


def downgrade():
    """Drop portfolio management tables."""
    op.drop_table('mcp_portfolio_positions')
    op.drop_table('mcp_portfolios')
```

**Testing:**
- Test upgrade: `alembic upgrade head`
- Test downgrade: `alembic downgrade -1`
- Verify indexes created: SQL query inspection
- Test with SQLite and PostgreSQL

---

### Feature D: MCP Tools Implementation

**Goal:** Implement 4 portfolio management tools following tool_registry.py pattern.

**Files to Create:**
- `maverick_mcp/api/routers/portfolio_management.py` - New tool implementations
- `maverick_mcp/api/services/portfolio_persistence_service.py` - Service layer
- `maverick_mcp/validation/portfolio_management.py` - Pydantic validation

**Service Layer Pattern:**

```python
# maverick_mcp/api/services/portfolio_persistence_service.py

class PortfolioPersistenceService(BaseService):
    """Service for portfolio CRUD operations."""

    async def get_or_create_default_portfolio(self) -> UserPortfolio:
        """Get the default portfolio, create if doesn't exist."""
        pass

    async def add_position(self, ticker: str, shares: Decimal,
                          price: Decimal, date: datetime,
                          notes: Optional[str]) -> PortfolioPosition:
        """Add or update position with cost averaging."""
        pass

    async def get_portfolio_with_live_data(self) -> dict:
        """Fetch portfolio with current market prices."""
        pass

    async def remove_position(self, ticker: str,
                            shares: Optional[Decimal]) -> bool:
        """Remove position or partial shares."""
        pass

    async def clear_portfolio(self) -> bool:
        """Delete all positions."""
        pass
```

**Tool Registration:**

```python
# Add to maverick_mcp/api/routers/tool_registry.py

def register_portfolio_management_tools(mcp: FastMCP) -> None:
    """Register portfolio management tools."""
    from maverick_mcp.api.routers.portfolio_management import (
        add_portfolio_position,
        get_my_portfolio,
        remove_portfolio_position,
        clear_my_portfolio
    )

    mcp.tool(name="portfolio_add_position")(add_portfolio_position)
    mcp.tool(name="portfolio_get_my_portfolio")(get_my_portfolio)
    mcp.tool(name="portfolio_remove_position")(remove_portfolio_position)
    mcp.tool(name="portfolio_clear")(clear_my_portfolio)
```

---

### Feature E: MCP Resource Implementation

**Goal:** Create `portfolio://my-holdings` resource for automatic AI context injection.

**File to Modify:**
- `maverick_mcp/api/server.py` - Add resource alongside existing health:// and dashboard:// resources

**Resource Implementation:**

```python
# Add to maverick_mcp/api/server.py (around line 823, near other resources)

@mcp.resource("portfolio://my-holdings")
def portfolio_holdings_resource() -> dict[str, Any]:
    """
    Portfolio holdings resource for AI context injection.

    Provides comprehensive portfolio context to AI agents including:
    - Current positions with live P&L
    - Portfolio metrics and diversification
    - Sector exposure analysis
    - Top/bottom performers

    This resource is automatically available to AI agents during conversations,
    enabling personalized analysis without requiring manual ticker input.
    """
    # Implementation using service layer with async handling
    pass
```

---

### Feature F: Phase 2 Tool Enhancements

**Goal:** Enhance existing tools to auto-detect portfolio holdings.

**Files to Modify:**
1. `maverick_mcp/api/routers/portfolio.py` - Enhance 3 existing tools
2. `maverick_mcp/validation/portfolio.py` - Update validation to allow optional parameters

**Enhancement Pattern:**
- Add optional parameters (tickers can be None)
- Check portfolio for holdings if no tickers provided
- Add position awareness to analysis results
- Maintain backward compatibility

---

## 4. Progress (Living Document Section)

| Date | Time | Item Completed / Status Update | Resulting Changes (LOC/Files) |
|:-----|:-----|:------------------------------|:------------------------------|
| 2025-11-01 | Start | Plan approved and documented | PORTFOLIO_PERSONALIZATION_PLAN.md created |
| TBD | TBD | Implementation begins | - |

_(This section will be updated during implementation)_

---

## 5. Surprises and Discoveries

_(Technical issues discovered during implementation will be documented here)_

**Anticipated Challenges:**
1. **MCP Resource Async Context:** Resources are sync functions but need async database calls - solved with event loop management (see existing health_resource pattern)
2. **Cost Basis Precision:** Financial calculations require Decimal precision, not floats - use Numeric(12,4) for prices, Numeric(20,8) for shares
3. **Portfolio Resource Performance:** Live price fetching could be slow - implement caching strategy, consider async batching
4. **Single User Assumption:** No user authentication means all operations use user_id="default" - acceptable for personal use

---

## 6. Decision Log

| Date | Decision | Rationale |
|:-----|:---------|:----------|
| 2025-11-01 | **Cost Basis Method: Average Cost** | Simplest for educational use, matches existing PortfolioManager, avoids tax accounting complexity |
| 2025-11-01 | **Table Names: mcp_portfolios, mcp_portfolio_positions** | Consistent with existing mcp_* naming convention for MCP-specific tables |
| 2025-11-01 | **User ID: "default" for all users** | Single-user personal-use design, consistent with auth disabled architecture |
| 2025-11-01 | **Numeric Precision: Numeric(12,4) for prices, Numeric(20,8) for shares** | Matches existing financial data patterns, supports fractional shares |
| 2025-11-01 | **Optional tickers parameter for Phase 2** | Enables "just works" UX while maintaining backward compatibility |
| 2025-11-01 | **MCP Resource for AI context** | Most elegant solution for automatic context injection without breaking tool contracts |
| 2025-11-01 | **Domain-Driven Design pattern** | Follows established MaverickMCP architecture, clean separation of concerns |

---

## 7. Implementation Phases

### Phase 1: Foundation (4-5 days)
**Files Created:** 8 new files
**Files Modified:** 3 existing files
**Estimated LOC:** ~2,500 lines
**Tests:** ~1,200 lines

### Phase 2: Integration (2-3 days)
**Files Modified:** 4 existing files
**Estimated LOC:** ~800 lines additional
**Tests:** ~600 lines additional

### Phase 3: Polish (1-2 days)
**Documentation:** PORTFOLIO.md (~300 lines)
**Performance:** Query optimization
**Testing:** Manual Claude Desktop validation

**Total Effort:** 7-10 days
**Total New Code:** ~3,500 lines (including tests)
**Total Tests:** ~1,800 lines

---

## 8. Risk Assessment

**Low Risk:**
- ✅ Follows established patterns
- ✅ No breaking changes to existing tools
- ✅ Optional Phase 2 enhancements
- ✅ Well-scoped feature

**Medium Risk:**
- ⚠️ MCP resource performance with live prices
- ⚠️ Migration compatibility (SQLite vs PostgreSQL)
- ⚠️ Edge cases in cost basis averaging

**Mitigation Strategies:**
1. **Performance:** Implement caching, batch price fetches, add timeout protection
2. **Migration:** Test with both SQLite and PostgreSQL, provide rollback path
3. **Edge Cases:** Comprehensive unit tests, property-based testing for arithmetic

---

## 9. Testing Strategy

**Unit Tests (~60% of test code):**
- Domain entity logic (Position, Portfolio)
- Cost basis averaging edge cases
- Numeric precision validation
- Business logic validation

**Integration Tests (~30% of test code):**
- Database CRUD operations
- Migration upgrade/downgrade
- Service layer with real database
- Cross-tool functionality

**Manual Tests (~10% of effort):**
- Claude Desktop end-to-end workflows
- Natural language interactions
- MCP resource visibility
- Tool integration scenarios

**Test Coverage Target:** 85%+

---

## 10. Success Metrics

**Functional Success:**
- [ ] All 4 new tools work in Claude Desktop
- [ ] Portfolio resource visible to AI agents
- [ ] Cost basis averaging accurate to 4 decimal places
- [ ] Migration works on SQLite and PostgreSQL
- [ ] 3 enhanced tools auto-detect portfolio

**Quality Success:**
- [ ] 85%+ test coverage
- [ ] Zero linting errors (ruff)
- [ ] Full type annotations (ty check passes)
- [ ] All pre-commit hooks pass

**UX Success:**
- [ ] "Analyze my portfolio" works without ticker input
- [ ] AI agents reference actual holdings in responses
- [ ] Natural language interactions feel seamless
- [ ] Error messages are clear and actionable

---

## 11. Related Documentation

- **Original Issue:** [#40 - Portfolio Personalization](https://github.com/wshobson/maverick-mcp/issues/40)
- **User Documentation:** `docs/PORTFOLIO.md` (to be created)
- **API Documentation:** Tool docstrings and MCP introspection
- **Testing Guide:** `tests/README.md` (to be updated)

---

This execution plan provides a comprehensive roadmap following the PLANS.md rubric structure. The implementation is well-scoped, follows established patterns, and delivers significant UX improvement while maintaining code quality and architectural integrity.
