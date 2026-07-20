"""Builds the FastMCP server: wires platform infrastructure and every domain's service and tools together. Second layer: imports every maverick.<domain> package, maverick.platform, and this package's prompts module."""

from fastmcp import FastMCP

from maverick.backtesting import backtesting_extra_available
from maverick.backtesting import tools as backtesting_tools
from maverick.market_data import tools as market_data_tools
from maverick.market_data.config import get_market_data_settings
from maverick.market_data.fetchers import YFinanceFetcher, build_mover_fetcher
from maverick.market_data.service import MarketDataService
from maverick.platform.cache import Cache
from maverick.platform.config import get_platform_settings
from maverick.platform.db import create_engine_from_settings
from maverick.portfolio import tools as portfolio_tools
from maverick.portfolio.service import PortfolioService
from maverick.portfolio.service_journal import JournalService
from maverick.research import research_extra_available
from maverick.research import tools as research_tools
from maverick.screening import tools as screening_tools
from maverick.screening.service import ScreeningService
from maverick.technical import tools as technical_tools
from maverick.technical.service import TechnicalService

_SERVER_NAME = "maverick-mcp"


def build_server() -> FastMCP:
    """Assemble every domain's service and tools onto one `FastMCP` instance.

    Assembly order is binding (see the phase 8 decision log and the server
    recon it restates):

    1. `get_platform_settings()` resolves one shared `Engine` (via
       `platform.db.create_engine_from_settings`) and one shared `Cache`.
    2. `MarketDataService` is constructed FIRST: its `__init__` eagerly
       creates its schema (every other domain's is lazy), and it needs a
       `YFinanceFetcher` plus a `MoverFetcher` built via
       `market_data.fetchers.build_mover_fetcher` -- a Phase 2 review catch
       that is easy to forget at assembly time.
    3. `ScreeningService`/`PortfolioService`/`TechnicalService` each inject
       that one `MarketDataService` instance.
    4. `JournalService` is portfolio's standalone sibling service (own
       engine, own schema); it is wired into `portfolio.tools.configure`'s
       optional `journal_service` parameter rather than composed inside
       `PortfolioService`.
    5. `BacktestingService`/`ResearchService` are constructed -- and their
       heavy-dependency service modules imported -- only when their extras
       are installed (`backtesting_extra_available()`/
       `research_extra_available()`). Each domain's own `tools.register()`
       already degrades to zero tools with one warning log when its extra
       is absent, so this function never raises for a base install.
    6. Per domain: `configure(...)` then `register(mcp)`.
    """
    settings = get_platform_settings()
    engine = create_engine_from_settings(settings.database)
    cache = Cache(settings.cache, redis_settings=settings.redis)

    yf = YFinanceFetcher(http_settings=settings.http)
    movers = build_mover_fetcher(get_market_data_settings(), yf)
    market_data = MarketDataService(engine, cache, yf, movers)

    screening = ScreeningService(engine, market_data)
    portfolio = PortfolioService(engine, market_data)
    technical = TechnicalService(market_data)
    journal = JournalService(engine)

    mcp = FastMCP(name=_SERVER_NAME)

    market_data_tools.configure(market_data)
    market_data_tools.register(mcp)

    screening_tools.configure(screening)
    screening_tools.register(mcp)

    portfolio_tools.configure(portfolio, journal_service=journal)
    portfolio_tools.register(mcp)

    technical_tools.configure(technical)
    technical_tools.register(mcp)

    if backtesting_extra_available():
        from maverick.backtesting.service import BacktestingService

        backtesting_tools.configure(BacktestingService(market_data))
    backtesting_tools.register(mcp)

    if research_extra_available():
        from maverick.research.service import ResearchService

        research_tools.configure(ResearchService())
    research_tools.register(mcp)

    return mcp
