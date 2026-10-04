---
name: maverick-eval-client
description: Eval-only client for Maverick MCP error analysis. Answers one user request using only the Maverick tools against a seeded scratch database. The controller spawns it once per eval case from evals/tool_surface; never use it for any other task.
model: claude-opus-5-5
maxTurns: 8
omitClaudeMd: true
permissionMode: bypassPermissions
tools:
  - mcp__maverick__backtesting_analyze_market_regimes
  - mcp__maverick__backtesting_backtest_portfolio
  - mcp__maverick__backtesting_compare_strategies
  - mcp__maverick__backtesting_create_strategy_ensemble
  - mcp__maverick__backtesting_list_strategies
  - mcp__maverick__backtesting_monte_carlo_simulation
  - mcp__maverick__backtesting_optimize_strategy
  - mcp__maverick__backtesting_parse_strategy
  - mcp__maverick__backtesting_run_backtest
  - mcp__maverick__backtesting_run_ml_strategy_backtest
  - mcp__maverick__backtesting_train_ml_predictor
  - mcp__maverick__backtesting_walk_forward_analysis
  - mcp__maverick__market_data_clear_market_cache
  - mcp__maverick__market_data_get_chart_links
  - mcp__maverick__market_data_get_market_overview
  - mcp__maverick__market_data_get_price_history
  - mcp__maverick__market_data_get_price_history_batch
  - mcp__maverick__market_data_get_quote
  - mcp__maverick__market_data_get_stock_fundamentals
  - mcp__maverick__portfolio_add_position
  - mcp__maverick__portfolio_check_position_risk
  - mcp__maverick__portfolio_clear_portfolio
  - mcp__maverick__portfolio_compare_tickers
  - mcp__maverick__portfolio_correlation_analysis
  - mcp__maverick__portfolio_get_my_portfolio
  - mcp__maverick__portfolio_get_regime_adjusted_sizing
  - mcp__maverick__portfolio_get_risk_alerts
  - mcp__maverick__portfolio_get_risk_dashboard
  - mcp__maverick__portfolio_get_strategy_performance
  - mcp__maverick__portfolio_journal_add_trade
  - mcp__maverick__portfolio_journal_close_trade
  - mcp__maverick__portfolio_journal_list_trades
  - mcp__maverick__portfolio_journal_review
  - mcp__maverick__portfolio_remove_position
  - mcp__maverick__portfolio_risk_adjusted_analysis
  - mcp__maverick__portfolio_watchlist_add
  - mcp__maverick__portfolio_watchlist_brief
  - mcp__maverick__portfolio_watchlist_create
  - mcp__maverick__portfolio_watchlist_list
  - mcp__maverick__portfolio_watchlist_remove
  - mcp__maverick__screening_get_all
  - mcp__maverick__screening_get_bearish
  - mcp__maverick__screening_get_bullish
  - mcp__maverick__screening_get_by_criteria
  - mcp__maverick__screening_get_supply_demand
  - mcp__maverick__screening_run_screens
  - mcp__maverick__technical_get_full_technical_analysis
  - mcp__maverick__technical_get_macd_analysis
  - mcp__maverick__technical_get_rsi_analysis
  - mcp__maverick__technical_get_support_resistance
disallowedTools:
  - mcp__maverick__research_analyze_company
  - mcp__maverick__research_analyze_sentiment
  - mcp__maverick__research_run_comprehensive
mcpServers:
  - maverick:
      type: stdio
      command: {REPO}/.venv/bin/python
      args: ["-m", "evals.tool_surface.agent_server"]
      env:
        PYTHONPATH: {REPO}
---
You are connected to the Maverick MCP server, which provides stock analysis tools.
