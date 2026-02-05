# Product Definition

## Project Name

**MaverickMCP**

## Description

Personal-use MCP server for Claude Desktop providing professional-grade stock analysis, technical indicators, stock screening, and portfolio tracking.

## Problem Statement

This project addresses common friction in day-to-day market analysis workflows:

1. **Too many tools, too little context** - Market data, indicators, screeners, and notes are often spread across multiple apps
2. **Slow iteration** - Ad-hoc analysis is repetitive and hard to automate
3. **Lack of repeatable outputs** - Analyses are rarely standardized and easy to compare over time

## Target Users

- A single primary user (the repo owner) using Claude Desktop for day-to-day investing research
- Advanced retail investors and technical-analysis oriented traders (secondary)

## Key Goals

1. **Fast, reliable analysis** inside Claude Desktop via stable MCP tools
2. **Accurate indicator and screening outputs** with deterministic logic
3. **Practical portfolio visibility** (positions, cost basis, P&L, correlations)
4. **Graceful degradation** when optional providers (Redis, Postgres, research APIs) are unavailable

## Success Metrics

- Tool reliability: MCP tools stay registered and respond consistently
- Latency: common queries return quickly (via caching where appropriate)
- Correctness: indicator calculations and screening rules match expectations
- Usability: outputs are structured, scannable, and easy to act on

