# Deep Research Feature

The deep research feature adds web-search-backed financial research to
MaverickMCP. It is designed for local educational analysis and should always
communicate uncertainty, cite sources in responses where applicable, and avoid
presenting generated conclusions as financial advice.

## Capabilities

- Comprehensive research over companies, sectors, market topics, and news.
- Persona-aware framing for conservative, moderate, aggressive, and day-trader
  analysis modes.
- Search-provider integration through Exa and optional Tavily support.
- Timeout-aware execution paths for quick, standard, and comprehensive research.
- Circuit-breaker and health-check behavior for external provider failures.
- MCP tools for comprehensive research, company research, market sentiment, and
  financial-news search.

## Main Components

- `maverick_mcp.agents.deep_research`: legacy-compatible research agent module.
- `maverick_mcp.agents.research.providers.exa`: Exa provider adapter.
- `maverick_mcp.agents.research.providers.tavily`: optional Tavily adapter.
- `maverick_mcp.api.routers.research`: research tool implementations.
- `maverick_mcp.api.routers.tool_registry`: prefixed research tool
  registration.

## Configuration

```bash
EXA_API_KEY=your_exa_key
TAVILY_API_KEY=your_tavily_key       # optional, when Tavily dependency is available
OPENROUTER_API_KEY=your_openrouter_key
```

Research should degrade gracefully when optional providers are missing. Unit
tests should mock external providers; real provider tests must be explicitly
marked and gated on API keys.

## MCP Tools

Registered research tools include:

- `research_comprehensive_research`
- `research_company_comprehensive`
- `research_analyze_market_sentiment`
- `research_search_financial_news`

Agent orchestration tools also expose research-oriented flows under the
`agents_*` tool namespace.

## Workflow

1. Classify the user query and desired research scope.
2. Allocate timeout and model budget.
3. Search with available providers.
4. Filter and score sources.
5. Synthesize findings for the requested persona.
6. Return structured output with caveats and provider diagnostics when useful.

## Testing

See `../testing/exa-research.md` and `../testing/speed.md` for provider and
timeout validation.
