# README search research

The README describes MaverickMCP as a stock market MCP server for local stock
analysis, portfolio tracking, and Python backtesting. The wording reflects the
implemented features and the search evidence below. Search results do not
establish product quality or financial accuracy.

## Research method

DataForSEO MCP queries ran on October 4, 2026, using Google, United States
location code 2840, and English. Keyword requests used the
[Keyword Overview API](https://docs.dataforseo.com/v3/dataforseo_labs/google/keyword_overview/live/).
Search result requests used the
[Google Organic Live API](https://docs.dataforseo.com/v3/serp/google/organic/live/advanced/)
with desktop results and depth 10.

| Search term | Approximate monthly searches | Keyword data updated |
| --- | --- | --- |
| yahoo finance mcp | 170 | September 13, 2026 |
| finance mcp server | 40 | September 16, 2026 |
| maverick mcp | 30 | September 14, 2026 |
| stock market mcp server | 10 | September 15, 2026 |
| python backtesting | 320 | September 13, 2026 |

The metrics are estimates, not traffic forecasts. Several narrower terms had
no returned data, which does not establish zero demand. Paid-search competition
was not used as a measure of organic search difficulty.

Keyword request IDs were `10041133-1402-0607-0000-c13616562cc6` and
`10041135-1402-0607-0000-06346959d319`. Search result request IDs were
`10041135-1402-0139-0000-89365c4586c8` for "stock market mcp server" and
`10041136-1402-0139-0000-2d8580056706` for "maverick mcp".

## Decisions

Use "stock market MCP server" in the title and explain Model Context Protocol
in the opening paragraphs. Describe the Yahoo Finance connection through
`yfinance` without suggesting an official Yahoo affiliation. Mention Python
backtesting in the overview and its own tool section because it is an optional
part of the product.

The branded search also returned unrelated projects, so the title includes the
stock-analysis purpose. Google AI summaries cited this repository in both
sampled searches, but the organic results did not place its repository URL in
the returned lists. Neither observation establishes a stable ranking.

Keep installation instructions, client setup links, and feature descriptions
near the start. Use plain headings and descriptive links. Remove unsupported
claims about real-time data, setup effort, and professional quality. Explain
the difference between current source and the older published release.

The content follows [Google's SEO starter guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide),
which recommends useful current content, natural wording, and descriptive
links. No ranking or traffic improvement is claimed. Repository metadata can
use the same product description, but no third-party directory submission or
new website is part of this change.
