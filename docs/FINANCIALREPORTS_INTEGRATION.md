# Data Source Suggestion: FinancialReports.eu MCP Integration

## Overview

[FinancialReports.eu](https://financialreports.eu) provides API access to **14M+ regulatory filings** from 35 official sources across 30+ countries. Since both Maverick and FinancialReports.eu use MCP, integration is a natural fit — adding regulatory filings as a new tool alongside Maverick's existing stock analysis capabilities.

## Why This Fits Maverick MCP

Maverick provides stock analysis via MCP. FinancialReports.eu adds:

- **Regulatory filings** — annual reports, interim results, ESG disclosures, M&A announcements
- **MCP-native** — FinancialReports.eu already has its own [MCP server](https://financialreports.eu), making it a natural companion to Maverick
- **Global coverage** — 35 regulators (SEC, FCA, Euronext, EDINET, etc.) across 30+ countries
- **Markdown endpoint** — `GET /filings/{id}/markdown/` returns LLM-ready text, perfect for Claude analysis
- **33,000+ companies** with ISIN identifiers

## Integration Approaches

### 1. MCP Tool Addition

Add filing-related tools to Maverick's MCP server:

```python
import requests

headers = {"X-API-Key": "your-api-key"}

# Fetch filings for a company
resp = requests.get("https://api.financialreports.eu/filings/",
    headers=headers,
    params={
        "company_isin": "US0378331005",  # Apple
        "categories": "2",               # Financial Reporting
        "page_size": 5
    }
)

# Get filing content as Markdown for Claude analysis
filing_id = resp.json()["results"][0]["id"]
content = requests.get(
    f"https://api.financialreports.eu/filings/{filing_id}/markdown/",
    headers=headers
).text
```

### 2. Companion MCP Server

Users can run FinancialReports.eu's MCP server alongside Maverick, giving Claude access to both stock analysis and regulatory filings in the same session.

### 3. Python SDK

```bash
pip install financial-reports-generated-client
```

```python
from financial_reports_client import Client
from financial_reports_client.api.filings import filings_list, filings_markdown_retrieve

client = Client(base_url="https://api.financialreports.eu")
client = client.with_headers({"X-API-Key": "your-api-key"})

filings = filings_list.sync(client=client, company_isin="US0378331005", categories="2")
content = filings_markdown_retrieve.sync(client=client, id=filings.results[0].id)
```

## API Details

| Property | Value |
|---|---|
| **Base URL** | `https://api.financialreports.eu` |
| **API Docs** | [docs.financialreports.eu](https://docs.financialreports.eu/) |
| **Authentication** | API key via `X-API-Key` header |
| **Python SDK** | `pip install financial-reports-generated-client` |
| **Rate Limiting** | Burst limit + monthly quota |
| **Companies** | 33,230+ |
| **Total Filings** | 14,135,359+ |
| **Sources** | 35 official regulators |

## Complementary Value

| Maverick MCP (current) | + FinancialReports.eu |
|---|---|
| Stock price analysis | Regulatory filing documents |
| Technical indicators | Annual report text (Markdown) |
| Portfolio tracking | Filing event timeline |
| US-focused data | 35 regulators, 30+ countries |
| — | ESG disclosures, M&A announcements |
