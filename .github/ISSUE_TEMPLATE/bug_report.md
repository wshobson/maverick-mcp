---
name: Bug Report
about: Create a report to help us improve MaverickMCP
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 💰 Financial Disclaimer Acknowledgment

- [ ] I understand this is educational software and not financial advice
- [ ] I am not expecting investment recommendations or guaranteed returns
- [ ] This bug report is about technical functionality, not financial performance

## 📋 Reproduction Steps

Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## 🎯 Expected Behavior

A clear and concise description of what you expected to happen.

## 📸 Screenshots

If applicable, add screenshots to help explain your problem.

## 💻 Environment Information

**Desktop/Server:**
 - OS: [e.g. macOS, Ubuntu, Windows]
 - Python Version: [e.g. 3.12.0]
 - MaverickMCP Version: [e.g. 1.1.0]
 - Installation Method: [e.g. uvx from release tag, Docker (GHCR), git clone]
 - Extras installed: [none, backtesting, research]

**MCP Client:**
 - Client and Version: [e.g. Claude Desktop, Claude Code, Cursor]
 - Transport: [STDIO, Streamable HTTP]
 - mcp-remote Version: [if bridging HTTP through mcp-remote]

**Dependencies:**
 - FastMCP Version: [e.g. 4.0.3]
 - Database: [SQLite, PostgreSQL]
 - Redis: [Yes/No, version if yes]

## 📋 Configuration

**Environment Variables (remove sensitive data):**
```
LLM_PROVIDER=***
DATABASE_URL=***
REDIS_HOST=***
# ... other relevant config
```

**Relevant .env settings:**
```
LOG_LEVEL=DEBUG
CACHE_ENABLED=true
# ... other settings
```

## 📊 Error Messages/Logs

**Error message:**
```
Paste the full error message here
```

**Server logs (if available):**
```
Paste relevant server logs here (remove API keys)
```

**Console/Terminal output:**
```
Paste terminal output here
```

## 🔧 Additional Context

- Are you using any specific financial data providers?
- What stock symbols were you analyzing when this occurred?
- Any specific time ranges or parameters involved?
- Any custom configuration or modifications?

## ✅ Pre-submission Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have removed all sensitive data (API keys, personal info)
- [ ] I can reproduce this bug consistently
- [ ] I have included relevant error messages and logs
- [ ] I understand this is educational software with no financial guarantees

## 🏷️ Bug Classification

**Severity:**
- [ ] Critical (crashes, data loss)
- [ ] High (major feature broken)
- [ ] Medium (feature partially working)
- [ ] Low (minor issue, workaround available)

**Component:**
- [ ] Data fetching (Yahoo Finance, finviz)
- [ ] Technical analysis calculations
- [ ] Stock screening
- [ ] Portfolio, watchlists, or trade journal
- [ ] Backtesting (`[backtesting]` extra)
- [ ] Research (`[research]` extra)
- [ ] Database operations
- [ ] Caching (Redis)
- [ ] MCP server/tools
- [ ] MCP client integration (Claude Desktop, Claude Code, Cursor, etc.)
- [ ] Installation/Setup

**Additional Labels:**
- [ ] documentation (if docs need updating)
- [ ] good first issue (if suitable for newcomers)
- [ ] help wanted (if community help is needed)