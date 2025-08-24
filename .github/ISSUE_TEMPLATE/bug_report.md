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
 - Python Version: [e.g. 3.11.5]
 - MaverickMCP Version: [e.g. 0.1.0]
 - Installation Method: [e.g. pip, uv, git clone]

**Claude Desktop (if applicable):**
 - Claude Desktop Version: [e.g. 1.0.0]
 - mcp-remote Version: [if using Claude Desktop]

**Dependencies:**
 - FastMCP Version: [e.g. 2.7.0]
 - FastAPI Version: [e.g. 0.115.0]
 - Database: [SQLite, PostgreSQL]
 - Redis: [Yes/No, version if yes]

## 📋 Configuration

**Environment Variables (remove sensitive data):**
```
TIINGO_API_KEY=***
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
- [ ] Data fetching (Tiingo, Yahoo Finance)
- [ ] Technical analysis calculations
- [ ] Stock screening
- [ ] Database operations
- [ ] Caching (Redis)
- [ ] MCP server/tools
- [ ] Claude Desktop integration
- [ ] Installation/Setup

**Additional Labels:**
- [ ] documentation (if docs need updating)
- [ ] good first issue (if suitable for newcomers)
- [ ] help wanted (if community help is needed)