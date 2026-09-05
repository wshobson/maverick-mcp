# Security Policy

MaverickMCP is a personal-use, local MCP server. It has no network
authentication by design: it runs on the user's own machine, binds its HTTP
transport to `127.0.0.1` unless started with `--host 0.0.0.0`, and reads API
keys only from environment variables. The engineering rules that follow from
that model live in `docs/SECURITY.md`.

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Report them through
[GitHub Security Advisories](https://github.com/wshobson/maverick-mcp/security/advisories/new).
Include the affected file paths, a commit or tag, reproduction steps, and the
impact you observed. You should receive a response within 48 hours.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.x     | yes       |
| < 1.0   | no        |

Versions before 1.0.0 shipped the legacy `maverick_mcp` package, which was
deleted at the v1.0.0 cutover and receives no fixes.

## Security model

- Local, single-user deployment. Remote deployment, authentication, and
  billing are out of scope until a design document reopens them.
- No secrets in the repository, in tool output, or in logs. Keys live in
  environment variables; see `.env.example`.
- Text fetched from third parties (market data, filings, web search) is
  untrusted input and is returned to the client labeled as data.
- Tool annotations such as `readOnlyHint` are hints for clients, not
  security guarantees.
- Dependencies are kept current through Dependabot, and `safety` is part of
  the dev extra.

## Security checklist for pull requests

- No hardcoded secrets or credentials.
- Input validated with Pydantic models at the tool boundary.
- Errors do not leak secrets or internal paths.
- API keys flow only through environment variables.
- No vulnerable dependencies introduced.
- Personal-use security model maintained: no auth or billing surface.
