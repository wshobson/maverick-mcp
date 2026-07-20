# Security posture

Engineering rules for this codebase. As of v1.0.0, `maverick/` is the whole
system; the legacy `maverick_mcp/` package (and any auth/billing surface it
carried) is deleted. Vulnerability reporting lives in the root `SECURITY.md`.

- The server has no authentication by design. It is a local, single-user
  tool. Remote deployment is out of scope until a design doc reopens it.
- Text fetched from third parties (news, filings, web search) is untrusted
  input. Tools return it labeled as data. Never blend it into instructions,
  tool descriptions, or prompts.
- Tool annotations are UX hints, not security guarantees.
- No secrets in tool output, logs, or error messages. API keys live in
  environment variables and never leave the process.
- Integrations with third-party data services require maintainer review of
  the provider itself, not just the code. See PR #209 for the precedent.
