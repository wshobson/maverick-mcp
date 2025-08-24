# Security Policy

## Reporting Security Vulnerabilities

The MaverickMCP team takes security seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

## Reporting a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via GitHub Security Advisories (recommended).

Please include:
- Type of vulnerability
- Full paths of affected source files
- Location of the affected code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Security Features

MaverickMCP implements security measures appropriate for personal-use software:

### Personal Use Security Model
- **Local Deployment**: Designed to run locally for individual users
- **No Network Authentication**: Simplicity over complex auth systems
- **Environment Variable Security**: All API keys stored as environment variables
- **Basic Rate Limiting**: Protection against excessive API calls

### Data Protection
- **Input Validation**: Comprehensive Pydantic validation on all inputs
- **SQL Injection Prevention**: SQLAlchemy ORM with parameterized queries
- **API Key Security**: Secure handling of financial data provider credentials
- **Local Data Storage**: All analysis data stored locally by default

### Infrastructure Security
- **Environment Variables**: All secrets externalized, no hardcoded credentials
- **Secure Headers**: HSTS, CSP, X-Frame-Options, X-Content-Type-Options
- **Audit Logging**: Comprehensive security event logging
- **Circuit Breakers**: Protection against cascade failures

## Security Best Practices for Contributors

### Configuration
- Never commit secrets or API keys
- Use environment variables for all sensitive configuration
- Follow the `.env.example` template
- Use strong, unique passwords for development databases

### Code Guidelines
- Always validate and sanitize user input
- Use parameterized queries (SQLAlchemy ORM)
- Implement proper error handling without exposing sensitive information
- Follow the principle of least privilege
- Add rate limiting to new endpoints

### Dependencies
- Keep dependencies up to date
- Review security advisories regularly
- Run `safety check` before releases
- Use `bandit` for static security analysis

## Security Checklist for Pull Requests

- [ ] No hardcoded secrets or credentials
- [ ] Input validation on all user-provided data
- [ ] Proper error handling without information leakage
- [ ] API key handling follows environment variable patterns
- [ ] Financial data handling includes appropriate disclaimers
- [ ] Security tests for new features
- [ ] No vulnerable dependencies introduced
- [ ] Personal-use security model maintained (no complex auth)

## Running Security Audits

### Dependency Scanning
```bash
# Install security tools
pip install safety bandit

# Check for known vulnerabilities
safety check

# Static security analysis
bandit -r maverick_mcp/
```

### Additional Security Tools
```bash
# OWASP dependency check
pip install pip-audit
pip-audit

# Advanced static analysis
pip install semgrep
semgrep --config=auto maverick_mcp/
```

## Security Headers Configuration

The application implements the following security headers:
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`

## Incident Response

In case of a security incident:

1. **Immediate Response**: Assess the severity and impact
2. **Containment**: Isolate affected systems
3. **Investigation**: Determine root cause and extent
4. **Remediation**: Fix the vulnerability
5. **Recovery**: Restore normal operations
6. **Post-Incident**: Document lessons learned

## Security Contacts

- **Primary**: [GitHub Security Advisories](https://github.com/wshobson/maverick-mcp/security) (Recommended)
- **Alternative**: [GitHub Issues](https://github.com/wshobson/maverick-mcp/issues) (Public security issues only)
- **Community**: [GitHub Discussions](https://github.com/wshobson/maverick-mcp/discussions)

## Acknowledgments

We would like to thank the following individuals for responsibly disclosing security issues:

*This list will be updated as vulnerabilities are reported and fixed.*

## Financial Data Security

### Investment Data Protection
- **Personal Investment Information**: Never share account details, positions, or personal financial data
- **API Keys**: Secure storage of financial data provider API keys (Tiingo, FRED, etc.)
- **Market Data**: Ensure compliance with data provider terms of service and usage restrictions
- **Analysis Results**: Be aware that financial analysis outputs may contain sensitive investment insights

### Compliance Considerations
- **Financial Regulations**: Users must comply with applicable securities laws (SEC, CFTC, etc.)
- **Data Privacy**: Market analysis and portfolio data should be treated as confidential
- **Audit Trails**: Financial analysis activities may need to be logged for regulatory purposes
- **Cross-border Data**: Consider regulations when using financial data across international boundaries

## Financial Disclaimer for Security Context

**IMPORTANT**: This security policy covers the technical security of the software only. The financial analysis and investment tools provided by MaverickMCP are for educational purposes only and do not constitute financial advice. Always consult with qualified financial professionals for investment decisions.

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [SEC Cybersecurity Guidelines](https://www.sec.gov/spotlight/cybersecurity)
- [Financial Data Security Best Practices](https://www.cisa.gov/financial-services)

---

Thank you for helping keep MaverickMCP and its users safe!