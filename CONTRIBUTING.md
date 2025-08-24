# Contributing to MaverickMCP

Welcome to MaverickMCP! We're excited to have you contribute to this open-source financial analysis MCP server.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Reporting Issues](#reporting-issues)
- [Financial Domain Guidelines](#financial-domain-guidelines)

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- PostgreSQL (optional, SQLite works for development)
- Redis (optional for development)

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/wshobson/maverick-mcp.git
   cd maverick-mcp
   ```

2. **Install dependencies**

   ```bash
   uv sync --extra dev
   ```

3. **Set up environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start development server**
   ```bash
   make dev
   # Or: ./scripts/start-backend.sh --dev
   ```

### Development Commands

- `make dev` - Start everything (recommended)
- `make test` - Run unit tests (5-10 seconds)
- `make lint` - Check code quality
- `make format` - Auto-format code
- `make typecheck` - Run type checking

### Pre-commit Hooks (Optional but Recommended)

We provide pre-commit hooks to ensure code quality:

```bash
# Install pre-commit (one time setup)
pip install pre-commit

# Install hooks for this repository
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

Pre-commit hooks will automatically run on every commit and include:

- Code formatting (ruff)
- Linting (ruff)
- Security scanning (bandit, safety)
- Custom financial domain validations

**Note**: Pre-commit hooks are optional for contributors but recommended for maintainers.

## Project Structure

MaverickMCP follows Domain-Driven Design (DDD) principles:

```
maverick_mcp/
├── api/           # FastAPI routers and server
├── domain/        # Core business logic (entities, services)
├── application/   # Use cases and DTOs
├── infrastructure/# External services (database, APIs)
├── auth/         # Authentication and security
├── config/       # Settings and configuration
└── tests/        # Test suite
```

## Running Tests

We use pytest with multiple test categories:

```bash
# Unit tests only (fast, ~5-10 seconds)
make test

# All tests including integration
make test-all

# Specific test
make test-specific TEST=test_name

# With coverage
pytest --cov=maverick_mcp
```

**Note**: Integration tests require PostgreSQL and Redis. They're excluded from CI by default.

## Code Style

We enforce strict code quality standards:

### Tools

- **ruff** for linting and formatting
- **pyright** for type checking
- **pytest** for testing

### Guidelines

1. **Type Hints**: Required for all functions and variables
2. **Docstrings**: Google-style docstrings for public APIs
3. **Error Handling**: Proper exception handling with specific error types
4. **Security**: Never hardcode secrets, always use environment variables

### Before Submitting

```bash
# Run all quality checks
make check  # Runs lint + typecheck

# Auto-format code
make format
```

## Making Changes

### Development Workflow

1. **Start with an issue** - Create or find an existing issue
2. **Create a branch** - Use descriptive branch names:

   - `feature/add-new-indicator`
   - `fix/authentication-bug`
   - `docs/improve-setup-guide`

3. **Make focused commits** - One logical change per commit
4. **Write tests** - Add tests for new features or bug fixes
5. **Update documentation** - Keep docs current with changes

### Financial Calculations

When working with financial logic:

- **Accuracy is critical** - Double-check all calculations
- **Use proper data types** - `Decimal` for currency, `float` for ratios
- **Include validation** - Validate input ranges and edge cases
- **Add comprehensive tests** - Test edge cases and boundary conditions
- **Document assumptions** - Explain the financial logic in docstrings

## Submitting Pull Requests

### Checklist

- [ ] Tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Documentation is updated
- [ ] Financial calculations are validated
- [ ] No hardcoded secrets or credentials

### PR Template

```markdown
## Description

Brief description of the changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Financial Impact

- [ ] No financial calculations affected
- [ ] Financial calculations verified for accuracy
- [ ] New financial calculations added with tests

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added (if applicable)
- [ ] Manual testing completed

## Screenshots (if applicable)
```

## Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Environment details** (Python version, OS, dependencies)
- **Reproduction steps** - Clear, minimal steps to reproduce
- **Expected vs actual behavior**
- **Error messages** - Full stack traces when available
- **Financial context** - If related to calculations or market data

### Feature Requests

Include:

- **Use case** - What problem does this solve?
- **Proposed solution** - How should it work?
- **Financial domain knowledge** - Any domain-specific requirements
- **Implementation considerations** - Technical constraints or preferences

## Financial Domain Guidelines

### Market Data

- **Respect rate limits** - All providers have API limits
- **Cache appropriately** - Balance freshness with performance
- **Handle market closures** - Account for weekends and holidays
- **Validate symbols** - Check ticker symbol formats

### Technical Indicators

- **Use established formulas** - Follow industry-standard calculations
- **Document data requirements** - Specify minimum periods needed
- **Handle edge cases** - Division by zero, insufficient data
- **Test with real market data** - Verify against known examples

### Risk Management

- **Position sizing** - Implement proper risk controls
- **Stop loss calculations** - Accurate risk/reward ratios
- **Portfolio limits** - Respect maximum position sizes
- **Backtesting accuracy** - Avoid look-ahead bias

### Financial Compliance for Contributors

- **Educational Purpose**: All financial calculations and analysis tools must be clearly marked as educational
- **No Investment Advice**: Never include language that could be construed as investment recommendations
- **Disclaimer Requirements**: Include appropriate disclaimers in docstrings for financial functions
- **Data Attribution**: Properly attribute data sources (Tiingo, Yahoo Finance, FRED, etc.)
- **Risk Warnings**: Include risk warnings in documentation for portfolio and trading-related features
- **Regulatory Awareness**: Be mindful of securities regulations (SEC, CFTC, international equivalents)

#### Financial Function Documentation Template

```python
def calculate_risk_metric(data: pd.DataFrame) -> float:
    """
    Calculate a financial risk metric.

    DISCLAIMER: This is for educational purposes only and does not
    constitute financial advice. Past performance does not guarantee
    future results.

    Args:
        data: Historical price data

    Returns:
        Risk metric value
    """
```

## Architecture Guidelines

### Domain-Driven Design

- **Domain layer** - Pure business logic, no external dependencies
- **Application layer** - Use cases and orchestration
- **Infrastructure layer** - Database, APIs, external services
- **API layer** - HTTP handlers, validation, serialization

### MCP Integration

- **Tool design** - Each tool should have a single, clear purpose
- **Resource management** - Implement proper caching and cleanup
- **Error handling** - Return meaningful error messages
- **Documentation** - Include usage examples and parameter descriptions

## Getting Help

- **Discussions** - Use GitHub Discussions for questions
- **Issues** - Create issues for bugs or feature requests
- **Code Review** - Participate in PR reviews to learn
- **Documentation** - Check existing docs and CLAUDE.md for project context

## Contributing to Open Source

### Community Standards

MaverickMCP follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/):

- **Be welcoming** - Help newcomers feel welcome
- **Be respectful** - Treat all contributors with respect
- **Be patient** - Allow time for responses and reviews
- **Be constructive** - Focus on improving the project
- **Be inclusive** - Welcome diverse perspectives and backgrounds

### Recognition

Contributors are recognized in multiple ways:

- **CHANGELOG.md** - All contributors listed in release notes
- **GitHub contributors** - Automatic recognition via commits
- **Special mentions** - Outstanding contributions highlighted in README
- **Hall of Fame** - Major contributors featured in documentation

### Continuous Integration

Our CI/CD pipeline ensures code quality:

- **Automated testing** - All PRs run comprehensive test suites
- **Security scanning** - Automated vulnerability detection
- **Code quality checks** - Linting, formatting, and type checking
- **Performance testing** - Benchmark validation on PRs
- **Documentation validation** - Ensures docs stay current

### Current Architecture (Simplified for Personal Use)

MaverickMCP has been cleaned up and simplified:

- **No Complex Auth**: Removed enterprise JWT/OAuth systems for simplicity
- **No Billing System**: Personal-use focused, no subscription management
- **Local First**: Designed to run locally with Claude Desktop
- **Educational Focus**: Built for learning and personal financial analysis
- **Clean Dependencies**: Removed unnecessary enterprise features

This makes the codebase much more approachable for contributors!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Financial Software Disclaimer for Contributors

**IMPORTANT**: By contributing to MaverickMCP, you acknowledge that:

- All financial analysis tools are for educational purposes only
- No content should be construed as investment advice or recommendations
- Contributors are not responsible for user investment decisions or outcomes
- All financial calculations should include appropriate disclaimers and risk warnings
- Data accuracy cannot be guaranteed and users must verify information independently

Contributors should review the full financial disclaimer in the LICENSE file and README.md.

## Recognition

Contributors will be acknowledged in our CHANGELOG and can be featured in project documentation. We appreciate all contributions, from code to documentation to issue reports!

---

Thank you for contributing to MaverickMCP! Your efforts help make sophisticated financial analysis tools accessible to everyone.
