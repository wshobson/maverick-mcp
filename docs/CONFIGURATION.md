# MaverickMCP Configuration Guide

This guide covers all configuration options for MaverickMCP, including the BYOK (Bring Your Own Key) LLM provider system.

## Configuration via Environment Variables

MaverickMCP uses environment variables for all configuration. Copy `.env.example` to `.env` and update the values for your setup.

```bash
cp .env.example .env
```

---

## LLM Configuration (BYOK - Bring Your Own Key)

MaverickMCP supports multiple LLM providers through a BYOK model. You can use OpenRouter (default), OpenAI, Anthropic, or any OpenAI-compatible endpoint.

### Provider Selection

| Variable | Default | Valid Values |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto`, `openrouter`, `openai`, `anthropic` |

- **`auto`** (default) - Automatically selects the first provider with an available API key. Resolution order: OpenRouter -> OpenAI -> Anthropic -> FakeListLLM (testing fallback).
- **`openrouter`** - Explicitly use OpenRouter. Requires `OPENROUTER_API_KEY`.
- **`openai`** - Explicitly use OpenAI. Requires `OPENAI_API_KEY`.
- **`anthropic`** - Explicitly use Anthropic. Requires `ANTHROPIC_API_KEY`.

### Base URL Override

Override the default API endpoint for any provider. This is useful for self-hosted models, corporate proxies, or alternative OpenAI-compatible services.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | *(none)* | Override OpenAI API base URL (e.g., `https://my-proxy.example.com/v1`) |
| `ANTHROPIC_BASE_URL` | *(none)* | Override Anthropic API base URL (e.g., `https://my-proxy.example.com`) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Override OpenRouter API base URL |

When a base URL is not set (or set to an empty string), the factory uses the provider's default endpoint.

### Model Override

Change the default model used for each provider.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_DEFAULT_MODEL` | `gpt-4o` | Default model for OpenAI provider |
| `ANTHROPIC_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Default model for Anthropic provider |

OpenRouter model selection is handled automatically by the intelligent model selector based on task type and cost preferences.

### Temperature

| Variable | Default | Range |
|---|---|---|
| `LLM_TEMPERATURE` | `0.3` | `0.0` - `1.0` |

Controls the sampling temperature for all providers. Lower values produce more deterministic outputs; higher values produce more creative responses.

### API Keys

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | No* | OpenRouter API key. Get one at [openrouter.ai](https://openrouter.ai). |
| `OPENAI_API_KEY` | No* | OpenAI API key. Get one at [platform.openai.com](https://platform.openai.com). |
| `ANTHROPIC_API_KEY` | No* | Anthropic API key. Get one at [console.anthropic.com](https://console.anthropic.com). |

\*At least one API key is required for LLM features. With `LLM_PROVIDER=auto`, the system uses whichever key is available.

---

## Usage Examples

### Example 1: OpenRouter (Default)

The simplest setup - just set an OpenRouter key and everything works automatically.

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
LLM_PROVIDER=auto
```

### Example 2: OpenAI Direct

Use OpenAI directly instead of through OpenRouter.

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

### Example 3: Anthropic Direct

Use Anthropic's Claude models directly.

```bash
# .env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

### Example 4: OpenAI-Compatible Endpoint (Self-Hosted)

Point MaverickMCP at a self-hosted LLM service such as LM Studio, Ollama, vLLM, or LiteLLM that exposes an OpenAI-compatible API.

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=not-needed             # Some proxies require a non-empty key
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_DEFAULT_MODEL=my-local-model
```

Common OpenAI-compatible endpoints:

| Service | Typical Base URL |
|---|---|
| LM Studio | `http://localhost:1234/v1` |
| Ollama (with OpenAI compat) | `http://localhost:11434/v1` |
| vLLM | `http://localhost:8000/v1` |
| LiteLLM Proxy | `http://localhost:4000/v1` |

### Example 5: Corporate Proxy

Route all LLM calls through a corporate proxy gateway.

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://llm-proxy.internal.example.com/v1
```

### Example 6: Anthropic via Proxy

Route Anthropic API calls through a custom endpoint.

```bash
# .env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
ANTHROPIC_BASE_URL=https://my-proxy.example.com
```

### Example 7: Custom OpenRouter Endpoint

Use an OpenRouter-compatible service or mirror.

```bash
# .env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://custom-openrouter.example.com/api/v1
```

### Example 8: Override Model and Temperature

Switch to a different model and increase creativity.

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
OPENAI_DEFAULT_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
```

---

## Auto-Detection Logic

When `LLM_PROVIDER=auto` (the default), the system checks for API keys in this order:

1. **OpenRouter** - if `OPENROUTER_API_KEY` is set -> uses OpenRouter with intelligent model selection
2. **OpenAI** - if `OPENAI_API_KEY` is set -> uses `ChatOpenAI` with the default or overridden model
3. **Anthropic** - if `ANTHROPIC_API_KEY` is set -> uses `ChatAnthropic` with the default or overridden model
4. **Fallback** - if no keys are set -> uses `FakeListLLM` (testing-only mock responses)

This means existing deployments that only set `OPENROUTER_API_KEY` continue to work exactly as before with no changes required.

---

## Full Environment Variable Reference

### Application

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `MaverickMCP` | Application name |
| `ENVIRONMENT` | `development` | Environment: `development` or `production` |
| `LOG_LEVEL` | `info` | Logging level |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API bind port |
| `API_DEBUG` | `false` | Enable debug mode |

### Database

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///maverick_mcp.db` | Database connection URL |

### Redis (Optional)

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_PASSWORD` | *(empty)* | Redis password |
| `REDIS_SSL` | `false` | Enable SSL for Redis |

### Data Providers

| Variable | Default | Description |
|---|---|---|
| `TIINGO_API_KEY` | *(required)* | Tiingo API key for market data |
| `EXA_API_KEY` | *(optional)* | Exa API key for web search |
| `TAVILY_API_KEY` | *(optional)* | Tavily API key for web search |

### Cache

| Variable | Default | Description |
|---|---|---|
| `CACHE_TTL_SECONDS` | `604800` | Cache TTL in seconds (7 days) |
| `CACHE_ENABLED` | `true` | Enable caching |
