# DeepResearchAgent Documentation

## Overview

The DeepResearchAgent provides comprehensive financial research capabilities using web search, content analysis, and AI-powered insights. It integrates seamlessly with the existing maverick-mcp architecture and adapts research depth and focus based on investor personas.

## Key Features

### 🔍 Comprehensive Research
- **Multi-Source Web Search**: Integrates Exa AI and Tavily for comprehensive coverage
- **Content Analysis**: AI-powered extraction of insights, sentiment, and key themes
- **Source Credibility**: Automatic scoring and validation of information sources
- **Citation Management**: Proper citations and reference tracking
- **Fact Validation**: Cross-referencing and validation of research claims

### 🎯 Persona-Aware Research
- **Conservative**: Focus on stability, dividends, risk factors, established companies
- **Moderate**: Balanced approach with growth and value considerations
- **Aggressive**: Emphasis on growth opportunities, momentum, high-return potential
- **Day Trader**: Short-term focus, liquidity, technical factors, immediate opportunities

### 🏗️ LangGraph 2025 Integration
- **State Management**: Comprehensive state tracking with `DeepResearchState`
- **Workflow Orchestration**: Multi-step research process with error handling
- **Streaming Support**: Real-time progress updates and streaming responses
- **Circuit Breaker**: Automatic failover and rate limiting protection

## Architecture

### Core Components

```
DeepResearchAgent
├── ResearchQueryAnalyzer     # Query analysis and strategy planning
├── WebSearchProvider         # Multi-provider search (Exa, Tavily)
├── ContentAnalyzer          # AI-powered content analysis
├── PersonaAdapter           # Persona-specific result filtering
└── CacheManager            # Intelligent caching and performance
```

### State Management

The `DeepResearchState` extends `BaseAgentState` with comprehensive tracking:

```python
class DeepResearchState(BaseAgentState):
    # Research parameters
    research_query: str
    research_scope: str  
    research_depth: str
    timeframe: str
    
    # Source management
    raw_sources: list[dict]
    processed_sources: list[dict]
    source_credibility: dict[str, float]
    
    # Content analysis
    extracted_content: dict[str, str]
    key_insights: list[dict]
    sentiment_analysis: dict
    
    # Research findings
    research_themes: list[dict]
    consensus_view: dict
    contrarian_views: list[dict]
    
    # Persona adaptation
    persona_focus_areas: list[str]
    actionable_insights: list[dict]
```

## Usage Examples

### Standalone Usage

```python
from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.providers.llm_factory import get_llm

# Initialize agent
llm = get_llm()
research_agent = DeepResearchAgent(
    llm=llm,
    persona="moderate",
    max_sources=50,
    research_depth="comprehensive"
)

# Company research
result = await research_agent.research_company_comprehensive(
    symbol="AAPL",
    session_id="research_session",
    include_competitive_analysis=True
)

# Market sentiment analysis
sentiment = await research_agent.analyze_market_sentiment(
    topic="artificial intelligence stocks",
    session_id="sentiment_session",
    timeframe="1w"
)

# Custom research
custom = await research_agent.research_topic(
    query="impact of Federal Reserve policy on tech stocks",
    session_id="custom_session",
    research_scope="comprehensive",
    timeframe="1m"
)
```

### SupervisorAgent Integration

```python
from maverick_mcp.agents.supervisor import SupervisorAgent

# Create supervisor with research agent
supervisor = SupervisorAgent(
    llm=llm,
    agents={
        "market": market_agent,
        "technical": technical_agent,
        "research": research_agent  # DeepResearchAgent
    },
    persona="moderate"
)

# Coordinated analysis
result = await supervisor.coordinate_agents(
    query="Should I invest in MSFT? I want comprehensive analysis",
    session_id="coordination_session"
)
```

### MCP Tools Integration

Available MCP tools for Claude Desktop:

1. **`comprehensive_research`** - Deep research on any financial topic
2. **`analyze_market_sentiment`** - Market sentiment analysis
3. **`research_company_comprehensive`** - Company fundamental analysis
4. **`search_financial_news`** - News search and analysis
5. **`validate_research_claims`** - Fact-checking and validation

#### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "maverick-research": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/research"]
    }
  }
}
```

#### Example Prompts

- "Research Tesla's competitive position in the EV market with comprehensive analysis"
- "Analyze current market sentiment for renewable energy stocks"
- "Perform fundamental analysis of Apple (AAPL) including competitive advantages"
- "Search for recent news about Federal Reserve interest rate decisions"

## Configuration

### Environment Variables

```bash
# Required API Keys
EXA_API_KEY=your_exa_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional Configuration
RESEARCH_MAX_SOURCES=50
RESEARCH_CACHE_TTL_HOURS=4
RESEARCH_DEPTH=comprehensive
```

### Settings

```python
from maverick_mcp.config.settings import get_settings

settings = get_settings()

# Research settings
research_config = settings.research
print(f"Max sources: {research_config.default_max_sources}")
print(f"Cache TTL: {research_config.cache_ttl_hours} hours")
print(f"Trusted domains: {research_config.trusted_domains}")
```

## Research Workflow

### 1. Query Analysis
- Classify research type (company, sector, market, news, fundamental)
- Determine appropriate search strategies and sources
- Set persona-specific focus areas and priorities

### 2. Search Execution
- Execute parallel searches across multiple providers
- Apply domain filtering and content type selection
- Handle rate limiting and error recovery

### 3. Content Processing
- Extract and clean content from sources
- Remove duplicates and low-quality sources
- Score sources for credibility and relevance

### 4. Content Analysis
- AI-powered insight extraction
- Sentiment analysis and trend detection
- Theme identification and cross-referencing

### 5. Persona Adaptation
- Filter insights for persona relevance
- Adjust risk assessments and recommendations
- Generate persona-specific action items

### 6. Result Synthesis
- Consolidate findings into coherent analysis
- Generate citations and source references
- Calculate confidence scores and quality metrics

## Persona Behaviors

### Conservative Investor
- **Focus**: Stability, dividends, established companies, risk factors
- **Sources**: Prioritize authoritative financial publications
- **Insights**: Emphasize capital preservation and low-risk opportunities
- **Actions**: More cautious recommendations with detailed risk analysis

### Moderate Investor  
- **Focus**: Balanced growth and value, diversification
- **Sources**: Mix of news, analysis, and fundamental reports
- **Insights**: Balanced view of opportunities and risks
- **Actions**: Moderate position sizing with measured recommendations

### Aggressive Investor
- **Focus**: Growth opportunities, momentum, high-return potential
- **Sources**: Include social media sentiment and trending analysis
- **Insights**: Emphasize upside potential and growth catalysts
- **Actions**: Larger position sizing with growth-focused recommendations

### Day Trader
- **Focus**: Short-term catalysts, technical factors, liquidity
- **Sources**: Real-time news, social sentiment, technical analysis
- **Insights**: Immediate trading opportunities and momentum indicators
- **Actions**: Quick-turn recommendations with tight risk controls

## Performance & Caching

### Intelligent Caching
- **Research Results**: 4-hour TTL for comprehensive research
- **Source Content**: 1-hour TTL for raw content
- **Sentiment Analysis**: 30-minute TTL for rapidly changing topics
- **Company Fundamentals**: 24-hour TTL for stable company data

### Rate Limiting
- **Exa AI**: Respects API rate limits with exponential backoff
- **Tavily**: Built-in rate limiting and request queuing
- **Content Analysis**: Batch processing to optimize LLM usage

### Performance Optimization
- **Parallel Search**: Concurrent execution across providers
- **Content Streaming**: Progressive result delivery
- **Circuit Breakers**: Automatic failover on provider issues
- **Connection Pooling**: Efficient network resource usage

## Error Handling

### Circuit Breaker Pattern
- Automatic provider failover on repeated failures
- Graceful degradation with partial results
- Recovery testing and automatic restoration

### Fallback Strategies
- Provider fallback (Exa → Tavily → Basic web search)
- Reduced scope fallback (comprehensive → standard → basic)
- Cached result fallback when live search fails

### Error Types
- `WebSearchError`: Search provider failures
- `ContentAnalysisError`: Content processing failures
- `ResearchError`: General research operation failures
- `CircuitBreakerError`: Circuit breaker activation

## Integration Points

### SupervisorAgent Routing
- Automatic routing for research-related queries
- Intelligent agent selection based on query complexity
- Result synthesis with technical and market analysis

### MCP Server Integration
- RESTful API endpoints for external access
- Standardized request/response formats
- Authentication and rate limiting support

### Database Integration
- Research result caching in PostgreSQL/SQLite
- Source credibility tracking and learning
- Historical research analysis and trends

## Best Practices

### Query Optimization
- Use specific, focused queries for better results
- Include timeframe context for temporal relevance
- Specify research depth based on needs (basic/standard/comprehensive)

### Persona Selection
- Choose persona that matches intended investment style
- Consider persona characteristics in result interpretation
- Use persona-specific insights for decision making

### Result Interpretation
- Review confidence scores and source diversity
- Consider contrarian views alongside consensus
- Validate critical claims through multiple sources

### Performance Tuning
- Adjust max_sources based on speed vs. comprehensiveness needs
- Use appropriate research_depth for the use case
- Monitor cache hit rates and adjust TTL settings

## Troubleshooting

### Common Issues

1. **No Results Found**
   - Check API key configuration
   - Verify internet connectivity
   - Try broader search terms

2. **Low Confidence Scores**
   - Increase max_sources parameter
   - Use longer timeframe for more data
   - Check for topic relevance and specificity

3. **Rate Limiting Errors**
   - Review API usage limits
   - Implement request spacing
   - Consider upgrading API plans

4. **Poor Persona Alignment**
   - Review persona characteristics
   - Adjust focus areas in research strategy
   - Consider custom persona configuration

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
research_agent = DeepResearchAgent(
    llm=llm,
    persona="moderate",
    research_depth="comprehensive"
)
```

## Future Enhancements

### Planned Features
- **Multi-language Support**: Research in multiple languages
- **PDF Analysis**: Direct analysis of earnings reports and filings
- **Real-time Alerts**: Research-based alert generation
- **Custom Personas**: User-defined persona characteristics
- **Research Collaboration**: Multi-user research sessions

### API Extensions
- **Batch Research**: Process multiple queries simultaneously
- **Research Templates**: Pre-configured research workflows
- **Historical Analysis**: Time-series research trend analysis
- **Integration APIs**: Third-party platform integrations

---

## Support

For questions, issues, or feature requests related to the DeepResearchAgent:

1. Check the troubleshooting section above
2. Review the example code in `/examples/deep_research_integration.py`
3. Enable debug logging for detailed error information
4. Consider the integration patterns with SupervisorAgent for complex workflows

The DeepResearchAgent is designed to provide institutional-quality research capabilities while maintaining the flexibility and persona-awareness that makes it suitable for individual investors across all experience levels.