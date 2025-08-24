## 📋 Pull Request Summary

**Brief description of changes:**
A concise description of what this PR accomplishes.

**Related issue(s):**
- Fixes #(issue)
- Closes #(issue)
- Addresses #(issue)

## 💰 Financial Disclaimer Acknowledgment

- [ ] I understand this is educational software and not financial advice
- [ ] Any financial analysis features include appropriate disclaimers
- [ ] This PR maintains the educational/personal-use focus of the project

## 🔄 Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update (improvements to documentation)
- [ ] 🔧 Refactor (code changes that neither fix bugs nor add features)
- [ ] ⚡ Performance improvement
- [ ] 🧹 Chore (dependencies, build tools, etc.)

## 🎯 Component Areas

**Primary areas affected:**
- [ ] Data fetching (Tiingo, Yahoo Finance, FRED)
- [ ] Technical analysis calculations
- [ ] Stock screening strategies  
- [ ] Portfolio analysis and optimization
- [ ] MCP server/tools implementation
- [ ] Database operations and models
- [ ] Caching (Redis/in-memory)
- [ ] Claude Desktop integration
- [ ] Development tools and setup
- [ ] Documentation and examples

## 🔧 Implementation Details

**Technical approach:**
Describe the technical approach and any architectural decisions.

**Key changes:**
- Changed X to improve Y
- Added new function Z for feature A
- Refactored B to better handle C

**Dependencies:**
- [ ] No new dependencies added
- [ ] New dependencies added (list below)
- [ ] Dependencies removed (list below)

**New dependencies added:**
- package-name==version (reason for adding)

## 🧪 Testing

**Testing performed:**
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated  
- [ ] Manual testing completed
- [ ] Tested with Claude Desktop
- [ ] Tested with different data sources
- [ ] Performance testing completed

**Test scenarios covered:**
- [ ] Happy path functionality
- [ ] Error handling and edge cases
- [ ] Data validation and sanitization
- [ ] API rate limiting compliance
- [ ] Database operations
- [ ] Cache behavior

**Manual testing:**
```bash
# Commands used for testing
make test
make test-integration
# etc.
```

## 📊 Financial Analysis Impact

**Financial calculations:**
- [ ] No financial calculations affected
- [ ] New financial calculations added (validated for accuracy)
- [ ] Existing calculations modified (thoroughly tested)
- [ ] All calculations include appropriate disclaimers

**Data providers:**
- [ ] No data provider changes
- [ ] New data provider integration
- [ ] Existing provider modifications
- [ ] Rate limiting compliance verified

**Market data handling:**
- [ ] Historical data processing
- [ ] Real-time data integration
- [ ] Technical indicator calculations
- [ ] Screening algorithm changes

## 🔒 Security Considerations

**Security checklist:**
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Error handling doesn't leak sensitive information
- [ ] API keys handled securely via environment variables
- [ ] SQL injection prevention verified
- [ ] Rate limiting respected for external APIs

## 📚 Documentation

**Documentation updates:**
- [ ] Code comments added/updated
- [ ] README.md updated (if needed)
- [ ] API documentation updated
- [ ] Examples/tutorials added
- [ ] Financial disclaimers included where appropriate

**Breaking changes documentation:**
- [ ] No breaking changes
- [ ] Breaking changes documented in PR description
- [ ] Migration guide provided
- [ ] CHANGELOG.md updated

## ✅ Pre-submission Checklist

**Code quality:**
- [ ] Code follows the project style guide
- [ ] Self-review of code completed
- [ ] Tests added/updated and passing
- [ ] No linting errors (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] All tests pass (`make test`)

**Financial software standards:**
- [ ] Financial disclaimers included where appropriate
- [ ] No investment advice or guarantees provided
- [ ] Educational purpose maintained
- [ ] Data accuracy considerations documented
- [ ] Risk warnings included for relevant features

**Community standards:**
- [ ] PR title is descriptive and follows convention
- [ ] Description clearly explains the changes
- [ ] Related issues are linked
- [ ] Screenshots/examples included (if applicable)
- [ ] Ready for review

## 📸 Screenshots/Examples

**Before and after (if applicable):**
<!-- Add screenshots, CLI output, or code examples -->

**New functionality examples:**
```python
# Example of new feature usage
result = new_function(symbol="AAPL", period=20)
print(result)
```

## 🤝 Review Guidance

**Areas needing special attention:**
- Focus on X because of Y
- Pay attention to Z implementation
- Verify A works correctly with B

**Questions for reviewers:**
- Does the implementation approach make sense?
- Are there any security concerns?
- Is the documentation clear and complete?
- Any suggestions for improvement?

## 🚀 Deployment Notes

**Environment considerations:**
- [ ] No environment changes required
- [ ] New environment variables needed (documented)
- [ ] Database migrations required
- [ ] Cache invalidation needed

**Rollback plan:**
- [ ] Changes are fully backward compatible
- [ ] Database migrations are reversible
- [ ] Rollback steps documented below

**Rollback steps (if needed):**
1. Step 1
2. Step 2

## 🎓 Educational Impact

**Learning value:**
- What financial concepts does this help teach?
- How does this improve the developer experience?
- What new capabilities does this enable for users?

**Community benefit:**
- Who will benefit from these changes?
- How does this advance the project's educational mission?
- Any potential for broader community impact?

---

**Additional Notes:**
Any other information that would be helpful for reviewers.