"""
Comprehensive validation models for Maverick-MCP API.

This module provides Pydantic models for validating all tool inputs,
API requests, and responses, ensuring data integrity and providing
clear error messages with standardized response formats.
"""

# Auth validation removed for personal use
from .base import (
    DateRangeMixin,
    DateString,
    DateValidator,
    PaginationMixin,
    Percentage,
    PositiveFloat,
    PositiveInt,
    StrictBaseModel,
    TickerSymbol,
    TickerValidator,
)

# Billing validation removed for personal use
from .data import (
    ClearCacheRequest,
    FetchStockDataRequest,
    GetChartLinksRequest,
    GetNewsRequest,
    GetStockInfoRequest,
    StockDataBatchRequest,
)

# Error imports removed - use maverick_mcp.exceptions instead
from .middleware import (
    RateLimitMiddleware,
    SecurityMiddleware,
    ValidationMiddleware,
)
from .portfolio import (
    CorrelationAnalysisRequest,
    PortfolioComparisonRequest,
    RiskAnalysisRequest,
)
from .responses import (
    BaseResponse,
    BatchOperationResult,
    BatchResponse,
    DataResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    ListResponse,
    RateLimitInfo,
    RateLimitResponse,
    ValidationErrorResponse,
    WebhookEvent,
    WebhookResponse,
    error_response,
    success_response,
    validation_error_response,
)
from .screening import (
    CustomScreeningRequest,
    MaverickScreeningRequest,
    SupplyDemandBreakoutRequest,
)
from .technical import (
    MACDAnalysisRequest,
    RSIAnalysisRequest,
    StockChartRequest,
    SupportResistanceRequest,
    TechnicalAnalysisRequest,
)

# Webhook validation removed for personal use

__all__ = [
    # Base validation
    "DateRangeMixin",
    "DateString",
    "DateValidator",
    "PaginationMixin",
    "Percentage",
    "PositiveFloat",
    "PositiveInt",
    "StrictBaseModel",
    "TickerSymbol",
    "TickerValidator",
    # Data validation
    "FetchStockDataRequest",
    "StockDataBatchRequest",
    "GetStockInfoRequest",
    "GetNewsRequest",
    "GetChartLinksRequest",
    "ClearCacheRequest",
    # Middleware
    "RateLimitMiddleware",
    "SecurityMiddleware",
    "ValidationMiddleware",
    # Portfolio validation
    "RiskAnalysisRequest",
    "PortfolioComparisonRequest",
    "CorrelationAnalysisRequest",
    # Response models
    "BaseResponse",
    "BatchOperationResult",
    "BatchResponse",
    "DataResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "HealthStatus",
    "ListResponse",
    "RateLimitInfo",
    "RateLimitResponse",
    "ValidationErrorResponse",
    "WebhookEvent",
    "WebhookResponse",
    "error_response",
    "success_response",
    "validation_error_response",
    # Screening validation
    "MaverickScreeningRequest",
    "SupplyDemandBreakoutRequest",
    "CustomScreeningRequest",
    # Technical validation
    "RSIAnalysisRequest",
    "MACDAnalysisRequest",
    "SupportResistanceRequest",
    "TechnicalAnalysisRequest",
    "StockChartRequest",
]
