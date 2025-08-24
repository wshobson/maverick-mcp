"""
Adapters to convert MCP tools to LangChain tools.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


def extract_tool_schema(func: Callable) -> type[BaseModel]:
    """
    Extract parameter schema from a function's signature and annotations.

    Args:
        func: Function to extract schema from

    Returns:
        Pydantic model representing the function's parameters
    """
    sig = inspect.signature(func)
    fields = {}

    for param_name, param in sig.parameters.items():
        if param_name in ["self", "cls"]:
            continue

        # Get type annotation
        param_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )

        # Get default value
        default = ... if param.default == inspect.Parameter.empty else param.default

        # Extract description from docstring if available
        description = f"Parameter {param_name}"
        if func.__doc__:
            # Simple extraction - could be improved with proper docstring parsing
            lines = func.__doc__.split("\n")
            for line in lines:
                if param_name in line and ":" in line:
                    description = line.split(":", 1)[1].strip()
                    break

        fields[param_name] = (
            param_type,
            Field(default=default, description=description),
        )

    # Create dynamic model
    model_name = f"{func.__name__.title()}Schema"
    return create_model(model_name, **fields)


def mcp_to_langchain_adapter(
    mcp_tool: Callable,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    return_direct: bool = False,
    persona_aware: bool = False,
) -> StructuredTool:
    """
    Convert an MCP tool function to a LangChain StructuredTool.

    Args:
        mcp_tool: The MCP tool function to convert
        name: Optional custom name for the tool
        description: Optional custom description
        args_schema: Optional Pydantic model for arguments
        return_direct: Whether to return tool output directly
        persona_aware: Whether this tool should be persona-aware

    Returns:
        LangChain StructuredTool
    """
    # Extract metadata
    tool_name = name or mcp_tool.__name__
    tool_description = description or mcp_tool.__doc__ or f"Tool: {tool_name}"

    # Extract or use provided schema
    if args_schema is None:
        args_schema = extract_tool_schema(mcp_tool)

    # Create wrapper function to handle any necessary conversions
    async def async_wrapper(**kwargs):
        """Async wrapper for MCP tool."""
        try:
            result = await mcp_tool(**kwargs)
            return _format_tool_result(result)
        except Exception as e:
            logger.error(f"Error in tool {tool_name}: {str(e)}")
            return {"error": str(e), "status": "error"}

    def sync_wrapper(**kwargs):
        """Sync wrapper for MCP tool."""
        try:
            result = mcp_tool(**kwargs)
            return _format_tool_result(result)
        except Exception as e:
            logger.error(f"Error in tool {tool_name}: {str(e)}")
            return {"error": str(e), "status": "error"}

    # Determine if tool is async
    is_async = inspect.iscoroutinefunction(mcp_tool)

    # Create the structured tool
    if is_async:
        tool = StructuredTool(
            name=tool_name,
            description=tool_description,
            coroutine=async_wrapper,
            args_schema=args_schema,
            return_direct=return_direct,
        )
    else:
        tool = StructuredTool(
            name=tool_name,
            description=tool_description,
            func=sync_wrapper,
            args_schema=args_schema,
            return_direct=return_direct,
        )

    # Mark if persona-aware
    if persona_aware:
        tool.metadata = {"persona_aware": True}

    return tool


def _format_tool_result(result: Any) -> str | dict[str, Any]:
    """
    Format tool result for LangChain compatibility.

    Args:
        result: Raw tool result

    Returns:
        Formatted result
    """
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        return result
    elif hasattr(result, "dict"):
        # Pydantic model
        return result.dict()
    else:
        # Convert to string as fallback
        return str(result)


def create_langchain_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    return_direct: bool = False,
    persona_aware: bool = False,
):
    """
    Decorator to create a LangChain tool from a function.

    Usage:
        @create_langchain_tool(name="stock_screener", persona_aware=True)
        def screen_stocks(strategy: str, limit: int = 20) -> dict:
            ...
    """

    def decorator(f: Callable) -> StructuredTool:
        return mcp_to_langchain_adapter(
            f,
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            persona_aware=persona_aware,
        )

    if func is None:
        return decorator
    else:
        return decorator(func)


# Example persona-aware tool wrapper
class PersonaAwareToolWrapper(BaseTool):
    """Wrapper to make any tool persona-aware."""

    wrapped_tool: BaseTool
    persona_adjuster: Callable | None = None
    persona: str | None = None

    def __init__(
        self,
        tool: BaseTool,
        adjuster: Callable | None = None,
        persona: str | None = None,
    ):
        super().__init__(
            name=f"persona_aware_{tool.name}", description=tool.description
        )
        self.wrapped_tool = tool
        self.persona_adjuster = adjuster
        self.persona = persona

    def _run(self, *args, **kwargs):
        """Run tool with persona adjustments."""
        # Apply persona adjustments if available
        if self.persona_adjuster and hasattr(self, "persona"):
            kwargs = self.persona_adjuster(kwargs, self.persona)

        return self.wrapped_tool._run(*args, **kwargs)

    async def _arun(self, *args, **kwargs):
        """Async run tool with persona adjustments."""
        # Apply persona adjustments if available
        if self.persona_adjuster and hasattr(self, "persona"):
            kwargs = self.persona_adjuster(kwargs, self.persona)

        return await self.wrapped_tool._arun(*args, **kwargs)
