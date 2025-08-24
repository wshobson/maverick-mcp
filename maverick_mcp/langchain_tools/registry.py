"""
Tool registry for managing MCP and LangChain tools.
"""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

from .adapters import mcp_to_langchain_adapter

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tools from different sources."""

    def __init__(self):
        self.mcp_tools: dict[str, Callable] = {}
        self.langchain_tools: dict[str, BaseTool] = {}
        self.tool_metadata: dict[str, dict[str, Any]] = {}

    def register_mcp_tool(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        persona_aware: bool = False,
        categories: list[str] | None = None,
    ) -> None:
        """
        Register an MCP tool function.

        Args:
            func: The MCP tool function
            name: Optional custom name
            description: Optional description
            persona_aware: Whether tool should adapt to personas
            categories: Tool categories for organization
        """
        tool_name = name or func.__name__
        self.mcp_tools[tool_name] = func

        # Store metadata
        self.tool_metadata[tool_name] = {
            "source": "mcp",
            "description": description or func.__doc__,
            "persona_aware": persona_aware,
            "categories": categories or [],
            "original_func": func,
        }

        logger.info(f"Registered MCP tool: {tool_name}")

    def register_langchain_tool(
        self, tool: BaseTool, categories: list[str] | None = None
    ) -> None:
        """
        Register a LangChain tool.

        Args:
            tool: The LangChain tool
            categories: Tool categories for organization
        """
        self.langchain_tools[tool.name] = tool

        # Store metadata
        self.tool_metadata[tool.name] = {
            "source": "langchain",
            "description": tool.description,
            "persona_aware": tool.metadata.get("persona_aware", False)
            if hasattr(tool, "metadata") and tool.metadata
            else False,
            "categories": categories or [],
            "tool_instance": tool,
        }

        logger.info(f"Registered LangChain tool: {tool.name}")

    def get_tool(self, name: str, as_langchain: bool = True) -> Any | None:
        """
        Get a tool by name.

        Args:
            name: Tool name
            as_langchain: Whether to return as LangChain tool

        Returns:
            Tool instance or function
        """
        # Check if it's already a LangChain tool
        if name in self.langchain_tools:
            return self.langchain_tools[name]

        # Check if it's an MCP tool
        if name in self.mcp_tools:
            if as_langchain:
                # Convert to LangChain tool on demand
                metadata = self.tool_metadata[name]
                return mcp_to_langchain_adapter(
                    self.mcp_tools[name],
                    name=name,
                    description=metadata["description"],
                    persona_aware=metadata["persona_aware"],
                )
            else:
                return self.mcp_tools[name]

        return None

    def get_tools_by_category(
        self, category: str, as_langchain: bool = True
    ) -> list[Any]:
        """
        Get all tools in a category.

        Args:
            category: Category name
            as_langchain: Whether to return as LangChain tools

        Returns:
            List of tools
        """
        tools = []

        for name, metadata in self.tool_metadata.items():
            if category in metadata.get("categories", []):
                tool = self.get_tool(name, as_langchain=as_langchain)
                if tool:
                    tools.append(tool)

        return tools

    def get_all_tools(self, as_langchain: bool = True) -> list[Any]:
        """
        Get all registered tools.

        Args:
            as_langchain: Whether to return as LangChain tools

        Returns:
            List of all tools
        """
        tools: list[Any] = []

        # Add all LangChain tools
        if as_langchain:
            tools.extend(self.langchain_tools.values())

        # Add all MCP tools
        for name in self.mcp_tools:
            if name not in self.langchain_tools:  # Avoid duplicates
                tool = self.get_tool(name, as_langchain=as_langchain)
                if tool:
                    tools.append(tool)

        return tools

    def get_persona_aware_tools(self, as_langchain: bool = True) -> list[Any]:
        """Get all persona-aware tools."""
        tools = []

        for name, metadata in self.tool_metadata.items():
            if metadata.get("persona_aware", False):
                tool = self.get_tool(name, as_langchain=as_langchain)
                if tool:
                    tools.append(tool)

        return tools

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """List all tools with their metadata."""
        return {
            name: {
                "description": meta["description"],
                "source": meta["source"],
                "persona_aware": meta["persona_aware"],
                "categories": meta["categories"],
            }
            for name, meta in self.tool_metadata.items()
        }


# Global registry instance
_tool_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        _initialize_default_tools()
    return _tool_registry


def _initialize_default_tools():
    """Initialize registry with default MCP tools."""
    get_tool_registry()

    try:
        # TODO: Fix router tool registration
        # The router tools are FastMCP FunctionTool instances, not plain Callables
        # Need to extract the underlying function or adapt the registration approach

        logger.info("Tool registry initialized (router tools registration pending)")

    except ImportError as e:
        logger.warning(f"Could not import default MCP tools: {e}")
