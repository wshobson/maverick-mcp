"""
Template for creating new MCP tools.

Copy this file and modify it to create new tools quickly.
"""

from typing import Any

from maverick_mcp.api.server import mcp
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


@mcp.tool()
async def tool_name(
    param1: str,
    param2: int = 10,
    param3: bool = True,
) -> dict[str, Any]:
    """
    Brief description of what this tool does.

    This tool performs [specific action] and returns [expected output].

    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 10)
        param3: Description of third parameter (default: True)

    Returns:
        dict containing:
        - result: The main result of the operation
        - status: Success/failure status
        - details: Additional details about the operation

    Raises:
        ValueError: If parameters are invalid
        Exception: For other errors
    """
    # Log tool execution
    logger.info(
        "Executing tool_name",
        extra={
            "param1": param1,
            "param2": param2,
            "param3": param3,
        },
    )

    try:
        # Validate inputs
        if not param1:
            raise ValueError("param1 cannot be empty")

        if param2 < 0:
            raise ValueError("param2 must be non-negative")

        # Main tool logic here
        # Example: Fetch data, process it, return results

        # For tools that need database access:
        # from maverick_mcp.data.models import get_db
        # db = next(get_db())
        # try:
        #     # Database operations
        # finally:
        #     db.close()

        # For tools that need async operations:
        # import asyncio
        # results = await asyncio.gather(
        #     async_operation1(),
        #     async_operation2(),
        # )

        # Prepare response
        result = {
            "result": f"Processed {param1} with settings {param2}, {param3}",
            "status": "success",
            "details": {
                "processed_at": "2024-01-01T00:00:00Z",
                "item_count": 42,
            },
        }

        logger.info(
            "Tool completed successfully",
            extra={"tool": "tool_name", "result_keys": list(result.keys())},
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error in tool_name: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": "validation",
        }
    except Exception as e:
        logger.error(
            f"Unexpected error in tool_name: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "error": str(e),
            "error_type": "unexpected",
        }


# Example of a tool that doesn't require authentication
@mcp.tool()
async def public_tool_name(query: str) -> dict[str, Any]:
    """
    A public tool that doesn't require authentication.

    Args:
        query: The query to process

    Returns:
        dict with query results
    """
    return {
        "query": query,
        "results": ["result1", "result2"],
        "count": 2,
    }
