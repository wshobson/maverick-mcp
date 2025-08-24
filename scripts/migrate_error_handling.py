#!/usr/bin/env python3
"""
Script to migrate generic exception handlers to specific ones in API routers.

This script analyzes all router files and replaces generic `except Exception as e:`
with specific exception handlers based on the context.
"""

import re
from pathlib import Path

# Router files to process
ROUTER_DIR = Path(__file__).parent.parent / "maverick_mcp" / "api" / "routers"

# Exception mapping based on common patterns
EXCEPTION_MAPPINGS = {
    # Database operations
    "get_db": [
        ("sqlalchemy.exc.OperationalError", "Database connection failed"),
        ("sqlalchemy.exc.IntegrityError", "Data integrity constraint violated"),
    ],
    "db.query": [
        ("sqlalchemy.exc.OperationalError", "Database query failed"),
        ("maverick_mcp.exceptions.DataNotFoundError", "Requested data not found"),
    ],
    "db.commit": [
        ("sqlalchemy.exc.IntegrityError", "Database constraint violation"),
        ("sqlalchemy.exc.OperationalError", "Database operation failed"),
    ],
    # Authentication/Authorization
    "verify_token": [
        ("maverick_mcp.exceptions.AuthenticationError", "Token verification failed"),
        ("jwt.exceptions.InvalidTokenError", "Invalid authentication token"),
    ],
    "get_current_user": [
        ("maverick_mcp.exceptions.AuthenticationError", "User authentication failed"),
        ("maverick_mcp.exceptions.DataNotFoundError", "User not found"),
    ],
    "check_credits": [
        ("maverick_mcp.exceptions.InsufficientCreditsError", "Insufficient credits"),
    ],
    # External API calls
    "requests.": [
        ("requests.exceptions.ConnectionError", "External API connection failed"),
        ("requests.exceptions.Timeout", "External API request timeout"),
        ("maverick_mcp.exceptions.APIRateLimitError", "API rate limit exceeded"),
    ],
    "stock_provider": [
        ("maverick_mcp.exceptions.DataProviderError", "Stock data provider error"),
        ("maverick_mcp.exceptions.DataNotFoundError", "Stock data not found"),
    ],
    "stripe": [
        ("stripe.error.StripeError", "Payment processing error"),
        ("maverick_mcp.validation.errors.PaymentRequiredError", "Payment required"),
    ],
    # Validation
    "pydantic": [
        ("pydantic.ValidationError", "Request validation failed"),
        ("maverick_mcp.exceptions.ValidationError", "Invalid input data"),
    ],
    "validate_": [
        ("maverick_mcp.exceptions.ValidationError", "Validation failed"),
        ("ValueError", "Invalid value provided"),
    ],
    # File operations
    "open(": [
        ("FileNotFoundError", "File not found"),
        ("PermissionError", "File access denied"),
        ("IOError", "File operation failed"),
    ],
    # JSON operations
    "json.": [
        ("json.JSONDecodeError", "Invalid JSON format"),
        ("ValueError", "JSON parsing failed"),
    ],
}

# Template for specific exception handling
EXCEPTION_TEMPLATE = """
    except {exception_type} as e:
        logger.error(
            "{error_message}: {{error}}",
            extra={{
                "error": str(e),
                "error_type": "{exception_type}",
                {extra_fields}
            }}
        )
        {error_response}
"""

ERROR_RESPONSE_TEMPLATES = {
    "AuthenticationError": "raise HTTPException(status_code=401, detail=str(e))",
    "AuthorizationError": "raise HTTPException(status_code=403, detail=str(e))",
    "ValidationError": "raise HTTPException(status_code=422, detail=str(e))",
    "DataNotFoundError": "raise HTTPException(status_code=404, detail=str(e))",
    "InsufficientCreditsError": "raise HTTPException(status_code=402, detail=str(e))",
    "IntegrityError": 'raise HTTPException(status_code=409, detail="Data conflict")',
    "OperationalError": 'raise HTTPException(status_code=503, detail="Service temporarily unavailable")',
    "Default": 'raise HTTPException(status_code=500, detail="Internal server error")',
}


def analyze_try_block(content: str, start_line: int, end_line: int) -> list[str]:
    """Analyze try block content to determine likely exceptions."""
    block_content = "\n".join(content.split("\n")[start_line:end_line])

    detected_patterns = []
    for pattern, exceptions in EXCEPTION_MAPPINGS.items():
        if pattern in block_content:
            detected_patterns.extend(exceptions)

    # If no patterns detected, use generic set
    if not detected_patterns:
        detected_patterns = [
            ("ValueError", "Invalid value"),
            ("KeyError", "Missing required field"),
            ("Exception", "Unexpected error"),
        ]

    return detected_patterns


def generate_specific_handlers(
    patterns: list[tuple[str, str]], indent: str = "    "
) -> str:
    """Generate specific exception handlers based on patterns."""
    handlers = []

    for exception_type, error_message in patterns[:-1]:  # All but last
        exception_name = exception_type.split(".")[-1]
        error_response = ERROR_RESPONSE_TEMPLATES.get(
            exception_name, ERROR_RESPONSE_TEMPLATES["Default"]
        )

        handler = f"""{indent}except {exception_type} as e:
{indent}    logger.warning(
{indent}        "{error_message}: {{error}}",
{indent}        extra={{
{indent}            "error": str(e),
{indent}            "error_type": "{exception_name}",
{indent}        }}
{indent}    )
{indent}    {error_response}"""
        handlers.append(handler)

    # Add final catch-all for unexpected errors
    handlers.append(f"""{indent}except Exception as e:
{indent}    logger.error(
{indent}        "Unexpected error occurred",
{indent}        exc_info=True,
{indent}        extra={{
{indent}            "error": str(e),
{indent}            "error_type": type(e).__name__,
{indent}        }}
{indent}    )
{indent}    # Send to monitoring
{indent}    from maverick_mcp.api.error_handling import handle_api_error
{indent}    if 'request' in locals():
{indent}        return handle_api_error(request, e)
{indent}    else:
{indent}        raise HTTPException(status_code=500, detail="Internal server error")""")

    return "\n".join(handlers)


def process_file(filepath: Path) -> tuple[bool, str]:
    """Process a single file to replace generic exception handlers."""
    with open(filepath) as f:
        content = f.read()

    # Find all generic exception handlers
    pattern = r"(\s*)except Exception as e:(.*?)(?=\n\s*(?:except|else|finally|$))"
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        return False, "No generic exception handlers found"

    # Process matches in reverse order to maintain positions
    modified_content = content
    modifications = 0

    for match in reversed(matches):
        indent = match.group(1)
        match.group(2)

        # Find the corresponding try block
        try_pattern = rf"{indent}try:.*?{re.escape(match.group(0))}"
        try_match = re.search(try_pattern, content[: match.end()], re.DOTALL)

        if try_match:
            # Analyze the try block
            patterns = analyze_try_block(content, try_match.start(), match.start())

            # Generate specific handlers
            new_handlers = generate_specific_handlers(patterns, indent)

            # Replace the generic handler
            modified_content = (
                modified_content[: match.start()]
                + new_handlers
                + modified_content[match.end() :]
            )
            modifications += 1

    # Add necessary imports at the top
    imports_to_add = [
        "from fastapi import HTTPException",
        "from maverick_mcp.utils.logging import get_logger",
    ]

    # Find existing imports
    import_section_end = 0
    for line in modified_content.split("\n"):
        if line.strip() and not line.startswith(("import ", "from ", "#", '"""')):
            break
        import_section_end += len(line) + 1

    # Add imports if not present
    for imp in imports_to_add:
        if imp not in modified_content:
            modified_content = (
                modified_content[:import_section_end]
                + imp
                + "\n"
                + modified_content[import_section_end:]
            )

    # Add logger initialization if not present
    if "logger = get_logger" not in modified_content:
        # Find the end of imports
        lines = modified_content.split("\n")
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(("import ", "from ", "#")):
                lines.insert(i, "\nlogger = get_logger(__name__)\n")
                break
        modified_content = "\n".join(lines)

    # Write back
    with open(filepath, "w") as f:
        f.write(modified_content)

    return True, f"Modified {modifications} exception handlers"


def main():
    """Main function to process all router files."""
    print("Starting migration of generic exception handlers...")

    router_files = list(ROUTER_DIR.glob("*.py"))
    router_files = [f for f in router_files if f.name != "__init__.py"]

    print(f"Found {len(router_files)} router files to process")

    results = []
    for filepath in router_files:
        print(f"\nProcessing {filepath.name}...")
        success, message = process_file(filepath)
        results.append((filepath.name, success, message))
        print(f"  {message}")

    # Summary
    print("\n" + "=" * 50)
    print("Migration Summary:")
    print("=" * 50)

    successful = sum(1 for _, success, _ in results if success)
    print(f"Successfully processed: {successful}/{len(results)} files")

    print("\nDetails:")
    for filename, success, message in results:
        status = "✓" if success else "✗"
        print(f"  {status} {filename}: {message}")

    print("\nNext steps:")
    print("1. Review the modified files for correctness")
    print("2. Run tests to ensure functionality is preserved")
    print("3. Update any custom exception handling logic as needed")


if __name__ == "__main__":
    main()
