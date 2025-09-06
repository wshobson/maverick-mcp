"""
Postman Collection Export Utility

Converts OpenAPI specifications to Postman collection format.
"""

from typing import Any


def convert_to_postman(openapi_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert OpenAPI specification to Postman collection format.

    Args:
        openapi_dict: OpenAPI specification dictionary

    Returns:
        Postman collection dictionary
    """
    info = openapi_dict.get("info", {})

    collection = {
        "info": {
            "name": info.get("title", "API Collection"),
            "description": info.get("description", "Exported from OpenAPI spec"),
            "version": info.get("version", "1.0.0"),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "item": [],
        "variable": [],
    }

    # Add server variables
    servers = openapi_dict.get("servers", [])
    if servers:
        collection["variable"].append(
            {
                "key": "baseUrl",
                "value": servers[0].get("url", "http://localhost:8000"),
                "type": "string",
            }
        )

    # Convert paths to Postman requests
    paths = openapi_dict.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                item = {
                    "name": operation.get("summary", f"{method.upper()} {path}"),
                    "request": {
                        "method": method.upper(),
                        "header": [],
                        "url": {
                            "raw": "{{baseUrl}}" + path,
                            "host": ["{{baseUrl}}"],
                            "path": path.split("/")[1:]
                            if path.startswith("/")
                            else path.split("/"),
                        },
                    },
                    "response": [],
                }

                # Add request body if present
                if "requestBody" in operation:
                    content = operation["requestBody"].get("content", {})
                    if "application/json" in content:
                        item["request"]["header"].append(
                            {
                                "key": "Content-Type",
                                "value": "application/json",
                                "type": "text",
                            }
                        )

                        # Add example body if available
                        schema = content["application/json"].get("schema", {})
                        if "example" in schema:
                            item["request"]["body"] = {
                                "mode": "raw",
                                "raw": str(schema["example"]),
                            }

                collection["item"].append(item)

    return collection
