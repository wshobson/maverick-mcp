"""
Insomnia Collection Export Utility

Converts OpenAPI specifications to Insomnia workspace format.
"""

import uuid
from typing import Any


def convert_to_insomnia(openapi_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert OpenAPI specification to Insomnia workspace format.

    Args:
        openapi_dict: OpenAPI specification dictionary

    Returns:
        Insomnia workspace dictionary
    """
    info = openapi_dict.get("info", {})

    workspace = {
        "_type": "export",
        "__export_format": 4,
        "__export_date": "2024-01-01T00:00:00.000Z",
        "__export_source": "maverick-mcp:openapi",
        "resources": [],
    }

    # Create workspace resource
    workspace_id = f"wrk_{uuid.uuid4().hex[:12]}"
    workspace["resources"].append(
        {
            "_id": workspace_id,
            "_type": "workspace",
            "name": info.get("title", "API Workspace"),
            "description": info.get("description", "Exported from OpenAPI spec"),
            "scope": "collection",
        }
    )

    # Create environment for base URL
    env_id = f"env_{uuid.uuid4().hex[:12]}"
    servers = openapi_dict.get("servers", [])
    base_url = (
        servers[0].get("url", "http://localhost:8000")
        if servers
        else "http://localhost:8000"
    )

    workspace["resources"].append(
        {
            "_id": env_id,
            "_type": "environment",
            "name": "Base Environment",
            "data": {"base_url": base_url},
            "dataPropertyOrder": {"&": ["base_url"]},
            "color": "#7d69cb",
            "isPrivate": False,
            "metaSortKey": 1,
            "parentId": workspace_id,
        }
    )

    # Convert paths to Insomnia requests
    paths = openapi_dict.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                request_id = f"req_{uuid.uuid4().hex[:12]}"

                request = {
                    "_id": request_id,
                    "_type": "request",
                    "parentId": workspace_id,
                    "name": operation.get("summary", f"{method.upper()} {path}"),
                    "description": operation.get("description", ""),
                    "url": "{{ _.base_url }}" + path,
                    "method": method.upper(),
                    "headers": [],
                    "parameters": [],
                    "body": {},
                    "authentication": {},
                }

                # Add request body if present
                if "requestBody" in operation:
                    content = operation["requestBody"].get("content", {})
                    if "application/json" in content:
                        request["headers"].append(
                            {"name": "Content-Type", "value": "application/json"}
                        )

                        request["body"] = {"mimeType": "application/json", "text": "{}"}

                        # Add example if available
                        schema = content["application/json"].get("schema", {})
                        if "example" in schema:
                            request["body"]["text"] = str(schema["example"])

                # Add query parameters if present
                if "parameters" in operation:
                    for param in operation["parameters"]:
                        if param.get("in") == "query":
                            request["parameters"].append(
                                {
                                    "name": param["name"],
                                    "value": "",
                                    "description": param.get("description", ""),
                                    "disabled": not param.get("required", False),
                                }
                            )

                workspace["resources"].append(request)

    return workspace
