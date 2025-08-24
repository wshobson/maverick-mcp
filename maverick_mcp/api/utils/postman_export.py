"""
Postman collection export utility.

Converts OpenAPI specification to Postman Collection v2.1 format.
"""

import json
import uuid
from typing import Any
from urllib.parse import urlparse


def convert_to_postman(openapi_spec: dict[str, Any]) -> dict[str, Any]:
    """
    Convert OpenAPI specification to Postman collection format.

    Args:
        openapi_spec: OpenAPI specification dictionary

    Returns:
        Postman collection v2.1 format dictionary
    """
    # Extract base information
    info = openapi_spec.get("info", {})
    servers = openapi_spec.get("servers", [])
    base_url = servers[0]["url"] if servers else "http://localhost:8000"

    # Parse base URL
    urlparse(base_url)

    # Create Postman collection structure
    collection = {
        "info": {
            "_postman_id": str(uuid.uuid4()),
            "name": info.get("title", "API Collection"),
            "description": info.get("description", ""),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "item": [],
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{access_token}}", "type": "string"}],
        },
        "variable": [
            {"key": "baseUrl", "value": base_url, "type": "string"},
            {"key": "access_token", "value": "", "type": "string"},
            {"key": "csrf_token", "value": "", "type": "string"},
        ],
    }

    # Group endpoints by tags
    tag_folders: dict[str, dict[str, Any]] = {}

    # Process paths
    paths = openapi_spec.get("paths", {})
    components = openapi_spec.get("components", {})

    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                # Get operation details
                operation_id = operation.get("operationId", f"{method}_{path}")
                summary = operation.get("summary", operation_id)
                description = operation.get("description", "")
                tags = operation.get("tags", ["Default"])

                # Create request object
                request = {
                    "name": summary,
                    "request": {
                        "method": method.upper(),
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json",
                                "type": "text",
                            },
                            {
                                "key": "Accept",
                                "value": "application/json",
                                "type": "text",
                            },
                        ],
                        "url": {
                            "raw": f"{{{{baseUrl}}}}{path}",
                            "host": ["{{baseUrl}}"],
                            "path": path.strip("/").split("/"),
                            "variable": [],
                        },
                        "description": description,
                    },
                    "response": [],
                }

                # Add CSRF token for state-changing operations
                if method in ["post", "put", "patch", "delete"]:
                    request["request"]["header"].append(
                        {
                            "key": "X-CSRF-Token",
                            "value": "{{csrf_token}}",
                            "type": "text",
                        }
                    )

                # Handle path parameters
                parameters = operation.get("parameters", [])
                for param in parameters:
                    if param.get("in") == "path":
                        request["request"]["url"]["variable"].append(
                            {
                                "key": param["name"],
                                "value": "",
                                "description": param.get("description", ""),
                            }
                        )
                    elif param.get("in") == "query":
                        if "query" not in request["request"]["url"]:
                            request["request"]["url"]["query"] = []
                        request["request"]["url"]["query"].append(
                            {
                                "key": param["name"],
                                "value": "",
                                "description": param.get("description", ""),
                                "disabled": not param.get("required", False),
                            }
                        )

                # Handle request body
                request_body = operation.get("requestBody", {})
                if request_body:
                    content = request_body.get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        example = content["application/json"].get("example")

                        if example:
                            request["request"]["body"] = {
                                "mode": "raw",
                                "raw": json.dumps(example, indent=2),
                                "options": {"raw": {"language": "json"}},
                            }
                        elif "$ref" in schema:
                            # Generate example from schema reference
                            ref_name = schema["$ref"].split("/")[-1]
                            schema_def = components.get("schemas", {}).get(ref_name, {})
                            example_body = generate_example_from_schema(schema_def)
                            request["request"]["body"] = {
                                "mode": "raw",
                                "raw": json.dumps(example_body, indent=2),
                                "options": {"raw": {"language": "json"}},
                            }

                # Add example responses
                responses = operation.get("responses", {})
                for status_code, response_def in responses.items():
                    if status_code.isdigit():
                        response_content = response_def.get("content", {})
                        if "application/json" in response_content:
                            example = response_content["application/json"].get(
                                "example"
                            )
                            if example:
                                request["response"].append(
                                    {
                                        "name": f"{status_code} - {response_def.get('description', '')}",
                                        "originalRequest": request["request"].copy(),
                                        "status": response_def.get("description", ""),
                                        "code": int(status_code),
                                        "_postman_previewlanguage": "json",
                                        "header": [
                                            {
                                                "key": "Content-Type",
                                                "value": "application/json",
                                            }
                                        ],
                                        "body": json.dumps(example, indent=2),
                                    }
                                )

                # Group by tag
                tag = tags[0] if tags else "Default"
                if tag not in tag_folders:
                    tag_folders[tag] = {
                        "name": tag,
                        "item": [],
                        "description": get_tag_description(openapi_spec, tag),
                    }

                tag_folders[tag]["item"].append(request)

    # Add auth endpoints as a special pre-request script
    auth_folder = {
        "name": "Authentication Setup",
        "item": [
            {
                "name": "Login (Run First)",
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "exec": [
                                "// Extract tokens from response",
                                "if (pm.response.code === 200) {",
                                "    const response = pm.response.json();",
                                "    ",
                                "    // Set CSRF token as variable",
                                "    pm.collectionVariables.set('csrf_token', response.csrf_token);",
                                "    ",
                                "    // Extract access token from cookies",
                                "    const cookies = pm.response.headers.get('Set-Cookie');",
                                "    if (cookies) {",
                                "        const accessTokenMatch = cookies.match(/access_token=([^;]+)/);",
                                "        if (accessTokenMatch) {",
                                "            pm.collectionVariables.set('access_token', accessTokenMatch[1]);",
                                "        }",
                                "    }",
                                "    ",
                                "    console.log('Authentication successful!');",
                                "}",
                            ],
                            "type": "text/javascript",
                        },
                    }
                ],
                "request": {
                    "method": "POST",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": {
                        "mode": "raw",
                        "raw": json.dumps(
                            {
                                "email": "{{user_email}}",
                                "password": "{{user_password}}",
                                "device_name": "Postman Client",
                            },
                            indent=2,
                        ),
                    },
                    "url": {
                        "raw": "{{baseUrl}}/auth/login",
                        "host": ["{{baseUrl}}"],
                        "path": ["auth", "login"],
                    },
                },
            }
        ],
    }

    # Add folders to collection
    collection["item"] = [auth_folder] + list(tag_folders.values())

    # Add additional variables
    collection["variable"].extend(
        [
            {"key": "user_email", "value": "test@example.com", "type": "string"},
            {"key": "user_password", "value": "your_password", "type": "string"},
        ]
    )

    return collection


def generate_example_from_schema(schema: dict[str, Any]) -> Any:
    """Generate example data from OpenAPI schema."""
    if schema.get("type") == "object":
        example = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            if (
                prop_name in required or len(example) < 5
            ):  # Include required + up to 5 fields
                example[prop_name] = generate_example_value(prop_schema)

        return example
    elif schema.get("type") == "array":
        item_schema = schema.get("items", {})
        return [generate_example_value(item_schema)]
    else:
        return generate_example_value(schema)


def generate_example_value(schema: dict[str, Any]) -> Any:
    """Generate example value based on schema type."""
    schema_type = schema.get("type", "string")
    schema_format = schema.get("format", "")

    # Check for example or default
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]

    # Generate based on type/format
    if schema_type == "string":
        if schema_format == "email":
            return "user@example.com"
        elif schema_format == "date-time":
            return "2024-01-15T10:30:00Z"
        elif schema_format == "date":
            return "2024-01-15"
        elif schema_format == "uuid":
            return "550e8400-e29b-41d4-a716-446655440000"
        elif "password" in schema.get("title", "").lower():
            return "********"
        else:
            return "string"
    elif schema_type == "integer":
        return 1
    elif schema_type == "number":
        return 1.0
    elif schema_type == "boolean":
        return True
    elif schema_type == "array":
        item_schema = schema.get("items", {})
        return [generate_example_value(item_schema)]
    elif schema_type == "object":
        return generate_example_from_schema(schema)
    else:
        return None


def get_tag_description(openapi_spec: dict[str, Any], tag_name: str) -> str:
    """Get description for a tag from OpenAPI spec."""
    tags = openapi_spec.get("tags", [])
    for tag in tags:
        if tag.get("name") == tag_name:
            return tag.get("description", "")
    return ""
