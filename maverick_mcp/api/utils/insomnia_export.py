"""
Insomnia collection export utility.

Converts OpenAPI specification to Insomnia v4 format.
"""

import json
import time
import uuid
from typing import Any


def convert_to_insomnia(openapi_spec: dict[str, Any]) -> dict[str, Any]:
    """
    Convert OpenAPI specification to Insomnia collection format.

    Args:
        openapi_spec: OpenAPI specification dictionary

    Returns:
        Insomnia v4 format dictionary
    """
    # Extract base information
    info = openapi_spec.get("info", {})
    servers = openapi_spec.get("servers", [])
    base_url = servers[0]["url"] if servers else "http://localhost:8000"

    # Create workspace ID
    workspace_id = f"wrk_{uuid.uuid4().hex[:8]}"
    base_env_id = f"env_{uuid.uuid4().hex[:8]}"

    # Create Insomnia export structure
    export = {
        "_type": "export",
        "__export_format": 4,
        "__export_date": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "__export_source": "maverickmcp.api",
        "resources": [],
    }

    # Add workspace
    export["resources"].append(
        {
            "_id": workspace_id,
            "_type": "workspace",
            "created": int(time.time() * 1000),
            "description": info.get("description", ""),
            "modified": int(time.time() * 1000),
            "name": info.get("title", "API Workspace"),
            "parentId": None,
            "scope": "collection",
        }
    )

    # Add base environment
    export["resources"].append(
        {
            "_id": base_env_id,
            "_type": "environment",
            "color": None,
            "created": int(time.time() * 1000),
            "data": {
                "base_url": base_url,
                "access_token": "",
                "csrf_token": "",
                "user_email": "test@example.com",
                "user_password": "your_password",
            },
            "dataPropertyOrder": {
                "&": [
                    "base_url",
                    "access_token",
                    "csrf_token",
                    "user_email",
                    "user_password",
                ]
            },
            "isPrivate": False,
            "metaSortKey": 1,
            "modified": int(time.time() * 1000),
            "name": "Base Environment",
            "parentId": workspace_id,
        }
    )

    # Add API spec
    api_spec_id = f"spc_{uuid.uuid4().hex[:8]}"
    export["resources"].append(
        {
            "_id": api_spec_id,
            "_type": "api_spec",
            "contents": json.dumps(openapi_spec),
            "contentType": "json",
            "created": int(time.time() * 1000),
            "fileName": "openapi.json",
            "modified": int(time.time() * 1000),
            "parentId": workspace_id,
        }
    )

    # Create request groups by tags
    tag_groups: dict[str, str] = {}
    paths = openapi_spec.get("paths", {})
    components = openapi_spec.get("components", {})

    # Add authentication folder
    auth_folder_id = f"fld_{uuid.uuid4().hex[:8]}"
    export["resources"].append(
        {
            "_id": auth_folder_id,
            "_type": "request_group",
            "created": int(time.time() * 1000),
            "description": "Authentication endpoints - run login first to get tokens",
            "environment": {},
            "environmentPropertyOrder": None,
            "metaSortKey": -1,
            "modified": int(time.time() * 1000),
            "name": "Authentication Setup",
            "parentId": workspace_id,
        }
    )

    # Add login request
    login_request_id = f"req_{uuid.uuid4().hex[:8]}"
    export["resources"].append(
        {
            "_id": login_request_id,
            "_type": "request",
            "authentication": {},
            "body": {
                "mimeType": "application/json",
                "text": json.dumps(
                    {
                        "email": "{{ _.user_email }}",
                        "password": "{{ _.user_password }}",
                        "device_name": "Insomnia Client",
                    },
                    indent=2,
                ),
            },
            "created": int(time.time() * 1000),
            "description": "Login to get access and refresh tokens",
            "headers": [{"name": "Content-Type", "value": "application/json"}],
            "isPrivate": False,
            "metaSortKey": -1,
            "method": "POST",
            "modified": int(time.time() * 1000),
            "name": "Login (Run First)",
            "parameters": [],
            "parentId": auth_folder_id,
            "settingDisableRenderRequestBody": False,
            "settingEncodeUrl": True,
            "settingRebuildPath": True,
            "settingSendCookies": True,
            "settingStoreCookies": True,
            "url": "{{ _.base_url }}/auth/login",
        }
    )

    # Process API paths
    sort_key = 1000
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                # Get operation details
                operation_id = operation.get("operationId", f"{method}_{path}")
                summary = operation.get("summary", operation_id)
                description = operation.get("description", "")
                tags = operation.get("tags", ["Default"])
                tag = tags[0] if tags else "Default"

                # Get or create tag folder
                if tag not in tag_groups:
                    folder_id = f"fld_{uuid.uuid4().hex[:8]}"
                    tag_groups[tag] = folder_id

                    export["resources"].append(
                        {
                            "_id": folder_id,
                            "_type": "request_group",
                            "created": int(time.time() * 1000),
                            "description": get_tag_description(openapi_spec, tag),
                            "environment": {},
                            "environmentPropertyOrder": None,
                            "metaSortKey": sort_key,
                            "modified": int(time.time() * 1000),
                            "name": tag,
                            "parentId": workspace_id,
                        }
                    )
                    sort_key += 1000

                # Create request
                request_id = f"req_{uuid.uuid4().hex[:8]}"
                request = {
                    "_id": request_id,
                    "_type": "request",
                    "authentication": {
                        "disabled": False,
                        "token": "{{ _.access_token }}",
                        "type": "bearer",
                    },
                    "body": {},
                    "created": int(time.time() * 1000),
                    "description": description,
                    "headers": [{"name": "Accept", "value": "application/json"}],
                    "isPrivate": False,
                    "metaSortKey": sort_key,
                    "method": method.upper(),
                    "modified": int(time.time() * 1000),
                    "name": summary,
                    "parameters": [],
                    "parentId": tag_groups[tag],
                    "settingDisableRenderRequestBody": False,
                    "settingEncodeUrl": True,
                    "settingRebuildPath": True,
                    "settingSendCookies": True,
                    "settingStoreCookies": True,
                    "url": f"{{{{ _.base_url }}}}{path}",
                }

                # Add CSRF token for state-changing operations
                if method in ["post", "put", "patch", "delete"]:
                    request["headers"].append(
                        {"name": "X-CSRF-Token", "value": "{{ _.csrf_token }}"}
                    )

                # Handle parameters
                parameters = operation.get("parameters", [])
                for param in parameters:
                    if param.get("in") == "query":
                        request["parameters"].append(
                            {
                                "disabled": not param.get("required", False),
                                "id": f"pair_{uuid.uuid4().hex[:8]}",
                                "name": param["name"],
                                "value": "",
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
                            request["body"] = {
                                "mimeType": "application/json",
                                "text": json.dumps(example, indent=2),
                            }
                            request["headers"].append(
                                {"name": "Content-Type", "value": "application/json"}
                            )
                        elif "$ref" in schema:
                            # Generate example from schema reference
                            ref_name = schema["$ref"].split("/")[-1]
                            schema_def = components.get("schemas", {}).get(ref_name, {})
                            example_body = generate_example_from_schema(schema_def)
                            request["body"] = {
                                "mimeType": "application/json",
                                "text": json.dumps(example_body, indent=2),
                            }
                            request["headers"].append(
                                {"name": "Content-Type", "value": "application/json"}
                            )

                export["resources"].append(request)
                sort_key += 100

    return export


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
