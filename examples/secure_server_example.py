"""
Example of how to use the new SecurityConfig to create secure servers.

This example demonstrates how to replace the dangerous CORS configuration
in server_multi.py with a secure, validated configuration.
"""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from maverick_mcp.config.security_utils import (
    check_security_config,
    create_secure_starlette_middleware,
    get_safe_cors_config,
    log_security_status,
)


def create_secure_multi_transport_app():
    """
    Example of how to create a secure multi-transport app.

    This replaces the dangerous CORS configuration in server_multi.py
    with a secure, validated configuration.
    """

    # Check security configuration first
    if not check_security_config():
        raise ValueError(
            "Security configuration validation failed! Check logs for details."
        )

    # Get secure middleware
    secure_middleware = create_secure_starlette_middleware()

    # Create routes (example)
    async def health_check(request):
        return JSONResponse({"status": "ok", "secure": True})

    routes = [Route("/health", endpoint=health_check)]

    # Create the Starlette app with secure middleware
    app = Starlette(
        routes=routes,
        middleware=secure_middleware,  # This replaces the dangerous wildcard CORS
    )

    return app


def demonstrate_cors_configurations():
    """Demonstrate different CORS configurations and their security implications."""

    print("=== CORS Configuration Examples ===\n")

    # DANGEROUS - What was in server_multi.py
    print("❌ DANGEROUS configuration (from server_multi.py):")
    dangerous_config = {
        "allow_origins": ["*"],
        "allow_credentials": True,  # This combination is dangerous!
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": ["*"],
    }
    print(f"  {dangerous_config}")
    print("  ⚠️  This allows ANY website to make authenticated requests!")
    print("  🔴 Security vulnerability: Credential theft, CSRF attacks\n")

    # SAFE - Using SecurityConfig
    print("✅ SAFE configuration (using SecurityConfig):")
    safe_config = get_safe_cors_config()
    print(f"  {safe_config}")
    print("  ✓ Specific origins only")
    print("  ✓ Validated configuration")
    print("  ✓ Environment-appropriate settings\n")

    # Log current security status
    log_security_status()


if __name__ == "__main__":
    print("Maverick MCP Security Configuration Example\n")

    # Demonstrate CORS configurations
    demonstrate_cors_configurations()

    # Create a secure app
    try:
        app = create_secure_multi_transport_app()
        print("✅ Secure app created successfully!")
        print("   - CORS configuration validated")
        print("   - Security headers applied")
        print("   - Environment-appropriate settings")
    except ValueError as e:
        print(f"❌ Failed to create secure app: {e}")
        print("   Check your environment variables and security configuration.")
