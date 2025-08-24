"""
Example showing how to fix the CORS vulnerability in server_multi.py.

This demonstrates the before/after of fixing the dangerous wildcard CORS
configuration with credentials.
"""

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from maverick_mcp.config.security_utils import create_secure_starlette_middleware


def dangerous_server_multi_app():
    """
    DANGEROUS - This is what was in server_multi.py

    ❌ DO NOT USE THIS CONFIGURATION ❌
    """

    # Example route
    async def health(request):
        return JSONResponse({"status": "ok", "secure": False})

    routes = [Route("/health", endpoint=health)]

    # ❌ DANGEROUS CONFIGURATION - What was in server_multi.py
    dangerous_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],  # ❌ Wildcard origin
            allow_credentials=True,  # ❌ With credentials = VULNERABILITY
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
    ]

    app = Starlette(routes=routes, middleware=dangerous_middleware)

    return app


def secure_server_multi_app():
    """
    SECURE - Using the new SecurityConfig

    ✅ This is the proper, secure way to configure CORS
    """

    # Example route
    async def health(request):
        return JSONResponse({"status": "ok", "secure": True})

    routes = [Route("/health", endpoint=health)]

    # ✅ SECURE CONFIGURATION - Using SecurityConfig
    secure_middleware = create_secure_starlette_middleware()

    app = Starlette(routes=routes, middleware=secure_middleware)

    return app


def compare_configurations():
    """Compare the dangerous vs secure configurations."""

    print("=== server_multi.py CORS Fix Comparison ===\n")

    print("❌ DANGEROUS (Original server_multi.py):")
    print("```python")
    print("Middleware(")
    print("    CORSMiddleware,")
    print("    allow_origins=['*'],          # ❌ Any website")
    print("    allow_credentials=True,       # ❌ Can steal credentials")
    print("    allow_methods=['*'],")
    print("    allow_headers=['*'],")
    print("    expose_headers=['*'],")
    print(")")
    print("```")
    print("🔴 VULNERABILITY: Any website can make authenticated requests!")
    print("🔴 ATTACK VECTORS: Credential theft, CSRF, session hijacking\n")

    print("✅ SECURE (Using SecurityConfig):")
    print("```python")
    print(
        "from maverick_mcp.config.security_utils import create_secure_starlette_middleware"
    )
    print("")
    print("secure_middleware = create_secure_starlette_middleware()")
    print("app = Starlette(routes=routes, middleware=secure_middleware)")
    print("```")
    print("✓ Environment-specific origins only")
    print("✓ Validated configuration")
    print("✓ Security headers included")
    print("✓ Rate limiting configured")
    print("✓ Trusted hosts validation\n")

    print("📋 MIGRATION STEPS:")
    print("1. Replace the dangerous CORSMiddleware configuration")
    print("2. Import create_secure_starlette_middleware")
    print("3. Use the secure middleware list")
    print("4. Test CORS with your frontend")
    print("5. Deploy with confidence!\n")


if __name__ == "__main__":
    compare_configurations()

    # Show that we can create both (but we shouldn't use the dangerous one!)
    print("Creating apps for demonstration...")

    try:
        dangerous_app = dangerous_server_multi_app()
        print("❌ Dangerous app created (for demonstration only)")
    except Exception as e:
        print(f"❌ Could not create dangerous app: {e}")

    try:
        secure_app = secure_server_multi_app()
        print("✅ Secure app created successfully")
    except Exception as e:
        print(f"❌ Could not create secure app: {e}")

    print("\n🎯 RECOMMENDATION: Use only the secure configuration in production!")
    print("   The dangerous configuration is a serious security vulnerability.")
