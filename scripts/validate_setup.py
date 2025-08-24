#!/usr/bin/env python3
"""
Validate setup for the Tiingo data loader.

This script checks that all required dependencies are installed
and provides setup instructions if anything is missing.
"""

import os
import sys
from pathlib import Path


def check_dependency(module_name, package_name=None, description=""):
    """Check if a dependency is available."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✅ {module_name}: Available")
        return True
    except ImportError:
        print(f"❌ {module_name}: Missing - {description}")
        print(f"   Install with: pip install {package_name}")
        return False


def check_environment():
    """Check Python environment and dependencies."""
    print("🐍 Python Environment Check")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()

    # Core dependencies
    print("📦 Core Dependencies")
    print("-" * 20)

    deps_ok = True

    # Required for async operations
    deps_ok &= check_dependency(
        "aiohttp", "aiohttp>=3.8.0", "HTTP client for async operations"
    )

    # Data processing
    deps_ok &= check_dependency(
        "pandas", "pandas>=2.0.0", "Data manipulation and analysis"
    )
    deps_ok &= check_dependency("numpy", "numpy>=1.24.0", "Numerical computing")

    # Technical indicators
    deps_ok &= check_dependency(
        "pandas_ta", "pandas-ta>=0.3.14b0", "Technical analysis indicators"
    )

    # Database
    deps_ok &= check_dependency(
        "sqlalchemy", "sqlalchemy>=2.0.0", "SQL toolkit and ORM"
    )
    deps_ok &= check_dependency(
        "psycopg2", "psycopg2-binary>=2.9.0", "PostgreSQL adapter"
    )

    print()

    # Optional dependencies
    print("🔧 Optional Dependencies")
    print("-" * 25)

    optional_deps = [
        ("requests", "requests>=2.28.0", "HTTP library for fallback operations"),
        ("pytest", "pytest>=7.0.0", "Testing framework"),
        ("asynctest", "asynctest>=0.13.0", "Async testing utilities"),
    ]

    for module, package, desc in optional_deps:
        check_dependency(module, package, desc)

    print()

    return deps_ok


def check_api_token():
    """Check if Tiingo API token is configured."""
    print("🔑 API Configuration")
    print("-" * 20)

    token = os.getenv("TIINGO_API_TOKEN")
    if token:
        print(f"✅ TIINGO_API_TOKEN: Set (length: {len(token)})")
        return True
    else:
        print("❌ TIINGO_API_TOKEN: Not set")
        print("   Get your free API token at: https://www.tiingo.com")
        print("   Set with: export TIINGO_API_TOKEN=your_token_here")
        return False


def check_database():
    """Check database connection."""
    print("\n🗄️  Database Configuration")
    print("-" * 26)

    db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if db_url:
        # Mask password in URL for display
        masked_url = db_url
        if "@" in db_url and "://" in db_url:
            parts = db_url.split("://", 1)
            if len(parts) == 2 and "@" in parts[1]:
                user_pass, host_db = parts[1].split("@", 1)
                if ":" in user_pass:
                    user, _ = user_pass.split(":", 1)
                    masked_url = f"{parts[0]}://{user}:****@{host_db}"

        print("✅ DATABASE_URL: Set")
        print(f"   URL: {masked_url}")

        # Try to connect if SQLAlchemy is available
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT 1"))
                result.fetchone()
            print("✅ Database connection: Success")
            return True
        except ImportError:
            print("⚠️  Database connection: Cannot test (SQLAlchemy not installed)")
            return True
        except Exception as e:
            print(f"❌ Database connection: Failed - {e}")
            return False
    else:
        print("❌ DATABASE_URL: Not set")
        print(
            "   Set with: export DATABASE_URL=postgresql://user:pass@localhost/maverick_mcp"
        )
        return False


def check_project_structure():
    """Check that we're in the right directory structure."""
    print("\n📁 Project Structure")
    print("-" * 20)

    current_dir = Path.cwd()
    script_dir = Path(__file__).parent

    print(f"Current directory: {current_dir}")
    print(f"Script directory: {script_dir}")

    # Check for expected files
    expected_files = [
        "load_tiingo_data.py",
        "tiingo_config.py",
        "load_example.py",
        "requirements_tiingo.txt",
    ]

    all_present = True
    for file in expected_files:
        file_path = script_dir / file
        if file_path.exists():
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")
            all_present = False

    # Check for parent project structure
    parent_files = [
        "../maverick_mcp/__init__.py",
        "../maverick_mcp/data/models.py",
        "../maverick_mcp/core/technical_analysis.py",
    ]

    print("\nParent project files:")
    for file in parent_files:
        file_path = script_dir / file
        if file_path.exists():
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")
            all_present = False

    return all_present


def provide_setup_instructions():
    """Provide setup instructions."""
    print("\n🚀 Setup Instructions")
    print("=" * 21)

    print("1. Install Python dependencies:")
    print("   pip install -r scripts/requirements_tiingo.txt")
    print()

    print("2. Get Tiingo API token:")
    print("   - Sign up at https://www.tiingo.com")
    print("   - Get your free API token from the dashboard")
    print("   - export TIINGO_API_TOKEN=your_token_here")
    print()

    print("3. Configure database:")
    print("   - Ensure PostgreSQL is running")
    print("   - Create maverick_mcp database")
    print("   - export DATABASE_URL=postgresql://user:pass@localhost/maverick_mcp")
    print()

    print("4. Test the setup:")
    print("   python3 scripts/validate_setup.py")
    print()

    print("5. Run a sample load:")
    print("   python3 scripts/load_tiingo_data.py --symbols AAPL,MSFT --years 1")


def main():
    """Main validation function."""
    print("Tiingo Data Loader Setup Validation")
    print("=" * 38)

    # Check all components
    deps_ok = check_environment()
    api_ok = check_api_token()
    db_ok = check_database()
    structure_ok = check_project_structure()

    print("\n" + "=" * 40)

    if deps_ok and api_ok and db_ok and structure_ok:
        print("🎉 Setup validation PASSED!")
        print("You can now use the Tiingo data loader.")
        print()
        print("Quick start:")
        print("  python3 scripts/load_tiingo_data.py --help")
        print("  python3 scripts/load_example.py")
        return 0
    else:
        print("❌ Setup validation FAILED!")
        print("Please fix the issues above before proceeding.")
        print()
        provide_setup_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
