#!/usr/bin/env python3
"""
Test runner for ExaSearch integration tests.

This script provides convenient commands to run different categories of ExaSearch
integration tests with proper filtering and reporting.

Usage:
    python scripts/run_exa_tests.py --unit         # Unit tests only
    python scripts/run_exa_tests.py --integration  # Integration tests
    python scripts/run_exa_tests.py --all          # All tests
    python scripts/run_exa_tests.py --quick        # Quick test suite
    python scripts/run_exa_tests.py --benchmark    # Run benchmarks
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ExaTestRunner:
    """Test runner for ExaSearch integration tests."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_file = self.project_root / "tests" / "test_exa_research_integration.py"
        self.benchmark_script = self.project_root / "scripts" / "benchmark_exa_research.py"
    
    def run_command(self, cmd: list[str]) -> int:
        """Run command and return exit code."""
        print(f"🚀 Running: {' '.join(cmd)}")
        print("="*60)
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except KeyboardInterrupt:
            print("\n⚠️ Test run interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return 1
    
    def run_unit_tests(self, verbose: bool = True) -> int:
        """Run unit tests only."""
        print("🧪 Running ExaSearch Unit Tests")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_file),
            "-m", "unit",
            "--tb=short",
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self.run_command(cmd)
    
    def run_integration_tests(self, verbose: bool = True) -> int:
        """Run integration tests."""
        print("🔗 Running ExaSearch Integration Tests")
        print("⚠️  Note: These tests require valid EXA_API_KEY environment variable")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_file),
            "-m", "integration",
            "--tb=short",
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self.run_command(cmd)
    
    def run_all_tests(self, verbose: bool = True) -> int:
        """Run all ExaSearch tests."""
        print("🎯 Running All ExaSearch Tests")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_file),
            "--tb=short",
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self.run_command(cmd)
    
    def run_quick_tests(self) -> int:
        """Run quick test suite (unit tests only, minimal output)."""
        print("⚡ Running Quick ExaSearch Test Suite")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_file),
            "-m", "unit and not slow",
            "-q",
            "--tb=line",
        ]
        
        return self.run_command(cmd)
    
    def run_slow_tests(self, verbose: bool = True) -> int:
        """Run slow/performance tests."""
        print("🐌 Running Slow ExaSearch Tests")
        print("⚠️  Note: These tests may take several minutes to complete")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_file),
            "-m", "slow",
            "--tb=short",
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self.run_command(cmd)
    
    def run_with_coverage(self) -> int:
        """Run tests with coverage reporting."""
        print("📊 Running ExaSearch Tests with Coverage")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_file),
            "-m", "unit",
            "--cov=maverick_mcp.agents.deep_research",
            "--cov=maverick_mcp.utils.parallel_research",
            "--cov=maverick_mcp.api.routers.research",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--tb=short",
            "-v",
        ]
        
        return self.run_command(cmd)
    
    def run_benchmark(self, quick: bool = False, **kwargs) -> int:
        """Run performance benchmarks."""
        print("🏁 Running ExaSearch Performance Benchmarks")
        
        if not self.benchmark_script.exists():
            print(f"❌ Benchmark script not found: {self.benchmark_script}")
            return 1
        
        cmd = ["python", str(self.benchmark_script)]
        
        if quick:
            cmd.append("--quick")
        
        # Add additional benchmark arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return self.run_command(cmd)
    
    def validate_environment(self) -> bool:
        """Validate test environment setup."""
        print("🔍 Validating Test Environment")
        
        # Check if test file exists
        if not self.test_file.exists():
            print(f"❌ Test file not found: {self.test_file}")
            return False
        
        # Check for required dependencies
        try:
            import pytest
            import maverick_mcp
            print("✅ Required dependencies available")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False
        
        # Check for optional dependencies
        optional_deps = {
            "exa_py": "ExaSearch integration",
            "psutil": "Performance monitoring",
        }
        
        for dep, description in optional_deps.items():
            try:
                __import__(dep)
                print(f"✅ Optional dependency available: {dep} ({description})")
            except ImportError:
                print(f"⚠️  Optional dependency missing: {dep} ({description})")
        
        # Check environment variables
        import os
        exa_key = os.getenv("EXA_API_KEY")
        if exa_key:
            print("✅ EXA_API_KEY environment variable configured")
        else:
            print("⚠️  EXA_API_KEY not configured - integration tests will be limited")
        
        return True
    
    def show_test_info(self):
        """Show information about available tests."""
        print("📋 ExaSearch Integration Test Suite Information")
        print("="*60)
        
        test_categories = {
            "Unit Tests": [
                "ExaSearchProvider initialization and configuration",
                "Timeout calculation and failure handling", 
                "DeepResearchAgent setup with ExaSearch",
                "Specialized subagent routing and execution",
                "Task distribution and parallel orchestration",
                "Content analysis and result processing",
            ],
            "Integration Tests": [
                "End-to-end research workflow with ExaSearch",
                "Multi-persona research consistency",
                "Parallel vs sequential execution comparison",
                "MCP tool endpoint validation",
                "Real ExaSearch API integration",
            ],
            "Performance Tests": [
                "Research depth performance comparison",
                "Parallel execution efficiency metrics", 
                "Timeout resilience validation",
                "Memory usage monitoring",
                "Concurrent request handling",
            ],
            "Benchmarks": [
                "Comprehensive performance analysis",
                "Cross-depth timing comparison", 
                "Focus area specialization metrics",
                "Parallel efficiency measurement",
                "Error rate and reliability stats",
            ]
        }
        
        for category, tests in test_categories.items():
            print(f"\n🏷️  {category}:")
            for test in tests:
                print(f"   • {test}")
        
        print(f"\n📁 Test Files:")
        print(f"   • Main Test Suite: {self.test_file}")
        print(f"   • Benchmark Script: {self.benchmark_script}")
        
        print(f"\n⚙️  Environment Requirements:")
        print(f"   • Python 3.12+")
        print(f"   • pytest and project dependencies")
        print(f"   • EXA_API_KEY environment variable (for integration tests)")
        print(f"   • Redis (optional, for caching tests)")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Run ExaSearch integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_exa_tests.py --unit              # Unit tests only
  python scripts/run_exa_tests.py --integration       # Integration tests
  python scripts/run_exa_tests.py --all --verbose     # All tests with verbose output
  python scripts/run_exa_tests.py --quick             # Quick test suite
  python scripts/run_exa_tests.py --coverage          # Tests with coverage report
  python scripts/run_exa_tests.py --benchmark         # Performance benchmarks
  python scripts/run_exa_tests.py --benchmark --quick # Quick benchmarks
  python scripts/run_exa_tests.py --info              # Show test information
        """
    )
    
    # Test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--quick", action="store_true", help="Run quick test suite")
    test_group.add_argument("--slow", action="store_true", help="Run slow/performance tests")
    test_group.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    test_group.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    test_group.add_argument("--info", action="store_true", help="Show test suite information")
    test_group.add_argument("--validate", action="store_true", help="Validate test environment")
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    
    # Benchmark options
    parser.add_argument("--depth", choices=["basic", "standard", "comprehensive", "exhaustive"], 
                       help="Benchmark specific research depth")
    parser.add_argument("--focus", choices=["fundamentals", "technicals", "sentiment", "competitive"],
                       help="Benchmark specific focus area")
    
    args = parser.parse_args()
    
    # Default to unit tests if no option specified
    if not any([args.unit, args.integration, args.all, args.quick, args.slow, 
               args.coverage, args.benchmark, args.info, args.validate]):
        args.unit = True
    
    runner = ExaTestRunner()
    
    # Handle info and validation commands
    if args.info:
        runner.show_test_info()
        return 0
    
    if args.validate:
        success = runner.validate_environment()
        return 0 if success else 1
    
    # Validate environment before running tests
    if not runner.validate_environment():
        print("❌ Environment validation failed")
        return 1
    
    verbose = args.verbose and not args.quiet
    exit_code = 0
    
    # Run selected test category
    if args.unit:
        exit_code = runner.run_unit_tests(verbose=verbose)
    elif args.integration:
        exit_code = runner.run_integration_tests(verbose=verbose)
    elif args.all:
        exit_code = runner.run_all_tests(verbose=verbose)
    elif args.quick:
        exit_code = runner.run_quick_tests()
    elif args.slow:
        exit_code = runner.run_slow_tests(verbose=verbose)
    elif args.coverage:
        exit_code = runner.run_with_coverage()
    elif args.benchmark:
        benchmark_kwargs = {}
        if args.depth:
            benchmark_kwargs['depth'] = args.depth
        if args.focus:
            benchmark_kwargs['focus'] = args.focus
        
        exit_code = runner.run_benchmark(quick=args.quick, **benchmark_kwargs)
    
    # Print final status
    if exit_code == 0:
        print("\n✅ All tests completed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())