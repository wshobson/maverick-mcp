#!/usr/bin/env python3
"""
Security maintenance script for MaverickMCP.

This script provides utilities for security maintenance tasks including:
- Audit log cleanup
- Rate limiting data cleanup
- Security report generation
- Threat analysis
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
# ruff: noqa: E402
from maverick_mcp.api.middleware.per_user_rate_limiting import (
    cleanup_redis_rate_limit_data,
)
from maverick_mcp.auth.audit_logger import audit_logger
from maverick_mcp.auth.audit_reports import (
    AuditReportGenerator,
    get_compliance_report,
    get_security_summary,
    get_threat_analysis,
)
from maverick_mcp.config.settings import get_settings
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def cleanup_audit_logs(retention_days: int = 90):
    """Clean up old audit logs."""
    logger.info(f"Starting audit log cleanup (retention: {retention_days} days)")

    try:
        deleted_count = await audit_logger.cleanup_old_logs()
        logger.info(f"Cleanup completed: {deleted_count} logs deleted")
        return deleted_count
    except Exception as e:
        logger.error(f"Error during audit log cleanup: {e}")
        return 0


async def cleanup_rate_limiting_data(retention_hours: int = 24):
    """Clean up old rate limiting data from Redis."""
    if not settings.auth.redis_enabled:
        logger.warning("Redis not enabled, skipping rate limiting cleanup")
        return

    logger.info(
        f"Starting rate limiting data cleanup (retention: {retention_hours} hours)"
    )

    try:
        await cleanup_redis_rate_limit_data(settings.auth.redis_url, retention_hours)
        logger.info("Rate limiting cleanup completed")
    except Exception as e:
        logger.error(f"Error during rate limiting cleanup: {e}")


def generate_security_report(days: int = 30, output_file: str = None):
    """Generate and optionally save security summary report."""
    logger.info(f"Generating security report for last {days} days")

    try:
        report = get_security_summary(days=days)

        if output_file:
            import json

            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Security report saved to {output_file}")
        else:
            print(json.dumps(report, indent=2, default=str))

        return report
    except Exception as e:
        logger.error(f"Error generating security report: {e}")
        return None


def generate_threat_analysis(days: int = 7, output_file: str = None):
    """Generate and optionally save threat analysis report."""
    logger.info(f"Generating threat analysis for last {days} days")

    try:
        report = get_threat_analysis(days=days)

        if output_file:
            import json

            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Threat analysis saved to {output_file}")
        else:
            print(json.dumps(report, indent=2, default=str))

        return report
    except Exception as e:
        logger.error(f"Error generating threat analysis: {e}")
        return None


def generate_compliance_report(days: int = 90, output_file: str = None):
    """Generate and optionally save compliance report."""
    logger.info(f"Generating compliance report for last {days} days")

    try:
        report = get_compliance_report(days=days)

        if output_file:
            import json

            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Compliance report saved to {output_file}")
        else:
            print(json.dumps(report, indent=2, default=str))

        return report
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        return None


def export_audit_data(days: int = 30, output_file: str = "audit_export.csv"):
    """Export audit data to CSV."""
    logger.info(f"Exporting audit data for last {days} days")

    try:
        generator = AuditReportGenerator()
        start_date = datetime.now(UTC) - timedelta(days=days)
        csv_data = generator.export_audit_data_csv(start_date=start_date)

        with open(output_file, "w") as f:
            f.write(csv_data)

        logger.info(f"Audit data exported to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error exporting audit data: {e}")
        return None


async def run_full_maintenance():
    """Run all maintenance tasks."""
    logger.info("Starting full security maintenance")

    tasks = [
        # Cleanup tasks
        cleanup_audit_logs(),
        cleanup_rate_limiting_data(),
    ]

    # Run cleanup tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} failed: {result}")
        else:
            logger.info(f"Task {i} completed successfully")

    # Generate reports
    logger.info("Generating security reports")

    # Security summary
    generate_security_report(days=30, output_file="security_summary.json")

    # Threat analysis
    generate_threat_analysis(days=7, output_file="threat_analysis.json")

    # Compliance report
    generate_compliance_report(days=90, output_file="compliance_report.json")

    # Export audit data
    export_audit_data(days=30, output_file="audit_export.csv")

    logger.info("Full security maintenance completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Security maintenance for MaverickMCP")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Cleanup commands
    cleanup_parser = subparsers.add_parser("cleanup", help="Run cleanup tasks")
    cleanup_parser.add_argument(
        "--audit-retention",
        type=int,
        default=90,
        help="Audit log retention in days (default: 90)",
    )
    cleanup_parser.add_argument(
        "--rate-limit-retention",
        type=int,
        default=24,
        help="Rate limiting data retention in hours (default: 24)",
    )

    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument(
        "--type",
        choices=["security", "threat", "compliance"],
        required=True,
        help="Type of report to generate",
    )
    report_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to include in report (default: 30)",
    )
    report_parser.add_argument(
        "--output", type=str, help="Output file (default: print to stdout)"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export audit data")
    export_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to export (default: 30)"
    )
    export_parser.add_argument(
        "--output",
        type=str,
        default="audit_export.csv",
        help="Output CSV file (default: audit_export.csv)",
    )

    # Full maintenance command
    subparsers.add_parser("full", help="Run full maintenance (cleanup + reports)")

    args = parser.parse_args()

    if args.command == "cleanup":

        async def run_cleanup():
            await cleanup_audit_logs(args.audit_retention)
            await cleanup_rate_limiting_data(args.rate_limit_retention)

        asyncio.run(run_cleanup())

    elif args.command == "report":
        if args.type == "security":
            generate_security_report(args.days, args.output)
        elif args.type == "threat":
            generate_threat_analysis(args.days, args.output)
        elif args.type == "compliance":
            generate_compliance_report(args.days, args.output)

    elif args.command == "export":
        export_audit_data(args.days, args.output)

    elif args.command == "full":
        asyncio.run(run_full_maintenance())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
