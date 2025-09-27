"""
Test script for Mailgun email integration.

Run this script to test your Mailgun configuration:
    python maverick_mcp/tests/test_mailgun_email.py
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.email_service import (
    MailgunService,
    send_api_key_email,
    send_welcome_email,
)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mailgun_config():
    """Test Mailgun configuration."""
    print("=" * 60)
    print("Testing Mailgun Configuration")
    print("=" * 60)

    print(f"Email Enabled: {settings.email.enabled}")
    print(f"Mailgun Domain: {settings.email.mailgun_domain}")
    print(f"From Address: {settings.email.from_address}")
    print(f"From Name: {settings.email.from_name}")
    print(f"API Key Set: {'Yes' if settings.email.mailgun_api_key else 'No'}")

    if not settings.email.mailgun_api_key:
        print("\n❌ Mailgun API key not configured!")
        print("Please set MAILGUN_API_KEY in your .env file")
        return False

    if not settings.email.mailgun_domain:
        print("\n❌ Mailgun domain not configured!")
        print("Please set MAILGUN_DOMAIN in your .env file")
        return False

    print("\n✅ Mailgun configuration looks good!")
    return True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_send_email():
    """Test sending a basic email."""
    print("\n" + "=" * 60)
    print("Testing Basic Email Send")
    print("=" * 60)

    # Get test email from environment or use default
    test_email = os.getenv("TEST_EMAIL", "test@example.com")

    service = MailgunService()

    success = await service.send_email(
        to=test_email,
        subject="Test Email from Maverick-MCP",
        text="This is a test email to verify Mailgun integration.",
        html="<h1>Test Email</h1><p>This is a test email to verify Mailgun integration.</p>",
        tags=["test", "integration"],
        metadata={"test": "true", "source": "test_script"},
    )

    if success:
        print(f"✅ Test email sent successfully to {test_email}")
    else:
        print(f"❌ Failed to send test email to {test_email}")

    return success


@pytest.mark.asyncio
@pytest.mark.integration
async def test_email_templates():
    """Test all email templates."""
    print("\n" + "=" * 60)
    print("Testing Email Templates")
    print("=" * 60)

    test_email = os.getenv("TEST_EMAIL", "test@example.com")
    test_name = "Test User"

    # Test welcome email
    print("\n1. Testing Welcome Email...")
    success = await send_welcome_email(test_email, test_name)
    print("✅ Welcome email sent" if success else "❌ Welcome email failed")

    # Test API key email
    print("\n2. Testing API Key Email...")
    success = await send_api_key_email(test_email, test_name, "test_1234567890")
    print("✅ API key email sent" if success else "❌ API key email failed")


async def main():
    """Run all tests."""
    print("\nMaverick-MCP Mailgun Email Test Suite")
    print("=====================================\n")

    # Test configuration
    if not await test_mailgun_config():
        print("\nPlease configure Mailgun before running tests.")
        print("See .env.mailgun.example for configuration details.")
        return

    # Ask if user wants to send test emails
    print("\nWould you like to send test emails? (y/n)")
    response = input().strip().lower()

    if response == "y":
        test_email = input(
            "Enter test email address (or press Enter for default): "
        ).strip()
        if test_email:
            os.environ["TEST_EMAIL"] = test_email

        # Send test emails
        await test_send_email()

        print("\nWould you like to test all email templates? (y/n)")
        if input().strip().lower() == "y":
            await test_email_templates()

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
