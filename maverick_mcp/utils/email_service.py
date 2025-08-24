"""
Email service using Mailgun for sending transactional emails.

This module provides a simple interface for sending emails through Mailgun's API.
"""

import httpx

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class MailgunService:
    """Service for sending emails through Mailgun API."""

    def __init__(self, domain: str | None = None, api_key: str | None = None):
        """
        Initialize Mailgun service.

        Args:
            domain: Mailgun domain (defaults to settings)
            api_key: Mailgun API key (defaults to settings)
        """
        self.domain = domain or settings.email.mailgun_domain
        self.api_key = api_key or settings.email.mailgun_api_key
        self.base_url = f"https://api.mailgun.net/v3/{self.domain}"
        self.from_email = settings.email.from_address
        self.from_name = settings.email.from_name

        # Validate configuration
        if not self.domain or not self.api_key:
            logger.warning("Mailgun not configured - email sending disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Mailgun service initialized for domain: {self.domain}")

    async def send_email(
        self,
        to: str | list[str],
        subject: str,
        text: str | None = None,
        html: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """
        Send an email using Mailgun API.

        Args:
            to: Recipient email address(es)
            subject: Email subject
            text: Plain text content
            html: HTML content
            cc: CC recipients
            bcc: BCC recipients
            reply_to: Reply-to address
            tags: Mailgun tags for tracking
            metadata: Custom metadata

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Email service disabled - skipping email send")
            # In development, log email content to stdout
            if settings.api.debug:
                logger.info(
                    "Email content (dev mode)",
                    extra={
                        "to": to,
                        "subject": subject,
                        "text": text[:200] if text else None,  # First 200 chars
                        "html_length": len(html) if html else 0,
                        "cc": cc,
                        "bcc": bcc,
                        "reply_to": reply_to,
                        "tags": tags,
                        "metadata": metadata,
                    },
                )
            return False

        if not text and not html:
            logger.error("Email must have either text or HTML content")
            return False

        # Prepare recipient list
        if isinstance(to, str):
            to = [to]

        # Build email data
        data = {
            "from": f"{self.from_name} <{self.from_email}>",
            "to": ",".join(to),
            "subject": subject,
        }

        if text:
            data["text"] = text
        if html:
            data["html"] = html
        if cc:
            data["cc"] = ",".join(cc)
        if bcc:
            data["bcc"] = ",".join(bcc)
        if reply_to:
            data["h:Reply-To"] = reply_to
        if tags:
            for tag in tags[:3]:  # Mailgun allows max 3 tags
                data["o:tag"] = tag

        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                data[f"v:{key}"] = value

        # Send email
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    auth=("api", self.api_key),
                    data=data,
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"Email sent successfully to {to}",
                        extra={
                            "message_id": result.get("id"),
                            "subject": subject,
                            "recipients": to,
                        },
                    )
                    # In debug mode, also log email content
                    if settings.api.debug:
                        logger.debug(
                            "Email content sent (debug mode)",
                            extra={
                                "to": to,
                                "subject": subject,
                                "text_preview": text[:200] if text else None,
                                "has_html": bool(html),
                                "message_id": result.get("id"),
                            },
                        )
                    return True
                else:
                    logger.error(
                        f"Failed to send email: {response.status_code}",
                        extra={
                            "response": response.text,
                            "recipients": to,
                            "subject": subject,
                        },
                    )
                    return False

        except Exception as e:
            logger.error(
                f"Error sending email: {str(e)}",
                extra={"recipients": to, "subject": subject},
                exc_info=True,
            )
            return False

    async def send_welcome_email(self, user_email: str, user_name: str) -> bool:
        """
        Send welcome email to new user.

        Args:
            user_email: User's email address
            user_name: User's name

        Returns:
            True if sent successfully
        """
        subject = f"Welcome to {settings.app_name}!"

        text = f"""
Hi {user_name},

Welcome to {settings.app_name}! We're excited to have you on board.

Your account has been successfully created and you can now access all our financial analysis tools.

To get started:
1. Log in to your account
2. Generate your API key from the dashboard
3. Start using our powerful MCP tools

If you have any questions, feel free to reach out to our support team.

Best regards,
The {settings.app_name} Team
        """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Welcome to {settings.app_name}</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #2c3e50;">Welcome to {settings.app_name}!</h1>

        <p>Hi {user_name},</p>

        <p>Welcome to {settings.app_name}! We're excited to have you on board.</p>

        <p>Your account has been successfully created and you can now access all our financial analysis tools.</p>

        <h2 style="color: #34495e;">Getting Started</h2>
        <ol>
            <li>Log in to your account</li>
            <li>Generate your API key from the dashboard</li>
            <li>Start using our powerful MCP tools</li>
        </ol>

        <p>If you have any questions, feel free to reach out to our support team.</p>

        <p>Best regards,<br>
        The {settings.app_name} Team</p>
    </div>
</body>
</html>
        """

        return await self.send_email(
            to=user_email,
            subject=subject,
            text=text,
            html=html,
            tags=["welcome", "onboarding"],
            metadata={"user_name": user_name},
        )

    async def send_api_key_email(
        self, user_email: str, user_name: str, api_key_prefix: str
    ) -> bool:
        """
        Send API key notification email.

        Args:
            user_email: User's email address
            user_name: User's name
            api_key_prefix: API key prefix for identification

        Returns:
            True if sent successfully
        """
        subject = "Your API Key Has Been Created"

        text = f"""
Hi {user_name},

Your API key has been successfully created!

API Key Prefix: {api_key_prefix}

Important:
- Keep your API key secure and never share it publicly
- You can manage your API keys from your dashboard
- Each key can be configured with specific rate limits and permissions

Happy coding!

The {settings.app_name} Team
        """

        return await self.send_email(
            to=user_email,
            subject=subject,
            text=text,
            tags=["api-key", "notification"],
            metadata={"api_key_prefix": api_key_prefix},
        )

    async def send_payment_failed_email(
        self, user_email: str, user_name: str, error_message: str
    ) -> bool:
        """
        Send payment failure notification.

        Args:
            user_email: User's email address
            user_name: User's name
            error_message: Payment error details

        Returns:
            True if sent successfully
        """
        subject = "Payment Failed - Action Required"

        text = f"""
Hi {user_name},

We were unable to process your payment for {settings.app_name}.

Error: {error_message}

Please update your payment method to continue using our services:
1. Log in to your account
2. Go to Billing settings
3. Update your payment information

Your service will continue for 7 days. After that, your API access may be suspended.

If you need assistance, please contact our support team.

The {settings.app_name} Team
        """

        return await self.send_email(
            to=user_email,
            subject=subject,
            text=text,
            tags=["payment", "failed", "billing"],
            metadata={"error": error_message},
        )


# Global email service instance
email_service = MailgunService()


# Convenience functions
async def send_email(
    to: str | list[str],
    subject: str,
    text: str | None = None,
    html: str | None = None,
    **kwargs,
) -> bool:
    """Send an email using the global email service."""
    return await email_service.send_email(to, subject, text, html, **kwargs)


async def send_welcome_email(user_email: str, user_name: str) -> bool:
    """Send welcome email to new user."""
    return await email_service.send_welcome_email(user_email, user_name)


async def send_api_key_email(
    user_email: str, user_name: str, api_key_prefix: str
) -> bool:
    """Send API key notification email."""
    return await email_service.send_api_key_email(user_email, user_name, api_key_prefix)


async def send_payment_failed_email(
    user_email: str, user_name: str, error_message: str
) -> bool:
    """Send payment failure notification."""
    return await email_service.send_payment_failed_email(
        user_email, user_name, error_message
    )
