"""Enhance audit logging capabilities

Revision ID: 007_enhance_audit_logging
Revises: f976356b6f07_add_refresh_tokens_and_audit_tables
Create Date: 2025-01-27 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "007_enhance_audit_logging"
down_revision = "fix_database_integrity"
branch_labels = None
depends_on = None


def upgrade():
    """Enhance audit logging with additional indexes and constraints."""

    # Add performance indexes for audit log queries
    op.create_index(
        "idx_auth_audit_log_event_type_created_at",
        "mcp_auth_audit_log",
        ["event_type", "created_at"],
    )

    op.create_index(
        "idx_auth_audit_log_user_id_created_at",
        "mcp_auth_audit_log",
        ["user_id", "created_at"],
    )

    op.create_index(
        "idx_auth_audit_log_ip_address_created_at",
        "mcp_auth_audit_log",
        ["ip_address", "created_at"],
    )

    op.create_index(
        "idx_auth_audit_log_success_created_at",
        "mcp_auth_audit_log",
        ["success", "created_at"],
    )

    # Add constraint for event_type to ensure valid values
    op.execute("""
        ALTER TABLE mcp_auth_audit_log
        ADD CONSTRAINT chk_audit_log_event_type
        CHECK (event_type IN (
            'login_success', 'login_failed', 'logout', 'password_change',
            'password_reset_request', 'password_reset_complete', 'token_refresh',
            'token_revoked', 'token_expired', 'api_key_created', 'api_key_deleted',
            'api_key_used', 'api_key_rate_limited', 'credits_purchased', 'credits_used',
            'credits_refunded', 'credits_expired', 'insufficient_credits',
            'suspicious_activity', 'rate_limit_exceeded', 'multiple_failed_logins',
            'unusual_access_pattern', 'privilege_escalation_attempt',
            'sensitive_data_access', 'bulk_data_export', 'pii_access',
            'system_error', 'configuration_change', 'backup_created', 'data_migration'
        ))
    """)

    # Add partial index for failed events (more efficient for security monitoring)
    op.execute("""
        CREATE INDEX idx_auth_audit_log_failed_events
        ON mcp_auth_audit_log (created_at, event_type, ip_address)
        WHERE success = false
    """)

    # Add partial index for security events
    op.execute("""
        CREATE INDEX idx_auth_audit_log_security_events
        ON mcp_auth_audit_log (created_at, user_id, ip_address)
        WHERE event_type IN (
            'suspicious_activity', 'rate_limit_exceeded', 'multiple_failed_logins',
            'unusual_access_pattern', 'privilege_escalation_attempt'
        )
    """)


def downgrade():
    """Remove audit logging enhancements."""

    # Drop indexes
    op.drop_index("idx_auth_audit_log_security_events", table_name="mcp_auth_audit_log")
    op.drop_index("idx_auth_audit_log_failed_events", table_name="mcp_auth_audit_log")
    op.drop_index(
        "idx_auth_audit_log_success_created_at", table_name="mcp_auth_audit_log"
    )
    op.drop_index(
        "idx_auth_audit_log_ip_address_created_at", table_name="mcp_auth_audit_log"
    )
    op.drop_index(
        "idx_auth_audit_log_user_id_created_at", table_name="mcp_auth_audit_log"
    )
    op.drop_index(
        "idx_auth_audit_log_event_type_created_at", table_name="mcp_auth_audit_log"
    )

    # Drop constraint
    op.drop_constraint("chk_audit_log_event_type", "mcp_auth_audit_log", type_="check")
