"""Legacy precision migration stub (billing tables removed in OSS build)."""

# Revision identifiers, used by Alembic.
revision = "e0c75b0bdadb"
down_revision = "add_stripe_webhook_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op migration for the open-source build."""
    pass


def downgrade() -> None:
    """No-op downgrade for the open-source build."""
    pass
