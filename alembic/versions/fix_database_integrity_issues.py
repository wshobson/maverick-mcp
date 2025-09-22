"""Legacy migration stub for OSS build without billing tables."""

# Revision identifiers, used by Alembic.
revision = "fix_database_integrity"
down_revision = "e0c75b0bdadb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op migration for the open-source build."""
    pass


def downgrade() -> None:
    """No-op downgrade for the open-source build."""
    pass
