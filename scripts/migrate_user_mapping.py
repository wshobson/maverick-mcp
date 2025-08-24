#!/usr/bin/env python3
"""
Data migration script to populate mcp_user_mapping table and verify data integrity.

This script should be run AFTER applying the database integrity migration.
It will:
1. Populate the mcp_user_mapping table from Django's users_customuser table
2. Check for orphaned records in MCP tables
3. Report any data integrity issues
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import DATABASE_URL


def main():
    """Run the user mapping data migration."""
    print("Starting user mapping data migration...")

    # Create database connection
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        try:
            # 1. Check if Django users table exists
            result = session.execute(
                text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'users_customuser'
                );
            """)
            )
            django_table_exists = result.scalar()

            if not django_table_exists:
                print("WARNING: Django users_customuser table not found!")
                print("This is expected in development environments.")
                print("Creating sample user mappings for testing...")

                # Create sample mappings for development/testing
                session.execute(
                    text("""
                    INSERT INTO mcp_user_mapping (user_id, django_user_id, email, is_active, created_at, updated_at)
                    VALUES
                        (1, 1, 'test@example.com', true, NOW(), NOW()),
                        (2, 2, 'demo@example.com', true, NOW(), NOW())
                    ON CONFLICT (user_id) DO NOTHING;
                """)
                )
                session.commit()
                print("Created sample user mappings.")
            else:
                # 2. Populate mcp_user_mapping from Django users
                print("Populating mcp_user_mapping from Django users...")

                result = session.execute(
                    text("""
                    INSERT INTO mcp_user_mapping (user_id, django_user_id, email, is_active, created_at, updated_at)
                    SELECT id, id, email, is_active, date_joined, date_joined
                    FROM users_customuser
                    ON CONFLICT (user_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW();
                """)
                )
                session.commit()

                rows_affected = result.rowcount
                print(f"Synchronized {rows_affected} users to mcp_user_mapping.")

            # 3. Check for orphaned records
            print("\nChecking for orphaned records...")

            orphan_checks = [
                ("mcp_user_credits", "user_id"),
                ("mcp_credit_transactions", "user_id"),
                ("mcp_requests", "user_id"),
                ("mcp_refresh_tokens", "user_id"),
                ("mcp_api_keys", "user_id"),
                ("mcp_auth_audit_log", "user_id"),  # This one is nullable
            ]

            has_orphans = False
            for table, column in orphan_checks:
                # Skip nullable columns
                if table == "mcp_auth_audit_log":
                    query = text(f"""
                        SELECT COUNT(*)
                        FROM {table}
                        WHERE {column} IS NOT NULL
                        AND {column} NOT IN (SELECT user_id FROM mcp_user_mapping);
                    """)
                else:
                    query = text(f"""
                        SELECT COUNT(*)
                        FROM {table}
                        WHERE {column} NOT IN (SELECT user_id FROM mcp_user_mapping);
                    """)

                result = session.execute(query)
                orphan_count = result.scalar()

                if orphan_count > 0:
                    has_orphans = True
                    print(f"WARNING: Found {orphan_count} orphaned records in {table}")

                    # Get sample of orphaned user IDs
                    sample_query = text(f"""
                        SELECT DISTINCT {column}
                        FROM {table}
                        WHERE {column} NOT IN (SELECT user_id FROM mcp_user_mapping)
                        {"AND " + column + " IS NOT NULL" if table == "mcp_auth_audit_log" else ""}
                        LIMIT 5;
                    """)

                    result = session.execute(sample_query)
                    orphan_ids = [row[0] for row in result]
                    print(f"  Sample orphaned user IDs: {orphan_ids}")

            if not has_orphans:
                print("✓ No orphaned records found!")

            # 4. Report user mapping statistics
            print("\nUser mapping statistics:")

            result = session.execute(text("SELECT COUNT(*) FROM mcp_user_mapping;"))
            total_mappings = result.scalar()
            print(f"  Total user mappings: {total_mappings}")

            result = session.execute(
                text("SELECT COUNT(*) FROM mcp_user_mapping WHERE is_active = true;")
            )
            active_mappings = result.scalar()
            print(f"  Active users: {active_mappings}")

            # 5. Check for high-value stocks that might exceed old precision
            print("\nChecking for high-value stock prices...")

            result = session.execute(
                text("""
                SELECT ticker_symbol, MAX(close_price) as max_price
                FROM stocks_stock s
                JOIN stocks_pricecache p ON s.stock_id = p.stock_id
                GROUP BY ticker_symbol
                HAVING MAX(close_price) > 99999.99
                ORDER BY max_price DESC
                LIMIT 10;
            """)
            )

            high_value_stocks = list(result)
            if high_value_stocks:
                print(
                    "Found stocks with prices that would exceed old Numeric(10,2) precision:"
                )
                for symbol, price in high_value_stocks:
                    print(f"  {symbol}: ${price:,.2f}")
            else:
                print("✓ No stocks found exceeding old precision limits.")

            print("\nData migration completed successfully!")

        except Exception as e:
            print(f"ERROR during migration: {e}")
            session.rollback()
            raise


if __name__ == "__main__":
    main()
