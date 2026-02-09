"""add_fts_gin_index

Revision ID: 7791001eb8a2
Revises: 8dceb919041e
Create Date: 2026-02-09 17:40:52.049941

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '7791001eb8a2'
down_revision: Union[str, Sequence[str], None] = '8dceb919041e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create GIN index for full-text search on document_chunks.content.

    Note: For production deployments with large existing datasets, consider
    using CREATE INDEX CONCURRENTLY to avoid locking the table. Alembic
    doesn't support CONCURRENTLY natively — use raw SQL with autocommit.
    """
    op.execute("""
        CREATE INDEX ix_document_chunks_content_fts
        ON document_chunks
        USING GIN (to_tsvector('english', content))
    """)


def downgrade() -> None:
    """Drop the full-text search GIN index."""
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_fts")
