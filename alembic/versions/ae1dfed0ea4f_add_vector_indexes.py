"""add_vector_indexes

Revision ID: ae1dfed0ea4f
Revises: b95b9c3ddf9b
Create Date: 2025-11-24 15:17:34.823653

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "ae1dfed0ea4f"
down_revision: Union[str, Sequence[str], None] = "b95b9c3ddf9b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create HNSW indexes for vector similarity search"""

    op.execute("""
        CREATE INDEX ix_chunks_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
