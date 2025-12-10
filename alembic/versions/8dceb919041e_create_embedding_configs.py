"""create_embedding_configs

Revision ID: 8dceb919041e
Revises: 929b0daef26b
Create Date: 2025-12-10 15:28:44.094645

"""

from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Column, String, Boolean, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func


# revision identifiers, used by Alembic.
revision: str = "8dceb919041e"
down_revision: Union[str, Sequence[str], None] = "929b0daef26b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "embedding_configs",
        Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            comment="Unique embedding config identifier",
        ),
        Column(
            "provider_name",
            String(50),
            nullable=False,
            unique=True,
            index=True,
            comment="Embedding provider name",
        ),
        Column(
            "config_data",
            JSONB,
            nullable=False,
            comment="Configuration data (base_url, api_key, model, dimension, etc.)",
        ),
        Column(
            "is_active",
            Boolean(),
            nullable=False,
            server_default="true",
            comment="Whether this configuration is active",
        ),
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            comment="Creation timestamp",
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            comment="Last update timestamp",
        ),
        CheckConstraint(
            "provider_name IN ('ollama', 'openai', 'vertex_ai', 'openai_compatible')",
            name="valid_embedding_provider_name",
        ),
        comment="Embedding provider configurations",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("embedding_configs")
