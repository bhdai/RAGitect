"""create_llm_provider_configs

Revision ID: 38b2df5424bf
Revises: 6a3bc9ddb157
Create Date: 2025-12-08 11:13:00.281305

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "38b2df5424bf"
down_revision: Union[str, Sequence[str], None] = "6a3bc9ddb157"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "llm_provider_configs",
        sa.Column(
            "id",
            sa.UUID(),
            nullable=False,
            comment="Unique configuration identifier",
        ),
        sa.Column(
            "provider_name",
            sa.String(length=50),
            nullable=False,
            comment="Provider name (ollama, openai, anthropic)",
        ),
        sa.Column(
            "config_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="Configuration data (base_url, api_key, model, etc.)",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
            comment="Whether this configuration is active",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Creation timestamp",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Last update timestamp",
        ),
        sa.CheckConstraint(
            "provider_name IN ('ollama', 'openai', 'anthropic')",
            name="valid_provider_name",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_name"),
        comment="LLM provider configurations with encrypted API keys",
    )
    op.create_index(
        op.f("ix_llm_provider_configs_provider_name"),
        "llm_provider_configs",
        ["provider_name"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_llm_provider_configs_provider_name"),
        table_name="llm_provider_configs",
    )
    op.drop_table("llm_provider_configs")
