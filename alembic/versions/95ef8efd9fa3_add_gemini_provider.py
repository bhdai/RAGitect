"""add_gemini_provider

Revision ID: 95ef8efd9fa3
Revises: 38b2df5424bf
Create Date: 2025-12-10 09:55:53.405818

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "95ef8efd9fa3"
down_revision: Union[str, Sequence[str], None] = "38b2df5424bf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("valid_provider_name", "llm_provider_configs")
    op.create_check_constraint(
        "valid_provider_name",
        "llm_provider_configs",
        "provider_name IN ('ollama', 'openai', 'anthropic', 'gemini')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("valid_provider_name", "llm_provider_configs")
    op.create_check_constraint(
        "valid_provider_name",
        "llm_provider_configs",
        "provider_name IN ('ollama', 'openai', 'anthropic')",
    )
