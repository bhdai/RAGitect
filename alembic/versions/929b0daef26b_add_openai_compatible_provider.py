"""add_openai_compatible_provider

Revision ID: 929b0daef26b
Revises: 95ef8efd9fa3
Create Date: 2025-12-10 13:34:04.468396

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "929b0daef26b"
down_revision: Union[str, Sequence[str], None] = "95ef8efd9fa3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop existing constraint
    op.drop_constraint("valid_provider_name", "llm_provider_configs", type_="check")

    # Recreate with openai_compatible added
    op.create_check_constraint(
        "valid_provider_name",
        "llm_provider_configs",
        "provider_name IN ('ollama', 'openai', 'anthropic', 'gemini', 'openai_compatible')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop new constraint
    op.drop_constraint("valid_provider_name", "llm_provider_configs", type_="check")

    # Recreate original without openai_compatible
    op.create_check_constraint(
        "valid_provider_name",
        "llm_provider_configs",
        "provider_name IN ('ollama', 'openai', 'anthropic', 'gemini')",
    )
