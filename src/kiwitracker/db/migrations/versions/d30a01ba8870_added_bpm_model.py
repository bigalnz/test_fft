"""Added BPM model

Revision ID: d30a01ba8870
Revises: 
Create Date: 2024-04-21 22:15:14.768730

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd30a01ba8870'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('bpm',
    sa.Column('bid', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dt', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('channel', sa.Integer(), nullable=False),
    sa.Column('bpm', sa.Float(), nullable=False),
    sa.Column('dbfs', sa.Float(), nullable=False),
    sa.Column('clipping', sa.Float(), nullable=False),
    sa.Column('duration', sa.Float(), nullable=False),
    sa.Column('snr', sa.Float(), nullable=False),
    sa.Column('lat', sa.Float(), nullable=False),
    sa.Column('lon', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('bid')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('bpm')
    # ### end Alembic commands ###
