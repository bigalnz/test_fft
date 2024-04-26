"""Added FT model

Revision ID: 2e1d034a41ef
Revises: eeee662837fc
Create Date: 2024-04-26 02:02:45.861684

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2e1d034a41ef'
down_revision: Union[str, None] = 'eeee662837fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('fast_telemetry',
    sa.Column('fid', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('channel', sa.Integer(), nullable=False),
    sa.Column('carrier_freq', sa.Float(), nullable=False),
    sa.Column('start_dt', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('end_dt', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('snr_min', sa.Float(), nullable=False),
    sa.Column('snr_max', sa.Float(), nullable=False),
    sa.Column('snr_mean', sa.Float(), nullable=False),
    sa.Column('dbfs_min', sa.Float(), nullable=False),
    sa.Column('dbfs_max', sa.Float(), nullable=False),
    sa.Column('dbfs_mean', sa.Float(), nullable=False),
    sa.Column('lat', sa.Float(), nullable=False),
    sa.Column('lon', sa.Float(), nullable=False),
    sa.Column('mode', sa.String(length=16), nullable=False),
    sa.Column('d1', sa.Integer(), nullable=False),
    sa.Column('d2', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('fid')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('fast_telemetry')
    # ### end Alembic commands ###
