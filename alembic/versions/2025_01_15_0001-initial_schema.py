"""Initial database schema

Revision ID: initial_schema
Revises:
Create Date: 2025-01-15 00:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""

    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255)),
        sa.Column('role', sa.String(50), nullable=False, server_default='user'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_users_id', 'users', ['id'])

    # Create projects table
    op.create_table(
        'projects',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('website', sa.String(500), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('status', sa.String(50), nullable=False, server_default='active', index=True),
        sa.Column('owner_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('extra_metadata', sa.JSON, server_default='{}'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_projects_id', 'projects', ['id'])

    # Create tasks table
    op.create_table(
        'tasks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id'), nullable=False, index=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending', index=True),
        sa.Column('priority', sa.String(50), nullable=False, server_default='medium', index=True),
        sa.Column('assignee_id', sa.String(36), sa.ForeignKey('users.id'), index=True),
        sa.Column('due_date', sa.DateTime),
        sa.Column('extra_metadata', sa.JSON, server_default='{}'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_tasks_id', 'tasks', ['id'])

    # Create webhooks table
    op.create_table(
        'webhooks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('url', sa.String(500), nullable=False),
        sa.Column('events', sa.JSON, nullable=False),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id'), index=True),
        sa.Column('description', sa.Text),
        sa.Column('status', sa.String(50), server_default='active', index=True),
        sa.Column('secret', sa.String(255)),
        sa.Column('last_triggered_at', sa.DateTime),
        sa.Column('success_count', sa.Integer, server_default='0'),
        sa.Column('error_count', sa.Integer, server_default='0'),
        sa.Column('extra_metadata', sa.JSON, server_default='{}'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_webhooks_id', 'webhooks', ['id'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('webhooks')
    op.drop_table('tasks')
    op.drop_table('projects')
    op.drop_table('users')
