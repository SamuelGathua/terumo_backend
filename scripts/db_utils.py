"""
scripts/db_utils.py - shared database helpers for the ABIS seeding scripts
(replaces the three copy-pasted get_async_engine() functions).
"""

import os

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def resolve_async_url() -> str:
    raw = (
        os.environ.get("ASYNC_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite+aiosqlite:///./blood_supply.db"
    )
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+asyncpg://", 1)
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    return raw


def get_async_engine() -> AsyncEngine:
    return create_async_engine(resolve_async_url(), echo=False)


def assert_safe_target(engine: AsyncEngine) -> None:
    """Refuse destructive seeding against anything except SQLite unless explicitly allowed."""
    backend = engine.url.get_backend_name()
    if backend != "sqlite" and os.environ.get("ALLOW_DB_RESET") != "1":
        raise SystemExit(
            f"Refusing to wipe tables on a '{backend}' database (host={engine.url.host}). "
            "Set ALLOW_DB_RESET=1 only if you really mean it."
        )
