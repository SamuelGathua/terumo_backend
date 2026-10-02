import json
import logging
import os
from typing import Any, AsyncGenerator, Optional
from dotenv import load_dotenv
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Load local environment variables if present
load_dotenv()

logger = logging.getLogger("abis.database")

# 1. Asynchronous Database Engine Configuration
# Environment provides ASYNC_DATABASE_URL as primary target per operational directives
raw_async_url = (
    os.getenv("ASYNC_DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or "sqlite+aiosqlite:///./blood_supply.db"
)

# Normalize PostgreSQL driver schemes to asyncpg
if raw_async_url.startswith("postgres://"):
    ASYNC_DATABASE_URL = raw_async_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_async_url.startswith("postgresql://") and not raw_async_url.startswith("postgresql+asyncpg://"):
    ASYNC_DATABASE_URL = raw_async_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = raw_async_url

connect_args = {}
if "sqlite" in ASYNC_DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency provider for FastAPI route handlers."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def apply_column_migrations(sync_conn) -> None:
    """Idempotently ensure required enrichment columns exist on both SQLite and PostgreSQL."""
    from sqlalchemy import inspect, text
    inspector = inspect(sync_conn)
    tables = inspector.get_table_names()

    migrations = [
        ("donors", "sex", "VARCHAR(1)"),
        ("donors", "donor_type", "VARCHAR(30)"),
        ("donors", "date_of_birth", "DATE"),
        ("facilities", "keph_level", "INTEGER"),
        ("transfusion_requests", "status", "VARCHAR(20) DEFAULT 'PENDING'"),
    ]

    for table, col, col_type in migrations:
        if table in tables:
            existing_cols = [c["name"] for c in inspector.get_columns(table)]
            if col not in existing_cols:
                logger.info(f"Applying schema migration: adding {table}.{col} ({col_type})...")
                sync_conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))

async def init_db() -> None:
    """Initialize database tables asynchronously and apply schema column migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(apply_column_migrations)


# 2. Redis Client & Caching Layer (redis.asyncio) with Resilient Local Memory Fallback
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client: Optional[aioredis.Redis] = None
_local_memory_cache: dict[str, tuple[float, str]] = {}

def get_redis_client() -> Optional[aioredis.Redis]:
    """Obtain or initialize the global asynchronous Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0
            )
        except Exception as e:
            logger.warning(f"Could not initialize Redis client connection pool: {e}")
            _redis_client = None
    return _redis_client

async def get_cached_json(key: str) -> Optional[Any]:
    """Retrieve and deserialize a JSON cached entry from Redis, with local memory fallback."""
    client = get_redis_client()
    if client is not None:
        try:
            data = await client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Redis cache lookup unreachable: {e}. Checking memory fallback.")

    # Check local memory fallback
    import time
    if key in _local_memory_cache:
        expire_at, serialized = _local_memory_cache[key]
        if time.time() < expire_at:
            return json.loads(serialized)
        else:
            del _local_memory_cache[key]
    return None

async def set_cached_json(key: str, value: Any, ttl_seconds: int = 900) -> None:
    """Serialize and cache a value in Redis with TTL (default 15 minutes = 900s), with memory fallback."""
    import time
    serialized = json.dumps(value, default=str)
    client = get_redis_client()
    if client is not None:
        try:
            await client.set(key, serialized, ex=ttl_seconds)
            return
        except Exception as e:
            logger.debug(f"Redis cache write unreachable: {e}. Storing in memory fallback.")

    # Store in local memory cache
    _local_memory_cache[key] = (time.time() + ttl_seconds, serialized)

async def delete_cached_keys(*keys: str) -> None:
    """Invalidate specific keys from Redis and local memory cache."""
    client = get_redis_client()
    if client is not None:
        try:
            for key in keys:
                await client.delete(key)
        except Exception as e:
            logger.debug(f"Redis cache delete unreachable: {e}")
    for key in keys:
        _local_memory_cache.pop(key, None)

