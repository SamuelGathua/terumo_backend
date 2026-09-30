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

async def init_db() -> None:
    """Initialize database tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# 2. Redis Client & Caching Layer (redis.asyncio)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client: Optional[aioredis.Redis] = None

def get_redis_client() -> Optional[aioredis.Redis]:
    """Obtain or initialize the global asynchronous Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0
            )
        except Exception as e:
            logger.warning(f"Could not initialize Redis client connection pool: {e}")
            _redis_client = None
    return _redis_client

async def get_cached_json(key: str) -> Optional[Any]:
    """Retrieve and deserialize a JSON cached entry from Redis."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        data = await client.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.debug(f"Redis cache lookup missed/failed for key '{key}': {e}")
    return None

async def set_cached_json(key: str, value: Any, ttl_seconds: int = 900) -> None:
    """Serialize and cache a value in Redis with a Time-To-Live (default 15 minutes = 900s)."""
    client = get_redis_client()
    if client is None:
        return
    try:
        serialized = json.dumps(value, default=str)
        await client.set(key, serialized, ex=ttl_seconds)
    except Exception as e:
        logger.debug(f"Redis cache write failed for key '{key}': {e}")
