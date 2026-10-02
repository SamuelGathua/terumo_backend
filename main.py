import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from routes import router
from ml_engine import predictive_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("abis.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler: initializes tables, migrates datasets, and trains ML models upon startup."""
    logger.info("Initializing database schemas...")
    try:
        await init_db()
        logger.info("Database schemas initialized.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    # Automated check: Migrate foundational data sheets to server database if unseeded
    try:
        from sqlalchemy import func, select
        from database import AsyncSessionLocal
        from models import Facility
        async with AsyncSessionLocal() as session:
            fac_count = (await session.execute(select(func.count(Facility.id)))).scalar()
        if fac_count < 100:
            logger.info(f"Database unseeded (found {fac_count} facilities). Migrating foundational datasets from data/ folder...")
            from scripts.migrate_all_data import run_full_migration
            mig_res = await run_full_migration()
            logger.info(f"Automated startup data migration completed: {mig_res}")
        else:
            logger.info(f"Database verified with {fac_count} facilities online.")
    except Exception as e:
        logger.warning(f"Startup migration notice: {e}")

    logger.info("Initializing Predictive Intelligence Engine...")
    try:
        # Load saved model if present; retrain in background otherwise to prevent Railway healthcheck timeouts
        if predictive_engine._try_load_artifact():
            logger.info(f"Loaded existing retention model: {predictive_engine.model_metrics.get('version')}")
        else:
            logger.info("No saved retention model found. Scheduling background training to prevent healthcheck timeout...")
            asyncio.create_task(predictive_engine.initialize_from_db(force_retrain=True))
    except Exception as e:
        logger.warning(f"Could not complete model initialization on startup: {e}. Heuristic fallback active.")

    yield
    logger.info("ABIS service shutting down.")

app = FastAPI(
    title="Adaptive Blood Infrastructure System (ABIS) API",
    description=(
        "Standardised, intelligent, and adaptable blood-bank information system connecting "
        "donor drives, laboratories, cold chain transit, and hospital transfusion centers. "
        "Built for Terumo BCT Africa Hackathon 2026."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enforce Environment-Specific CORS Security per AGENTS.md
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
SECRET_KEY = os.getenv("SECRET_KEY", "abis-hackathon-development-key-default")

if ENVIRONMENT == "production":
    allowed_origins = [
        os.getenv("FRONTEND_URL", "https://terumo-frontend.vercel.app"),
        "https://samuelgathua.github.io",
    ]
    logger.info(f"Production environment detected. Restricting CORS to: {allowed_origins}")
else:
    allowed_origins = ["*"]
    logger.info("Development environment detected. Permissive CORS enabled for local prototyping.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular routes
app.include_router(router, prefix="/api/v1")
app.include_router(router) # Direct root mount for convenience

@app.get("/", tags=["System"])
async def root():
    return {
        "project": "Adaptive Blood Infrastructure System (ABIS)",
        "event": "Terumo BCT Africa Hackathon 2026",
        "documentation": "/docs",
        "health_check": "/healthz/",
        "model_trained": predictive_engine.is_trained,
        "environment": ENVIRONMENT,
        "status": "Operational",
    }
