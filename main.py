import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("abis.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler: initializes tables upon service startup."""
    logger.info("Initializing database schemas...")
    try:
        await init_db()
        logger.info("Database schemas initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
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

# Enforce Environment-Specific CORS Security
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
        "environment": ENVIRONMENT,
        "status": "Operational",
    }
