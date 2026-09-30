"""
scripts/seed_demand.py - Asynchronous Time-Series Blood Demand Simulation Pipeline
=================================================================================

Generates 2 years (730 days) of daily historical and real-time hospital transfusion
orders across at least 3 distinct facilities using a discrete Ornstein-Uhlenbeck
mean-reverting stochastic random walk process:

    D_t = D_{t-1} + theta * (mu - D_{t-1}) + sigma * epsilon_t

Mathematical Constraints:
    - units_requested strictly floored at 0 (non-negative integer constraint).
    - Incorporates weekend cyclical shocks (Friday/Saturday emergency trauma surges).
"""

import asyncio
import datetime
import logging
import os
import sys
import numpy as np
from sqlalchemy import insert, select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add parent directory to sys.path for direct script execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models import TransfusionRequest, Facility
from database import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("abis.seed_demand")


def get_async_engine():
    """Securely resolve the asynchronous database engine."""
    raw_async_url = (
        os.environ.get("ASYNC_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite+aiosqlite:///./blood_supply.db"
    )

    if raw_async_url.startswith("postgres://"):
        async_url = raw_async_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif raw_async_url.startswith("postgresql://") and not raw_async_url.startswith("postgresql+asyncpg://"):
        async_url = raw_async_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        async_url = raw_async_url

    connect_args = {}
    if "sqlite" in async_url:
        connect_args["check_same_thread"] = False

    return create_async_engine(async_url, echo=False, connect_args=connect_args)


def simulate_facility_demand_series(
    facility_id: str,
    baseline_mean: float,
    reversion_speed: float,
    volatility: float,
    total_days: int = 730
) -> list:
    """
    Simulate daily demand using a mean-reverting stochastic random walk:
        D_t = D_{t-1} + theta * (mu - D_{t-1}) + sigma * epsilon_t
    with strict non-negative floor enforcement and weekend trauma variance.
    """
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=total_days)

    records = []
    current_demand = baseline_mean
    blood_types = ["ALL", "O+", "A+", "B+", "O-"]
    blood_type_weights = [0.40, 0.30, 0.15, 0.10, 0.05]

    urgencies = ["ROUTINE", "EMERGENCY", "MASS_TRANSFUSION"]
    urgency_weights = [0.75, 0.20, 0.05]

    np.random.seed(hash(facility_id) % (2**32))

    for day_offset in range(total_days):
        current_date = start_date + datetime.timedelta(days=day_offset)
        weekday = current_date.weekday() # 4=Friday, 5=Saturday

        # Inject weekend emergency trauma shock factor (+20% variance)
        weekend_multiplier = 1.20 if weekday in (4, 5) else 1.0

        # Mean reversion calculation
        shock = np.random.normal(loc=0.0, scale=volatility)
        drift = reversion_speed * (baseline_mean - current_demand)
        current_demand = (current_demand + drift + shock) * (1.0 if weekday not in (4, 5) else 1.05)

        # Strictly enforce non-negative floor
        units_requested = max(0, int(round(current_demand * weekend_multiplier)))

        # Assign clinical distribution attributes
        blood_type = str(np.random.choice(blood_types, p=blood_type_weights))
        urgency = str(np.random.choice(urgencies, p=urgency_weights))

        records.append({
            "request_date": datetime.datetime.combine(current_date, datetime.time(hour=8, minute=0)),
            "requesting_facility_id": facility_id,
            "blood_type_requested": blood_type,
            "units_requested": units_requested,
            "urgency_level": urgency,
            "created_at": datetime.datetime.utcnow(),
        })

    return records


async def seed_demand_data(total_days: int = 730):
    """Generate 2 years of daily hospital orders and bulk insert into PostgreSQL."""
    facilities_config = [
        {
            "id": "HOSP-NAIROBI-01",
            "name": "Kenyatta National Referral Hospital",
            "region": "Nairobi",
            "type": "HOSPITAL",
            "mean": 88.0,
            "theta": 0.26,
            "sigma": 15.0,
            "capacity": 1200,
            "current_units": 650,
        },
        {
            "id": "HOSP-KISUMU-02",
            "name": "Jaramogi Oginga Odinga Teaching & Referral Hospital",
            "region": "Kisumu",
            "type": "HOSPITAL",
            "mean": 46.0,
            "theta": 0.30,
            "sigma": 8.5,
            "capacity": 600,
            "current_units": 280,
        },
        {
            "id": "CLINIC-MOMBASA-03",
            "name": "Coast General Teaching & Referral Hospital",
            "region": "Mombasa",
            "type": "HOSPITAL",
            "mean": 28.0,
            "theta": 0.35,
            "sigma": 6.0,
            "capacity": 400,
            "current_units": 90, # Low capacity to demonstrate rebalancing trigger
        },
        {
            "id": "REGIONAL-HUB-01",
            "name": "Nairobi Regional Blood Transfusion Center",
            "region": "Nairobi",
            "type": "BLOOD_BANK",
            "mean": 120.0,
            "theta": 0.20,
            "sigma": 18.0,
            "capacity": 3000,
            "current_units": 2400, # High surplus for rebalancing
        }
    ]

    all_request_records = []
    facility_records = []

    for fac in facilities_config:
        facility_records.append({
            "id": fac["id"],
            "name": fac["name"],
            "facility_type": fac["type"],
            "region": fac["region"],
            "latitude": 0.0,
            "longitude": 0.0,
            "inventory_capacity": fac["capacity"],
            "current_inventory_units": fac["current_units"],
        })

        if fac["type"] == "HOSPITAL":
            logger.info(f"Simulating {total_days} days of demand for facility: {fac['id']} (mu={fac['mean']})...")
            fac_records = simulate_facility_demand_series(
                facility_id=fac["id"],
                baseline_mean=fac["mean"],
                reversion_speed=fac["theta"],
                volatility=fac["sigma"],
                total_days=total_days
            )
            all_request_records.extend(fac_records)

    engine = get_async_engine()
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    logger.info("Connecting to database and verifying tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        # Seed Facilities topology
        logger.info(f"Upserting {len(facility_records)} network facilities...")
        for fac in facility_records:
            existing = await session.get(Facility, fac["id"])
            if not existing:
                session.add(Facility(**fac))
        await session.commit()

        # Bulk insert daily transfusion requests
        logger.info(f"Bulk-inserting {len(all_request_records)} daily transfusion requests in chunks of 1,000...")
        chunk_size = 1000
        for idx in range(0, len(all_request_records), chunk_size):
            chunk = all_request_records[idx:idx + chunk_size]
            await session.execute(insert(TransfusionRequest), chunk)
            await session.commit()
            logger.info(f"Inserted demand batch {idx // chunk_size + 1} ({len(chunk)} records).")

        # Verify insertion count
        count_stmt = select(func.count(TransfusionRequest.request_id))
        total_requests = (await session.execute(count_stmt)).scalar()
        logger.info(f"Database population confirmed! Total transfusion orders in ledger: {total_requests}")

    await engine.dispose()
    logger.info("scripts/seed_demand.py completed successfully.")


if __name__ == "__main__":
    asyncio.run(seed_demand_data(total_days=730))
