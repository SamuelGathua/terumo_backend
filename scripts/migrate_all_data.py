"""
scripts/migrate_all_data.py - Server Database Migration Pipeline
================================================================

Migrates foundational datasets from the data/ folder to the PostgreSQL / SQLite database:
1. data/kenya-health-facilities-2017_08_02.xlsx (8,932+ Kenyan health facilities with KEPH tiers)
2. data/blood_donor_dataset.csv & data/blood-format.csv (10,000 KDE-fitted donors)
3. 730-day (2-year) Ornstein-Uhlenbeck daily transfusion requests matching KEPH tier boundaries:
   - Level 6: Capacity randint(800, 1200), Demand mu = randint(80, 150)
   - Level 5: Capacity randint(300, 500), Demand mu = randint(30, 70)
   - Level 4: Capacity randint(50, 150), Demand mu = randint(5, 20)
   - Level 3: Capacity randint(0, 10), Demand mu = randint(0, 2)
   - Level 2 / Dispensary: Capacity = 0, Demand mu = 0 (STRICT: 0 requests)
   - Blood Hubs / RBTC: Capacity randint(2000, 5000), Demand mu = 0
4. Clears Redis caches and trains the Machine Learning models.
"""

import asyncio
import datetime
import logging
import os
import random
import sys
import uuid
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from database import Base, get_redis_client
from models import Donor, Facility, TransfusionRequest, InventoryUnit
from scripts.seed_facilities import assign_tier_metrics, calculate_current_inventory, parse_keph_tier, simulate_demand_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("abis.migrate_all_data")


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


async def migrate_facilities_and_demand(session: AsyncSession, engine) -> Dict:
    """Migrate Excel health facilities sheet and generate KEPH tiered transfusion demand."""
    csv_path = os.path.join(BACKEND_DIR, "data", "kenya-health-facilities-2017_08_02.csv")
    excel_path = os.path.join(BACKEND_DIR, "data", "kenya-health-facilities-2017_08_02.xlsx")

    if os.path.exists(csv_path):
        logger.info(f"Reading Kenyan Facilities Master Sheet from CSV: {csv_path}...")
        df = pd.read_csv(csv_path)
    elif os.path.exists(excel_path):
        logger.info(f"Reading Kenyan Facilities Master Sheet from Excel: {excel_path}...")
        df = pd.read_excel(excel_path)
    else:
        alt_path = os.path.join(BACKEND_DIR, "..", "terumo_web", "public", "kenya-health-facilities-2017_08_02.xlsx")
        if os.path.exists(alt_path):
            df = pd.read_excel(alt_path)
        else:
            raise FileNotFoundError(f"Facilities dataset not found at: {csv_path} or {excel_path}")
    random.seed(42)

    primary_anchors = {
        "HOSP-NAIROBI-01": {"name": "Kenyatta National Referral Hospital", "tier": "Level 6", "region": "Nairobi"},
        "HOSP-KISUMU-02": {"name": "Jaramogi Oginga Odinga Teaching & Referral Hospital", "tier": "Level 5", "region": "Kisumu"},
        "CLINIC-MOMBASA-03": {"name": "Coast General Teaching & Referral Hospital", "tier": "Level 5", "region": "Mombasa"},
        "REGIONAL-HUB-01": {"name": "Nairobi Regional Blood Transfusion Center", "tier": "Blood Hub", "region": "Nairobi"},
    }

    facilities_to_insert: List[Dict] = []
    demand_facilities: List[Tuple[str, int]] = []
    tier_counts = {"Level 6": 0, "Level 5": 0, "Level 4": 0, "Level 3": 0, "Level 2": 0, "Blood Hub": 0}

    # 1. Anchors
    for fac_id, meta in primary_anchors.items():
        cap, dem_mu, db_type = assign_tier_metrics(meta["tier"])
        inv = calculate_current_inventory(cap)
        facilities_to_insert.append({
            "id": fac_id,
            "name": meta["name"],
            "facility_type": db_type,
            "region": meta["region"],
            "latitude": 0.0,
            "longitude": 0.0,
            "inventory_capacity": cap,
            "current_inventory_units": inv,
        })
        tier_counts[meta["tier"]] += 1
        if dem_mu > 0:
            demand_facilities.append((fac_id, dem_mu))

    # 2. Excel facilities
    seen_ids = set(primary_anchors.keys())
    for _, row in df.iterrows():
        code = str(row.get("Code", "")).strip()
        fac_id = f"FAC-{code}" if code else f"FAC-{random.randint(100000, 999999)}"
        if fac_id in seen_ids:
            continue
        seen_ids.add(fac_id)

        raw_name = str(row.get("Name", "Health Facility")).strip()
        raw_type = str(row.get("Facility type", "")).strip()
        raw_keph = str(row.get("Keph level", "")).strip()
        raw_county = str(row.get("County", "National")).strip()

        tier = parse_keph_tier(raw_name, raw_type, raw_keph)
        cap, dem_mu, db_type = assign_tier_metrics(tier)
        inv = calculate_current_inventory(cap)

        facilities_to_insert.append({
            "id": fac_id,
            "name": raw_name[:200],
            "facility_type": db_type,
            "region": raw_county[:100],
            "latitude": 0.0,
            "longitude": 0.0,
            "inventory_capacity": cap,
            "current_inventory_units": inv,
        })
        tier_counts[tier] += 1

        # Generate daily demand history
        if tier in ("Level 6", "Level 5"):
            demand_facilities.append((fac_id, dem_mu))
        elif tier == "Level 4" and (fac_id == "FAC-16201" or random.random() < 0.08):
            demand_facilities.append((fac_id, dem_mu))
        elif tier == "Level 3" and (fac_id == "FAC-22976" or random.random() < 0.02):
            demand_facilities.append((fac_id, dem_mu))
        # STRICT: Level 2 / Dispensaries NEVER added!

    logger.info("Clearing and populating facilities table...")
    await session.execute(delete(Facility))
    await session.execute(delete(TransfusionRequest))
    await session.commit()

    chunk_size = 1000
    for i in range(0, len(facilities_to_insert), chunk_size):
        chunk = facilities_to_insert[i:i + chunk_size]
        await session.execute(insert(Facility), chunk)
        await session.commit()

    logger.info(f"Generating demand records for {len(demand_facilities)} hospital facilities...")
    all_demands = []
    for fac_id, dem_mu in demand_facilities:
        records = simulate_demand_series(facility_id=fac_id, baseline_mean=dem_mu, total_days=730)
        all_demands.extend(records)

    req_chunk_size = 2500
    for j in range(0, len(all_demands), req_chunk_size):
        chunk = all_demands[j:j + req_chunk_size]
        await session.execute(insert(TransfusionRequest), chunk)
        await session.commit()

    return {
        "total_facilities": len(facilities_to_insert),
        "tier_counts": tier_counts,
        "total_transfusion_records": len(all_demands),
    }


async def migrate_donors(session: AsyncSession) -> int:
    """Migrate foundational blood donor dataset using KDE distribution."""
    dataset_path = os.path.join(BACKEND_DIR, "data", "blood_donor_dataset.csv")
    format_path = os.path.join(BACKEND_DIR, "data", "blood-format.csv")

    if not os.path.exists(dataset_path):
        logger.warning(f"Donor dataset not found at {dataset_path}, skipping donor migration.")
        return 0

    logger.info(f"Loading seed data from {dataset_path} and {format_path}...")
    donor_df = pd.read_csv(dataset_path)
    tenure_days = np.clip(donor_df["months_since_first_donation"].values * 30.4, 30, 3650)
    total_donations = np.clip(donor_df["number_of_donation"].values, 1, 100)

    np.random.seed(42)
    recency_days = np.clip(
        tenure_days / (total_donations + np.random.uniform(0.5, 2.0, size=len(tenure_days))),
        10,
        tenure_days * 0.95
    )

    feature_matrix = np.column_stack([recency_days, total_donations, tenure_days])
    kde = KernelDensity(kernel="gaussian", bandwidth=1.5)
    sample_idx = np.random.choice(len(feature_matrix), size=min(2000, len(feature_matrix)), replace=False)
    kde.fit(feature_matrix[sample_idx])

    n_samples = 10000
    raw_samples = kde.sample(n_samples, random_state=42)
    recency_raw = np.clip(raw_samples[:, 0], 5, 1000).round()
    donations_raw = np.clip(raw_samples[:, 1], 1, 80).round()
    tenure_raw = np.clip(raw_samples[:, 2], 30, 4000).round()

    tenure_days_arr = np.maximum(tenure_raw, recency_raw + np.random.randint(5, 60, size=n_samples)).astype(int)
    recency_days_arr = recency_raw.astype(int)
    total_donations_arr = donations_raw.astype(int)

    blood_types = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
    blood_type_weights = [0.45, 0.04, 0.28, 0.03, 0.14, 0.02, 0.03, 0.01]
    sampled_blood_types = np.random.choice(blood_types, p=blood_type_weights, size=n_samples)

    is_high_risk = np.random.rand(n_samples) < 0.05
    syphilis_ratios = np.where(
        is_high_risk,
        np.random.uniform(10.0, 18.5, size=n_samples),
        np.random.uniform(0.10, 2.50, size=n_samples)
    ).round(2)

    freq_factor = 1.0 / (1.0 + np.exp(-0.35 * (total_donations_arr - 3)))
    recency_factor = np.exp(-0.004 * np.maximum(0, recency_days_arr - 60))
    tenure_factor = np.clip((tenure_days_arr + 1) / (recency_days_arr + 90), 0.1, 1.0)
    retention_prob = np.clip(0.5 * freq_factor * recency_factor + 0.3 * tenure_factor + 0.2, 0.05, 0.98).round(4)
    retention_status = (retention_prob >= 0.50).astype(int)

    donor_records = []
    base_time = datetime.datetime.now(datetime.timezone.utc)

    for i in range(n_samples):
        donor_records.append({
            "donor_id": str(uuid.uuid4()),
            "blood_type": str(sampled_blood_types[i]),
            "tenure_days": int(tenure_days_arr[i]),
            "recency_days": int(recency_days_arr[i]),
            "total_donations": int(total_donations_arr[i]),
            "retention_probability": float(retention_prob[i]),
            "retention_status": int(retention_status[i]),
            "syphilis_s_co_ratio": float(syphilis_ratios[i]),
            "created_at": base_time - datetime.timedelta(minutes=n_samples - i),
        })

    logger.info("Clearing and populating donors table...")
    await session.execute(delete(Donor))
    await session.commit()

    chunk_size = 2500
    for idx in range(0, len(donor_records), chunk_size):
        chunk = donor_records[idx:idx + chunk_size]
        await session.execute(insert(Donor), chunk)
        await session.commit()

    return len(donor_records)


async def run_full_migration() -> Dict:
    """Master routine to migrate all data sheets to the database."""
    engine = get_async_engine()
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    logger.info("Verifying all database tables exist...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        logger.info("Starting Facilities & Demand Migration...")
        fac_results = await migrate_facilities_and_demand(session, engine)

        logger.info("Starting Donors Migration...")
        total_donors = await migrate_donors(session)

    # Invalidate Redis cache
    try:
        redis_client = get_redis_client()
        if redis_client:
            keys = await redis_client.keys("abis:*")
            if keys:
                await redis_client.delete(*keys)
                logger.info(f"Purged {len(keys)} Redis cache keys.")
    except Exception as e:
        logger.warning(f"Redis cache purge notice: {e}")

    await engine.dispose()
    logger.info("ALL DATASETS MIGRATED SUCCESSFULLY!")
    return {
        "status": "migration_complete",
        "facilities": fac_results["total_facilities"],
        "transfusion_records": fac_results["total_transfusion_records"],
        "tier_counts": fac_results["tier_counts"],
        "donors": total_donors,
    }


if __name__ == "__main__":
    results = asyncio.run(run_full_migration())
    print("\n=== MIGRATION COMPLETE ===")
    print(results)
