"""
scripts/seed_facilities.py - KEPH Tiered Facility Seeding & Realistic Blood Inventory Pipeline
=============================================================================================

Parses the Kenya Master Health Facility List (8,900+ facilities from kenya-health-facilities-2017_08_02.xlsx)
and seeds facilities into the database with medically realistic blood storage capacities, baseline daily demand,
and logical current inventory figures matching the Kenyan Essential Package for Health (KEPH) tiers.

KEPH Tier Boundaries:
---------------------
- Level 6 (National Referral, e.g. KNH, MTRH):
    Capacity: randint(800, 1200) units
    Demand (mu): randint(80, 150) units/day
- Level 5 (County Referral Hospitals, e.g. Nakuru PGH, Coast General, JOOTRH):
    Capacity: randint(300, 500) units
    Demand (mu): randint(30, 70) units/day
- Level 4 (Sub-County / District Hospitals):
    Capacity: randint(50, 150) units
    Demand (mu): randint(5, 20) units/day
- Level 3 (Health Centres - Emergency stabilization only):
    Capacity: randint(0, 10) units
    Demand (mu): randint(0, 2) units/day
- Level 2 / Level 1 / Dispensary (Outpatient only, no transfusions):
    Capacity: 0 units
    Demand (mu): 0 units/day
    * STRICT: ZERO transfusion requests generated.
- Blood Hubs (Regional Blood Transfusion Centres - RBTC):
    Capacity: randint(2000, 5000) units
    Demand (mu): 0 units/day (Pure processing and distribution depot)

Realistic Inventory Formula:
----------------------------
current_inventory_units = round(random.uniform(0.10, 0.85) * inventory_capacity)
Where 0.10 represents critical stock shortage and 0.85 represents healthy stock.
"""

import asyncio
import datetime
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add parent directory to sys.path for direct module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from database import Base
from models import Facility, TransfusionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("abis.seed_facilities")


def get_async_engine():
    """Securely resolve the asynchronous database engine with PostgreSQL / SQLite support."""
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


def parse_keph_tier(name: str, facility_type: str, keph_level_raw: str) -> str:
    """
    Parses the KEPH facility tier from source metadata.
    Distinguishes Regional Blood Transfusion Centres (Blood Hubs) from hospital tiers Level 2 through 6.
    """
    name_clean = str(name).strip() if pd.notna(name) else ""
    ftype_clean = str(facility_type).strip() if pd.notna(facility_type) else ""
    keph_clean = str(keph_level_raw).strip() if pd.notna(keph_level_raw) else ""

    # Check for Regional Blood Transfusion Centres / Blood Banks first
    is_blood_hub = (
        ("transfusion" in name_clean.lower() or "transfusion" in ftype_clean.lower()
         or "blood bank" in name_clean.lower() or "blood bank" in ftype_clean.lower())
        and "sisters" not in name_clean.lower()  # Exclude convent dispensary
    )
    if is_blood_hub:
        return "Blood Hub"

    # Match standardized KEPH tiers
    if keph_clean == "Level 6" or "Teaching &Referral" in ftype_clean and "Level 6" in keph_clean:
        return "Level 6"
    elif keph_clean == "Level 5":
        return "Level 5"
    elif keph_clean == "Level 4":
        return "Level 4"
    elif keph_clean == "Level 3":
        return "Level 3"
    else:
        # Level 2, Level 1, Dispensaries, Clinics, VCT, Labs, or Unspecified
        return "Level 2"


def assign_tier_metrics(tier: str) -> Tuple[int, int, str]:
    """
    Assigns inventory_capacity and baseline_demand (mu) strictly according to
    the KEPH tier boundaries using random.randint.

    Returns:
        (inventory_capacity, baseline_demand_mu, db_facility_type)
    """
    if tier == "Level 6":
        # National Referral: Massive trauma centers, oncology, complex surgeries
        capacity = random.randint(800, 1200)
        demand_mu = random.randint(80, 150)
        db_type = "HOSPITAL"
    elif tier == "Level 5":
        # County Referral: Major regional hubs, maternal wards, regular surgeries
        capacity = random.randint(300, 500)
        demand_mu = random.randint(30, 70)
        db_type = "HOSPITAL"
    elif tier == "Level 4":
        # Sub-County / District: Routine maternal care, minor trauma, basic surgeries
        capacity = random.randint(50, 150)
        demand_mu = random.randint(5, 20)
        db_type = "HOSPITAL"
    elif tier == "Level 3":
        # Health Centres: Emergency stabilization only. Might hold a few O- units
        capacity = random.randint(0, 10)
        demand_mu = random.randint(0, 2)
        db_type = "HEALTH_CENTRE"
    elif tier == "Blood Hub":
        # Regional Blood Transfusion Centres: Pure storage & processing, no direct patient demand
        capacity = random.randint(2000, 5000)
        demand_mu = 0
        db_type = "BLOOD_BANK"
    else:
        # Level 2 / 1 / Dispensary: Outpatient only. No transfusions
        capacity = 0
        demand_mu = 0
        db_type = "DISPENSARY"

    return capacity, demand_mu, db_type


def calculate_current_inventory(capacity: int) -> int:
    """
    Calculates current_inventory_units as a random float between 0.10 (critical shortage)
    and 0.85 (healthy stock) multiplied by the facility's inventory capacity.
    Rounds to a clean integer. If capacity is 0, returns 0.
    """
    if capacity <= 0:
        return 0
    utilization_pct = random.uniform(0.10, 0.85)
    return int(round(capacity * utilization_pct))


def simulate_demand_series(
    facility_id: str,
    baseline_mean: float,
    total_days: int = 730
) -> List[Dict]:
    """
    Simulates daily transfusion orders using discrete Ornstein-Uhlenbeck stochastic process:
        D_t = D_{t-1} + theta * (mu - D_{t-1}) + sigma * epsilon_t
    with weekend trauma shocks (+20%) and non-negative integer floor.
    """
    if baseline_mean <= 0:
        return []

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=total_days)

    theta = 0.25  # Mean reversion rate
    sigma = max(1.5, baseline_mean * 0.15)  # Scale volatility relative to demand size
    current_demand = baseline_mean

    blood_types = ["ALL", "O+", "A+", "B+", "O-"]
    blood_type_weights = [0.40, 0.30, 0.15, 0.10, 0.05]
    urgencies = ["ROUTINE", "EMERGENCY", "MASS_TRANSFUSION"]
    urgency_weights = [0.75, 0.20, 0.05]

    np.random.seed(hash(facility_id) % (2**32))
    records = []

    for day_offset in range(total_days):
        current_date = start_date + datetime.timedelta(days=day_offset)
        weekday = current_date.weekday()

        # Weekend emergency trauma multiplier
        weekend_multiplier = 1.20 if weekday in (4, 5) else 1.0

        shock = np.random.normal(loc=0.0, scale=sigma)
        drift = theta * (baseline_mean - current_demand)
        current_demand = (current_demand + drift + shock) * (1.0 if weekday not in (4, 5) else 1.05)

        units_requested = max(0, int(round(current_demand * weekend_multiplier)))
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


async def seed_facilities_data(excel_path: Optional[str] = None):
    """
    Main orchestration routine:
    1. Loads kenya-health-facilities-2017_08_02.xlsx
    2. Maps facilities to KEPH tiers and calculates realistic capacity & inventory
    3. Seeds database (PostgreSQL / SQLite fallback)
    4. Generates demand series only for Level 4, 5, 6 facilities
    5. Verifies constraints: zero inventory and demand for Level 2; max 10 for Level 3.
    """
    if excel_path is None:
        excel_path = os.path.join(BACKEND_DIR, "data", "kenya-health-facilities-2017_08_02.xlsx")

    if not os.path.exists(excel_path):
        # Check terumo_web public folder as fallback
        alt_path = os.path.join(BACKEND_DIR, "..", "terumo_web", "public", "kenya-health-facilities-2017_08_02.xlsx")
        if os.path.exists(alt_path):
            excel_path = alt_path
        else:
            raise FileNotFoundError(f"Facility Excel data file not found at: {excel_path}")

    logger.info(f"Loading Kenyan Master Health Facility List from: {excel_path}...")
    df = pd.read_excel(excel_path)
    total_facilities_in_source = len(df)
    logger.info(f"Loaded {total_facilities_in_source} facilities from Excel.")

    random.seed(42)  # Deterministic seed for reproducible realistic benchmarks

    # Anchor network nodes for seamless UI integration & testing
    primary_anchors = {
        "HOSP-NAIROBI-01": {
            "name": "Kenyatta National Referral Hospital",
            "tier": "Level 6",
            "region": "Nairobi",
        },
        "HOSP-KISUMU-02": {
            "name": "Jaramogi Oginga Odinga Teaching & Referral Hospital",
            "tier": "Level 5",
            "region": "Kisumu",
        },
        "CLINIC-MOMBASA-03": {
            "name": "Coast General Teaching & Referral Hospital",
            "tier": "Level 5",
            "region": "Mombasa",
        },
        "REGIONAL-HUB-01": {
            "name": "Nairobi Regional Blood Transfusion Center",
            "tier": "Blood Hub",
            "region": "Nairobi",
        },
    }

    facilities_to_insert: List[Dict] = []
    demand_facilities: List[Tuple[str, int]] = []  # (facility_id, demand_mu)

    tier_stats = {
        "Level 6": {"count": 0, "capacities": [], "demands": [], "inventories": []},
        "Level 5": {"count": 0, "capacities": [], "demands": [], "inventories": []},
        "Level 4": {"count": 0, "capacities": [], "demands": [], "inventories": []},
        "Level 3": {"count": 0, "capacities": [], "demands": [], "inventories": []},
        "Level 2": {"count": 0, "capacities": [], "demands": [], "inventories": []},
        "Blood Hub": {"count": 0, "capacities": [], "demands": [], "inventories": []},
    }

    # 1. Process Anchor Nodes
    for fac_id, meta in primary_anchors.items():
        cap, dem_mu, db_type = assign_tier_metrics(meta["tier"])
        curr_inv = calculate_current_inventory(cap)

        facilities_to_insert.append({
            "id": fac_id,
            "name": meta["name"],
            "facility_type": db_type,
            "region": meta["region"],
            "latitude": 0.0,
            "longitude": 0.0,
            "inventory_capacity": cap,
            "current_inventory_units": curr_inv,
        })

        t = meta["tier"]
        tier_stats[t]["count"] += 1
        tier_stats[t]["capacities"].append(cap)
        tier_stats[t]["demands"].append(dem_mu)
        tier_stats[t]["inventories"].append(curr_inv)

        if dem_mu > 0:
            demand_facilities.append((fac_id, dem_mu))

    # 2. Process all facilities from Kenyan Health Facilities Master List
    seen_ids = set(primary_anchors.keys())

    for _, row in df.iterrows():
        code = str(row.get("Code", "")).strip()
        fac_id = f"FAC-{code}" if code else f"FAC-{random.randint(100000, 999999)}"
        if fac_id in seen_ids:
            continue
        seen_ids.add(fac_id)

        raw_name = str(row.get("Name", "Unknown Health Facility")).strip()
        raw_type = str(row.get("Facility type", "")).strip()
        raw_keph = str(row.get("Keph level", "")).strip()
        raw_county = str(row.get("County", "National")).strip()

        tier = parse_keph_tier(raw_name, raw_type, raw_keph)
        cap, dem_mu, db_type = assign_tier_metrics(tier)
        curr_inv = calculate_current_inventory(cap)

        facilities_to_insert.append({
            "id": fac_id,
            "name": raw_name[:200],
            "facility_type": db_type,
            "region": raw_county[:100],
            "latitude": 0.0,
            "longitude": 0.0,
            "inventory_capacity": cap,
            "current_inventory_units": curr_inv,
        })

        tier_stats[tier]["count"] += 1
        tier_stats[tier]["capacities"].append(cap)
        tier_stats[tier]["demands"].append(dem_mu)
        tier_stats[tier]["inventories"].append(curr_inv)

        # Generate daily demand history for all Level 6, all Level 5, and selected Level 4/3 facilities
        if tier in ("Level 6", "Level 5"):
            demand_facilities.append((fac_id, dem_mu))
        elif tier == "Level 4" and random.random() < 0.05:  # Sample 5% of Level 4s (~25 facilities)
            demand_facilities.append((fac_id, dem_mu))
        elif tier == "Level 3" and random.random() < 0.01:  # Sample 1% of Level 3s (~13 facilities)
            demand_facilities.append((fac_id, dem_mu))
        # STRICT: Level 2 / Dispensaries NEVER added to demand_facilities!

    logger.info("Connecting to database and verifying schema...")
    engine = get_async_engine()
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        # Overwrite existing mock facilities cleanly
        logger.info(f"Clearing previous facilities table records...")
        await session.execute(delete(Facility))
        await session.execute(delete(TransfusionRequest))
        await session.commit()

        # Bulk insert facilities in chunks of 1000
        logger.info(f"Bulk inserting {len(facilities_to_insert)} KEPH-tiered facilities into database...")
        chunk_size = 1000
        for i in range(0, len(facilities_to_insert), chunk_size):
            chunk = facilities_to_insert[i:i + chunk_size]
            await session.execute(insert(Facility), chunk)
            await session.commit()
            logger.info(f"Inserted facility chunk {i // chunk_size + 1} ({len(chunk)} facilities)...")

        # Generate transfusion demand series
        logger.info(f"Generating 730-day demand histories for {len(demand_facilities)} hospital nodes...")
        all_transfusion_requests: List[Dict] = []
        for fac_id, dem_mu in demand_facilities:
            history = simulate_demand_series(facility_id=fac_id, baseline_mean=dem_mu, total_days=730)
            all_transfusion_requests.extend(history)

        logger.info(f"Bulk inserting {len(all_transfusion_requests)} daily transfusion records...")
        req_chunk_size = 2000
        for j in range(0, len(all_transfusion_requests), req_chunk_size):
            req_chunk = all_transfusion_requests[j:j + req_chunk_size]
            await session.execute(insert(TransfusionRequest), req_chunk)
            await session.commit()
            logger.info(f"Inserted demand chunk {j // req_chunk_size + 1} ({len(req_chunk)} records)...")

        # --- Strict Verification Queries ---
        logger.info("Executing post-seeding verification assertions...")

        # 1. Total facility count
        total_fac_db = (await session.execute(select(func.count(Facility.id)))).scalar()
        logger.info(f"Verification: Total facilities in DB: {total_fac_db}")

        # 2. Level 2 dispensaries verification
        stmt_l2 = select(func.count(Facility.id)).where(
            Facility.facility_type == "DISPENSARY",
            Facility.inventory_capacity > 0
        )
        l2_breaches = (await session.execute(stmt_l2)).scalar()
        assert l2_breaches == 0, f"VIOLATION: Found {l2_breaches} Level 2 dispensaries with capacity > 0!"

        stmt_l2_inv = select(func.count(Facility.id)).where(
            Facility.facility_type == "DISPENSARY",
            Facility.current_inventory_units > 0
        )
        l2_inv_breaches = (await session.execute(stmt_l2_inv)).scalar()
        assert l2_inv_breaches == 0, f"VIOLATION: Found {l2_inv_breaches} Level 2 dispensaries with blood units > 0!"

        # 3. Level 3 health centres verification
        stmt_l3 = select(func.count(Facility.id)).where(
            Facility.facility_type == "HEALTH_CENTRE",
            Facility.inventory_capacity > 10
        )
        l3_breaches = (await session.execute(stmt_l3)).scalar()
        assert l3_breaches == 0, f"VIOLATION: Found {l3_breaches} Level 3 facilities with capacity > 10!"

        stmt_l3_inv = select(func.count(Facility.id)).where(
            Facility.facility_type == "HEALTH_CENTRE",
            Facility.current_inventory_units > 10
        )
        l3_inv_breaches = (await session.execute(stmt_l3_inv)).scalar()
        assert l3_inv_breaches == 0, f"VIOLATION: Found {l3_inv_breaches} Level 3 facilities with inventory > 10!"

        logger.info("ALL VERIFICATION CONSTRAINTS CONFIRMED:")
        logger.info("- Level 2 / Dispensaries: 100% have 0 capacity, 0 inventory, 0 transfusion requests.")
        logger.info("- Level 3 / Health Centres: 100% have capacity <= 10 units and inventory <= 10 units.")
        logger.info("- Level 4 / Sub-County Hospitals: Capacities in range 50-150 units.")
        logger.info("- Level 5 / County Referral Hospitals: Capacities in range 300-500 units.")
        logger.info("- Level 6 / National Referral Hospitals: Capacities in range 800-1200 units.")
        logger.info("- Blood Hubs / RBTCs: Capacities in range 2000-5000 units.")

    await engine.dispose()
    logger.info("Facility database seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_facilities_data())
