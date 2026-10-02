"""
scripts/seed_donors.py - Asynchronous Event-First Synthetic Donor Ledger Seeder
=============================================================================

Generates 10,000 realistic donor records and their longitudinal donation event
history using the latent-behaviour simulator (scripts/donor_simulation.py).

Features and labels for predictive intelligence are derived dynamically from
actual donation events rather than hardcoded synthetic formulas.

Safety:
    Guarded by scripts/db_utils.assert_safe_target(): refuses to wipe tables on
    production/remote PostgreSQL databases unless ALLOW_DB_RESET=1 is explicitly set.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Ensure backend root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from database import Base, apply_column_migrations
from models import DonationEvent, Donor
from scripts.db_utils import assert_safe_target, get_async_engine
from scripts.donor_simulation import SimConfig, simulate_donor_ledger, validate_ledger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("abis.seed_donors")


async def generate_and_seed_donors(
    n_donors: int = 10000,
    history_days: int = 1460,
    seed: int = 42,
) -> dict:
    """
    Simulates longitudinal donor behavior and seeds donors and donation_events.
    """
    engine = get_async_engine()

    # Safety Guard: refuse destructive wiping on non-SQLite databases without explicit confirmation
    assert_safe_target(engine)

    logger.info("Initializing database schema and checking enrichment columns...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(apply_column_migrations)

    # 1. Generate longitudinal behavior ledger
    logger.info(f"Simulating event-first ledger for {n_donors} donors over {history_days} days (seed={seed})...")
    cfg = SimConfig(n_donors=n_donors, history_days=history_days, seed=seed)
    donors, events = simulate_donor_ledger(cfg)

    # 2. Validate hard invariants
    logger.info("Validating ledger invariants...")
    stats = validate_ledger(donors, events)
    logger.info("Ledger validated successfully: %s", stats)

    # 3. Bulk insert via AsyncSession
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        logger.info("Clearing existing donation_events and donors records...")
        await session.execute(delete(DonationEvent))
        await session.execute(delete(Donor))
        await session.commit()

        # Insert Donors
        logger.info(f"Bulk-inserting {len(donors)} donor profiles in batches of 2,500...")
        donor_batch_size = 2500
        for i in range(0, len(donors), donor_batch_size):
            chunk = donors[i : i + donor_batch_size]
            await session.execute(insert(Donor), chunk)
            await session.commit()
            logger.info(f"Inserted donor batch {i // donor_batch_size + 1} ({len(chunk)} records).")

        # Insert Donation Events
        logger.info(f"Bulk-inserting {len(events)} donation events in batches of 5,000...")
        event_batch_size = 5000
        for j in range(0, len(events), event_batch_size):
            chunk = events[j : j + event_batch_size]
            await session.execute(insert(DonationEvent), chunk)
            await session.commit()
            logger.info(f"Inserted event batch {j // event_batch_size + 1} ({len(chunk)} records).")

        # Verification Queries
        total_donors = (await session.execute(select(func.count(Donor.donor_id)))).scalar()
        total_events = (await session.execute(select(func.count(DonationEvent.event_id)))).scalar()
        synced_events = (
            await session.execute(
                select(func.count(DonationEvent.event_id)).where(DonationEvent.sync_status == "SYNCED")
            )
        ).scalar()

        logger.info(
            f"Seeding complete! Database online with {total_donors} donors, "
            f"{total_events} total events ({synced_events} SYNCED)."
        )

    await engine.dispose()
    return {
        "donors_seeded": total_donors,
        "events_seeded": total_events,
        "synced_events": synced_events,
        "validation_stats": stats,
    }


if __name__ == "__main__":
    asyncio.run(generate_and_seed_donors())
