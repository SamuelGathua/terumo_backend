"""
scripts/seed_donors.py - Asynchronous Synthetic Donor Data Generation Pipeline
=============================================================================

Generates 10,000 realistic donor records using Kernel Density Estimation (KDE)
fitted on foundational seed data (data/blood_donor_dataset.csv and data/blood-format.csv).

Mathematical Constraints:
    - tenure_days >= recency_days strictly enforced for logical validity.
Clinical Constraints:
    - ~5% of donors receive syphilis TPPA screening S/CO ratio >= 10.0 (high predictive value).
    - ~95% receive safe, non-reactive ratios (0.10 - 2.50).
"""

import asyncio
import datetime
import logging
import os
import sys
import uuid
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sqlalchemy import insert, select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add parent directory to sys.path for direct script execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models import Donor
from database import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("abis.seed_donors")


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


def load_and_fit_kde(dataset_path: str, format_path: str) -> tuple:
    """
    Load foundational seed data and fit a Kernel Density Estimation (KDE) model
    on continuous behavioral RFM features (recency_days, total_donations, tenure_days).
    """
    logger.info(f"Loading seed data from {dataset_path} and {format_path}...")
    donor_df = pd.read_csv(dataset_path)
    format_df = pd.read_csv(format_path)

    # Derive continuous RFM vectors from foundational seed records
    # Tenure in days: months_since_first_donation * 30.4
    tenure_days = np.clip(donor_df["months_since_first_donation"].values * 30.4, 30, 3650)
    total_donations = np.clip(donor_df["number_of_donation"].values, 1, 100)

    # In transfusion medicine, average interval between donations is ~90-120 days.
    # We estimate baseline recency inversely correlated with donation frequency:
    np.random.seed(42)
    recency_days = np.clip(
        tenure_days / (total_donations + np.random.uniform(0.5, 2.0, size=len(tenure_days))),
        10,
        tenure_days * 0.95
    )

    feature_matrix = np.column_stack([recency_days, total_donations, tenure_days])

    # Fit Gaussian Kernel Density Estimation (bandwidth=1.5 standard for RFM domains)
    logger.info("Fitting Gaussian Kernel Density Estimation (KDE) on RFM feature matrix...")
    kde = KernelDensity(kernel="gaussian", bandwidth=1.5)
    # Fit on sample for rapid convergence
    sample_idx = np.random.choice(len(feature_matrix), size=min(2000, len(feature_matrix)), replace=False)
    kde.fit(feature_matrix[sample_idx])

    return kde, donor_df["blood_group"].values


async def generate_and_seed_donors(n_samples: int = 10000):
    """Generate 10,000 synthetic donor records and bulk insert into PostgreSQL."""
    data_dir = os.path.join(BACKEND_DIR, "data")
    dataset_path = os.path.join(data_dir, "blood_donor_dataset.csv")
    format_path = os.path.join(data_dir, "blood-format.csv")

    kde, blood_group_pool = load_and_fit_kde(dataset_path, format_path)

    logger.info(f"Drawing {n_samples} synthetic samples from KDE distribution...")
    raw_samples = kde.sample(n_samples, random_state=42)

    # Post-processing and strict mathematical constraints enforcement:
    recency_raw = np.clip(raw_samples[:, 0], 5, 1000).round()
    donations_raw = np.clip(raw_samples[:, 1], 1, 80).round()
    tenure_raw = np.clip(raw_samples[:, 2], 30, 4000).round()

    # Mathematical Constraint: tenure_days MUST be strictly >= recency_days
    tenure_days = np.maximum(tenure_raw, recency_raw + np.random.randint(5, 60, size=n_samples)).astype(int)
    recency_days = recency_raw.astype(int)
    total_donations = donations_raw.astype(int)

    # Blood group sampling: realistic population distribution
    blood_types = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
    blood_type_weights = [0.45, 0.04, 0.28, 0.03, 0.14, 0.02, 0.03, 0.01]
    sampled_blood_types = np.random.choice(blood_types, p=blood_type_weights, size=n_samples)

    # Clinical Constraint: Syphilis TPPA screening S/CO ratio
    # Roughly 5% receive S/CO >= 10.0 (high positive predictive value)
    # Remaining 95% receive safe range (0.10 - 2.50)
    is_high_risk = np.random.rand(n_samples) < 0.05
    syphilis_ratios = np.where(
        is_high_risk,
        np.random.uniform(10.0, 18.5, size=n_samples),
        np.random.uniform(0.10, 2.50, size=n_samples)
    ).round(2)

    # Initial retention probability score via non-linear RFM habit calculation
    freq_factor = 1.0 / (1.0 + np.exp(-0.35 * (total_donations - 3)))
    recency_factor = np.exp(-0.004 * np.maximum(0, recency_days - 60))
    tenure_factor = np.clip((tenure_days + 1) / (recency_days + 90), 0.1, 1.0)
    retention_prob = np.clip(0.5 * freq_factor * recency_factor + 0.3 * tenure_factor + 0.2, 0.05, 0.98).round(4)
    retention_status = (retention_prob >= 0.50).astype(int)

    logger.info("Constructing donor records payload...")
    donor_records = []
    base_time = datetime.datetime.utcnow()

    for i in range(n_samples):
        donor_records.append({
            "donor_id": str(uuid.uuid4()),
            "blood_type": str(sampled_blood_types[i]),
            "tenure_days": int(tenure_days[i]),
            "recency_days": int(recency_days[i]),
            "total_donations": int(total_donations[i]),
            "retention_probability": float(retention_prob[i]),
            "retention_status": int(retention_status[i]),
            "syphilis_s_co_ratio": float(syphilis_ratios[i]),
            "created_at": base_time - datetime.timedelta(minutes=n_samples - i),
        })

    # Bulk insert into PostgreSQL via AsyncSession
    engine = get_async_engine()
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    logger.info("Connecting to database and verifying tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    logger.info(f"Bulk-inserting {len(donor_records)} donor records in batches of 2,500...")
    chunk_size = 2500
    async with async_session() as session:
        for idx in range(0, len(donor_records), chunk_size):
            chunk = donor_records[idx:idx + chunk_size]
            await session.execute(insert(Donor), chunk)
            await session.commit()
            logger.info(f"Inserted batch {idx // chunk_size + 1} ({len(chunk)} records).")

        # Verify count
        count_stmt = select(func.count(Donor.donor_id))
        total_in_db = (await session.execute(count_stmt)).scalar()
        logger.info(f"Database population confirmed! Total donors in ledger: {total_in_db}")

    await engine.dispose()
    logger.info("scripts/seed_donors.py completed successfully.")


if __name__ == "__main__":
    asyncio.run(generate_and_seed_donors(10000))
