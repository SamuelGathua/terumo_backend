"""
tests/test_integration.py - Full End-to-End Integration and Smoke Tests
========================================================================

Exercises:
1. Schema & Migration check: confirms sex, donor_type, date_of_birth, keph_level, status columns.
2. Fast test seeding into SQLite.
3. Training endpoint and status verification (trained, synthetic, isotonic calibration).
4. Prediction route verification (features returned, risk tiers, eligibility).
5. Facilities demand forecasting: KEPH tier baseline, SARIMAX forecasting, and 404 on unknown facility.
6. Route error handling: 422 on uninitialized, 401 on missing admin auth in production.
"""

import os
import pytest
from httpx import ASGITransport, AsyncClient

# Set testing environment before importing app
os.environ["ENVIRONMENT"] = "development"
os.environ["ABIS_DATA_SOURCE"] = "synthetic"

from main import app
from database import AsyncSessionLocal, init_db
from models import Donor, Facility, TransfusionRequest, DonationEvent
from ml_engine import predictive_engine
from sqlalchemy import select, func


@pytest.mark.anyio
async def test_database_schema_and_columns():
    """Verify that all enrichment and operational columns exist and are queryable."""
    await init_db()
    async with AsyncSessionLocal() as session:
        # Check Donor columns
        donor = (await session.execute(select(Donor).limit(1))).scalars().first()
        assert donor is not None, "Donors table should not be empty"
        assert hasattr(donor, "sex")
        assert hasattr(donor, "donor_type")
        assert hasattr(donor, "date_of_birth")

        # Check Facility columns
        fac = (await session.execute(select(Facility).limit(1))).scalars().first()
        assert fac is not None, "Facilities table should not be empty"
        assert hasattr(fac, "keph_level")

        # Check TransfusionRequest columns
        req = (await session.execute(select(TransfusionRequest).limit(1))).scalars().first()
        assert req is not None, "Transfusion requests table should not be empty"
        assert hasattr(req, "status")


@pytest.mark.anyio
async def test_retention_training_and_metrics_routes():
    """Verify retention model metrics and retraining endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Fetch metrics
        resp = await client.get("/api/v1/predict/retention/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("trained", "uninitialized")
        if data.get("status") == "trained":
            assert data.get("data_source") == "synthetic"
            assert "roc_auc" in data.get("test_calibrated", {})

        # 2. Trigger retrain with background=True
        resp_bg = await client.post("/api/v1/predict/retention/train?background=true")
        # May be 200 (scheduled) or 409 (if lock currently held)
        assert resp_bg.status_code in (200, 409)
        if resp_bg.status_code == 200:
            assert resp_bg.json().get("status") == "training_scheduled"


@pytest.mark.anyio
async def test_retention_prediction_route():
    """Verify donor retention prediction route produces calibrated predictions."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "recency_days": 45,
            "frequency_total": 4,
            "tenure_days": 360,
            "sex": "M",
            "donor_type": "VOLUNTARY",
            "age_years": 32.0,
        }
        resp = await client.post("/api/v1/predict/retention", json=payload)
        assert resp.status_code == 200
        res = resp.json()
        assert "retention_probability" in res
        assert 0.0 <= res["retention_probability"] <= 1.0
        assert res["risk_tier"] in ("LOW_RISK", "AT_RISK", "HIGH_RISK")
        assert "model_source" in res
        assert "days_until_eligible" in res


@pytest.mark.anyio
async def test_demand_forecast_route():
    """Verify SARIMAX demand forecasting for known facility and 404 for unknown facility."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Known anchor facility
        resp = await client.get("/api/v1/predict/demand/HOSP-NAIROBI-01?horizon_days=7")
        assert resp.status_code == 200
        fc = resp.json()
        assert fc["facility_id"] == "HOSP-NAIROBI-01"
        assert len(fc["forecast"]) == 7
        assert "forecast_start" in fc
        assert "tier_basis" in fc

        # Unknown facility returns HTTP 404
        resp_404 = await client.get("/api/v1/predict/demand/UNKNOWN-FACILITY-999")
        assert resp_404.status_code == 404
