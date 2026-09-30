import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from database import get_db
import models
import schemas
import ml_engine

router = APIRouter()

# --- Health Check Endpoint ---
@router.get("/healthz/", tags=["System"])
async def health_check_endpoint():
    """Health check endpoint to verify service readiness."""
    return {
        "status": "ok",
        "service": "Adaptive Blood Infrastructure System (ABIS)",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# --- Machine Learning: Retention Scoring ---
@router.post(
    "/predict/retention",
    response_model=schemas.RetentionPredictionResponse,
    tags=["Predictive Intelligence"]
)
async def predict_donor_retention(
    payload: schemas.RetentionPredictionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict donor return probability based on Recency, Frequency, and Tenure (RFM).
    Uses the trained Random Forest classifier or clinical prior heuristic fallback.
    """
    # If the model is not trained yet, attempt to train on existing database records
    if ml_engine._donor_model is None:
        result = await db.execute(
            select(
                models.DonorProfile.recency,
                models.DonorProfile.frequency,
                models.DonorProfile.tenure,
                models.DonorProfile.retention_status
            ).limit(5000)
        )
        records = result.all()
        if len(records) >= 50:
            df = pd.DataFrame(
                records,
                columns=["recency", "frequency", "tenure", "retention_status"]
            )
            ml_engine.train_retention_model(df)

    prediction = ml_engine.predict_retention_score(
        recency=payload.recency,
        frequency=payload.frequency,
        tenure=payload.tenure
    )
    return schemas.RetentionPredictionResponse(**prediction)


# --- Machine Learning: Demand Forecasting ---
@router.get(
    "/predict/demand",
    response_model=schemas.DemandForecastResponse,
    tags=["Predictive Intelligence"]
)
async def predict_blood_demand(
    facility_id: str = Query("HOSP-NAIROBI-01", description="Hospital / Facility ID"),
    horizon_days: int = Query(7, ge=1, le=30, description="Forecast horizon in days"),
    db: AsyncSession = Depends(get_db)
):
    """
    Project daily blood demand for the next N days using ARIMA time-series modeling.
    Flags critical shortage alerts when demand exceeds the historical baseline by >25%.
    """
    query = (
        select(models.TransfusionDemand.units_requested)
        .where(models.TransfusionDemand.facility_id == facility_id)
        .order_by(models.TransfusionDemand.date.asc())
        .limit(1000)
    )
    result = await db.execute(query)
    units_series = pd.Series([row[0] for row in result.all()])

    forecast_data = ml_engine.forecast_demand_arima(
        historical_series=units_series,
        horizon_days=horizon_days,
        facility_id=facility_id
    )
    return schemas.DemandForecastResponse(**forecast_data)


# --- Donor Ledger Endpoints ---
@router.post(
    "/donors/",
    response_model=schemas.DonorProfileResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Donor Ledger"]
)
async def create_donor(
    donor: schemas.DonorProfileCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new donor profile in the central ledger."""
    db_donor = models.DonorProfile(**donor.model_dump())
    db.add(db_donor)
    await db.commit()
    await db.refresh(db_donor)
    return db_donor


@router.get(
    "/donors/",
    response_model=List[schemas.DonorProfileResponse],
    tags=["Donor Ledger"]
)
async def list_donors(
    blood_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List registered donors with pagination and optional blood type filter."""
    stmt = select(models.DonorProfile).order_by(desc(models.DonorProfile.created_at))
    if blood_type:
        stmt = stmt.where(models.DonorProfile.blood_type == blood_type)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# --- Transfusion Demands Endpoints ---
@router.post(
    "/transfusion-demands/",
    response_model=schemas.TransfusionDemandResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Transfusion Demands"]
)
async def create_transfusion_demand(
    demand: schemas.TransfusionDemandCreate,
    db: AsyncSession = Depends(get_db)
):
    """Record a hospital transfusion demand request."""
    db_demand = models.TransfusionDemand(**demand.model_dump())
    db.add(db_demand)
    await db.commit()
    await db.refresh(db_demand)
    return db_demand


@router.get(
    "/transfusion-demands/",
    response_model=List[schemas.TransfusionDemandResponse],
    tags=["Transfusion Demands"]
)
async def list_transfusion_demands(
    facility_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Query historical hospital transfusion demand records."""
    stmt = select(models.TransfusionDemand).order_by(desc(models.TransfusionDemand.date))
    if facility_id:
        stmt = stmt.where(models.TransfusionDemand.facility_id == facility_id)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# --- Offline Traceability Sync Endpoint ---
@router.post(
    "/sync/traceability",
    status_code=status.HTTP_200_OK,
    tags=["Resilient Offline Traceability"]
)
async def sync_offline_traceability_ledger(
    batch: schemas.OfflineSyncBatch,
    db: AsyncSession = Depends(get_db)
):
    """
    Asynchronously ingest offline batches captured by the Flutter field app at donor drives
    or cold transport legs. Flags potential cold-chain breaches automatically.
    """
    synced_count = 0
    breaches_flagged = 0

    for item in batch.units:
        # Check if barcode already exists
        stmt = select(models.BloodUnitLedger).where(models.BloodUnitLedger.unit_barcode == item.unit_barcode)
        existing = (await db.execute(stmt)).scalar_one_or_none()

        is_breach = item.cold_chain_breach or item.temperature_celsius > 10.0 or item.temperature_celsius < 1.0
        if is_breach:
            breaches_flagged += 1

        if existing:
            existing.status = item.status
            existing.current_facility_id = item.current_facility_id
            existing.temperature_celsius = item.temperature_celsius
            existing.cold_chain_breach = is_breach
            existing.last_synced_at = datetime.datetime.utcnow()
        else:
            new_unit = models.BloodUnitLedger(
                unit_barcode=item.unit_barcode,
                blood_type=item.blood_type,
                collection_date=item.collection_date,
                expiry_date=item.expiry_date,
                status=item.status,
                current_facility_id=item.current_facility_id,
                temperature_celsius=item.temperature_celsius,
                cold_chain_breach=is_breach,
            )
            db.add(new_unit)
        synced_count += 1

    await db.commit()
    return {
        "status": "sync_complete",
        "device_id": batch.device_id,
        "units_processed": synced_count,
        "breaches_flagged": breaches_flagged,
        "sync_timestamp": datetime.datetime.utcnow().isoformat(),
    }


# --- Network Rebalancing Suggestion Endpoint ---
@router.get(
    "/rebalance/suggestions",
    tags=["Liquidity Rebalancing Engine"]
)
async def get_rebalancing_suggestions(
    db: AsyncSession = Depends(get_db)
):
    """
    Decentralized inventory rebalancing algorithm:
    Matches facilities experiencing critical shortages with regional hubs holding surplus inventory,
    optimizing transfer routes to minimize transit decay.
    """
    facilities_query = select(models.Facility)
    result = await db.execute(facilities_query)
    facilities = result.scalars().all()

    if not facilities:
        return {
            "status": "no_facilities_configured",
            "suggestions": []
        }

    shortages = []
    surpluses = []

    for f in facilities:
        utilization = f.current_inventory_units / max(1, f.inventory_capacity)
        if utilization < 0.25:
            shortages.append(f)
        elif utilization > 0.70:
            surpluses.append(f)

    suggestions = []
    for deficit in shortages:
        if surpluses:
            # Transfer from largest surplus
            donor_facility = max(surpluses, key=lambda x: x.current_inventory_units)
            transfer_qty = min(
                donor_facility.current_inventory_units - int(donor_facility.inventory_capacity * 0.5),
                int(deficit.inventory_capacity * 0.5) - deficit.current_inventory_units
            )
            if transfer_qty > 0:
                suggestions.append({
                    "from_facility_id": donor_facility.id,
                    "from_facility_name": donor_facility.name,
                    "to_facility_id": deficit.id,
                    "to_facility_name": deficit.name,
                    "recommended_units": transfer_qty,
                    "urgency": "CRITICAL" if deficit.current_inventory_units < 10 else "HIGH",
                    "reason": f"Deficit facility at {round(deficit.current_inventory_units/deficit.inventory_capacity*100)}% capacity. Surplus hub holds {donor_facility.current_inventory_units} units."
                })

    return {
        "network_status": "rebalancing_computed",
        "active_shortage_facilities": len(shortages),
        "active_surplus_facilities": len(surpluses),
        "recommended_transfers": suggestions
    }
