import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from database import get_db, get_cached_json, set_cached_json
import models
import schemas
import ml_engine

logger = logging.getLogger("abis.routes")
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
    Uses the trained Random Forest classifier with heuristic fallback.
    """
    if ml_engine._donor_model is None:
        result = await db.execute(
            select(
                models.Donor.recency_days.label("recency"),
                models.Donor.total_donations.label("frequency"),
                models.Donor.tenure_days.label("tenure"),
                models.Donor.retention_status
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
        recency=payload.recency_days,
        frequency=payload.frequency_total,
        tenure=payload.tenure_days
    )
    return schemas.RetentionPredictionResponse(**prediction)


# --- Machine Learning: Demand Forecasting with Redis Caching ---
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
    Results are cached in Redis with a 15-minute (900 seconds) TTL to minimize redundant computation.
    """
    cache_key = f"abis:demand_forecast:{facility_id}:{horizon_days}"
    cached_data = await get_cached_json(cache_key)
    if cached_data:
        cached_data["cached"] = True
        return schemas.DemandForecastResponse(**cached_data)

    query = (
        select(models.TransfusionRequest.units_requested)
        .where(models.TransfusionRequest.requesting_facility_id == facility_id)
        .order_by(models.TransfusionRequest.request_date.asc())
        .limit(1000)
    )
    result = await db.execute(query)
    units_series = pd.Series([row[0] for row in result.all()])

    forecast_data = ml_engine.forecast_demand_arima(
        historical_series=units_series,
        horizon_days=horizon_days,
        facility_id=facility_id
    )
    forecast_data["cached"] = False

    # Cache for 15 minutes (900 seconds) in Redis
    await set_cached_json(cache_key, forecast_data, ttl_seconds=900)

    return schemas.DemandForecastResponse(**forecast_data)


# --- 1. Donors Endpoints ---
@router.post(
    "/donors/",
    response_model=schemas.DonorResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["1. Donors Ledger"]
)
async def create_donor(
    donor: schemas.DonorCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new donor profile in the central ledger."""
    db_donor = models.Donor(**donor.model_dump())
    db.add(db_donor)
    await db.commit()
    await db.refresh(db_donor)
    return db_donor


@router.get(
    "/donors/",
    response_model=List[schemas.DonorResponse],
    tags=["1. Donors Ledger"]
)
async def list_donors(
    blood_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List registered donors with pagination and optional blood type filter."""
    stmt = select(models.Donor).order_by(desc(models.Donor.created_at))
    if blood_type:
        stmt = stmt.where(models.Donor.blood_type == blood_type)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# --- 2. Donation Events Endpoints ---
@router.post(
    "/events/",
    response_model=schemas.DonationEventResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["2. Donation Events"]
)
async def record_donation_event(
    event: schemas.DonationEventCreate,
    db: AsyncSession = Depends(get_db)
):
    """Record a blood collection event from a donor drive or clinic."""
    db_event = models.DonationEvent(**event.model_dump(exclude_unset=True))
    db.add(db_event)
    await db.commit()
    await db.refresh(db_event)
    return db_event


@router.get(
    "/events/",
    response_model=List[schemas.DonationEventResponse],
    tags=["2. Donation Events"]
)
async def list_donation_events(
    location_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List donation collection events."""
    stmt = select(models.DonationEvent).order_by(desc(models.DonationEvent.collection_timestamp))
    if location_id:
        stmt = stmt.where(models.DonationEvent.location_id == location_id)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# --- 3. Screening Results Endpoints ---
@router.post(
    "/screening/",
    response_model=schemas.ScreeningResultResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["3. AI Diagnostic Screening"]
)
async def create_screening_result(
    result_in: schemas.ScreeningResultCreate,
    db: AsyncSession = Depends(get_db)
):
    """Log serological laboratory test result with automated TPPA prediction."""
    # Clinical heuristic: S/CO >= 10.0 yields 98.4% confirmatory positive
    auto_tppa = result_in.tppa_predicted_status or (result_in.syphilis_s_co_ratio >= 10.0)
    data = result_in.model_dump()
    data["tppa_predicted_status"] = auto_tppa

    db_result = models.ScreeningResult(**data)
    db.add(db_result)
    await db.commit()
    await db.refresh(db_result)
    return db_result


# --- 4. Inventory Units Endpoints ---
@router.post(
    "/inventory/",
    response_model=schemas.InventoryUnitResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["4. Inventory Management"]
)
async def create_inventory_unit(
    unit: schemas.InventoryUnitCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a physical barcode blood unit in the inventory ledger."""
    db_unit = models.InventoryUnit(**unit.model_dump())
    db.add(db_unit)
    await db.commit()
    await db.refresh(db_unit)
    return db_unit


@router.get(
    "/inventory/",
    response_model=List[schemas.InventoryUnitResponse],
    tags=["4. Inventory Management"]
)
async def list_inventory_units(
    facility_id: Optional[str] = None,
    product_type: Optional[str] = None,
    status_filter: Optional[str] = Query("AVAILABLE", alias="status"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Query inventory units by facility, product type, and status."""
    stmt = select(models.InventoryUnit)
    if facility_id:
        stmt = stmt.where(models.InventoryUnit.current_facility_id == facility_id)
    if product_type:
        stmt = stmt.where(models.InventoryUnit.product_type == product_type)
    if status_filter:
        stmt = stmt.where(models.InventoryUnit.status == status_filter)
    stmt = stmt.order_by(models.InventoryUnit.expiry_date.asc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# --- 5. Transfusion Requests Endpoints ---
@router.post(
    "/transfusion-requests/",
    response_model=schemas.TransfusionRequestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["5. Transfusion Demand Engine"]
)
async def create_transfusion_request(
    request_in: schemas.TransfusionRequestCreate,
    db: AsyncSession = Depends(get_db)
):
    """Record an incoming hospital blood demand order."""
    db_request = models.TransfusionRequest(**request_in.model_dump())
    db.add(db_request)
    await db.commit()
    await db.refresh(db_request)
    return db_request


@router.get(
    "/transfusion-requests/",
    response_model=List[schemas.TransfusionRequestResponse],
    tags=["5. Transfusion Demand Engine"]
)
async def list_transfusion_requests(
    facility_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List historical hospital blood orders."""
    stmt = select(models.TransfusionRequest).order_by(desc(models.TransfusionRequest.request_date))
    if facility_id:
        stmt = stmt.where(models.TransfusionRequest.requesting_facility_id == facility_id)
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
    """Asynchronously ingest offline batches captured by the Flutter field app at donor drives."""
    synced_count = 0
    breaches_flagged = 0

    for item in batch.units:
        stmt = select(models.InventoryUnit).where(models.InventoryUnit.unit_id == item.unit_barcode)
        existing = (await db.execute(stmt)).scalar_one_or_none()

        is_breach = item.cold_chain_breach or item.temperature_celsius > 10.0 or item.temperature_celsius < 1.0
        if is_breach:
            breaches_flagged += 1

        if existing:
            existing.status = item.status
            existing.current_facility_id = item.current_facility_id
        else:
            new_unit = models.InventoryUnit(
                unit_id=item.unit_barcode,
                event_id="BATCH-SYNC-EVENT",
                product_type="WHOLE_BLOOD",
                expiry_date=datetime.datetime.combine(item.expiry_date, datetime.time.min),
                current_facility_id=item.current_facility_id,
                status=item.status,
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


# --- Network Rebalancing Suggestion Endpoint with Redis Caching ---
@router.get(
    "/rebalance/suggestions",
    tags=["Liquidity Rebalancing Engine"]
)
async def get_rebalancing_suggestions(
    db: AsyncSession = Depends(get_db)
):
    """Decentralized inventory rebalancing algorithm with Redis caching."""
    cache_key = "abis:rebalance:suggestions"
    cached = await get_cached_json(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    facilities_query = select(models.Facility)
    result = await db.execute(facilities_query)
    facilities = result.scalars().all()

    if not facilities:
        return {"status": "no_facilities_configured", "suggestions": []}

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

    response_data = {
        "network_status": "rebalancing_computed",
        "active_shortage_facilities": len(shortages),
        "active_surplus_facilities": len(surpluses),
        "recommended_transfers": suggestions,
        "cached": False
    }

    # Cache for 5 minutes (300 seconds)
    await set_cached_json(cache_key, response_data, ttl_seconds=300)
    return response_data
