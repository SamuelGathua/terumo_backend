import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from database import get_db, get_cached_json, set_cached_json
import models
import schemas
from ml_engine import predictive_engine

logger = logging.getLogger("abis.routes")
router = APIRouter()

# --- Health Check Endpoint ---
@router.get("/healthz/", tags=["System"])
async def health_check_endpoint():
    """Health check endpoint to verify service and model readiness."""
    return {
        "status": "ok",
        "service": "Adaptive Blood Infrastructure System (ABIS)",
        "model_trained": predictive_engine.is_trained,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# --- Task 1: Predictive Retention Engine ---
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
    Uses the trained Random Forest classifier with clinical risk tiering.
    """
    # Ensure model is initialized if not yet trained
    if not predictive_engine.is_trained:
        await predictive_engine.initialize_from_db()

    prediction = predictive_engine.predict_donor_retention(
        recency_days=payload.recency_days,
        total_donations=payload.frequency_total,
        tenure_days=payload.tenure_days
    )
    return schemas.RetentionPredictionResponse(**prediction)


@router.get(
    "/predict/retention/metrics",
    tags=["Predictive Intelligence"]
)
async def get_retention_model_metrics():
    """Retrieve evaluation metrics for the trained Random Forest donor retention model."""
    if not predictive_engine.is_trained:
        await predictive_engine.initialize_from_db()
    return predictive_engine.model_metrics


# --- Task 2: Time-Series Demand Forecasting with Redis Caching ---
@router.get(
    "/predict/demand/{facility_id}",
    response_model=schemas.DemandForecastResponse,
    tags=["Predictive Intelligence"]
)
async def predict_blood_demand_by_facility(
    facility_id: str = Path(..., description="Unique requesting facility ID (e.g. HOSP-NAIROBI-01)"),
    horizon_days: int = Query(7, ge=1, le=30, description="Forecast horizon in days (default: 7)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Project daily blood demand for the next 7 days using ARIMA time-series modeling.
    Wrapped in an asynchronous Redis cache layer with a 15-minute (900 seconds) TTL.
    """
    cache_key = f"abis:demand_forecast:{facility_id}:{horizon_days}"

    # 1. Check Redis Cache first
    cached_result = await get_cached_json(cache_key)
    if cached_result:
        cached_result["cached"] = True
        logger.info(f"Redis Cache HIT for key '{cache_key}'")
        return schemas.DemandForecastResponse(**cached_result)

    logger.info(f"Redis Cache MISS for key '{cache_key}'. Executing ARIMA time-series forecast...")

    # 2. Compute ARIMA Forecast via ml_engine
    forecast_data = await predictive_engine.forecast_facility_demand(
        facility_id=facility_id,
        horizon_days=horizon_days
    )
    forecast_data["cached"] = False

    # 3. Store JSON result in Redis with a 900-second (15 minutes) TTL
    await set_cached_json(cache_key, forecast_data, ttl_seconds=900)

    return schemas.DemandForecastResponse(**forecast_data)


@router.get(
    "/predict/demand",
    response_model=schemas.DemandForecastResponse,
    tags=["Predictive Intelligence"]
)
async def predict_blood_demand_query(
    facility_id: str = Query("HOSP-NAIROBI-01", description="Hospital / Facility ID"),
    horizon_days: int = Query(7, ge=1, le=30, description="Forecast horizon in days"),
    db: AsyncSession = Depends(get_db)
):
    """Query parameter variant of demand forecast endpoint with Redis caching."""
    return await predict_blood_demand_by_facility(facility_id=facility_id, horizon_days=horizon_days, db=db)


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


# --- Network Rebalancing Suggestion Endpoint ---
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

    await set_cached_json(cache_key, response_data, ttl_seconds=300)
    return response_data


# --- Database Administration & Dataset Migration ---
@router.post(
    "/admin/migrate",
    tags=["Database Administration"]
)
@router.get(
    "/admin/migrate",
    tags=["Database Administration"]
)
async def trigger_database_migration():
    """
    Migrates all foundational data sheets from the data/ folder to the PostgreSQL server database:
    1. kenya-health-facilities-2017_08_02.xlsx (8,932+ Kenyan health facilities with KEPH tiers)
    2. blood_donor_dataset.csv (10,000 KDE-fitted donors)
    3. 730-day (2-year) Ornstein-Uhlenbeck daily transfusion requests matching KEPH tier boundaries
    4. Purges all Redis cache keys.
    """
    from scripts.migrate_all_data import run_full_migration
    results = await run_full_migration()
    try:
        await predictive_engine.initialize_from_db()
    except Exception as e:
        logger.warning(f"Notice re-initializing predictive engine: {e}")
    return results


@router.post(
    "/admin/purge-cache",
    tags=["Database Administration"]
)
@router.get(
    "/admin/purge-cache",
    tags=["Database Administration"]
)
async def purge_redis_cache():
    """Invalidate all cached demand forecasts and rebalancing suggestions."""
    from database import get_redis_client
    try:
        client = get_redis_client()
        if client:
            keys = await client.keys("abis:*")
            if keys:
                await client.delete(*keys)
                return {"status": "cache_purged", "keys_deleted": len(keys)}
        return {"status": "cache_clean", "keys_deleted": 0}
    except Exception as e:
        return {"status": "cache_purge_notice", "error": str(e)}

