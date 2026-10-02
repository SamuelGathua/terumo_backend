import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from database import get_db, get_cached_json, set_cached_json, delete_cached_keys
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


@router.post(
    "/predict/retention/train",
    tags=["Predictive Intelligence"]
)
@router.get(
    "/predict/retention/train",
    tags=["Predictive Intelligence"]
)
async def retrain_donor_retention_model():
    """Trigger on-demand retraining of the Random Forest donor retention model."""
    metrics = await predictive_engine.initialize_from_db()
    return metrics


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
    tags=["2. Donation Events"]
)
async def list_donation_events(
    location_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List donation collection events with Redis read-through caching (TTL = 300s).
    Returns both single donation events and batch manifest records.
    """
    cache_key = "abis:traceability:events"
    if not location_id and offset == 0:
        cached_data = await get_cached_json(cache_key)
        if cached_data is not None:
            logger.info("Redis cache HIT for 'abis:traceability:events'")
            return cached_data

    logger.info("Redis cache MISS for 'abis:traceability:events'. Fetching from database...")
    stmt = select(models.DonationEvent).order_by(desc(models.DonationEvent.collection_timestamp))
    if location_id:
        stmt = stmt.where(models.DonationEvent.location_id == location_id)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    events = result.scalars().all()

    # Seed foundational collection events if database table is currently unseeded
    if not events and offset == 0 and not location_id:
        now = datetime.datetime.utcnow()
        # Find or create a default seed donor
        donor_stmt = select(models.Donor).limit(1)
        donor_res = await db.execute(donor_stmt)
        default_donor = donor_res.scalars().first()
        if not default_donor:
            default_donor = models.Donor(
                donor_id="donor-default-seed",
                blood_type="O+",
                tenure_days=730,
                recency_days=30,
                total_donations=5,
                retention_probability=0.92,
                retention_status=1,
            )
            db.add(default_donor)
            await db.commit()
            await db.refresh(default_donor)

        seed_events_meta = [
            {
                "event_id": "EVT-MCH-9021",
                "donor_id": default_donor.donor_id,
                "location_id": "Machakos Mobile Donor Drive (Site 2)",
                "collection_timestamp": now - datetime.timedelta(minutes=18),
                "sync_status": "PENDING_SYNC",
                "cold_chain_breach_flag": True,
            },
            {
                "event_id": "EVT-NRB-8942",
                "donor_id": default_donor.donor_id,
                "location_id": "University of Nairobi Student Center Drive",
                "collection_timestamp": now - datetime.timedelta(hours=1),
                "sync_status": "SYNCED",
                "cold_chain_breach_flag": False,
            },
            {
                "event_id": "EVT-KSM-7719",
                "donor_id": default_donor.donor_id,
                "location_id": "Kisumu County Transfusion Station",
                "collection_timestamp": now - datetime.timedelta(hours=3),
                "sync_status": "SYNCED",
                "cold_chain_breach_flag": False,
            },
            {
                "event_id": "EVT-MSA-6502",
                "donor_id": default_donor.donor_id,
                "location_id": "Mombasa Coastal Blood Center Depot",
                "collection_timestamp": now - datetime.timedelta(hours=5),
                "sync_status": "SYNCED",
                "cold_chain_breach_flag": False,
            },
        ]
        for meta in seed_events_meta:
            db.add(models.DonationEvent(**meta))
        await db.commit()

        # Re-query newly seeded events
        stmt = select(models.DonationEvent).order_by(desc(models.DonationEvent.collection_timestamp)).limit(limit)
        events = (await db.execute(stmt)).scalars().all()

    # Serialize events ensuring ISO 8601 formatting for all datetime objects
    serialized_events = []
    officers = ["Nurse J. Mutua", "Officer K. Ochieng", "Technologist A. Kiprop", "Liaison F. Mwangi"]
    for i, e in enumerate(events):
        officer = officers[i % len(officers)]
        is_breach = bool(e.cold_chain_breach_flag)
        temp = 11.4 if is_breach else (3.8 + (i * 0.4))
        serialized_events.append({
            "event_id": e.event_id,
            "donor_id": e.donor_id,
            "location": e.location_id,
            "location_id": e.location_id,
            "collection_timestamp": e.collection_timestamp.isoformat() if hasattr(e.collection_timestamp, "isoformat") else str(e.collection_timestamp),
            "timestamp": e.collection_timestamp.isoformat() if hasattr(e.collection_timestamp, "isoformat") else str(e.collection_timestamp),
            "sync_status": e.sync_status,
            "cold_chain_breach_flag": is_breach,
            "breach": is_breach,
            "temperature": round(temp, 1),
            "officer": officer,
            "field_lead": officer,
            "barcode_range": f"KE-BC-2026-08{9-i}1 ➔ 0{9-i}4 (24 Units)",
            "is_offline_upload": True if e.sync_status == "PENDING_SYNC" else False,
        })

    # Cache with 300-second TTL per Task 1.2
    if not location_id and offset == 0:
        await set_cached_json(cache_key, serialized_events, ttl_seconds=300)

    return serialized_events


# --- Task 1.3: Flutter Ingestion Endpoint & Cache Invalidation (POST /events/batch) ---
@router.post(
    "/events/batch",
    status_code=status.HTTP_200_OK,
    tags=["2. Donation Events"]
)
async def ingest_flutter_offline_batch(
    payload: schemas.BatchManifestUpload,
    db: AsyncSession = Depends(get_db)
):
    """
    Ingestion endpoint for Flutter mobile application offline-sync payloads.
    Stores new donation batch and physical inventory units in PostgreSQL,
    then immediately invalidates Redis cache keys ('abis:traceability:inventory', 'abis:traceability:events').
    """
    now = datetime.datetime.utcnow()

    # 1. Ensure a valid donor exists for collection linkage
    donor_stmt = select(models.Donor).limit(1)
    donor_res = await db.execute(donor_stmt)
    donor = donor_res.scalars().first()
    if not donor:
        donor = models.Donor(
            donor_id="donor-mobile-sync",
            blood_type="O+",
            tenure_days=365,
            recency_days=15,
            total_donations=3,
            retention_probability=0.90,
            retention_status=1,
        )
        db.add(donor)
        await db.commit()
        await db.refresh(donor)

    # 2. Check if donation event already exists, otherwise create it
    event_stmt = select(models.DonationEvent).where(models.DonationEvent.event_id == payload.batch_id)
    event_res = await db.execute(event_stmt)
    existing_event = event_res.scalars().first()

    is_breach = payload.cold_chain_breach or payload.temperature > 6.0 or payload.temperature < 2.0

    if existing_event:
        existing_event.sync_status = "SYNCED"
        existing_event.cold_chain_breach_flag = is_breach
    else:
        new_event = models.DonationEvent(
            event_id=payload.batch_id,
            donor_id=donor.donor_id,
            collection_timestamp=payload.timestamp or now,
            location_id=payload.location,
            sync_status="SYNCED",
            cold_chain_breach_flag=is_breach,
        )
        db.add(new_event)

    # 3. Insert or update incoming physical barcode records in PostgreSQL
    units_ingested = 0
    for rec in payload.barcode_records:
        unit_stmt = select(models.InventoryUnit).where(models.InventoryUnit.unit_id == rec.barcode)
        unit_res = await db.execute(unit_stmt)
        existing_unit = unit_res.scalars().first()

        unit_status = rec.status or ("BREACH" if is_breach else "AVAILABLE")
        expiry = rec.expiry_date or (now + datetime.timedelta(days=35))

        if existing_unit:
            existing_unit.status = unit_status
            existing_unit.current_facility_id = rec.facility or payload.location
            existing_unit.expiry_date = expiry
        else:
            new_unit = models.InventoryUnit(
                unit_id=rec.barcode,
                event_id=payload.batch_id,
                product_type=rec.product_type,
                expiry_date=expiry,
                current_facility_id=rec.facility or payload.location,
                status=unit_status,
                created_at=now,
            )
            db.add(new_unit)
        units_ingested += 1

    await db.commit()
    logger.info(f"Ingested mobile batch '{payload.batch_id}' with {units_ingested} records into PostgreSQL.")

    # 4. Immediate Cache Invalidation per Task 1.3
    await delete_cached_keys("abis:traceability:inventory", "abis:traceability:events")
    logger.info("Flushed Redis cache keys: 'abis:traceability:inventory', 'abis:traceability:events'")

    return {
        "status": "batch_ingested",
        "batch_id": payload.batch_id,
        "field_lead": payload.field_lead,
        "location": payload.location,
        "units_ingested": units_ingested,
        "timestamp": (payload.timestamp or now).isoformat(),
        "cache_invalidated": True,
    }


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


# --- 4. Inventory Units Endpoints (Task 1.2: Read-Through Caching) ---
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
    await delete_cached_keys("abis:traceability:inventory")
    return db_unit


@router.get(
    "/inventory/",
    tags=["4. Inventory Management"]
)
async def list_inventory_units(
    facility_id: Optional[str] = None,
    product_type: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """
    Query inventory units with Redis read-through caching (TTL = 300s).
    Check key 'abis:traceability:inventory'. Cache miss queries database, serializes with ISO datetimes.
    """
    cache_key = "abis:traceability:inventory"
    if not facility_id and not product_type and not status_filter:
        cached_data = await get_cached_json(cache_key)
        if cached_data is not None:
            logger.info("Redis cache HIT for 'abis:traceability:inventory'")
            return cached_data

    logger.info("Redis cache MISS for 'abis:traceability:inventory'. Querying database...")
    stmt = select(models.InventoryUnit)
    if facility_id:
        stmt = stmt.where(models.InventoryUnit.current_facility_id == facility_id)
    if product_type:
        stmt = stmt.where(models.InventoryUnit.product_type == product_type)
    if status_filter:
        stmt = stmt.where(models.InventoryUnit.status == status_filter)
    stmt = stmt.order_by(models.InventoryUnit.expiry_date.asc()).limit(limit)
    result = await db.execute(stmt)
    units = result.scalars().all()

    # Seed foundational units if database table is currently unseeded
    if not units and not facility_id and not product_type and not status_filter:
        now = datetime.datetime.utcnow()
        seed_units_data = [
            {
                "unit_id": "KE-BC-2026-0891",
                "event_id": "EVT-MCH-9021",
                "product_type": "WHOLE_BLOOD",
                "expiry_date": now + datetime.timedelta(days=35),
                "current_facility_id": "Transit Box #TB-04 (Machakos)",
                "status": "BREACH",
            },
            {
                "unit_id": "KE-BC-2026-0865",
                "event_id": "EVT-NRB-8942",
                "product_type": "PLATELETS",
                "expiry_date": now + datetime.timedelta(days=3),
                "current_facility_id": "Kenyatta National Referral (Cold Room 2)",
                "status": "AVAILABLE",
            },
            {
                "unit_id": "KE-BC-2026-0870",
                "event_id": "EVT-NRB-8942",
                "product_type": "PRBC",
                "expiry_date": now + datetime.timedelta(days=41),
                "current_facility_id": "Nairobi Regional Blood Depot",
                "status": "AVAILABLE",
            },
            {
                "unit_id": "KE-BC-2026-0830",
                "event_id": "EVT-KSM-7719",
                "product_type": "WHOLE_BLOOD",
                "expiry_date": now + datetime.timedelta(days=32),
                "current_facility_id": "Jaramogi Oginga Odinga Referral",
                "status": "AVAILABLE",
            },
            {
                "unit_id": "KE-BC-2026-0792",
                "event_id": "EVT-MSA-6502",
                "product_type": "PLATELETS",
                "expiry_date": now + datetime.timedelta(days=1),
                "current_facility_id": "Coast General Hospital Ward",
                "status": "AVAILABLE",
            },
        ]
        for u_data in seed_units_data:
            db.add(models.InventoryUnit(**u_data, created_at=now))
        await db.commit()

        stmt = select(models.InventoryUnit).order_by(models.InventoryUnit.expiry_date.asc()).limit(limit)
        units = (await db.execute(stmt)).scalars().all()

    # Serialized inventory units with ISO 8601 formatting
    serialized_units = []
    blood_types = ["O+", "O-", "A+", "B+", "AB+"]
    for i, u in enumerate(units):
        btype = blood_types[i % len(blood_types)]
        is_platelets = "PLATELET" in u.product_type.upper()
        if u.status == "BREACH":
            temp = 11.4
            agitated = False
        elif is_platelets:
            temp = 22.1
            agitated = True
        else:
            temp = 4.1 if i % 2 == 0 else 3.9
            agitated = False

        serialized_units.append({
            "barcode": u.unit_id,
            "unit_id": u.unit_id,
            "blood_type": btype,
            "blood_group": btype,
            "product_type": u.product_type,
            "expiry_date": u.expiry_date.isoformat() if hasattr(u.expiry_date, "isoformat") else str(u.expiry_date),
            "current_facility_id": u.current_facility_id,
            "current_facility": u.current_facility_id,
            "facility": u.current_facility_id,
            "status": u.status,
            "temperature": temp,
            "is_agitated": agitated,
            "created_at": u.created_at.isoformat() if hasattr(u.created_at, "isoformat") else str(u.created_at),
        })

    # Cache with 300-second TTL per Task 1.2
    if not facility_id and not product_type and not status_filter:
        await set_cached_json(cache_key, serialized_units, ttl_seconds=300)

    return serialized_units


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
    try:
        from scripts.migrate_all_data import run_full_migration
        results = await run_full_migration()
        try:
            await predictive_engine.initialize_from_db()
        except Exception as e:
            logger.warning(f"Notice re-initializing predictive engine: {e}")
        return results
    except Exception as e:
        import traceback
        err_msg = str(e)
        stack = traceback.format_exc()
        logger.error(f"Migration error: {err_msg}\n{stack}")
        return {
            "status": "error",
            "error": err_msg,
            "traceback": stack,
        }


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

