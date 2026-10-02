import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# --- 1. Donor Schemas ---
class DonorBase(BaseModel):
    blood_type: str = Field(..., pattern="^(A|B|AB|O)[+-]$", description="Blood group (e.g. O+, O-, A+).")
    tenure_days: int = Field(..., ge=0, description="Days elapsed since first donation.")
    recency_days: int = Field(..., ge=0, description="Days elapsed since last donation.")
    total_donations: int = Field(..., ge=1, description="Cumulative count of donations (frequency).")
    syphilis_s_co_ratio: Optional[float] = Field(0.5, ge=0.0, description="Syphilis TPPA screening S/CO ratio.")

class DonorCreate(DonorBase):
    retention_status: Optional[int] = Field(1, ge=0, le=1)
    retention_probability: Optional[float] = Field(0.5, ge=0.0, le=1.0)

class DonorResponse(DonorBase):
    donor_id: str
    retention_probability: float
    retention_status: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# Alias for backward compatibility
DonorProfileResponse = DonorResponse
DonorProfileCreate = DonorCreate


# --- 2. Donation Event Schemas ---
class DonationEventBase(BaseModel):
    donor_id: str
    location_id: str
    sync_status: Optional[str] = "SYNCED"
    cold_chain_breach_flag: Optional[bool] = False

class DonationEventCreate(DonationEventBase):
    collection_timestamp: Optional[datetime.datetime] = None

class DonationEventResponse(DonationEventBase):
    event_id: str
    collection_timestamp: datetime.datetime
    location: Optional[str] = None
    field_lead: Optional[str] = None
    officer: Optional[str] = None
    barcode_range: Optional[str] = None
    temperature: Optional[float] = 4.2
    breach: Optional[bool] = False
    is_offline_upload: Optional[bool] = False
    units_count: Optional[int] = None

    class Config:
        from_attributes = True


# --- 3. Screening Result Schemas ---
class ScreeningResultBase(BaseModel):
    event_id: str
    syphilis_s_co_ratio: float = Field(..., ge=0.0)
    dual_reagent_positive: bool = False
    tppa_predicted_status: bool = False

class ScreeningResultCreate(ScreeningResultBase):
    pass

class ScreeningResultResponse(ScreeningResultBase):
    test_id: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


# --- 4. Inventory Unit Schemas ---
class InventoryUnitBase(BaseModel):
    unit_id: str = Field(..., description="Blood bag barcode identifier.")
    event_id: str
    product_type: str = "WHOLE_BLOOD"
    expiry_date: datetime.datetime
    current_facility_id: str
    status: str = "AVAILABLE"

class InventoryUnitCreate(InventoryUnitBase):
    pass

class InventoryUnitResponse(InventoryUnitBase):
    created_at: datetime.datetime
    blood_type: Optional[str] = "O+"
    temperature: Optional[float] = 4.0
    is_agitated: Optional[bool] = None
    current_facility: Optional[str] = None

    class Config:
        from_attributes = True


# --- 5. Transfusion Request Schemas ---
class TransfusionRequestBase(BaseModel):
    request_date: datetime.datetime
    requesting_facility_id: str = Field(..., min_length=2, max_length=100)
    blood_type_requested: Optional[str] = Field("ALL", max_length=10)
    units_requested: int = Field(..., ge=0)
    urgency_level: Optional[str] = Field("ROUTINE", description="ROUTINE | EMERGENCY | MASS_TRANSFUSION")
    status: Optional[str] = Field("PENDING", description="PENDING | FULFILLED | CANCELLED")

class TransfusionRequestCreate(TransfusionRequestBase):
    pass

class TransfusionRequestResponse(TransfusionRequestBase):
    request_id: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# Alias for backward compatibility
TransfusionDemandResponse = TransfusionRequestResponse
TransfusionDemandCreate = TransfusionRequestCreate


# --- Machine Learning Schemas ---
class RetentionPredictionRequest(BaseModel):
    recency_days: int = Field(..., ge=0, description="Recency in days since last donation.")
    frequency_total: int = Field(..., ge=1, description="Cumulative total donations.")
    tenure_days: int = Field(..., ge=0, description="Tenure in days since first donation.")
    # Optional enrichment fields (activate additional model features when present)
    sex: Optional[str] = Field(None, description="Biological sex: M or F (affects eligibility interval).")
    donor_type: Optional[str] = Field(None, description="VOLUNTARY or FAMILY_REPLACEMENT.")
    age_years: Optional[float] = Field(None, ge=16.0, le=70.0, description="Donor age in years.")

class RetentionPredictionResponse(BaseModel):
    retention_probability: float = Field(..., ge=0.0, le=1.0)
    retention_status: int = Field(..., description="Binary classification (1 = Likely Retained, 0 = At Risk).")
    risk_tier: str = Field(..., description="Risk categorization: LOW_RISK, AT_RISK, HIGH_RISK.")
    recommended_action: str = Field(..., description="Clinical engagement recommendation for blood coordinator.")
    # Extended fields from v2 engine
    horizon_days: Optional[int] = Field(None, description="Forecast horizon in days the probability covers.")
    days_until_eligible: Optional[int] = Field(None, description="Days until the donor is next eligible to donate.")
    model_source: Optional[str] = Field(None, description="Model type used for the prediction.")

class DemandForecastPoint(BaseModel):
    date: datetime.date
    predicted_units: float
    confidence_lower_95: float
    confidence_upper_95: float

class DemandForecastResponse(BaseModel):
    facility_id: str
    forecast_horizon_days: int
    baseline_daily_mean: float
    stochastic_volatility: float
    forecast: List[DemandForecastPoint]
    rebalance_alert: Optional[str] = None
    cached: Optional[bool] = False
    # Extended fields from v2 SARIMAX engine
    forecast_start: Optional[str] = None
    model: Optional[str] = None
    interval_method: Optional[str] = None
    alert_level: Optional[str] = None
    history_days: Optional[int] = None
    tier_basis: Optional[str] = None

# --- Offline Sync Traceability Schemas ---
class BloodUnitSyncItem(BaseModel):
    unit_barcode: str
    blood_type: str
    collection_date: datetime.date
    expiry_date: datetime.date
    status: str
    current_facility_id: str
    temperature_celsius: float
    cold_chain_breach: bool = False

class OfflineSyncBatch(BaseModel):
    device_id: str
    sync_timestamp: datetime.datetime
    units: List[BloodUnitSyncItem]

# --- Flutter Batch Ingestion Schemas (POST /events/batch) ---
class BarcodeRecordItem(BaseModel):
    barcode: str = Field(..., description="Unique barcode ID e.g. KE-BC-2026-0891")
    blood_type: str = Field("O+", description="Blood group ABO/Rh")
    product_type: str = Field("WHOLE_BLOOD", description="WHOLE_BLOOD | PLATELETS | PRBC | FFP")
    expiry_date: Optional[datetime.datetime] = None
    facility: Optional[str] = "Transit Box #TB-04 (Machakos)"
    temperature: Optional[float] = 4.2
    status: Optional[str] = "AVAILABLE"
    is_agitated: Optional[bool] = None

class BatchManifestUpload(BaseModel):
    batch_id: str = Field(..., description="Unique batch ID, e.g. EVT-MCH-9021")
    field_lead: str = Field(..., description="Field lead nurse or officer, e.g. Nurse J. Mutua")
    location: str = Field(..., description="Mobile drive location, e.g. Machakos Mobile Donor Drive (Site 2)")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    temperature: float = Field(4.2, description="Transit telemetry temperature in Celsius")
    cold_chain_breach: bool = Field(False, description="Flag for temperature excursion")
    barcode_records: List[BarcodeRecordItem] = Field(..., description="List of unit barcode records from remote drive")


# --- 7. Overview Dashboard Aggregator Schemas ---
class HeaderAlerts(BaseModel):
    critical_shortages: int
    active_breaches: int
    total_alerts: int
    summary: str

class KpiMetric(BaseModel):
    value: int
    unit: str = "units"
    change_pct: float
    comparison_text: str
    sparkline: List[float]

class DashboardKpis(BaseModel):
    total_inventory: KpiMetric
    daily_collection_rate: KpiMetric
    pending_requests: KpiMetric

class SupplyDemandPoint(BaseModel):
    date: str
    supply_units: int
    demand_units: int
    supply: Optional[int] = None
    demand: Optional[int] = None

class PriorityRequestItem(BaseModel):
    id: int
    code: str
    code_bg: Optional[str] = None
    facility: str
    blood_type: str
    bloodType: Optional[str] = None
    amount: str
    units: int
    status: str
    badge_style: Optional[str] = None

class LiveActivityItem(BaseModel):
    id: str
    type: str  # "received" | "transit" | "breach" | "mobile_sync"
    title: str
    text: str
    time: str
    timestamp: Optional[str] = None

class OverviewSummaryResponse(BaseModel):
    header_alerts: HeaderAlerts
    kpis: DashboardKpis
    supply_vs_demand: List[SupplyDemandPoint]
    inventory_by_type: dict[str, int]
    priority_requests: List[PriorityRequestItem]
    live_activity: List[LiveActivityItem]
    cached: Optional[bool] = False

