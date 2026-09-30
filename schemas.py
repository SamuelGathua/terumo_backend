import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# --- 1. Donor Schemas ---
class DonorBase(BaseModel):
    blood_type: str = Field(..., pattern="^(A|B|AB|O)[+-]$", description="Blood group (e.g. O+, O-, A+).")
    tenure_days: int = Field(..., ge=0, description="Days elapsed since first donation.")
    recency_days: int = Field(..., ge=0, description="Days elapsed since last donation.")
    total_donations: int = Field(..., ge=1, description="Cumulative count of donations (frequency).")

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

    class Config:
        from_attributes = True


# --- 5. Transfusion Request Schemas ---
class TransfusionRequestBase(BaseModel):
    request_date: datetime.datetime
    requesting_facility_id: str = Field(..., min_length=2, max_length=100)
    blood_type_requested: Optional[str] = Field("ALL", max_length=10)
    units_requested: int = Field(..., ge=0)
    urgency_level: Optional[str] = Field("ROUTINE", description="ROUTINE | EMERGENCY | MASS_TRANSFUSION")

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

class RetentionPredictionResponse(BaseModel):
    retention_probability: float = Field(..., ge=0.0, le=1.0)
    retention_status: int = Field(..., description="Binary classification (1 = Likely Retained, 0 = At Risk).")
    risk_tier: str = Field(..., description="Risk categorization: LOW_RISK, MODERATE_RISK, HIGH_RISK.")
    recommended_action: str = Field(..., description="Clinical engagement recommendation for blood coordinator.")

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
