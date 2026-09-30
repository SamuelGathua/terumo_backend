import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Donor Schemas ---
class DonorProfileBase(BaseModel):
    recency: int = Field(..., ge=0, description="Days elapsed since the donor's last donation.")
    frequency: int = Field(..., ge=1, description="Total lifetime donation count.")
    tenure: int = Field(..., ge=0, description="Days elapsed since donor's first donation.")
    blood_type: str = Field(..., pattern="^(A|B|AB|O)[+-]$", description="Blood group (e.g. O+, O-, A+).")
    syphilis_s_co_ratio: Optional[float] = Field(0.5, ge=0.0, description="Screening S/CO ratio.")

class DonorProfileCreate(DonorProfileBase):
    retention_status: Optional[int] = Field(1, ge=0, le=1)

class DonorProfileResponse(DonorProfileBase):
    id: str
    retention_status: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# --- Transfusion Demand Schemas ---
class TransfusionDemandBase(BaseModel):
    date: datetime.date
    units_requested: int = Field(..., ge=0, description="Blood units requested.")
    facility_id: str = Field(..., min_length=2, max_length=100)
    blood_type: Optional[str] = Field("ALL", max_length=10)

class TransfusionDemandCreate(TransfusionDemandBase):
    pass

class TransfusionDemandResponse(TransfusionDemandBase):
    id: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# --- Machine Learning Request / Response Schemas ---
class RetentionPredictionRequest(BaseModel):
    recency: int = Field(..., ge=0, description="Recency in days since last donation.")
    frequency: int = Field(..., ge=1, description="Cumulative donation frequency.")
    tenure: int = Field(..., ge=0, description="Tenure in days since first donation.")

class RetentionPredictionResponse(BaseModel):
    retention_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability that donor returns.")
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
