import datetime
import uuid
from typing import Optional
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class Donor(Base):
    """
    1. donors (The Behavioral & Demographic Baseline)
    Stores unique donor profiles and calculated RFM metrics required by the Random Forest model.
    """
    __tablename__ = "donors"

    donor_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    blood_type: Mapped[str] = mapped_column(
        String(10), nullable=False, doc="Categorical blood group (A+, O-, B+, AB+, etc.)."
    )
    tenure_days: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Days elapsed since the donor's first recorded donation."
    )
    recency_days: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Days elapsed since the donor's last donation."
    )
    total_donations: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, doc="Cumulative count of successful donations (frequency)."
    )
    retention_probability: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, doc="Predicted probability (0.0 to 1.0) of donor returning."
    )
    retention_status: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, doc="Ground-truth binary classification (1 = Retained, 0 = Lapsed)."
    )
    syphilis_s_co_ratio: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, doc="Syphilis TPPA screening signal-to-cutoff (S/CO) ratio."
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    # --- Optional enrichment columns (migration: ALTER TABLE donors ADD COLUMN ... NULL) --------
    # Activate richer model features when populated; ml_engine checks hasattr() before querying.
    sex: Mapped[Optional[str]] = mapped_column(
        String(1), nullable=True, default=None,
        doc="Biological sex: M (male) or F (female). Affects donation eligibility interval."
    )
    donor_type: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True, default=None,
        doc="VOLUNTARY | FAMILY_REPLACEMENT | AUTOLOGOUS"
    )
    date_of_birth: Mapped[Optional[datetime.date]] = mapped_column(
        Date, nullable=True, default=None,
        doc="Date of birth (for age feature). Store in ISO 8601 format."
    )

    # Relationships
    donation_events: Mapped[list["DonationEvent"]] = relationship(
        "DonationEvent", back_populates="donor", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_donors_blood_type", "blood_type"),
        Index("ix_donors_retention_prob", "retention_probability"),
    )

    def __repr__(self) -> str:
        return f"<Donor(id='{self.donor_id}', blood_type='{self.blood_type}', recency={self.recency_days}, freq={self.total_donations}, prob={self.retention_probability})>"


class DonationEvent(Base):
    """
    2. donation_events (The Offline-First Traceability Ledger)
    Tracks the physical collection event. Primary target for Flutter app asynchronous syncing.
    """
    __tablename__ = "donation_events"

    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    donor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("donors.donor_id", ondelete="CASCADE"), nullable=False, index=True
    )
    collection_timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    location_id: Mapped[str] = mapped_column(
        String(100), nullable=False, doc="Identifier of mobile drive or static collection site."
    )
    sync_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="SYNCED", doc="PENDING | SYNCED"
    )
    cold_chain_breach_flag: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, doc="Flagged if transit times or temps between collection and lab exceed safety parameters."
    )

    # Relationships
    donor: Mapped["Donor"] = relationship("Donor", back_populates="donation_events")
    screening_results: Mapped[list["ScreeningResult"]] = relationship(
        "ScreeningResult", back_populates="donation_event", cascade="all, delete-orphan"
    )
    inventory_units: Mapped[list["InventoryUnit"]] = relationship(
        "InventoryUnit", back_populates="donation_event", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_events_location", "location_id"),
        Index("ix_events_sync_status", "sync_status"),
    )

    def __repr__(self) -> str:
        return f"<DonationEvent(id='{self.event_id}', donor='{self.donor_id}', sync='{self.sync_status}', breach={self.cold_chain_breach_flag})>"


class ScreeningResult(Base):
    """
    3. screening_results (The AI Diagnostic Layer)
    Isolates laboratory testing data from collection events for AI screening optimization.
    """
    __tablename__ = "screening_results"

    test_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("donation_events.event_id", ondelete="CASCADE"), nullable=False, index=True
    )
    syphilis_s_co_ratio: Mapped[float] = mapped_column(
        Float, nullable=False, doc="Continuous signal-to-cutoff (S/CO) ratio."
    )
    dual_reagent_positive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, doc="Flags if both initial screening reagents reacted (89.6% confirmatory rate)."
    )
    tppa_predicted_status: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, doc="ML predicted confirmatory status (e.g. S/CO >= 10.0 -> 98.4% positive)."
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    donation_event: Mapped["DonationEvent"] = relationship("DonationEvent", back_populates="screening_results")

    def __repr__(self) -> str:
        return f"<ScreeningResult(test_id='{self.test_id}', s_co={self.syphilis_s_co_ratio}, tppa_pred={self.tppa_predicted_status})>"


class InventoryUnit(Base):
    """
    4. inventory_units (The Supply Rebalancing Target)
    Represents physical bags of blood currently in the system, tracking perishability and location.
    """
    __tablename__ = "inventory_units"

    unit_id: Mapped[str] = mapped_column(
        String(100), primary_key=True, doc="Physical barcode identifier on the blood bag."
    )
    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("donation_events.event_id", ondelete="CASCADE"), nullable=False, index=True
    )
    product_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="WHOLE_BLOOD", doc="WHOLE_BLOOD | PLATELETS | PRBC | FFP"
    )
    expiry_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, doc="Platelets expire in 5-7 days; RBCs in 35-42 days."
    )
    current_facility_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, doc="Facility where unit is currently stored."
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="AVAILABLE", doc="AVAILABLE | IN_TRANSIT | TRANSFUSED | DISCARDED"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    donation_event: Mapped["DonationEvent"] = relationship("DonationEvent", back_populates="inventory_units")

    __table_args__ = (
        Index("ix_inventory_status_facility", "status", "current_facility_id"),
        Index("ix_inventory_expiry", "expiry_date"),
    )

    def __repr__(self) -> str:
        return f"<InventoryUnit(barcode='{self.unit_id}', product='{self.product_type}', status='{self.status}', facility='{self.current_facility_id}')>"


class TransfusionRequest(Base):
    """
    5. transfusion_requests (The Time-Series Demand Engine)
    Logs daily historical and real-time hospital orders feeding ARIMA/SARIMA forecasting models.
    """
    __tablename__ = "transfusion_requests"

    request_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, index=True, doc="Timestamp/date of the hospital transfusion order."
    )
    requesting_facility_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, doc="Hospital or clinic identifier placing the order."
    )
    blood_type_requested: Mapped[str] = mapped_column(
        String(10), nullable=False, default="ALL", doc="Requested ABO/Rh blood type or ALL."
    )
    units_requested: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Volume of blood units ordered."
    )
    urgency_level: Mapped[str] = mapped_column(
        String(30), nullable=False, default="ROUTINE", doc="ROUTINE | EMERGENCY | MASS_TRANSFUSION"
    )
    status: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, default="PENDING", doc="PENDING | FULFILLED | CANCELLED"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_requests_facility_date", "requesting_facility_id", "request_date"),
    )

    def __repr__(self) -> str:
        return f"<TransfusionRequest(id={self.request_id}, facility='{self.requesting_facility_id}', units={self.units_requested}, urgency='{self.urgency_level}')>"


class Facility(Base):
    """
    Facility Network Model.
    Regional nodes across the network (Hospitals, Blood Hubs, Cold Storage Depots).
    """
    __tablename__ = "facilities"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    facility_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="HOSPITAL", doc="HOSPITAL | BLOOD_BANK | COLD_ROOM | MOBILE_DRIVE"
    )
    region: Mapped[str] = mapped_column(String(100), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    longitude: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    inventory_capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=500)
    current_inventory_units: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    # KEPH level (1-6): enables tier-based demand baselines without an unreliable capacity proxy.
    # Migration: ALTER TABLE facilities ADD COLUMN keph_level INTEGER NULL;
    keph_level: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=None,
        doc="Kenya Essential Package for Health tier (1=Community, 6=National Referral)."
    )

    def __repr__(self) -> str:
        return f"<Facility(id='{self.id}', name='{self.name}', region='{self.region}')>"


# Aliases for backward-compatibility with prior code references
DonorProfile = Donor
TransfusionDemand = TransfusionRequest
BloodUnitLedger = InventoryUnit
