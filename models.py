import datetime
import uuid
from typing import Optional
from sqlalchemy import (
    Boolean,
    Column,
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

class DonorProfile(Base):
    """
    DonorProfile ORM Model.
    Represents donor behavioral features (RFM metrics), serological screening indicators,
    and retention history for machine learning retention classification.
    """
    __tablename__ = "donor_profiles"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    recency: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Days elapsed since the donor's last donation."
    )
    frequency: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Total lifetime donation count."
    )
    tenure: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Days elapsed since donor's first recorded donation."
    )
    blood_type: Mapped[str] = mapped_column(
        String(10), nullable=False, doc="ABO/Rh blood type (e.g., O+, O-, A+)."
    )
    retention_status: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, doc="Target binary classification (0 = Lapsed, 1 = Retained)."
    )
    syphilis_s_co_ratio: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, doc="Syphilis TPPA screening signal-to-cutoff (S/CO) ratio."
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_donor_retention", "retention_status"),
        Index("ix_donor_blood_type", "blood_type"),
    )

    def __repr__(self) -> str:
        return f"<DonorProfile(id='{self.id}', blood_type='{self.blood_type}', recency={self.recency}, frequency={self.frequency}, retained={self.retention_status})>"


class TransfusionDemand(Base):
    """
    TransfusionDemand ORM Model.
    Captures historical and incoming hospital blood transfusion consumption requests
    to feed stochastic time-series forecasting (ARIMA / SARIMA) and regional rebalancing.
    """
    __tablename__ = "transfusion_demands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    units_requested: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Number of whole blood/component units requested."
    )
    facility_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, doc="Unique identifier of requesting hospital or clinic."
    )
    blood_type: Mapped[str] = mapped_column(
        String(10), nullable=False, default="ALL", doc="Target blood group or 'ALL'."
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_facility_date", "facility_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<TransfusionDemand(facility='{self.facility_id}', date={self.date}, units={self.units_requested})>"


class Facility(Base):
    """
    Facility ORM Model.
    Represents healthcare facilities across the regional distribution network
    (Hospitals, Regional Blood Banks, Cold Storage Depots, Mobile Donor Drives).
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

    def __repr__(self) -> str:
        return f"<Facility(id='{self.id}', name='{self.name}', region='{self.region}')>"


class BloodUnitLedger(Base):
    """
    BloodUnitLedger ORM Model.
    Supports offline-first unit traceability, cold-chain temperature telemetry,
    and chain-of-custody tracking across transport legs.
    """
    __tablename__ = "blood_unit_ledger"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    unit_barcode: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    blood_type: Mapped[str] = mapped_column(String(10), nullable=False)
    collection_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="COLLECTED", doc="COLLECTED | TESTED_SAFE | IN_TRANSIT | TRANSFUSED | EXPIRED"
    )
    current_facility_id: Mapped[str] = mapped_column(String(100), nullable=False)
    temperature_celsius: Mapped[float] = mapped_column(Float, nullable=False, default=4.0)
    cold_chain_breach: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_synced_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<BloodUnitLedger(barcode='{self.unit_barcode}', type='{self.blood_type}', status='{self.status}')>"
