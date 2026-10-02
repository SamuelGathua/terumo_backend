"""
ml_engine.py - Predictive Intelligence Engine for ABIS (v2)
===========================================================

1. Donor retention  (RandomForest + probability calibration)
   Question answered: "what is the probability this donor donates again within H days?"
   (H = ABIS_RETENTION_HORIZON_DAYS, default 180).
   * Features AND labels are rebuilt point-in-time from the donation_events ledger
     (see retention_data.py). The stored donors.retention_status / retention_probability
     columns are never read.
   * Temporal, purged train / calibration / test split (no random split, no label leakage).
   * Isotonic (or Platt for small data) calibration, so the 0.40 / 0.70 tier cut-offs mean
     what they say. class_weight is NOT used - it would distort the probabilities.
   * Metrics are computed on the final, untouched test period and compared with simple
     baselines. They carry label_source / data_source so nobody mistakes demo numbers
     (data_source != "real") for validated performance.

2. Demand forecasting  (SARIMAX, model chosen per facility)
   * Daily series aggregated in Africa/Nairobi time; days with no orders are zeros.
   * Differencing is decided by an ADF test (not assumed); a weekly-seasonal candidate is
     compared by AIC against the non-seasonal one (same differencing, so AIC is comparable).
   * Rolling-origin backtest against a seasonal-naive baseline: backtest_facility_demand().
   * Sparse / constant histories fall back to a DETERMINISTIC tier-prior projection whose
     band is labelled as a heuristic, not a statistical 95% interval.

All CPU-bound work (training, SARIMAX fits) runs in worker threads so the event loop stays free.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import select
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from database import AsyncSessionLocal
import models
import retention_data as rd

logger = logging.getLogger("abis.ml_engine")
logger.setLevel(logging.INFO)

# ------------------------------- configuration ---------------------------------------
TZ_NAME = "Africa/Nairobi"
STORED_TZ = "UTC"  # timezone of the naive timestamps stored in the database
RETENTION_HORIZON_DAYS = int(os.environ.get("ABIS_RETENTION_HORIZON_DAYS", "180"))
MODEL_DIR = os.environ.get("ABIS_MODEL_DIR", "./model_store")
DATA_SOURCE = os.environ.get("ABIS_DATA_SOURCE", "synthetic")  # "real" | "synthetic"
TIER_THRESHOLDS = (0.40, 0.70)  # AT_RISK below 0.70, HIGH_RISK below 0.40 (calibrated probabilities)
MIN_TRAIN_ROWS = 1_000
MIN_CALIB_ISOTONIC = 1_000

MIN_HISTORY_DAYS = 28        # below this, use the tier-prior projection
SEASONAL_MIN_DAYS = 56       # need >= 8 weeks to try a weekly seasonal term
BASELINE_WINDOW_DAYS = 56    # trailing window used for baselines / surge alerts
HISTORY_DAYS = 730
FORECAST_CACHE_TTL_S = 900   # 15 minutes (aligned with Redis cache TTL)
MAX_CONCURRENT_FITS = 2
SURGE_RATIO = 1.25
SURGE_MIN_EXCESS_UNITS = 3.0

# KEPH level -> (baseline daily units mu, volatility sigma)
KEPH_BASELINES: Dict[int, Tuple[float, float]] = {
    6: (115.0, 18.0),
    5: (48.0, 8.5),
    4: (12.0, 3.0),
    3: (1.0, 0.5),
    2: (0.0, 0.0),
    1: (0.0, 0.0),
}
NO_DEMAND_TYPES = {"BLOOD_BANK", "COLD_ROOM", "MOBILE_DRIVE", "DISPENSARY"}
_warned_legacy_tier = False


class UnknownFacilityError(KeyError):
    """Raised instead of guessing a tier from the facility-id string. Map to HTTP 404 in the API."""


# ------------------------------- small helpers --------------------------------------
def _safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(
        (idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
        for b in range(bins) if (idx == b).any()
    ))


def _prob_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": round(_safe_auc(y, p), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "ece": round(_ece(y, p), 4),
    }


def _calibrate(bundle: Dict[str, Any], raw: np.ndarray) -> np.ndarray:
    cal = bundle["calibrator"]
    if bundle["calibration"] == "isotonic":
        return np.asarray(cal.predict(raw), dtype=float)
    return cal.predict_proba(raw.reshape(-1, 1))[:, 1]


class PredictiveIntelligenceEngine:
    """Singleton managing donor-retention scoring and facility demand forecasting."""

    _instance: Optional["PredictiveIntelligenceEngine"] = None

    def __new__(cls) -> "PredictiveIntelligenceEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.retention_bundle: Optional[Dict[str, Any]] = None
        self.model_metrics: Dict[str, Any] = {}
        self.is_trained: bool = False
        self._last_artifact_mtime: float = 0.0
        self._forecast_cache: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
        self._fit_semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = True

    @property
    def donor_model(self):  # backwards-compatible accessor
        return self.retention_bundle["model"] if self.retention_bundle else None

    def _semaphore(self) -> asyncio.Semaphore:
        if self._fit_semaphore is None:  # created lazily so it binds to the running loop
            self._fit_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FITS)
        return self._fit_semaphore

    # ==================================================================================
    # 1. DONOR RETENTION
    # ==================================================================================
    async def initialize_from_db(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Load the saved model if one exists, otherwise train from the ledger and persist it."""
        if not force_retrain and self._try_load_artifact():
            return self.model_metrics
        logger.info("Training donor-retention model from donation_events...")
        events, donors = await self._load_donation_data()
        return await asyncio.to_thread(self._train_retention, events, donors)

    async def _load_donation_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        donor_cols = ["donor_id"] + [c for c in ("sex", "date_of_birth", "donor_type") if hasattr(models.Donor, c)]
        async with AsyncSessionLocal() as session:
            ev_rows = (
                await session.execute(
                    select(models.DonationEvent.donor_id, models.DonationEvent.collection_timestamp).where(
                        models.DonationEvent.sync_status == "SYNCED"
                    )
                )
            ).all()
            donor_rows = (await session.execute(select(*[getattr(models.Donor, c) for c in donor_cols]))).all()
        events = pd.DataFrame([tuple(r) for r in ev_rows], columns=["donor_id", "collection_timestamp"])
        donors = pd.DataFrame([tuple(r) for r in donor_rows], columns=donor_cols) if len(donor_cols) > 1 else None
        return events, donors

    def _uninitialized(self, reason: str) -> Dict[str, Any]:
        logger.warning("Retention model not trained: %s. Using heuristic prior.", reason)
        self.retention_bundle, self.is_trained = None, False
        self.model_metrics = {"status": "uninitialized", "reason": reason, "model_source": "heuristic_prior"}
        return self.model_metrics

    def _train_retention(self, events: pd.DataFrame, donors: Optional[pd.DataFrame]) -> Dict[str, Any]:
        H = RETENTION_HORIZON_DAYS
        if events.empty:
            return self._uninitialized("no SYNCED donation events")

        as_of = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        cutoffs = rd.make_cutoffs(events, H, as_of)
        try:
            split = rd.temporal_split(cutoffs, H)
        except ValueError as exc:
            return self._uninitialized(str(exc))

        frames = {
            name: rd.build_training_set(events, cs, H, donors, observed_until=as_of)
            for name, cs in split.items()
        }
        for name, f in frames.items():
            if f.empty or f[rd.LABEL_COL].nunique() < 2:
                return self._uninitialized(name + " split does not contain both outcomes")
        train, calib, test = frames["train"], frames["calib"], frames["test"]
        if len(train) < MIN_TRAIN_ROWS:
            return self._uninitialized("only " + str(len(train)) + " training rows (need " + str(MIN_TRAIN_ROWS) + ")")

        feats = rd.select_features(train)
        fill = train[feats].median().fillna(0.0).to_dict()

        def mat(f: pd.DataFrame) -> np.ndarray:
            return f[feats].fillna(fill).to_numpy(dtype=float)

        y_tr, y_ca, y_te = (f[rd.LABEL_COL].to_numpy() for f in (train, calib, test))

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42
        )
        clf.fit(mat(train), y_tr)

        raw_ca = clf.predict_proba(mat(calib))[:, 1]
        if len(calib) >= MIN_CALIB_ISOTONIC:
            calibration, calibrator = "isotonic", IsotonicRegression(out_of_bounds="clip").fit(raw_ca, y_ca)
        else:
            calibration, calibrator = "platt", LogisticRegression(C=1e6).fit(raw_ca.reshape(-1, 1), y_ca)

        bundle: Dict[str, Any] = {
            "model": clf,
            "calibrator": calibrator,
            "calibration": calibration,
            "features": feats,
            "fill_values": fill,
            "horizon_days": H,
        }
        raw_te = clf.predict_proba(mat(test))[:, 1]
        p_te = _calibrate(bundle, raw_te)

        recency_te = test["recency_days"].to_numpy(dtype=float)
        heuristic_te = self._heuristic_array(recency_te, test["total_donations"].to_numpy(), test["tenure_days"].to_numpy())
        pred = (p_te >= 0.5).astype(int)

        metrics: Dict[str, Any] = {
            "status": "trained",
            "model_source": "random_forest_" + calibration,
            "label_source": "point_in_time_donation_events",
            "label_definition": "donated again within " + str(H) + " days of the cutoff date",
            "horizon_days": H,
            "data_source": DATA_SOURCE,
            "validated_on_real_data": DATA_SOURCE == "real",
            "split_cutoffs": {k: [str(v[0].date()), str(v[-1].date())] for k, v in split.items()},
            "rows": {"train": len(train), "calibration": len(calib), "test": len(test)},
            "prevalence": {
                "train": round(float(y_tr.mean()), 4),
                "calibration": round(float(y_ca.mean()), 4),
                "test": round(float(y_te.mean()), 4),
            },
            "test_calibrated": _prob_metrics(y_te, p_te),
            "test_uncalibrated": _prob_metrics(y_te, raw_te),
            "test_at_0.5": {
                "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            },
            "baselines_test_auc": {
                "recency_only": round(_safe_auc(y_te, -recency_te), 4),
                "hand_written_heuristic": round(_safe_auc(y_te, heuristic_te), 4),
            },
            "feature_importance": {f: round(float(v), 4) for f, v in zip(feats, clf.feature_importances_)},
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if DATA_SOURCE != "real":
            metrics["warning"] = (
                "Trained on data not flagged ABIS_DATA_SOURCE=real (e.g. the simulated ledger). "
                "These metrics describe the simulator, not donor behaviour in Kenya."
            )
        bundle["metrics"] = metrics
        bundle["version"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        clf.set_params(n_jobs=1)  # single-row inference is much faster without joblib fan-out
        self.retention_bundle, self.model_metrics, self.is_trained = bundle, metrics, True
        self._save_artifact(bundle)
        logger.info(
            "Retention model trained: %s",
            {k: metrics[k] for k in ("model_source", "test_calibrated", "baselines_test_auc")},
        )
        return metrics

    # ---------------------------- persistence ------------------------------------------
    def _artifact_path(self, tag: str) -> str:
        return os.path.join(MODEL_DIR, "retention_" + tag + ".joblib")

    def _save_artifact(self, bundle: Dict[str, Any]) -> None:
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            tmp = self._artifact_path("latest") + ".tmp"
            joblib.dump(bundle, tmp)
            joblib.dump(bundle, self._artifact_path(bundle["version"]))
            latest_path = self._artifact_path("latest")
            os.replace(tmp, latest_path)
            self._last_artifact_mtime = os.path.getmtime(latest_path)
        except Exception as exc:  # persistence is best-effort; the in-memory model still works
            logger.error("Could not persist retention model: %s", exc)

    def _try_load_artifact(self) -> bool:
        path = self._artifact_path("latest")
        if not os.path.exists(path):
            return False
        try:
            bundle = joblib.load(path)  # only ever load artifacts written by this service
            if bundle.get("horizon_days") != RETENTION_HORIZON_DAYS or "features" not in bundle:
                logger.info("Saved retention model does not match current configuration; retraining.")
                return False
            bundle["model"].set_params(n_jobs=1)
            self.retention_bundle, self.model_metrics, self.is_trained = bundle, bundle["metrics"], True
            self._last_artifact_mtime = os.path.getmtime(path)
            logger.info("Loaded retention model %s from %s", bundle.get("version"), path)
            return True
        except Exception as exc:
            logger.warning("Could not load saved retention model (%s); retraining.", exc)
            return False

    def reload_if_newer(self) -> bool:
        """Multi-worker sync: reload latest saved model artifact if another worker updated it on disk."""
        if not self.is_trained and self.retention_bundle is None:
            return False
        path = self._artifact_path("latest")
        if not os.path.exists(path):
            return False
        try:
            mtime = os.path.getmtime(path)
            if mtime > self._last_artifact_mtime:
                logger.info("Found newer retention model artifact on disk (mtime: %s); reloading...", mtime)
                return self._try_load_artifact()
        except Exception as exc:
            logger.debug("Error checking artifact mtime: %s", exc)
        return False

    # ---------------------------- inference --------------------------------------------
    def predict_donor_retention(
        self,
        recency_days: int,
        total_donations: int,
        tenure_days: int,
        sex: Optional[str] = None,
        donor_type: Optional[str] = None,
        age_years: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calibrated P(donor returns within the horizon), risk tier, and a recommended action."""
        self.reload_if_newer()
        recency_days = max(0, int(recency_days))
        total_donations = max(1, int(total_donations))
        tenure_days = max(int(tenure_days), recency_days)  # same clamp as training (engineer_features)
        min_interval = rd.MIN_INTERVAL_DAYS.get((sex or "").upper(), rd.DEFAULT_MIN_INTERVAL_DAYS)

        prob: Optional[float] = None
        source = "heuristic_prior"
        bundle = self.retention_bundle
        if bundle is not None and self.is_trained:
            try:
                row = rd.engineer_features([recency_days], [total_donations], [tenure_days], min_interval)
                row["is_female"] = np.nan if not sex else float(sex.upper() == "F")
                row["is_family_replacement"] = np.nan if not donor_type else float(donor_type.upper() == "FAMILY_REPLACEMENT")
                row["age_years"] = np.nan if age_years is None else float(age_years)
                X = row.reindex(columns=bundle["features"]).fillna(bundle["fill_values"]).to_numpy(dtype=float)
                raw = bundle["model"].predict_proba(X)[:, 1]
                prob = float(_calibrate(bundle, raw)[0])
                source = self.model_metrics.get("model_source", "random_forest")
            except Exception as exc:
                logger.error("Inference error: %s. Falling back to heuristic prior.", exc)
        if prob is None:
            prob = self._heuristic_retention(recency_days, total_donations, tenure_days)

        low, high = TIER_THRESHOLDS[0], TIER_THRESHOLDS[1]
        if prob >= high:
            tier, action = "LOW_RISK", "Donor actively engaged. Dispatch scheduled SMS reminder for upcoming mobile drive."
        elif prob >= low:
            tier, action = "AT_RISK", "Lapse warning. Dispatch personalized WhatsApp engagement with local community patient impact story."
        else:
            tier, action = "HIGH_RISK", "Critical attrition risk. Flag for direct call liaison coordinator with transport subsidy."

        days_until_eligible = max(0, min_interval - recency_days)
        if days_until_eligible > 0:
            action = "Not yet eligible to donate (eligible in " + str(days_until_eligible) + " days) - schedule the reminder for that date. " + action

        return {
            "retention_probability": round(prob, 4),
            "retention_status": 1 if prob >= 0.5 else 0,
            "risk_tier": tier,
            "recommended_action": action,
            "horizon_days": RETENTION_HORIZON_DAYS,
            "days_until_eligible": days_until_eligible,
            "model_source": source,
        }

    @staticmethod
    def _heuristic_array(recency, frequency, tenure) -> np.ndarray:
        rec = np.asarray(recency, dtype=float)
        freq = np.asarray(frequency, dtype=float)
        ten = np.asarray(tenure, dtype=float)
        freq_factor = 1.0 / (1.0 + np.exp(-0.45 * (freq - 3)))
        recency_factor = np.exp(-0.0055 * np.maximum(0.0, rec - 45))
        velocity_factor = np.minimum(1.0, (freq * 365.0) / np.maximum(30.0, ten) / 3.0)
        score = 0.50 * (freq_factor * recency_factor) + 0.35 * recency_factor + 0.15 * velocity_factor
        return np.clip(score, 0.05, 0.98)

    def _heuristic_retention(self, recency: int, frequency: int, tenure: int) -> float:
        """Hand-written prior used ONLY when no trained model is available."""
        return float(self._heuristic_array([recency], [frequency], [tenure])[0])

    # ==================================================================================
    # 2. DEMAND FORECASTING
    # ==================================================================================
    @staticmethod
    def _local_today() -> dt.date:
        return pd.Timestamp.now(tz=TZ_NAME).tz_localize(None).normalize().date()

    @staticmethod
    def build_daily_series(timestamps, units, start: dt.date, end: dt.date) -> pd.Series:
        """
        Aggregate order rows to calendar days in Africa/Nairobi time. Days without orders
        are 0 (no row == no order). Leading days before the first order are trimmed so a
        new facility is not padded with fake zeros. Returns an EMPTY series if no rows.
        """
        if len(timestamps) == 0:
            return pd.Series(dtype=float)
        ts = pd.to_datetime(pd.Series(list(timestamps)))
        local = ts.dt.tz_localize(STORED_TZ).dt.tz_convert(TZ_NAME).dt.tz_localize(None).dt.normalize()
        daily = pd.Series(np.asarray(units, dtype=float), index=pd.DatetimeIndex(local)).groupby(level=0).sum()
        daily = daily[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))]
        if daily.empty:
            return pd.Series(dtype=float)
        idx = pd.date_range(max(pd.Timestamp(start), daily.index.min()), pd.Timestamp(end), freq="D")
        return daily.reindex(idx, fill_value=0.0)

    async def fetch_facility_historical_demand(self, facility_id: str, days: int = HISTORY_DAYS) -> pd.Series:
        """Daily units requested (complete days only: today is excluded because it is still filling up)."""
        today = self._local_today()
        start = today - dt.timedelta(days=days)
        TR = models.TransfusionRequest
        stmt = select(TR.request_date, TR.units_requested).where(
            TR.requesting_facility_id == facility_id,
            TR.request_date >= dt.datetime.combine(start - dt.timedelta(days=1), dt.time.min),
        )
        if hasattr(TR, "status"):  # cancelled orders are not demand
            stmt = stmt.where(TR.status != "CANCELLED")
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(stmt)).all()
        return self.build_daily_series([r[0] for r in rows], [r[1] for r in rows], start, today - dt.timedelta(days=1))

    async def get_facility_profile(self, facility_id: str) -> Dict[str, Any]:
        async with AsyncSessionLocal() as session:
            fac = (await session.execute(select(models.Facility).where(models.Facility.id == facility_id))).scalar_one_or_none()
            if fac is None:
                raise UnknownFacilityError(facility_id)
            return {
                "facility_type": fac.facility_type,
                "keph_level": getattr(fac, "keph_level", None),
                "capacity": fac.inventory_capacity,
                "inventory": getattr(fac, "current_inventory_units", None),
            }

    @staticmethod
    def tier_baseline(profile: Dict[str, Any]) -> Tuple[float, float, str]:
        """(mu, sigma, basis). Uses keph_level when stored; capacity is a legacy, unreliable proxy."""
        global _warned_legacy_tier
        ftype = (profile.get("facility_type") or "").upper()
        if ftype in NO_DEMAND_TYPES:
            return 0.0, 0.0, "facility_type:" + ftype
        level = profile.get("keph_level")
        if level is not None:
            mu, sigma = KEPH_BASELINES.get(int(level), KEPH_BASELINES[4])
            return mu, sigma, "keph_level:" + str(level)
        if not _warned_legacy_tier:
            logger.warning(
                "facilities.keph_level is empty - inferring tier from inventory_capacity (unreliable). Persist keph_level."
            )
            _warned_legacy_tier = True
        cap = profile.get("capacity") or 0
        if cap <= 0:
            return 0.0, 0.0, "legacy_capacity:0"
        if ftype == "HEALTH_CENTRE" or cap <= 10:
            return (*KEPH_BASELINES[3], "legacy_capacity")
        if cap <= 150:
            return (*KEPH_BASELINES[4], "legacy_capacity")
        if cap <= 500:
            return (*KEPH_BASELINES[5], "legacy_capacity")
        return (*KEPH_BASELINES[6], "legacy_capacity")

    # ---------------------------- forecasting core (sync, thread-safe) -------------------
    @staticmethod
    def _dow_factors(series: pd.Series, weeks: int = 8) -> np.ndarray:
        """Multiplicative day-of-week profile (index 0 = Monday), mean-normalised to 1."""
        if not isinstance(series.index, pd.DatetimeIndex):
            return np.ones(7)
        tail = series.iloc[-7 * weeks:]
        if len(tail) < 28 or tail.mean() <= 0:
            return np.ones(7)
        f = (tail.groupby(tail.index.dayofweek).mean().reindex(range(7)) / tail.mean()).fillna(1.0).to_numpy()
        return f / f.mean()

    @staticmethod
    def _candidate_specs(y: np.ndarray):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # statsmodels announces a future return-type change
                d = 0 if adfuller(y, autolag="AIC")[1] < 0.05 else 1
        except Exception:
            d = 0  # demand counts are usually level-stationary; fall back to no differencing
        trend = "c" if d == 0 else "n"
        cands = [((1, d, 1), (0, 0, 0, 0))]
        if len(y) >= SEASONAL_MIN_DAYS:
            cands.append(((1, d, 1), (1, 0, 1, 7)))
        return d, trend, cands

    @staticmethod
    def _fit_spec(y: np.ndarray, order, seasonal, trend):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend).fit(disp=False, maxiter=200)

    def _select_spec(self, y: np.ndarray):
        """Returns (fitted_result, order, seasonal, trend) with the lowest AIC among same-differencing candidates."""
        _, trend, cands = self._candidate_specs(y)
        best = None
        for order, seasonal in cands:
            try:
                res = self._fit_spec(y, order, seasonal, trend)
                if np.isfinite(res.aic) and (best is None or res.aic < best[0].aic):
                    best = (res, order, seasonal, trend)
            except Exception as exc:
                logger.info("SARIMAX%sx%s failed: %s", order, seasonal, exc)
        return best

    @staticmethod
    def _zero_projection(facility_id: str, horizon: int, start: dt.date, basis: str) -> Dict[str, Any]:
        pts = [
            {
                "date": (start + dt.timedelta(days=i)).isoformat(),
                "predicted_units": 0.0,
                "confidence_lower_95": 0.0,
                "confidence_upper_95": 0.0,
            }
            for i in range(horizon)
        ]
        return {
            "facility_id": facility_id, "forecast_horizon_days": horizon, "baseline_daily_mean": 0.0,
            "stochastic_volatility": 0.0, "forecast": pts, "rebalance_alert": None, "alert_level": None,
            "model": "no_patient_demand", "interval_method": "n/a", "tier_basis": basis,
            "forecast_start": start.isoformat(),
        }

    def _heuristic_projection(
        self,
        facility_id: str,
        horizon: int,
        mean_val: float,
        std_val: float,
        start: dt.date,
        dow: Optional[np.ndarray] = None,
        model: str = "tier_prior",
        basis: str = "",
        inventory: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deterministic: flat mean x day-of-week profile, with a +/-1.96 sigma band that is NOT a statistical interval."""
        if mean_val <= 0.0:
            return self._zero_projection(facility_id, horizon, start, basis)
        if not (std_val > 0.0):  # constant history is not "no demand"
            std_val = max(0.25 * mean_val, 0.5)
        dow = np.ones(7) if dow is None else dow
        pts, total = [], 0.0
        for i in range(horizon):
            d = start + dt.timedelta(days=i)
            pred = max(0.0, mean_val * float(dow[d.weekday()]))
            total += pred
            pts.append({
                "date": d.isoformat(), "predicted_units": round(pred, 1),
                "confidence_lower_95": round(max(0.0, pred - 1.96 * std_val), 1),
                "confidence_upper_95": round(pred + 1.96 * std_val, 1),
            })
        out = {
            "facility_id": facility_id, "forecast_horizon_days": horizon,
            "baseline_daily_mean": round(mean_val, 2), "stochastic_volatility": round(std_val, 2),
            "forecast": pts, "rebalance_alert": None, "alert_level": None, "model": model,
            "interval_method": "heuristic_band_not_statistical", "tier_basis": basis,
            "forecast_start": start.isoformat(),
        }
        self._attach_alert(
            out,
            total,
            mean_val * sum(float(dow[(start + dt.timedelta(days=i)).weekday()]) for i in range(horizon)),
            inventory,
        )
        return out

    @staticmethod
    def _attach_alert(out: Dict[str, Any], projected: float, expected: float, inventory: Optional[int]) -> None:
        h, fid = out["forecast_horizon_days"], out["facility_id"]
        surge = expected > 0 and projected > expected * SURGE_RATIO and (projected - expected) >= SURGE_MIN_EXCESS_UNITS
        stock = inventory is not None and projected > inventory
        if not (surge or stock):
            return
        parts = []
        if surge:
            parts.append(
                "projected " + str(h) + "-day demand (" + str(round(projected)) + " units) is "
                + str(round((projected / expected - 1) * 100)) + "% above "
                + "the trailing " + str(BASELINE_WINDOW_DAYS // 7) + "-week baseline ("
                + str(round(expected)) + " units)"
            )
        if stock:
            parts.append(
                "projected demand exceeds on-hand inventory (" + str(inventory) + " units) before resupply"
            )
        out["alert_level"] = "STOCK_RISK" if stock else "SURGE"
        out["rebalance_alert"] = (
            out["alert_level"] + " for " + fid + ": " + "; ".join(parts)
            + ". Review inter-facility rebalancing (no action has been taken automatically)."
        )

    def _forecast_from_series(
        self,
        facility_id: str,
        series: pd.Series,
        horizon: int,
        tier_mu: float,
        tier_sigma: float,
        basis: str,
        inventory: Optional[int],
        start: dt.date,
    ) -> Dict[str, Any]:
        n = len(series)
        if n < MIN_HISTORY_DAYS:
            mean_val = float(series.mean()) if n >= 7 else tier_mu
            std_val = float(series.std()) if n >= 7 else tier_sigma
            logger.info("Sparse history for %s (%d days): tier prior blend.", facility_id, n)
            return self._heuristic_projection(
                facility_id, horizon, mean_val, std_val, start, None, "tier_prior_sparse_history", basis, inventory
            )

        y = series.to_numpy(dtype=float)
        dow = self._dow_factors(series)
        tail = series.iloc[-BASELINE_WINDOW_DAYS:]
        base_mean, base_std = float(tail.mean()), float(tail.std())

        sel = None if np.ptp(y) == 0 else self._select_spec(y)
        if sel is None:
            return self._heuristic_projection(
                facility_id, horizon, base_mean, base_std, start, dow, "trailing_mean_fallback", basis, inventory
            )

        res, order, seasonal, _ = sel
        fc = res.get_forecast(steps=horizon)
        mean_fc = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)

        points, total = [], 0.0
        for i in range(horizon):
            d = start + dt.timedelta(days=i)
            pred = max(0.0, float(mean_fc[i]))
            total += pred
            points.append({
                "date": d.isoformat(), "predicted_units": round(pred, 1),
                "confidence_lower_95": round(max(0.0, float(ci[i, 0])), 1),
                "confidence_upper_95": round(max(pred, float(ci[i, 1])), 1),
            })

        expected = sum(base_mean * float(dow[(start + dt.timedelta(days=i)).weekday()]) for i in range(horizon))
        out = {
            "facility_id": facility_id, "forecast_horizon_days": horizon,
            "baseline_daily_mean": round(base_mean, 2), "stochastic_volatility": round(base_std, 2),
            "forecast": points, "rebalance_alert": None, "alert_level": None,
            "model": "SARIMAX" + str(order) + "x" + str(seasonal), "history_days": n,
            "interval_method": "model_gaussian_95", "tier_basis": basis, "forecast_start": start.isoformat(),
        }
        self._attach_alert(out, total, expected, inventory)
        return out

    async def forecast_facility_demand(self, facility_id: str, horizon_days: int = 7, use_cache: bool = True) -> Dict[str, Any]:
        """
        Forecast daily blood demand for the next horizon_days, starting TODAY (Nairobi date;
        history covers complete days up to yesterday). Raises UnknownFacilityError for unknown ids.
        """
        horizon = int(np.clip(horizon_days, 1, 56))
        key = (facility_id, horizon)
        hit = self._forecast_cache.get(key)
        if use_cache and hit and time.monotonic() - hit[0] < FORECAST_CACHE_TTL_S:
            return copy.deepcopy(hit[1])

        profile = await self.get_facility_profile(facility_id)
        mu, sigma, basis = self.tier_baseline(profile)
        start = self._local_today()

        if mu <= 0.0:
            result = self._zero_projection(facility_id, horizon, start, basis)
        else:
            series = await self.fetch_facility_historical_demand(facility_id)
            async with self._semaphore():
                result = await asyncio.to_thread(
                    self._forecast_from_series,
                    facility_id, series, horizon, mu, sigma, basis, profile.get("inventory"), start,
                )
        self._forecast_cache[key] = (time.monotonic(), result)
        return copy.deepcopy(result)

    # ---------------------------- backtesting ------------------------------------------
    def _rolling_origin_backtest(self, y: np.ndarray, horizon: int, n_origins: int) -> Dict[str, Any]:
        n = len(y)
        origins = [n - horizon * (n_origins - j) for j in range(n_origins)]
        if origins[0] < SEASONAL_MIN_DAYS:
            return {"status": "insufficient_history", "history_days": n}
        sel = self._select_spec(y[: origins[0]])  # spec chosen only from data before the first origin
        if sel is None:
            return {"status": "model_fit_failed"}
        _, order, seasonal, trend = sel

        err_m, err_s, hits, tot = [], [], 0, 0
        for o in origins:
            train, actual = y[:o], y[o: o + horizon]
            try:
                res = self._fit_spec(train, order, seasonal, trend)
                fc = res.get_forecast(steps=horizon)
                m, ci = np.asarray(fc.predicted_mean), np.asarray(fc.conf_int(alpha=0.05))
            except Exception:
                continue
            snaive = np.array([train[len(train) - 7 + (h % 7)] for h in range(horizon)])
            err_m.append(np.abs(np.maximum(m, 0) - actual))
            err_s.append(np.abs(snaive - actual))
            hits += int(np.sum((actual >= ci[:, 0]) & (actual <= ci[:, 1])))
            tot += horizon
        if not err_m:
            return {"status": "model_fit_failed"}
        mae_m = float(np.mean(np.concatenate(err_m)))
        mae_s = float(np.mean(np.concatenate(err_s)))
        scale = float(np.mean(np.abs(y[7: origins[0]] - y[: origins[0] - 7])))
        return {
            "status": "ok",
            "model": "SARIMAX" + str(order) + "x" + str(seasonal),
            "horizon_days": horizon,
            "origins": len(err_m),
            "mae_model": round(mae_m, 3),
            "mae_seasonal_naive": round(mae_s, 3),
            "mase": round(mae_m / scale, 3) if scale > 0 else None,
            "skill_vs_seasonal_naive": round(1 - mae_m / mae_s, 3) if mae_s > 0 else None,
            "interval_coverage_95": round(hits / tot, 3) if tot else None,
        }

    async def backtest_facility_demand(self, facility_id: str, horizon_days: int = 7, n_origins: int = 8) -> Dict[str, Any]:
        """Offline diagnostic: is the model better than 'same as last week'? Positive skill = yes."""
        series = await self.fetch_facility_historical_demand(facility_id)
        async with self._semaphore():
            return await asyncio.to_thread(
                self._rolling_origin_backtest, series.to_numpy(dtype=float), horizon_days, n_origins
            )


# Global singleton instance
predictive_engine = PredictiveIntelligenceEngine()


# ------------------- module-level convenience functions (backwards compatible) --------
async def initialize_ml_engine():
    return await predictive_engine.initialize_from_db()


def predict_retention_score(
    recency: int, frequency: int, tenure: int,
    sex: Optional[str] = None,
    donor_type: Optional[str] = None,
    age_years: Optional[float] = None,
):
    return predictive_engine.predict_donor_retention(recency, frequency, tenure, sex, donor_type, age_years)


def forecast_demand_heuristic(
    historical_series: pd.Series, horizon_days: int = 7, facility_id: str = "REGIONAL-NETWORK"
):
    """Deterministic heuristic projection (flat mean x weekday profile). NOT an ARIMA forecast."""
    s = historical_series
    mean_val = float(s.mean()) if len(s) > 0 else 50.0
    std_val = float(s.std()) if len(s) > 1 else 10.0
    start = predictive_engine._local_today()
    return predictive_engine._heuristic_projection(
        facility_id, horizon_days, mean_val, std_val, start,
        predictive_engine._dow_factors(s), "heuristic_projection", "caller_supplied",
    )


forecast_demand_arima = forecast_demand_heuristic  # deprecated alias: the old name was misleading
