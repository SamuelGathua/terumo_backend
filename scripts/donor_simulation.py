"""
scripts/donor_simulation.py - Event-first synthetic donor ledger (pure numpy, no database)
===========================================================================================

The old seeder drew (recency, frequency, tenure) from a KDE and then computed the label as
a noisy formula of those same three numbers - so any model "learned" the formula.

This simulator works the other way round. It generates the BEHAVIOUR (individual donation
events over time) from latent traits, and everything else is derived from events:

    latent engagement e ~ Beta(2, 3)           (never stored, never a feature)
    after donation k the donor comes back with p = sigmoid(base[type] + 3*(e-0.4)
                                                 + 0.5*min(k-1, 6) + 0.015*(age-28))
    if they come back, the gap is  min_interval[sex] + Gamma(2, 40*(1.6-e)) days
    otherwise they never donate again ("permanent lapse")

Retention labels are then produced by retention_data.build_training_frame (features at a
cutoff date, label = returned within H days), exactly as they would be from real data.

IMPORTANT: every parameter below is an ASSUMPTION chosen to give plausible shapes
(low first-time return, habit formation, longer gaps for low engagement). None is
calibrated to Kenyan data. Metrics from a model trained on this ledger are demo numbers.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from retention_data import MIN_INTERVAL_DAYS

BLOOD_TYPES = ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")
BLOOD_TYPE_WEIGHTS = (0.45, 0.04, 0.28, 0.03, 0.14, 0.02, 0.03, 0.01)

LOCATIONS = (
    "REGIONAL-HUB-01",
    "MOBILE-DRIVE-NAIROBI",
    "MOBILE-DRIVE-KISUMU",
    "MOBILE-DRIVE-MOMBASA",
    "MOBILE-DRIVE-NAKURU",
)
LOCATION_WEIGHTS = (0.30, 0.25, 0.15, 0.15, 0.15)


@dataclass(frozen=True)
class SimConfig:
    n_donors: int = 10_000
    history_days: int = 1460          # ~4 years: enough for purged train/calib/test at H=180
    seed: int = 42
    p_female: float = 0.30
    p_family_replacement: float = 0.25
    base_logit_voluntary: float = -0.6
    base_logit_family: float = -1.8
    engagement_effect: float = 3.0
    habit_effect: float = 0.5
    habit_cap: int = 6
    age_effect: float = 0.015
    gap_shape: float = 2.0
    gap_scale_days: float = 40.0
    pending_window_days: int = 7      # recent events may still be un-synced (offline-first app)
    pending_share: float = 0.20
    breach_rate: float = 0.02
    active_window_days: int = 365     # only used for the legacy donors.retention_status column


def _expit(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_donor_ledger(cfg: SimConfig, today: Optional[dt.date] = None) -> Tuple[List[dict], List[dict]]:
    """Returns (donor_rows, event_rows) as plain-Python dicts ready for bulk insert."""
    today = today or dt.date.today()
    rng = np.random.default_rng(cfg.seed)
    start = today - dt.timedelta(days=cfg.history_days)

    donors: List[dict] = []
    events: List[dict] = []

    for _ in range(cfg.n_donors):
        donor_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))
        sex = "F" if rng.random() < cfg.p_female else "M"
        donor_type = "FAMILY_REPLACEMENT" if rng.random() < cfg.p_family_replacement else "VOLUNTARY"
        engagement = float(rng.beta(2.0, 3.0))
        age_first = int(np.clip(rng.normal(28, 9), 18, 65))
        base = cfg.base_logit_family if donor_type == "FAMILY_REPLACEMENT" else cfg.base_logit_voluntary
        min_gap = MIN_INTERVAL_DAYS[sex]

        # --- behaviour: a chain of donations until the donor stops or we hit "today" -----
        day = start + dt.timedelta(days=int(rng.integers(0, cfg.history_days)))
        donor_days: List[dt.date] = []
        k = 0
        while day < today:
            k += 1
            donor_days.append(day)
            logit = (
                base
                + cfg.engagement_effect * (engagement - 0.4)
                + cfg.habit_effect * min(k - 1, cfg.habit_cap)
                + cfg.age_effect * (age_first - 28)
            )
            if rng.random() >= _expit(logit):
                break  # permanent lapse after this donation
            scale = cfg.gap_scale_days * (1.6 - engagement)
            day = day + dt.timedelta(days=min_gap + int(rng.gamma(cfg.gap_shape, scale)))

        # --- ledger rows ------------------------------------------------------------------
        for d in donor_days:
            hour = int(rng.integers(5, 14))  # naive UTC -> 08:00-16:59 Africa/Nairobi
            ts = dt.datetime.combine(d, dt.time(hour, int(rng.integers(0, 60))))
            recent = (today - d).days <= cfg.pending_window_days
            events.append(
                {
                    "event_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)),
                    "donor_id": donor_id,
                    "collection_timestamp": ts,
                    "location_id": str(rng.choice(LOCATIONS, p=LOCATION_WEIGHTS)),
                    "sync_status": "PENDING" if (recent and rng.random() < cfg.pending_share) else "SYNCED",
                    "cold_chain_breach_flag": bool(rng.random() < cfg.breach_rate),
                }
            )

        # --- donor profile + legacy snapshot columns, all derived from the events ---------
        first, last = donor_days[0], donor_days[-1]
        high_reactive = rng.random() < 0.05  # legacy column; belongs on screening_results
        s_co = rng.uniform(10.0, 18.5) if high_reactive else rng.uniform(0.10, 2.50)
        donors.append(
            {
                "donor_id": donor_id,
                "blood_type": str(rng.choice(BLOOD_TYPES, p=BLOOD_TYPE_WEIGHTS)),
                "tenure_days": (today - first).days,
                "recency_days": (today - last).days,
                "total_donations": len(donor_days),
                # Legacy columns. NOT labels and NOT read by ml_engine:
                "retention_probability": 0.5,
                "retention_status": int((today - last).days <= cfg.active_window_days),
                "syphilis_s_co_ratio": round(float(s_co), 2),
                "created_at": dt.datetime.combine(first, dt.time(8, 0)),
                # New columns (inserted only if the Donor model has them):
                "sex": sex,
                "donor_type": donor_type,
                "date_of_birth": first - dt.timedelta(days=int(age_first * 365.25 + rng.integers(0, 365))),
            }
        )

    return donors, events


def validate_ledger(donors: List[dict], events: List[dict], today: Optional[dt.date] = None) -> Dict[str, float]:
    """Hard invariants (raise on violation) plus a few summary statistics worth eyeballing."""
    today = today or dt.date.today()
    sex = {d["donor_id"]: d["sex"] for d in donors}
    days_by_donor: Dict[str, List[dt.date]] = defaultdict(list)
    for e in events:
        days_by_donor[e["donor_id"]].append(e["collection_timestamp"].date())

    gap_violations = 0
    for did, days in days_by_donor.items():
        days.sort()
        mi = MIN_INTERVAL_DAYS[sex[did]]
        gap_violations += sum(1 for a, b in zip(days, days[1:]) if (b - a).days < mi)

    future_events = sum(1 for e in events if e["collection_timestamp"].date() >= today)
    bad_tenure = sum(1 for d in donors if d["tenure_days"] < d["recency_days"])
    bad_single = sum(1 for d in donors if d["total_donations"] == 1 and d["tenure_days"] != d["recency_days"])
    if gap_violations or future_events or bad_tenure or bad_single:
        raise ValueError(
            "Ledger invariants violated: gaps<min_interval="
            + str(gap_violations)
            + ", future_events="
            + str(future_events)
            + ", tenure<recency="
            + str(bad_tenure)
            + ", single-donation tenure!=recency="
            + str(bad_single)
        )

    old = [d for d in donors if d["tenure_days"] >= 540]  # first donation long ago -> negligible censoring
    return {
        "donors": len(donors),
        "events": len(events),
        "mean_donations_per_donor": round(len(events) / max(len(donors), 1), 2),
        "share_with_repeat_donation": round(float(np.mean([d["total_donations"] > 1 for d in donors])), 3),
        "return_rate_after_first_donation_old_cohort": round(
            float(np.mean([d["total_donations"] > 1 for d in old])) if old else float("nan"), 3
        ),
        "share_active_last_365d": round(float(np.mean([d["recency_days"] <= 365 for d in donors])), 3),
    }
