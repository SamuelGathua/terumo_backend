"""
retention_data.py - Point-in-time dataset construction for donor-retention modelling
====================================================================================

Why this module exists
----------------------
Training on a *stored* ``retention_status`` column that sits on the same row as
``recency_days`` / ``total_donations`` / ``tenure_days`` lets the label be a formula
of (or leak from) the features. This module rebuilds BOTH features and label from the
immutable ``donation_events`` ledger instead:

    features = what was knowable on a cutoff date T      (events with day <= T)
    label    = did the donor donate again in (T, T + H]?  (H = horizon, default 180 d)

Splits are temporal and purged: no training label window reaches into the period that
a later split's features were computed from.

Conventions
-----------
* ``events``: DataFrame[donor_id, collection_timestamp]. Pass SYNCED events only.
  Timestamps are naive UTC (tz-aware values are converted). Resolution is the day.
* ``donors`` (optional): DataFrame[donor_id, sex?, donor_type?, date_of_birth?].
  Missing columns are skipped, so the pipeline also runs on the current schema.
* Everything here is pure pandas/numpy - no database access - so it can be unit-tested.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HORIZON_DAYS_DEFAULT = 180
# Donors silent for longer than this are effectively gone. Keeping them in the training pool
# adds thousands of trivial negatives (inflating AUC) and makes prevalence drift as the
# ledger grows. Scoring is not filtered; the model just predicts "very low" for them.
MAX_RECENCY_DAYS_DEFAULT = 730

# ASSUMPTION - verify against the current Kenya NBTS donor-eligibility guideline.
MIN_INTERVAL_DAYS: Dict[str, int] = {"M": 90, "F": 120}
DEFAULT_MIN_INTERVAL_DAYS = 90

LABEL_COL = "label"
CUTOFF_COL = "cutoff"

# Candidate model inputs, in a fixed order. select_features() keeps those that exist.
FEATURE_ORDER = [
    "recency_days",
    "total_donations",
    "tenure_days",
    "avg_interval_days",
    "overdue_ratio",
    "is_repeat",
    "days_past_eligibility",
    "is_female",
    "is_family_replacement",
    "age_years",
]


# ----------------------------------------------------------------------------------
# Feature engineering (shared by training AND inference so they cannot drift apart)
# ----------------------------------------------------------------------------------
def engineer_features(recency, frequency, tenure, min_interval=DEFAULT_MIN_INTERVAL_DAYS) -> pd.DataFrame:
    """
    Vectorised RFM feature builder. Everything except the optional donor attributes is
    derivable from (recency, frequency, tenure), so the live API can call the model with
    the same three numbers it uses today.

    * tenure is clamped to >= recency (a donor cannot have been active for less time
      than has passed since their last donation).
    * avg_interval_days = (tenure - recency) / (n - 1) -- gap between first and last
      donation, spread over n-1 intervals. Sentinel -1 for first-time donors.
    * overdue_ratio = recency / avg_interval -- >1 means "later than their norm".
    * days_past_eligibility = recency - min_interval -- negative = not yet eligible.
    """
    rec = np.asarray(recency, dtype=float)
    n = np.asarray(frequency, dtype=float)
    ten = np.maximum(np.asarray(tenure, dtype=float), rec)
    mi = np.broadcast_to(np.asarray(min_interval, dtype=float), rec.shape)

    repeat = n > 1
    span = np.maximum(ten - rec, 0.0)
    avg_interval = np.where(repeat, span / np.maximum(n - 1.0, 1.0), -1.0)
    overdue = np.where(repeat, rec / np.maximum(avg_interval, 1.0), -1.0)

    return pd.DataFrame(
        {
            "recency_days": rec,
            "total_donations": n,
            "tenure_days": ten,
            "avg_interval_days": avg_interval,
            "overdue_ratio": overdue,
            "is_repeat": repeat.astype(float),
            "days_past_eligibility": rec - mi,
        }
    )


# ----------------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------------
def _prep_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[["donor_id", "collection_timestamp"]].copy()
    ts = pd.to_datetime(ev["collection_timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    ev["day"] = ts.dt.normalize()
    return ev[["donor_id", "day"]]


def _index_donors(donors: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if donors is None:
        return None
    if "donor_id" in donors.columns:
        donors = donors.set_index("donor_id")
    return donors


def _snapshot(ev: pd.DataFrame, as_of: pd.Timestamp, donors: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Features for every donor with >= 1 donation on/before as_of. ev is prepared."""
    past = ev[ev["day"] <= as_of]
    if past.empty:
        return pd.DataFrame(columns=FEATURE_ORDER)

    g = past.groupby("donor_id")["day"].agg(first_day="min", last_day="max", n="count")
    recency = (as_of - g["last_day"]).dt.days.to_numpy()
    tenure = (as_of - g["first_day"]).dt.days.to_numpy()
    ids = g.index.to_series()

    min_interval = DEFAULT_MIN_INTERVAL_DAYS
    if donors is not None and "sex" in donors.columns:
        sex = ids.map(donors["sex"])
        min_interval = sex.map(MIN_INTERVAL_DAYS).fillna(DEFAULT_MIN_INTERVAL_DAYS).to_numpy()

    feats = engineer_features(recency, g["n"].to_numpy(), tenure, min_interval)
    feats.index = g.index

    if donors is not None:
        if "sex" in donors.columns:
            sex = ids.map(donors["sex"])
            feats["is_female"] = np.where(sex.isna(), np.nan, (sex == "F").astype(float)).astype(float)
        if "donor_type" in donors.columns:
            dtype_ = ids.map(donors["donor_type"])
            feats["is_family_replacement"] = np.where(
                dtype_.isna(), np.nan, (dtype_ == "FAMILY_REPLACEMENT").astype(float)
            ).astype(float)
        if "date_of_birth" in donors.columns:
            dob = pd.to_datetime(ids.map(donors["date_of_birth"]))
            feats["age_years"] = ((as_of - dob).dt.days / 365.25).to_numpy()
    return feats


def _labels(ev: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int, index: pd.Index) -> np.ndarray:
    end = cutoff + pd.Timedelta(days=horizon_days)
    returned = ev.loc[(ev["day"] > cutoff) & (ev["day"] <= end), "donor_id"].unique()
    return index.isin(returned).astype(int)


# ----------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------
def snapshot_features(events: pd.DataFrame, as_of, donors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Features as known on as_of. Also the right input for batch-scoring today's donors."""
    return _snapshot(_prep_events(events), pd.Timestamp(as_of).normalize(), _index_donors(donors))


def build_training_frame(
    events: pd.DataFrame,
    cutoff,
    horizon_days: int = HORIZON_DAYS_DEFAULT,
    donors: Optional[pd.DataFrame] = None,
    observed_until: Optional[pd.Timestamp] = None,
    max_recency_days: Optional[int] = MAX_RECENCY_DAYS_DEFAULT,
) -> pd.DataFrame:
    """One row per donor active on/before cutoff: features(cutoff) + label(returned in (cutoff, cutoff+H])."""
    cutoff = pd.Timestamp(cutoff).normalize()
    if observed_until is not None and cutoff + pd.Timedelta(days=horizon_days) > pd.Timestamp(observed_until):
        raise ValueError(
            "Label window for cutoff "
            + str(cutoff.date())
            + " extends past the observed data ("
            + str(pd.Timestamp(observed_until).date())
            + "): labels would be right-censored."
        )
    ev = _prep_events(events)
    feats = _snapshot(ev, cutoff, _index_donors(donors))
    if max_recency_days is not None:
        feats = feats[feats["recency_days"] <= max_recency_days].copy()
    feats[LABEL_COL] = _labels(ev, cutoff, horizon_days, feats.index)
    feats[CUTOFF_COL] = cutoff
    return feats


def make_cutoffs(
    events: pd.DataFrame,
    horizon_days: int,
    as_of,
    step_days: int = 30,
    min_history_days: int = 90,
    sync_lag_days: int = 7,
) -> List[pd.Timestamp]:
    """
    Evenly spaced cutoffs whose full label window is observed. The newest cutoff is
    as_of - sync_lag - H (offline-first devices sync late, so the last days are incomplete).
    """
    if events.empty:
        return []
    ev = _prep_events(events)
    first = ev["day"].min()
    last_valid = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=sync_lag_days + horizon_days)
    start = first + pd.Timedelta(days=min_history_days)
    if last_valid < start:
        return []
    n_steps = (last_valid - start).days // step_days
    return [last_valid - pd.Timedelta(days=step_days * i) for i in range(n_steps, -1, -1)]


def temporal_split(
    cutoffs: List[pd.Timestamp], horizon_days: int, n_test: int = 3, n_calib: int = 3
) -> Dict[str, List[pd.Timestamp]]:
    """
    train | calib | test, ordered in time and PURGED: every cutoff in an earlier split
    satisfies cutoff + H <= first cutoff of the next split, so its label window never
    overlaps the period used to compute a later split's features.
    """
    cs = sorted(cutoffs)
    h = pd.Timedelta(days=horizon_days)
    if len(cs) < n_test + n_calib + 2:
        raise ValueError(
            "Only " + str(len(cs)) + " usable cutoffs - need more donation history for a purged temporal split."
        )
    test = cs[-n_test:]
    calib_pool = [c for c in cs if c + h <= test[0]]
    calib = calib_pool[-n_calib:]
    train = [c for c in cs if calib and c + h <= calib[0]]
    if not (train and calib and test):
        raise ValueError("History too short to produce non-empty train/calibration/test splits after purging.")
    return {"train": train, "calib": calib, "test": test}


def build_training_set(
    events: pd.DataFrame,
    cutoffs: List[pd.Timestamp],
    horizon_days: int,
    donors: Optional[pd.DataFrame] = None,
    observed_until: Optional[pd.Timestamp] = None,
    max_recency_days: Optional[int] = MAX_RECENCY_DAYS_DEFAULT,
) -> pd.DataFrame:
    ev = _prep_events(events)
    dx = _index_donors(donors)
    frames = []
    for c in cutoffs:
        c = pd.Timestamp(c).normalize()
        if observed_until is not None and c + pd.Timedelta(days=horizon_days) > pd.Timestamp(observed_until):
            raise ValueError("cutoff " + str(c.date()) + " has a censored label window")
        f = _snapshot(ev, c, dx)
        if max_recency_days is not None and not f.empty:
            f = f[f["recency_days"] <= max_recency_days].copy()
        if f.empty:
            continue
        f[LABEL_COL] = _labels(ev, c, horizon_days, f.index)
        f[CUTOFF_COL] = c
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=FEATURE_ORDER + [LABEL_COL, CUTOFF_COL])
    return pd.concat(frames)


def select_features(frame: pd.DataFrame) -> List[str]:
    """Model inputs present (and not entirely missing) in this frame, in a fixed order."""
    return [c for c in FEATURE_ORDER if c in frame.columns and frame[c].notna().any()]
