"""
tests/test_retention_forecast.py
=================================
Pure-Python tests for:
  * retention_data - feature engineering, cutoffs, splits
  * scripts/donor_simulation - ledger invariants
  * ml_engine (offline) - heuristic prior, series builder, tier baseline, zero projection

No database. No network. All assertions are deterministic from synthetic data.
"""

from __future__ import annotations

import datetime as dt
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make sure the parent (terumo_backend/) is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retention_data as rd
from scripts.donor_simulation import SimConfig, simulate_donor_ledger, validate_ledger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = dt.date(2025, 6, 30)


def _make_events(donor_days: dict) -> pd.DataFrame:
    """donor_days = {donor_id: [date, ...]}"""
    rows = []
    for did, days in donor_days.items():
        for d in days:
            rows.append({"donor_id": did, "collection_timestamp": pd.Timestamp(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# retention_data: engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_single_donation(self):
        f = rd.engineer_features([60], [1], [60])
        assert f.iloc[0]["is_repeat"] == 0.0
        assert f.iloc[0]["avg_interval_days"] == -1.0  # sentinel for first-time donors
        assert f.iloc[0]["overdue_ratio"] == -1.0
        assert f.iloc[0]["days_past_eligibility"] == 60 - rd.DEFAULT_MIN_INTERVAL_DAYS

    def test_repeat_donor_avg_interval(self):
        # 3 donations over 270 days: first at day 0, last at day 210, recency = 60
        # span = tenure - recency = (270-60) = 210; avg = 210/2 = 105
        f = rd.engineer_features([60], [3], [270])
        assert abs(f.iloc[0]["avg_interval_days"] - 105.0) < 1e-6

    def test_tenure_clamped_to_recency(self):
        # Caller passes tenure < recency (impossible, should be corrected)
        f = rd.engineer_features([100], [1], [50])
        assert f.iloc[0]["tenure_days"] == 100.0

    def test_vectorised(self):
        recency = [30, 60, 120]
        freq = [5, 2, 1]
        tenure = [365, 200, 120]
        f = rd.engineer_features(recency, freq, tenure)
        assert len(f) == 3
        assert all(f["recency_days"] == recency)


# ---------------------------------------------------------------------------
# retention_data: _prep_events tz handling
# ---------------------------------------------------------------------------

class TestPrepEvents:
    def test_naive_timestamps_passthrough(self):
        ev = pd.DataFrame({
            "donor_id": ["D1"],
            "collection_timestamp": [pd.Timestamp("2024-01-15 08:00:00")]
        })
        out = rd._prep_events(ev)
        assert out.iloc[0]["day"] == pd.Timestamp("2024-01-15")

    def test_tz_aware_stripped(self):
        ev = pd.DataFrame({
            "donor_id": ["D1"],
            "collection_timestamp": [pd.Timestamp("2024-01-15 08:00:00", tz="Africa/Nairobi")]
        })
        out = rd._prep_events(ev)
        # tz-converted to UTC, then stripped - still the same calendar date at 08:00 Nairobi = 05:00 UTC
        assert out.iloc[0]["day"].tzinfo is None


# ---------------------------------------------------------------------------
# retention_data: snapshot_features
# ---------------------------------------------------------------------------

class TestSnapshotFeatures:
    def test_no_donor_after_cutoff(self):
        ev = _make_events({"D1": [dt.date(2024, 1, 10)]})
        snap = rd.snapshot_features(ev, as_of=dt.date(2024, 1, 9))
        assert snap.empty

    def test_recency_computation(self):
        ev = _make_events({"D1": [dt.date(2024, 1, 1), dt.date(2024, 3, 1)]})
        snap = rd.snapshot_features(ev, as_of=dt.date(2024, 4, 1))
        row = snap.loc["D1"]
        assert row["recency_days"] == 31  # days from 2024-03-01 to 2024-04-01
        assert row["total_donations"] == 2

    def test_multiple_donors(self):
        ev = _make_events({
            "A": [dt.date(2024, 1, 1)],
            "B": [dt.date(2024, 2, 1), dt.date(2024, 3, 1)],
        })
        snap = rd.snapshot_features(ev, as_of=dt.date(2024, 4, 1))
        assert set(snap.index) == {"A", "B"}


# ---------------------------------------------------------------------------
# retention_data: make_cutoffs
# ---------------------------------------------------------------------------

class TestMakeCutoffs:
    def _events(self, n_days=700):
        start = pd.Timestamp(TODAY) - pd.Timedelta(days=n_days)
        rows = [{"donor_id": "D1", "collection_timestamp": start + pd.Timedelta(days=i)}
                for i in range(0, n_days, 5)]
        return pd.DataFrame(rows)

    def test_cutoffs_ordered(self):
        ev = self._events(700)
        cuts = rd.make_cutoffs(ev, horizon_days=180, as_of=TODAY)
        assert cuts == sorted(cuts)

    def test_cutoffs_not_empty(self):
        ev = self._events(700)
        cuts = rd.make_cutoffs(ev, horizon_days=180, as_of=TODAY)
        assert len(cuts) > 0

    def test_empty_events(self):
        ev = pd.DataFrame(columns=["donor_id", "collection_timestamp"])
        cuts = rd.make_cutoffs(ev, horizon_days=180, as_of=TODAY)
        assert cuts == []

    def test_label_windows_observed(self):
        ev = self._events(700)
        H = 180
        cuts = rd.make_cutoffs(ev, horizon_days=H, as_of=TODAY)
        for c in cuts:
            assert c + pd.Timedelta(days=H) <= pd.Timestamp(TODAY)


# ---------------------------------------------------------------------------
# retention_data: temporal_split
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def _enough_cutoffs(self, n=20):
        base = pd.Timestamp("2023-01-01")
        return [base + pd.Timedelta(days=30 * i) for i in range(n)]

    def test_purged(self):
        cs = self._enough_cutoffs(20)
        H = 180
        split = rd.temporal_split(cs, H, n_test=3, n_calib=3)
        h = pd.Timedelta(days=H)
        # last train cutoff + H <= first calib cutoff
        assert split["train"][-1] + h <= split["calib"][0]
        # last calib cutoff + H <= first test cutoff
        assert split["calib"][-1] + h <= split["test"][0]

    def test_raises_too_few_cutoffs(self):
        cs = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
        with pytest.raises(ValueError):
            rd.temporal_split(cs, 180)

    def test_all_partitions_non_empty(self):
        cs = self._enough_cutoffs(20)
        split = rd.temporal_split(cs, 180)
        assert split["train"] and split["calib"] and split["test"]


# ---------------------------------------------------------------------------
# retention_data: labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_returned_within_horizon_is_1(self):
        cutoff = pd.Timestamp("2024-01-01")
        ev = pd.DataFrame({
            "donor_id": ["D1"],
            "day": [pd.Timestamp("2024-02-01")]
        })
        idx = pd.Index(["D1"])
        assert rd._labels(ev, cutoff, 180, idx)[0] == 1

    def test_returned_after_horizon_is_0(self):
        cutoff = pd.Timestamp("2024-01-01")
        ev = pd.DataFrame({
            "donor_id": ["D1"],
            "day": [pd.Timestamp("2024-08-01")]  # > 180 days away
        })
        idx = pd.Index(["D1"])
        assert rd._labels(ev, cutoff, 180, idx)[0] == 0

    def test_on_cutoff_day_is_0(self):
        cutoff = pd.Timestamp("2024-01-01")
        ev = pd.DataFrame({"donor_id": ["D1"], "day": [cutoff]})
        idx = pd.Index(["D1"])
        assert rd._labels(ev, cutoff, 180, idx)[0] == 0  # same day is NOT in (cutoff, cutoff+H]


# ---------------------------------------------------------------------------
# donor_simulation
# ---------------------------------------------------------------------------

class TestDonorSimulation:
    def test_small_ledger_no_violations(self):
        cfg = SimConfig(n_donors=500, history_days=730, seed=0)
        donors, events = simulate_donor_ledger(cfg, today=TODAY)
        stats = validate_ledger(donors, events, today=TODAY)
        assert stats["donors"] == 500
        assert stats["events"] >= 500  # at least one event per donor

    def test_min_gap_respected(self):
        cfg = SimConfig(n_donors=200, history_days=730, seed=1)
        donors, events = simulate_donor_ledger(cfg, today=TODAY)
        # validate_ledger raises if any gap < min_interval
        validate_ledger(donors, events, today=TODAY)

    def test_repeat_donors_exist(self):
        cfg = SimConfig(n_donors=1000, history_days=730, seed=42)
        donors, events = simulate_donor_ledger(cfg, today=TODAY)
        stats = validate_ledger(donors, events, today=TODAY)
        assert stats["share_with_repeat_donation"] > 0.0


# ---------------------------------------------------------------------------
# ml_engine (offline) - heuristic prior
# ---------------------------------------------------------------------------

class TestHeuristicPrior:
    def _engine(self):
        from ml_engine import PredictiveIntelligenceEngine
        # Reset singleton for tests
        PredictiveIntelligenceEngine._instance = None
        eng = PredictiveIntelligenceEngine()
        return eng

    def test_very_recent_high_freq_high_prob(self):
        eng = self._engine()
        result = eng.predict_donor_retention(recency_days=10, total_donations=12, tenure_days=1000)
        assert result["retention_probability"] > 0.7, "Active, frequent donor should score HIGH"

    def test_very_lapsed_low_prob(self):
        eng = self._engine()
        result = eng.predict_donor_retention(recency_days=700, total_donations=1, tenure_days=700)
        assert result["retention_probability"] < 0.4, "Lapsed first-time donor should score LOW"

    def test_not_yet_eligible(self):
        eng = self._engine()
        result = eng.predict_donor_retention(recency_days=30, total_donations=5, tenure_days=500, sex="F")
        # Female min_interval = 120; 30 < 120 -> days_until_eligible > 0
        assert result["days_until_eligible"] == 90
        assert "eligible" in result["recommended_action"].lower()

    def test_risk_tiers_match_probability(self):
        eng = self._engine()
        for recency, expected_tier in [(10, "LOW_RISK"), (200, "HIGH_RISK")]:
            result = eng.predict_donor_retention(recency_days=recency, total_donations=2, tenure_days=400)
            # Verify tier boundaries are consistent
            prob = result["retention_probability"]
            if prob >= 0.70:
                assert result["risk_tier"] == "LOW_RISK"
            elif prob >= 0.40:
                assert result["risk_tier"] == "AT_RISK"
            else:
                assert result["risk_tier"] == "HIGH_RISK"


# ---------------------------------------------------------------------------
# ml_engine (offline) - build_daily_series
# ---------------------------------------------------------------------------

class TestBuildDailySeries:
    def _eng(self):
        from ml_engine import PredictiveIntelligenceEngine
        PredictiveIntelligenceEngine._instance = None
        return PredictiveIntelligenceEngine()

    def test_empty_timestamps_returns_empty(self):
        eng = self._eng()
        s = eng.build_daily_series([], [], dt.date(2024, 1, 1), dt.date(2024, 1, 10))
        assert s.empty

    def test_zeros_filled_on_missing_days(self):
        eng = self._eng()
        ts = [dt.datetime(2024, 1, 1, 10, 0), dt.datetime(2024, 1, 5, 10, 0)]
        units = [3.0, 7.0]
        s = eng.build_daily_series(ts, units, dt.date(2024, 1, 1), dt.date(2024, 1, 5))
        assert len(s) == 5
        assert s.iloc[0] == 3.0
        assert s.iloc[-1] == 7.0
        assert s.iloc[1] == 0.0

    def test_aggregation_same_day(self):
        eng = self._eng()
        ts = [dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 14, 0)]
        units = [5.0, 10.0]
        s = eng.build_daily_series(ts, units, dt.date(2024, 1, 1), dt.date(2024, 1, 1))
        assert s.iloc[0] == 15.0


# ---------------------------------------------------------------------------
# ml_engine: _zero_projection
# ---------------------------------------------------------------------------

class TestZeroProjection:
    def test_structure(self):
        from ml_engine import PredictiveIntelligenceEngine
        PredictiveIntelligenceEngine._instance = None
        eng = PredictiveIntelligenceEngine()
        out = eng._zero_projection("FAC-TEST", 7, dt.date(2025, 1, 1), "test_basis")
        assert out["baseline_daily_mean"] == 0.0
        assert len(out["forecast"]) == 7
        assert all(p["predicted_units"] == 0.0 for p in out["forecast"])
        assert out["rebalance_alert"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
