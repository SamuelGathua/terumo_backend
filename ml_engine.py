"""
ml_engine.py - Machine Learning & Statistical Inference Engine for ABIS
========================================================================

Chain-of-Thought Rationale:
1. Donor Retention Model (Random Forest Classifier):
   - Theoretical Motivation: Blood donor retention is driven by multidimensional non-linear
     interactions between Recency (decay of motivation), Frequency (habit formation), and
     Tenure (commitment duration). A Random Forest ensemble captures these non-linear thresholds
     without making Gaussian distribution assumptions.
   - Target Variable: Retention Status (1 = Returned/Active donor, 0 = Lapsed donor).
   - Heuristic Fallback: Logistic decay formulation based on empirical transfusion medicine baselines
     if model weights are not yet fitted from the database ledger.

2. Demand Forecasting Model (Time-Series & Stochastic Random Walk):
   - Theoretical Motivation: Daily blood requests exhibit mean reversion around facility capacity
     combined with Poisson/Gaussian stochastic shocks (emergency trauma events, maternal hemorrhages).
   - Mean Reversion Formula: D_t = D_{t-1} + theta * (mu - D_{t-1}) + sigma * epsilon_t
   - Forecast Architecture: ARIMA(p,d,q) fits the autoregressive autocorrelation and seasonal trends,
     projecting 7-day lookahead trajectories with 95% confidence intervals to identify critical shortage windows.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("abis.ml_engine")
logger.setLevel(logging.INFO)

# Global model cache to avoid cold-start delays on every inference call
_donor_model: Optional[RandomForestClassifier] = None


def train_retention_model(donor_df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on donor behavioral features (RFM).

    Features:
        - recency: Days since last donation (negative correlation with retention)
        - frequency: Cumulative donations count (positive correlation with habituation)
        - tenure: Days since first donation (loyalty factor)
    Target:
        - retention_status: 1 (retained) vs 0 (lapsed)
    """
    global _donor_model

    required_cols = {"recency", "frequency", "tenure", "retention_status"}
    if not required_cols.issubset(set(donor_df.columns)):
        raise ValueError(f"DataFrame missing required columns: {required_cols - set(donor_df.columns)}")

    X = donor_df[["recency", "frequency", "tenure"]].values
    y = donor_df["retention_status"].values.astype(int)

    # Balance class weights to account for donor drop-off skew
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X, y)
    _donor_model = clf
    logger.info(f"Random Forest Retention model successfully trained on {len(X)} records.")
    return clf


def predict_retention_score(
    recency: int,
    frequency: int,
    tenure: int,
    model: Optional[RandomForestClassifier] = None
) -> Dict[str, Any]:
    """
    Predict probability of donor retention with fallback heuristic logic.
    """
    global _donor_model
    active_model = model or _donor_model

    if active_model is not None:
        try:
            sample = np.array([[recency, frequency, tenure]])
            # Probability of class 1 (Retained)
            prob_retained = float(active_model.predict_proba(sample)[0][1])
        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Falling back to clinical heuristic.")
            prob_retained = _heuristic_retention_score(recency, frequency, tenure)
    else:
        # Heuristic prior
        prob_retained = _heuristic_retention_score(recency, frequency, tenure)

    # Risk categorization
    if prob_retained >= 0.70:
        risk_tier = "LOW_RISK"
        recommendation = "Standard automated SMS invitation for next scheduled mobile donor drive."
    elif prob_retained >= 0.40:
        risk_tier = "MODERATE_RISK"
        recommendation = "Targeted personalized WhatsApp message with community impact story."
    else:
        risk_tier = "HIGH_RISK"
        recommendation = "Proactive phone recall by donor liaison coordinator with transport assistance."

    return {
        "retention_probability": round(prob_retained, 4),
        "retention_status": 1 if prob_retained >= 0.5 else 0,
        "risk_tier": risk_tier,
        "recommended_action": recommendation,
    }


def _heuristic_retention_score(recency: int, frequency: int, tenure: int) -> float:
    """
    Deterministic clinical heuristic for donor retention:
    - High frequency (>5) and low recency (<90 days) -> High return probability (~85-95%)
    - High recency (>365 days) -> High lapse probability (<25%)
    """
    # Sigmoidal habit score from frequency
    freq_factor = 1.0 / (1.0 + np.exp(-0.4 * (frequency - 2)))
    # Recency decay penalty
    recency_penalty = np.exp(-0.005 * max(0, recency - 60))
    # Tenure loyalty ratio
    tenure_ratio = min(1.0, (tenure + 1) / (recency + 60))

    score = 0.5 * freq_factor * recency_penalty + 0.3 * tenure_ratio + 0.2 * (1.0 - min(1.0, recency / 730.0))
    return float(np.clip(score, 0.05, 0.98))


def forecast_demand_arima(
    historical_series: pd.Series,
    horizon_days: int = 7,
    facility_id: str = "REGIONAL-NETWORK"
) -> Dict[str, Any]:
    """
    Forecast hospital blood demand for the next N days.
    Uses ARIMA time-series modeling with a mean-reverting stochastic fallback.
    """
    today = datetime.date.today()
    clean_series = historical_series.dropna().astype(float)

    if len(clean_series) < 14:
        # If insufficient data points, fallback to mean-reverting stochastic forecast
        baseline_mean = float(clean_series.mean()) if len(clean_series) > 0 else 65.0
        stochastic_sigma = float(clean_series.std()) if len(clean_series) > 1 else 12.0
        return _stochastic_fallback_forecast(baseline_mean, stochastic_sigma, horizon_days, facility_id, today)

    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Fit ARIMA(1, 1, 1) or simple AR(1)
        model = ARIMA(clean_series.values, order=(1, 1, 1))
        model_fit = model.fit()

        forecast_res = model_fit.get_forecast(steps=horizon_days)
        mean_forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        points = []
        for i in range(horizon_days):
            target_date = today + datetime.timedelta(days=i + 1)
            pred = max(0.0, float(mean_forecast[i]))
            lower = max(0.0, float(conf_int[i, 0]))
            upper = max(pred, float(conf_int[i, 1]))
            points.append({
                "date": target_date,
                "predicted_units": round(pred, 1),
                "confidence_lower_95": round(lower, 1),
                "confidence_upper_95": round(upper, 1),
            })

        baseline_mean = float(clean_series.mean())
        volatility = float(clean_series.std())

        # Check for rebalance alert: if projected cumulative 7-day demand spikes 25% above normal
        total_projected = sum(p["predicted_units"] for p in points)
        expected_normal = baseline_mean * horizon_days
        alert = None
        if total_projected > expected_normal * 1.25:
            alert = (
                f"HIGH DEFICIT ALERT: Projected 7-day demand ({round(total_projected)} units) "
                f"exceeds historical baseline by {round(((total_projected/expected_normal)-1)*100)}%. "
                f"Immediate inter-facility rebalancing recommended."
            )

        return {
            "facility_id": facility_id,
            "forecast_horizon_days": horizon_days,
            "baseline_daily_mean": round(baseline_mean, 2),
            "stochastic_volatility": round(volatility, 2),
            "forecast": points,
            "rebalance_alert": alert,
        }

    except Exception as e:
        logger.error(f"ARIMA fitting failed: {e}. Executing stochastic fallback.")
        baseline_mean = float(clean_series.mean()) if len(clean_series) > 0 else 65.0
        stochastic_sigma = float(clean_series.std()) if len(clean_series) > 1 else 12.0
        return _stochastic_fallback_forecast(baseline_mean, stochastic_sigma, horizon_days, facility_id, today)


def _stochastic_fallback_forecast(
    baseline_mean: float,
    sigma: float,
    horizon_days: int,
    facility_id: str,
    start_date: datetime.date
) -> Dict[str, Any]:
    """
    Mean-reverting stochastic projection when series history is brief.
    D_t = D_{t-1} + theta * (mu - D_{t-1}) + sigma * epsilon
    """
    theta = 0.35  # Mean reversion speed
    current_val = baseline_mean
    points = []

    np.random.seed(42)
    for i in range(horizon_days):
        target_date = start_date + datetime.timedelta(days=i + 1)
        shock = np.random.normal(0, sigma * 0.5)
        current_val = current_val + theta * (baseline_mean - current_val) + shock
        pred = max(5.0, round(float(current_val), 1))
        lower = max(0.0, round(pred - 1.96 * (sigma * 0.7), 1))
        upper = round(pred + 1.96 * (sigma * 0.7), 1)

        points.append({
            "date": target_date,
            "predicted_units": pred,
            "confidence_lower_95": lower,
            "confidence_upper_95": upper,
        })

    return {
        "facility_id": facility_id,
        "forecast_horizon_days": horizon_days,
        "baseline_daily_mean": round(baseline_mean, 2),
        "stochastic_volatility": round(sigma, 2),
        "forecast": points,
        "rebalance_alert": None,
    }
