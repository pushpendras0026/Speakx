"""
Stage 3: Feature Normalization & Engineering 

Transformations applied (per user_data_ingestion.md & PDF spec):

A. coins_balance   → log₅(1 + coins_balance)           [Virtual Economy] 
B. streak_current  → log₂(1 + streak_current)           [Gamification] 
C. preferred_hour  → 6 time-window labels (PDF spec)    [Temporal]
D. region          → passthrough (kept as-is)            [Demographic]  
E. All numeric cols → Min-Max normalization → [0, 1]     [Scaling]   

Time windows (from PDF, Timing Optimizer):
  early_morning  06:00 – 08:59
  mid_morning    09:00 – 11:59
  afternoon      12:00 – 14:59
  late_afternoon 15:00 – 17:59
  evening        18:00 – 20:59
  night          21:00 – 23:59  (and 00:00 – 05:59)
"""

import numpy as np
import pandas as pd

# ── Time-window mapping (PDF spec) ────────────────────────────────────────────
# Boundaries: [start, end)  — end is exclusive
TIME_WINDOW_BINS = [
    (6,  9,  "early_morning"),    # 06:00 – 08:59
    (9,  12, "mid_morning"),      # 09:00 – 11:59
    (12, 15, "afternoon"),        # 12:00 – 14:59
    (15, 18, "late_afternoon"),   # 15:00 – 17:59
    (18, 21, "evening"),          # 18:00 – 20:59
    # 21-23 and 00-05 → night
]


def _hour_to_window(hour: int) -> str:
    """Map a raw 0-23 hour to its time-window label (PDF spec)."""
    for start, end, label in TIME_WINDOW_BINS:
        if start <= hour < end:
            return label
    return "night"  # 21:00 – 05:59


def _log_base(x: float, base: float) -> float:
    """Compute log_base(1 + x), safe for x = 0."""
    return np.log1p(x) / np.log(base)


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-Max scale a series to [0, 1]. If constant, returns 0.0 for all."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


# Numeric columns to normalize after log-scaling
# Columns that are ALREADY in [0,1] (rates/scores) are also re-normalized so
# the full feature space is consistently bounded.
COLS_TO_NORMALIZE = [
    "sessions_last_7d",
    "exercises_completed_7d",
    "days_since_signup",
    "coins_balance_scaled",   # post log₅ transform
    "streak_scaled",          # post log₂ transform
    "notif_open_rate_30d",
    "motivation_score",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalisation and binning steps.

    New columns added:
      coins_balance_scaled        : log₅(1 + coins_balance)
      streak_scaled               : log₂(1 + streak_current)
      time_window                 : psycho-behavioural label for preferred_hour
      <col>_norm  for each col    : Min-Max scaled version in [0, 1]

    Original raw columns are preserved for auditability.
    """
    df = df.copy()

    # ── A. Virtual Economy: Log base-5 ───────────────────────────────────────
    df["coins_balance_scaled"] = df["coins_balance"].astype(float).apply(
        lambda x: _log_base(x, 5)
    )

    # ── B. Gamification: Log base-2 ──────────────────────────────────────────
    df["streak_scaled"] = df["streak_current"].astype(float).apply(
        lambda x: _log_base(x, 2)
    )

    # ── C. Temporal: Time-window binning (PDF spec) ───────────────────────────
    df["time_window"] = df["preferred_hour"].astype(int).apply(_hour_to_window)

    # ── D. Region: passthrough ───────────────────────────────────────────────
    # (already present, no transform)

    # ── E. Min-Max normalization → [0, 1] ────────────────────────────────────
    for col in COLS_TO_NORMALIZE:
        if col in df.columns:
            df[f"{col}_norm"] = _minmax_normalize(df[col].astype(float))

    return df
