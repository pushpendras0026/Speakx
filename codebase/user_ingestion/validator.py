"""
Stage 1: Ingestion & Validation
Schema enforcement and type coercion for behavioral CSV data.
"""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Schema definition ──────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "user_id",
    "lifecycle_stage",
    "days_since_signup",
    "age_band",
    "region",
    "sessions_last_7d",
    "exercises_completed_7d",
    "streak_current",
    "coins_balance",
    "feature_ai_tutor_used",
    "feature_leaderboard_viewed",
    "feature_progress_checked",
    "preferred_hour",
    "notif_open_rate_30d",
    "motivation_score",
]

LIFECYCLE_VALUES = {"trial", "paid", "churned", "inactive"}
# age_band is dataset-specific (values differ across datasets).
# Unknown values are kept as-is — only truly null/empty entries are flagged.

DTYPE_MAP = {
    "days_since_signup":          "int",
    "sessions_last_7d":           "int",
    "exercises_completed_7d":     "int",
    "streak_current":             "int",
    "coins_balance":              "int",
    "preferred_hour":             "int",
    "notif_open_rate_30d":        "float",
    "motivation_score":           "float",
    "feature_ai_tutor_used":      "bool",
    "feature_leaderboard_viewed": "bool",
    "feature_progress_checked":   "bool",
}

NON_NEGATIVE_INT_COLS = [
    "sessions_last_7d", "exercises_completed_7d",
    "streak_current", "coins_balance", "days_since_signup",
]

RANGE_CONSTRAINTS = {
    "preferred_hour":      (0, 23),
    "notif_open_rate_30d": (0.0, 1.0),
    "motivation_score":    (0.0, 1.0),
}


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Convert string booleans (true/false/1/0) to actual bool (nullable)."""
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return series.astype(str).str.strip().str.lower().map(mapping)


def validate_and_coerce(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Validate schema, coerce types, flag constraint violations.

    Returns
    -------
    clean_df  : DataFrame with correct dtypes; rows with HARD violations removed.
    warnings  : List of warning strings for soft violations (range, categoricals).
    """
    warnings = []

    # ── 1. Header check ───────────────────────────────────────────────────────
    missing_headers = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_headers:
        raise ValueError(
            f"Pipeline failure: missing required columns: {missing_headers}"
        )

    # Work on a copy
    df = df.copy()

    # ── 2. Normalise string columns ───────────────────────────────────────────
    for col in ["lifecycle_stage", "age_band", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["user_id"] = df["user_id"].astype(str).str.strip()

    # ── 3. Unique user_id check ───────────────────────────────────────────────
    dupes = df[df["user_id"].duplicated(keep=False)]["user_id"].unique()
    if len(dupes):
        warnings.append(f"Duplicate user_ids detected (kept first): {list(dupes)}")
        df = df.drop_duplicates(subset="user_id", keep="first")

    # ── 4. Boolean coercion ───────────────────────────────────────────────────
    for col in ["feature_ai_tutor_used", "feature_leaderboard_viewed", "feature_progress_checked"]:
        df[col] = _coerce_bool(df[col])

    # ── 5. Numeric coercion (leave NaN for imputer) ───────────────────────────
    int_cols   = [c for c, t in DTYPE_MAP.items() if t == "int"]
    float_cols = [c for c, t in DTYPE_MAP.items() if t == "float"]

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 6. Hard constraint: non-negative integers ─────────────────────────────
    for col in NON_NEGATIVE_INT_COLS:
        bad = df[col].notna() & (df[col] < 0)
        if bad.any():
            bad_ids = df.loc[bad, "user_id"].tolist()
            warnings.append(
                f"Column '{col}' has negative values for users {bad_ids}; "
                "setting to NaN for imputation."
            )
            df.loc[bad, col] = float("nan")

    # ── 7. Soft constraints: categorical values ───────────────────────────────
    invalid_lc = ~df["lifecycle_stage"].isin(LIFECYCLE_VALUES)
    if invalid_lc.any():
        bad_ids = df.loc[invalid_lc, "user_id"].tolist()
        warnings.append(
            f"Unknown lifecycle_stage values for users {bad_ids}; "
            "defaulting to 'trial'."
        )
        df.loc[invalid_lc, "lifecycle_stage"] = "trial"

    # age_band: only flag rows where the value is null/empty after normalisation
    blank_ab = df["age_band"].isin(["", "nan", "none"])
    if blank_ab.any():
        df.loc[blank_ab, "age_band"] = "unknown"
        warnings.append(
            f"{blank_ab.sum()} rows had blank age_band; set to 'unknown'."
        )

    # ── 8. Range constraints ──────────────────────────────────────────────────
    for col, (lo, hi) in RANGE_CONSTRAINTS.items():
        out = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if out.any():
            bad_ids = df.loc[out, "user_id"].tolist()
            warnings.append(
                f"Column '{col}' out of range [{lo}, {hi}] for users {bad_ids}; "
                "setting to NaN for imputation."
            )
            df.loc[out, col] = float("nan")

    # ── 9. preferred_hour must be integer 0-23 ───────────────────────────────
    df["preferred_hour"] = df["preferred_hour"].round().astype("Int64")

    # Log summary
    n_missing = df[REQUIRED_COLUMNS].isna().sum()
    missing_summary = n_missing[n_missing > 0].to_dict()
    if missing_summary:
        logger.info("Missing value counts after validation: %s", missing_summary)

    return df, warnings
