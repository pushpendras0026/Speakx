"""
Stage 2: Hybrid Imputation (Segment-First / Hierarchical KNN) (knn data filling) 

Strategy
--------
Users are grouped by immutable attributes: lifecycle_stage × age_band × region.

Within each group:
  - If the group has >= 3 members  → KNN imputer (k=3, Euclidean distance on
    numeric behavioural features).
  - If the group has  < 3 members  → Group-median fallback.

Since N < 60 (per spec), dropping rows is prohibited.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)

# Numeric columns eligible for imputation
NUMERIC_IMPUTE_COLS = [
    "sessions_last_7d",
    "exercises_completed_7d",
    "streak_current",
    "coins_balance",
    "preferred_hour",
    "notif_open_rate_30d",
    "motivation_score",
    "days_since_signup",
]

# Group keys (immutable attributes)
GROUP_KEYS = ["lifecycle_stage", "age_band", "region"]

# KNN parameter (spec says k=3)
KNN_K = 3
KNN_MIN_GROUP_SIZE = KNN_K  # need at least 3 rows to run KNN


def _impute_group_knn(group: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Apply KNN imputer within a group."""
    imputer = KNNImputer(n_neighbors=KNN_K, metric="nan_euclidean")
    group = group.copy()
    group[cols] = imputer.fit_transform(group[cols])
    return group


def _impute_group_median(group: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Fallback: fill NaN with group median for each column."""
    group = group.copy()
    for col in cols:
        if group[col].isna().any():
            median_val = group[col].median()
            if pd.isna(median_val):
                # If entire group has no data for this column, use global fallback later
                continue
            group[col] = group[col].fillna(median_val)
    return group


def _round_int_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Round imputed values back to integer for integer-typed columns."""
    int_cols = [
        "sessions_last_7d", "exercises_completed_7d",
        "streak_current", "coins_balance",
        "preferred_hour", "days_since_signup",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().clip(lower=0).astype("Int64")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply hierarchical imputation to the dataframe.

    Returns a new DataFrame with no NaN in numeric columns.
    Boolean feature columns (feature_*) are filled with False.
    """
    df = df.copy()

    # Separate boolean columns — fill with False (most conservative assumption)
    bool_cols = ["feature_ai_tutor_used", "feature_leaderboard_viewed", "feature_progress_checked"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).infer_objects(copy=False)

    # Only impute columns that are actually present & numeric
    cols_to_impute = [c for c in NUMERIC_IMPUTE_COLS if c in df.columns]

    # Cast to float for sklearn compatibility
    df[cols_to_impute] = df[cols_to_impute].astype(float)

    results = []
    group_stats = []

    for keys, group in df.groupby(GROUP_KEYS, observed=True):
        n = len(group)
        has_missing = group[cols_to_impute].isna().any().any()

        if not has_missing:
            results.append(group)
            continue

        if n >= KNN_MIN_GROUP_SIZE:
            method = "KNN"
            group = _impute_group_knn(group, cols_to_impute)
        else:
            method = "Median"
            group = _impute_group_median(group, cols_to_impute)

        group_stats.append({
            "group": keys,
            "n_users": n,
            "method": method,
        })
        results.append(group)

    df_imputed = pd.concat(results).sort_index()

    # Global fallback: if any NaN still remain (entire group had no data),
    # use dataset-wide median.
    still_missing = df_imputed[cols_to_impute].isna().sum()
    if still_missing.any():
        for col in cols_to_impute:
            if df_imputed[col].isna().any():
                global_median = df_imputed[col].median()
                if pd.isna(global_median):
                    global_median = 0
                df_imputed[col] = df_imputed[col].fillna(global_median)
                logger.warning(
                    "Used global median (%.2f) for '%s' — no group data available.",
                    global_median, col
                )

    # Log imputation summary
    for stat in group_stats:
        logger.info(
            "Group %s | n=%d | method=%s",
            stat["group"], stat["n_users"], stat["method"]
        )

    # Restore integer dtypes
    df_imputed = _round_int_cols(df_imputed)

    # Clip preferred_hour back to 0–23
    df_imputed["preferred_hour"] = (
        df_imputed["preferred_hour"].astype(float)
        .clip(0, 23).round().astype("Int64")
    )

    return df_imputed
