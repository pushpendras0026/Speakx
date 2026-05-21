"""
User Data Ingestion Pipeline  –  Project Aurora (SpeakX)
=========================================================
Covers Task 1, Points 2 & 3:
  2. Ingestion & Validation        (Stage 1)
  2. Hybrid KNN Imputation         (Stage 2)
  3. Feature Engineering           (Stage 3)
  4. Intelligence Generation       (Stage 4)
       - Activeness Score  (Adaptive Sigmoid)
       - Churn Risk Signal (Contextual)
       - Propensity Scores (4-Drive Model)

Usage   
-----
    python user_data_ingestion.py --input data/input/user_behavioral_data.csv \
                                  --output data/output/user_profiles.csv

The output CSV is a fully processed user profile table ready for
downstream MECE segmentation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from user_ingestion.validator import validate_and_coerce
from user_ingestion.imputer import impute_missing
from user_ingestion.feature_engineer import engineer_features
from user_ingestion.scorer import score_users

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.ingestion")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Execute all three ingestion stages and write the result CSV.

    Parameters
    ----------
    input_path  : Path to raw behavioral CSV.
    output_path : Path for the processed user-profiles CSV.

    Returns
    -------
    Processed DataFrame.
    """

    # ── Stage 0: Load ─────────────────────────────────────────────────────────
    logger.info("Loading input file: %s", input_path)
    raw_df = pd.read_csv(input_path, dtype=str)           # read everything as str first
    logger.info("Loaded %d rows × %d columns", *raw_df.shape)

    # ── Stage 1: Validation & Schema Enforcement ──────────────────────────────
    logger.info("Stage 1 – Validation & Schema Enforcement")
    validated_df, warnings = validate_and_coerce(raw_df)

    if warnings:
        for w in warnings:
            logger.warning("  [VALIDATION] %s", w)
    else:
        logger.info("  No validation warnings – all rows pass schema checks.")

    n_rejected = len(raw_df) - len(validated_df)
    logger.info("  Rows after validation: %d  (rejected: %d)", len(validated_df), n_rejected)

    # ── Stage 2: Hybrid Imputation ────────────────────────────────────────────
    logger.info("Stage 2 – Hybrid KNN Imputation")
    missing_before = validated_df.isna().sum().sum()
    imputed_df = impute_missing(validated_df)
    missing_after = imputed_df.isna().sum().sum()
    logger.info(
        "  Missing values: %d → %d  (imputed %d cells)",
        missing_before, missing_after, missing_before - missing_after
    )

    # ── Stage 3: Feature Engineering ─────────────────────────────────────────
    logger.info("Stage 3 – Feature Normalization & Engineering")
    engineered_df = engineer_features(imputed_df)

    new_features = [
        "coins_balance_scaled", "streak_scaled", "time_window",
        "sessions_last_7d_norm", "exercises_completed_7d_norm",
        "days_since_signup_norm", "coins_balance_scaled_norm",
        "streak_scaled_norm", "notif_open_rate_30d_norm", "motivation_score_norm",
    ]
    logger.info("  New features added: %s", new_features)

    # ── Stage 4: Intelligence Generation ─────────────────────────────────────
    logger.info("Stage 4 – Intelligence Generation (Scoring)")
    scored_df = score_users(engineered_df)
    logger.info(
        "  Scores computed | activeness: mean=%.3f | churn_risk: mean=%.3f",
        scored_df["activeness_score"].mean(),
        scored_df["churn_risk"].mean(),
    )

    # ── Persist output ────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_path, index=False)
    logger.info("User profiles written to: %s", out_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(scored_df)

    return scored_df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable pipeline summary to stdout."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total users processed : {len(df)}")
    print(f"\nLifecycle distribution:")
    print(df["lifecycle_stage"].value_counts().to_string())
    print(f"\nAge band distribution:")
    print(df["age_band"].value_counts().to_string())
    print(f"\nRegion distribution:")
    print(df["region"].value_counts().to_string())
    print(f"\nTime-window distribution (preferred_hour → bins):")
    print(df["time_window"].value_counts().to_string())
    print(f"\nFeature stats (after engineering):")
    stat_cols = [
        "sessions_last_7d",       "sessions_last_7d_norm",
        "exercises_completed_7d", "exercises_completed_7d_norm",
        "streak_current",         "streak_scaled",       "streak_scaled_norm",
        "coins_balance",          "coins_balance_scaled","coins_balance_scaled_norm",
        "notif_open_rate_30d",    "notif_open_rate_30d_norm",
        "motivation_score",       "motivation_score_norm",
    ]
    print(df[stat_cols].describe().round(3).to_string())

    print(f"\nIntelligence scores:")
    score_cols = [
        "activeness_score", "churn_risk",
        "propensity_gamification", "propensity_learning",
        "propensity_achievement", "propensity_social",
    ]
    print(df[score_cols].describe().round(3).to_string())

    print(f"\nDominant drive distribution:")
    print(df["dominant_drive"].value_counts().to_string())
    print("=" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aurora – User Data Ingestion Pipeline (Task 1, Points 2 & 3)"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/input/user_behavioral_data.csv",
        help="Path to raw behavioral CSV (default: data/input/user_behavioral_data.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/output/user_profiles.csv",
        help="Path for processed user-profiles CSV (default: data/output/user_profiles.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.input, args.output)
