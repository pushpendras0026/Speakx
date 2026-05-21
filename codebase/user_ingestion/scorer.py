"""
Stage 4: Intelligence Generation (Scoring Engines)

Three compound indicators computed per user, exactly as defined in
user_data_ingestion.md § 4.

──────────────────────────────────────────────────────────────────
I.  ACTIVENESS SCORE  (Adaptive Sigmoid)
    R    = 0.7 × exercises_completed_7d  +  0.3 × sessions_last_7d
    μ    = median(R) across dataset
    σ    = std(R)    across dataset
    score = 1 / (1 + exp(−((R − μ) / σ) × 1.7))

II. CHURN RISK SIGNAL  (Contextual)
    base = 0.4(1 − Activeness) + 0.3(1 − Motivation) + 0.3(1 − OpenRate)
    Lifecycle multiplier Ms:
      Trial  day 0-3  → 1.2
      Trial  day 4-7  → 1.1
      Paid   day 8-14 → 1.0
      Paid   day 15+  → 0.9
      Churned/Inactive → 1.0  (not in spec; treated as standard baseline)
    total_risk = min(1.0, Ms × base)

III. PROPENSITY SCORES  (4-Drive Model)  — ||x|| = _norm columns
    gamification = 0.5 × ||coins|| + 0.5 × ||streak||
    learning     = 0.6 × ai_tutor_used(0/1) + 0.4 × ||exercises||
    achievement  = 0.4 × progress_checked(0/1) + 0.6 × ||motivation||
    social       = 0.7 × leaderboard_viewed(0/1) + 0.3 × tier_weight
      tier_weight: tier1→1.0, tier2→0.7, tier3→0.4
    dominant_drive = argmax of the four propensity scores
──────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Tier weight lookup ─────────────────────────────────────────────────────────
TIER_WEIGHT = {
    "tier1": 1.0,
    "tier2": 0.7,
    "tier3": 0.4,
}

# ── Lifecycle multiplier logic ─────────────────────────────────────────────────
def _lifecycle_multiplier(lifecycle: str, days: int) -> float:
    lc = lifecycle.lower()
    if lc == "trial":
        if days <= 3:
            return 1.2
        elif days <= 7:
            return 1.1
        else:
            return 1.1   # trial beyond day 7 — keep expiry risk
    elif lc == "paid":
        if days <= 14:
            return 1.0
        else:
            return 0.9
    else:
        # churned / inactive — standard baseline
        return 1.0


# ── I. Activeness Score ────────────────────────────────────────────────────────

def compute_activeness(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive sigmoid activeness score in (0, 1).
    Uses raw exercise and session counts (not normalized) for R,
    so μ and σ reflect true behavioral scale.
    """
    exercises = df["exercises_completed_7d"].astype(float)
    sessions  = df["sessions_last_7d"].astype(float)

    R  = 0.7 * exercises + 0.3 * sessions
    mu = R.median()
    sigma = R.std(ddof=0)          # population std (consistent across dataset sizes)

    if sigma == 0:
        # All users identical engagement — return 0.5 for everyone
        logger.warning("σ(R) = 0; all users receive activeness = 0.5")
        return pd.Series(0.5, index=df.index)

    z = ((R - mu) / sigma) * 1.7
    score = 1.0 / (1.0 + np.exp(-z))

    logger.info(
        "Activeness | R: mean=%.2f, μ=%.2f, σ=%.2f | score: mean=%.3f",
        R.mean(), mu, sigma, score.mean()
    )
    return score.rename("activeness_score")


# ── II. Churn Risk Signal ──────────────────────────────────────────────────────

def compute_churn_risk(df: pd.DataFrame, activeness: pd.Series) -> pd.Series:
    """
    Contextual churn risk in [0, 1].
    Uses normalized motivation and open-rate (already in [0,1]).
    """
    motivation = df["motivation_score"].astype(float)
    open_rate  = df["notif_open_rate_30d"].astype(float)

    base_risk = (
        0.4 * (1.0 - activeness) +
        0.3 * (1.0 - motivation) +
        0.3 * (1.0 - open_rate)
    )

    multipliers = df.apply(
        lambda row: _lifecycle_multiplier(row["lifecycle_stage"], int(row["days_since_signup"])),
        axis=1,
    )

    total_risk = (multipliers * base_risk).clip(upper=1.0)

    logger.info(
        "Churn Risk | base: mean=%.3f | total: mean=%.3f (after multipliers)",
        base_risk.mean(), total_risk.mean()
    )
    return total_risk.rename("churn_risk")


# ── III. Propensity Scores ─────────────────────────────────────────────────────

def compute_propensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Four propensity scores, each in [0, 1], plus the dominant drive label.
    Uses _norm columns produced by feature_engineer.py.
    """
    coins_norm    = df["coins_balance_scaled_norm"].astype(float)
    streak_norm   = df["streak_scaled_norm"].astype(float)
    exercises_norm = df["exercises_completed_7d_norm"].astype(float)
    motivation_norm = df["motivation_score_norm"].astype(float)

    # Boolean feature flags → 0.0 / 1.0
    ai_used      = df["feature_ai_tutor_used"].astype(float)
    leaderboard  = df["feature_leaderboard_viewed"].astype(float)
    progress     = df["feature_progress_checked"].astype(float)

    # Tier weight per user
    tier_weight = df["region"].str.lower().map(TIER_WEIGHT).fillna(0.7)

    gamification = 0.5 * coins_norm  + 0.5 * streak_norm
    learning     = 0.6 * ai_used     + 0.4 * exercises_norm
    achievement  = 0.4 * progress    + 0.6 * motivation_norm
    social       = 0.7 * leaderboard + 0.3 * tier_weight

    propensity_df = pd.DataFrame({
        "propensity_gamification": gamification,
        "propensity_learning":     learning,
        "propensity_achievement":  achievement,
        "propensity_social":       social,
    }, index=df.index)

    drive_cols = ["propensity_gamification", "propensity_learning",
                  "propensity_achievement", "propensity_social"]
    propensity_df["dominant_drive"] = propensity_df[drive_cols].idxmax(axis=1).str.replace(
        "propensity_", "", regex=False
    )

    logger.info(
        "Propensity | dominant drive distribution:\n%s",
        propensity_df["dominant_drive"].value_counts().to_string()
    )
    return propensity_df


# ── Master scorer entry point ──────────────────────────────────────────────────

def score_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three intelligence signals and append them to df.

    Adds columns:
      activeness_score          float  (0–1)
      churn_risk                float  (0–1)
      propensity_gamification   float  (0–1)
      propensity_learning       float  (0–1)
      propensity_achievement    float  (0–1)
      propensity_social         float  (0–1)
      dominant_drive            str
    """
    df = df.copy()

    activeness  = compute_activeness(df)
    churn_risk  = compute_churn_risk(df, activeness)
    propensity  = compute_propensity(df)

    df["activeness_score"] = activeness.round(4)
    df["churn_risk"]       = churn_risk.round(4)
    df = pd.concat([df, propensity.round(4)], axis=1)

    return df
