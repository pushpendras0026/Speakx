"""
Theme Engine  –  Project Aurora (SpeakX)
=========================================
Task 2A, Part 1 — Deterministic Octalysis Core Drive scoring.

Reads:
  - iteration_0_before_learning/user_segments.csv

For each segment, computes 8 Octalysis Core Drive scores (CD1–CD8),
selects the top 3 drives above the 70% threshold of the top score, and assigns:
  primary_theme, secondary_theme, tertiary_theme

Outputs:
  - iteration_0_before_learning/communication_themes.csv

Usage
-----
    python theme_engine.py \
        --segments  iteration_0_before_learning/user_segments.csv \
        --output    iteration_0_before_learning/communication_themes.csv
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.theme_engine")

# ── Octalysis Core Drive name map ──────────────────────────────────────────────
CD_NAMES: dict[str, str] = {
    "CD1": "Epic_Meaning",
    "CD2": "Accomplishment",
    "CD3": "Creativity_and_Feedback",
    "CD4": "Ownership_and_Possession",
    "CD5": "Social_Influence",
    "CD6": "Scarcity_and_Impatience",
    "CD7": "Unpredictability",
    "CD8": "Loss_and_Avoidance",
}

# Reverse map: theme name → CD integer (e.g. "Social_Influence" → 5)
# Derived from CD_NAMES so no hardcoding is needed here.
CD_NUM_BY_NAME: dict[str, int] = {
    name: int(key[2:]) for key, name in CD_NAMES.items()
}

THEME_SELECT_THRESHOLD = 0.7   # drives with score >= top_score * 0.7 are eligible

THEMES_CSV_FIELDNAMES = [
    "segment_id",
    "primary_theme",
    "secondary_theme",
    "tertiary_theme",
    "tone_preferences",  # pipe-joined tones from allowed_tone_hook_matrix (primary + secondary)
    "hooks",             # pipe-joined hooks from allowed_tone_hook_matrix (primary + secondary)
    "cd1_score",
    "cd2_score",
    "cd3_score",
    "cd4_score",
    "cd5_score",
    "cd6_score",
    "cd7_score",
    "cd8_score",
]


# ══════════════════════════════════════════════════════════════════════════════
# Core Drive scoring
# ══════════════════════════════════════════════════════════════════════════════

def compute_cd_scores(seg_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute all 8 Octalysis Core Drive scores for one segment's user rows.

    All input columns are confirmed 0-1 normalised in user_segments.csv.
    Boolean feature columns are cast to float before arithmetic.
    """
    feat_ai   = seg_df["feature_ai_tutor_used"].astype(float)
    feat_lead = seg_df["feature_leaderboard_viewed"].astype(float)
    feat_prog = seg_df["feature_progress_checked"].astype(float)

    cd1 = float(seg_df["motivation_score_norm"].mean())

    cd2 = float(
        (
            0.4 * seg_df["propensity_achievement"]
            + 0.3 * feat_prog
            + 0.3 * seg_df["exercises_completed_7d_norm"]
        ).mean()
    )

    cd3 = float((0.6 * feat_ai + 0.4 * seg_df["propensity_learning"]).mean())

    cd4 = float(seg_df["coins_balance_scaled_norm"].mean())

    cd5 = float(
        (0.6 * seg_df["propensity_social"] + 0.4 * feat_lead).mean()
    )

    cd6 = float((seg_df["lifecycle_stage"] == "trial").astype(float).mean())

    cd7 = float((1.0 - seg_df["notif_open_rate_30d_norm"]).mean())

    cd8 = float(
        (0.5 * seg_df["churn_risk"] + 0.5 * seg_df["streak_scaled_norm"]).mean()
    )

    return {
        "CD1": round(cd1, 4),
        "CD2": round(cd2, 4),
        "CD3": round(cd3, 4),
        "CD4": round(cd4, 4),
        "CD5": round(cd5, 4),
        "CD6": round(cd6, 4),
        "CD7": round(cd7, 4),
        "CD8": round(cd8, 4),
    }


def select_themes(scores: dict[str, float]) -> tuple[str, str, str]:
    """
    Select primary, secondary, tertiary theme names from 8 CD scores.
    Eligibility threshold: score >= top_score * THEME_SELECT_THRESHOLD.
    Pads with 'none' if fewer than 3 qualify.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranked[0][1]
    threshold = top_score * THEME_SELECT_THRESHOLD if top_score > 0 else 0.0

    eligible = [CD_NAMES[k] for k, v in ranked if v >= threshold]

    while len(eligible) < 3:
        eligible.append("none")

    return eligible[0], eligible[1], eligible[2]


def _lookup_tone_hooks(theme_name: str, tone_matrix: dict) -> tuple[str, str]:
    """
    Look up allowed tones and hooks for a CD theme from the tone matrix.

    The matrix structure is:
      {
        "allowed_tones":   [{"tone": "...", ...}, ...],
        "disallowed_tones":[...],
        "hook_taxonomy":   [{"primary_octalysis_drive": <int>, "example_hook_copy": "...", ...}, ...]
      }

    tone_preferences: names of all allowed tones (applicable to any theme).
    hooks: example_hook_copy from hook_taxonomy entries whose primary OR secondary
           octalysis drive number matches the CD number of this theme.

    Returns (pipe-joined tone names, pipe-joined hook copy strings).
    """
    # All allowed tone names — these are not CD-specific; all are permitted tones
    allowed_tones = tone_matrix.get("allowed_tones", [])
    tones = [entry.get("tone", "") for entry in allowed_tones if entry.get("tone")]

    # Hook feature names (snake_case) whose drive matches this theme's CD number.
    # Storing the feature name (not the example copy) so downstream consumers
    # can use it verbatim as a hook_type label and look up the example separately.
    cd_num = CD_NUM_BY_NAME.get(theme_name)
    hook_taxonomy = tone_matrix.get("hook_taxonomy", [])
    hook_names: list[str] = []
    if cd_num is not None:
        for entry in hook_taxonomy:
            if (entry.get("primary_octalysis_drive") == cd_num
                    or entry.get("secondary_octalysis_drive") == cd_num):
                feature = entry.get("feature", "")
                if feature:
                    snake = re.sub(r"[^A-Za-z0-9]+", "_", feature).strip("_").lower()
                    hook_names.append(snake)

    return "|".join(tones), "|".join(hook_names)


# ══════════════════════════════════════════════════════════════════════════════
# Write communication_themes.csv
# ══════════════════════════════════════════════════════════════════════════════

def write_themes(rows: list[dict], output_path: str) -> None:
    """Write communication_themes.csv with csv.QUOTE_ALL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=THEMES_CSV_FIELDNAMES,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("communication_themes.csv written → %s  (%d rows)", out, len(rows))


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_theme_engine(
    segments_path: str,
    output_path: str,
    tone_matrix_path: str = "",
) -> None:
    """
    Full Theme Engine pipeline.
    1. Load user_segments.csv
    2. Load allowed_tone_hook_matrix.json (if provided)
    3. Group by segment_id → compute 8 CD scores
    4. Select top-3 themes + look up tone_preferences / hooks from matrix
    5. Write communication_themes.csv
    """
    seg_path = Path(segments_path)
    if not seg_path.exists():
        raise FileNotFoundError(
            f"user_segments.csv not found at {seg_path}. "
            "Run segmentation.py first."
        )

    # Load tone matrix (graceful fallback to empty dict if not available)
    tone_matrix: dict = {}
    if tone_matrix_path:
        tm_path = Path(tone_matrix_path)
        if tm_path.exists():
            tone_matrix = json.loads(tm_path.read_text(encoding="utf-8"))
            logger.info("Loaded tone matrix from %s", tm_path)
        else:
            logger.warning(
                "tone_matrix_path provided (%s) but file not found — "
                "tone_preferences and hooks will be empty.",
                tone_matrix_path,
            )

    df = pd.read_csv(seg_path)
    segment_ids = sorted(df["segment_id"].unique())
    logger.info(
        "Loaded user_segments.csv: %d users, %d segments",
        len(df), len(segment_ids),
    )

    rows: list[dict] = []
    for seg_id in segment_ids:
        seg_df = df[df["segment_id"] == seg_id]
        scores = compute_cd_scores(seg_df)
        primary, secondary, tertiary = select_themes(scores)

        # Look up tones and hooks for primary + secondary themes
        primary_tones, primary_hooks     = _lookup_tone_hooks(primary,   tone_matrix)
        secondary_tones, secondary_hooks = _lookup_tone_hooks(secondary, tone_matrix)

        # Merge and de-duplicate, preserving order (primary first)
        def _merge_pipe(a: str, b: str) -> str:
            seen: set = set()
            parts: list[str] = []
            for item in (a + "|" + b).split("|"):
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    parts.append(item)
            return "|".join(parts)

        tone_preferences = _merge_pipe(primary_tones, secondary_tones)
        hooks_combined   = _merge_pipe(primary_hooks, secondary_hooks)

        rows.append({
            "segment_id":       int(seg_id),
            "primary_theme":    primary,
            "secondary_theme":  secondary,
            "tertiary_theme":   tertiary,
            "tone_preferences": tone_preferences,
            "hooks":            hooks_combined,
            "cd1_score": scores["CD1"],
            "cd2_score": scores["CD2"],
            "cd3_score": scores["CD3"],
            "cd4_score": scores["CD4"],
            "cd5_score": scores["CD5"],
            "cd6_score": scores["CD6"],
            "cd7_score": scores["CD7"],
            "cd8_score": scores["CD8"],
        })
        logger.info(
            "  Segment_%d  →  %s | %s | %s  "
            "(CD scores max=%.3f)",
            seg_id, primary, secondary, tertiary,
            max(scores.values()),
        )

    write_themes(rows, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  THEME ENGINE COMPLETE")
    print("=" * 70)
    print(f"  Segments processed : {len(rows)}")
    print(f"  Output             : {output_path}")
    print()
    print(f"  {'Seg ID':<8} {'Primary Theme':<32} {'Secondary Theme'}")
    print(f"  {'------':<8} {'-------------':<32} {'---------------'}")
    for r in rows:
        print(f"  {r['segment_id']:<8} {r['primary_theme']:<32} {r['secondary_theme']}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – Theme Engine (Task 2A, Part 1)"
    )
    p.add_argument(
        "--segments", "-s",
        default="iteration_0_before_learning/user_segments.csv",
        help="Path to user_segments.csv from segmentation.py",
    )
    p.add_argument(
        "--output", "-o",
        default="iteration_0_before_learning/communication_themes.csv",
        help="Output path for communication_themes.csv",
    )
    p.add_argument(
        "--tone-matrix", "-m",
        default="iteration_0_before_learning/allowed_tone_hook_matrix.json",
        help="Path to allowed_tone_hook_matrix.json from kb_ingestion.py",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_theme_engine(
        segments_path=args.segments,
        output_path=args.output,
        tone_matrix_path=args.tone_matrix,
    )
