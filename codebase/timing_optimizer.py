"""
Timing Optimizer  –  Project Aurora (SpeakX)
=============================================
Task 2B — Frequency + Timing Intelligence.

Reads:
  - pipeline_cache/segment_summary.csv (avg_activeness per segment for frequency banding)
  - iteration_0_before_learning/user_segments.csv  (time_window per user for timing distribution)

Computes per-segment:
  1. base_frequency  — Frequency Optimizer (activeness banding)
  2. guardrail_applied — Uninstall guardrail (STUB for iteration_0; see TODO below)
  3. primary / secondary / tertiary time windows — Temperature-Scaled Softmax (T=1.5)
  4. allocation_distribution — Largest Remainder Method integer split

Outputs:
  - iteration_0_before_learning/timing_recommendations.csv

Usage
-----
    python timing_optimizer.py \
        --summary   pipeline_cache/segment_summary.csv \
        --segments  iteration_0_before_learning/user_segments.csv \
        --output    iteration_0_before_learning/timing_recommendations.csv
"""

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.timing_optimizer")

# ── Constants ──────────────────────────────────────────────────────────────────
SOFTMAX_TEMPERATURE = 1.5

TIME_WINDOWS = [
    "early_morning",
    "mid_morning",
    "afternoon",
    "late_afternoon",
    "evening",
    "night",
]

# Frequency bands: (min_exclusive, max_inclusive, base_freq)
# Evaluated top-down; first match wins.
FREQUENCY_BANDS = [
    (0.7,  math.inf, 8),
    (0.4,  0.7,      5),
    (0.0,  0.4,      3),
]

TIMING_CSV_FIELDNAMES = [
    "segment_id",
    "base_frequency",
    "guardrail_applied",
    "primary_window",
    "secondary_window",
    "tertiary_window",
    "allocation_distribution",
    "expected_ctr",         # Iteration 0 baseline stub — updated dynamically in Task 3
    "expected_engagement",  # Iteration 0 baseline stub — updated dynamically in Task 3
]


# ══════════════════════════════════════════════════════════════════════════════
# Frequency Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def compute_base_frequency(avg_activeness: float) -> int:
    """Assign daily notification quota based on activeness band."""
    for lo, hi, freq in FREQUENCY_BANDS:
        if avg_activeness > lo:
            return freq
    return 3  # fallback for edge cases


def compute_expected_metrics(sum_row: "pd.Series") -> tuple[float, float]:
    """
    Compute data-driven expected_ctr and expected_engagement for a segment
    using Iteration 0 baseline behavioral features from segment_summary.csv.

    Engagement score = weighted blend:
      50% avg_activeness   (active users engage more)
      30% 1-avg_churn_risk (lower churn ↔ higher engagement)
      20% max propensity   (strongest motivational driver)

    Scales are calibrated to realistic EdTech push-notification ranges:
      expected_ctr        → [0.02, 0.15]
      expected_engagement → [0.05, 0.45]
    """
    avg_act   = float(sum_row.get("avg_activeness",   0.5))
    avg_churn = float(sum_row.get("avg_churn_risk",   0.5))
    max_prop  = max(
        float(sum_row.get("avg_gamification", 0)),
        float(sum_row.get("avg_learning",     0)),
        float(sum_row.get("avg_achievement",  0)),
        float(sum_row.get("avg_social",       0)),
    )

    engagement_score = (
        0.50 * avg_act
        + 0.30 * (1.0 - avg_churn)
        + 0.20 * max_prop
    )

    expected_engagement = round(
        max(0.05, min(0.45, engagement_score * 0.45 + 0.02)), 4
    )
    expected_ctr = round(
        max(0.02, min(0.15, engagement_score * 0.15 + 0.01)), 4
    )
    return expected_ctr, expected_engagement


# TODO [POST-ITERATION_0]: Replace this stub with actual per-segment uninstall_rate data.
# Logic (when data is available):
#   if segment_uninstall_rate > 0.02:
#       return max(1, base_freq - 2), True
# Until then, the guardrail is never applied.
def _uninstall_guardrail_stub(segment_id: int, base_freq: int) -> tuple[int, bool]:
    """
    Stub: uninstall_rate data not available in iteration_0.
    Returns (base_freq, False) — no penalty applied.
    """
    return base_freq, False


# ══════════════════════════════════════════════════════════════════════════════
# Timing Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def _temperature_softmax(v: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    """
    Temperature-Scaled Softmax.

    P_i = exp(v_i / T) / sum_j(exp(v_j / T))

    Subtracts max before exponentiating for numerical stability
    (prevents overflow without changing the result).
    """
    scaled = v / temperature
    scaled = scaled - scaled.max()          # numerical stability
    exp_v = np.exp(scaled)
    return exp_v / exp_v.sum()


def _largest_remainder_method(probs: np.ndarray, total: int) -> list[int]:
    """
    Distribute `total` notifications across len(probs) slots using
    the Largest Remainder Method. Guarantees sum(result) == total.
    """
    fractional   = probs * total
    integer_part = np.floor(fractional).astype(int)
    remainder    = total - int(integer_part.sum())

    # Assign the leftover units to the slots with the largest fractional parts
    fractional_parts = fractional - integer_part
    order = np.argsort(fractional_parts)[::-1]
    for i in range(remainder):
        integer_part[order[i]] += 1

    return integer_part.tolist()


def compute_timing_distribution(seg_df: pd.DataFrame, base_frequency: int) -> dict:
    """
    Compute top-3 time windows and discrete notification allocation for one segment.

    Uses:
      - seg_df["time_window"] (already bucketed by feature_engineer.py)
      - Temperature-Scaled Softmax (T=1.5)
      - Top-3 selection + L1 re-normalization
      - Largest Remainder Method for integer allocation
    """
    # ── Step 1: Raw frequency vector ──────────────────────────────────────────
    window_counts = seg_df["time_window"].value_counts()
    counts = np.array(
        [float(window_counts.get(w, 0)) for w in TIME_WINDOWS],
        dtype=float,
    )

    # ── Step 1b: Normalize counts to proportions ──────────────────────────────
    # Raw counts can be in the hundreds. Passing them directly into softmax with
    # T=1.5 causes winner-take-all collapse: after dividing by T the spread is
    # still ~40 log-units wide, so only the top bin survives exponentiation.
    # Normalising to proportions first constrains inputs to [0, 1], which is the
    # range the temperature parameter was designed for.
    total = counts.sum()
    if total > 0:
        counts = counts / total
    else:
        counts = np.ones(len(TIME_WINDOWS)) / len(TIME_WINDOWS)

    # ── Step 2: Temperature-Scaled Softmax ────────────────────────────────────
    softmax_probs = _temperature_softmax(counts)

    # ── Step 3: Top-3 selection ───────────────────────────────────────────────
    top3_indices  = np.argsort(softmax_probs)[::-1][:3]
    top3_windows  = [TIME_WINDOWS[i] for i in top3_indices]
    top3_probs    = softmax_probs[top3_indices]

    # ── Step 4: L1 re-normalize ───────────────────────────────────────────────
    top3_probs_renorm = top3_probs / top3_probs.sum()

    # ── Step 5: Discrete allocation via LRM ──────────────────────────────────
    allocation = _largest_remainder_method(top3_probs_renorm, base_frequency)

    return {
        "primary_window":         top3_windows[0],
        "secondary_window":       top3_windows[1],
        "tertiary_window":        top3_windows[2],
        "allocation_distribution": ":".join(str(n) for n in allocation),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Write timing_recommendations.csv
# ══════════════════════════════════════════════════════════════════════════════

def write_timing(rows: list[dict], output_path: str) -> None:
    """Write timing_recommendations.csv with csv.QUOTE_ALL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TIMING_CSV_FIELDNAMES,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "timing_recommendations.csv written → %s  (%d rows)", out, len(rows)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_timing_optimizer(
    summary_path: str,
    segments_path: str,
    output_path: str,
) -> None:
    """
    Full Timing Optimizer pipeline.
    1. Load segment_summary.csv + user_segments.csv
    2. Per segment: compute base_frequency → apply guardrail stub → timing distribution
    3. Write timing_recommendations.csv
    """
    sum_path = Path(summary_path)
    seg_path = Path(segments_path)

    if not sum_path.exists():
        raise FileNotFoundError(
            f"segment_summary.csv not found at {sum_path}. "
            "Run segmentation.py first."
        )
    if not seg_path.exists():
        raise FileNotFoundError(
            f"user_segments.csv not found at {seg_path}. "
            "Run segmentation.py first."
        )

    summary_df  = pd.read_csv(sum_path)
    segments_df = pd.read_csv(seg_path)

    logger.info(
        "Loaded segment_summary: %d segments | user_segments: %d users",
        len(summary_df), len(segments_df),
    )

    rows: list[dict] = []
    for _, sum_row in summary_df.iterrows():
        seg_id       = int(sum_row["segment_id"])
        avg_active   = float(sum_row["avg_activeness"])
        base_freq    = compute_base_frequency(avg_active)
        final_freq, guardrail = _uninstall_guardrail_stub(seg_id, base_freq)

        seg_df   = segments_df[segments_df["segment_id"] == seg_id]
        timing   = compute_timing_distribution(seg_df, final_freq)

        exp_ctr, exp_eng = compute_expected_metrics(sum_row)

        row = {
            "segment_id":            seg_id,
            "base_frequency":        final_freq,
            "guardrail_applied":     str(guardrail),
            **timing,
            "expected_ctr":          f"{exp_ctr:.4f}",
            "expected_engagement":   f"{exp_eng:.4f}",
        }
        rows.append(row)
        logger.info(
            "  Segment_%d  →  freq=%d  |  %s:%s:%s  |  alloc=%s",
            seg_id, final_freq,
            timing["primary_window"], timing["secondary_window"], timing["tertiary_window"],
            timing["allocation_distribution"],
        )

    write_timing(rows, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TIMING OPTIMIZER COMPLETE")
    print("=" * 70)
    print(f"  Segments processed : {len(rows)}")
    print(f"  Output             : {output_path}")
    print()
    print(f"  {'Seg':<5} {'Freq':<6} {'Primary Window':<18} {'Allocation'}")
    print(f"  {'---':<5} {'----':<6} {'--------------':<18} {'----------'}")
    for r in rows:
        print(
            f"  {r['segment_id']:<5} {r['base_frequency']:<6} "
            f"{r['primary_window']:<18} {r['allocation_distribution']}"
        )
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – Timing Optimizer (Task 2B)"
    )
    p.add_argument(
        "--summary", "-m",
        default="codebase/pipeline_cache/segment_summary.csv",
        help="Path to segment_summary.csv from segmentation.py",
    )
    p.add_argument(
        "--segments", "-s",
        default="iteration_0_before_learning/user_segments.csv",
        help="Path to user_segments.csv from segmentation.py",
    )
    p.add_argument(
        "--output", "-o",
        default="iteration_0_before_learning/timing_recommendations.csv",
        help="Output path for timing_recommendations.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_timing_optimizer(
        summary_path=args.summary,
        segments_path=args.segments,
        output_path=args.output,
    )
