"""
Schedule Generator V2  –  Project Aurora (SpeakX)
===================================================
Task 3 (Part 6) — UCB1-Based Routing Engine

Extends the Lifecycle-Aware Routing Engine from schedule_generator.py with
one critical change: instead of random/round-robin template selection,
templates are ranked by their UCB1 score and the top N are chosen.

Reads:
  - iteration_0_before_learning/user_segments.csv         (users + segment assignments)
  - iteration_1_after_learning/timing_recommendations.csv (EMA-updated timing)
  - iteration_1_after_learning/message_templates.csv    (V0 + V1 template library)
  - pipeline_cache/ucb_scores.csv              (UCB1 score per template)

UCB score rules (applied before template access):
  BAD  templates → UCB = -1.0   → permanently suppressed (never selected)
  NEW  templates → UCB = 999.0  → forced to front (cold-start exploration)
  REST templates → UCB = ctr + c * sqrt( ln(N_seg) / sends_i )

Outputs:
  - iteration_1_after_learning/user_notification_schedule.csv

Usage
-----
    python codebase/schedule_generator_v2.py \
        --segments    iteration_0_before_learning/user_segments.csv \
        --timing      iteration_1_after_learning/timing_recommendations.csv \
        --templates   iteration_1_after_learning/message_templates.csv \
        --ucb-scores  pipeline_cache/ucb_scores.csv \
        --output      iteration_1_after_learning/user_notification_schedule.csv
"""

import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.schedule_generator_v2")

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_NOTIF_SLOTS: int      = 9
MAX_PUSH_SLOTS: int       = 3      # OS push budget — at most 3 Push per user per day
DEFAULT_PREFERRED_HOUR: int = 18   # fallback when preferred_hour is missing/NaN
SPACING_HOURS: int        = 24 // MAX_NOTIF_SLOTS   # = 2 h → 9 unique times
SUPPRESSED_SCORE          = -1.0   # BAD templates are never selected

# Chronological milestone order (same as schedule_generator.py)
GOAL_MILESTONES = ["Day 1", "Day 3", "Week 1", "Week 2", "Week 4"]

# Standard Time Windows — hour → window label mapping
TIME_WINDOWS: list[tuple[int, int, str]] = [
    (6,  8,  "early_morning"),
    (9,  11, "mid_morning"),
    (12, 14, "afternoon"),
    (15, 17, "late_afternoon"),
    (18, 20, "evening"),
    (21, 23, "night"),
    (0,  5,  "night"),        # late-night wraps into "night"
]

# ── Fieldnames ─────────────────────────────────────────────────────────────────
_NOTIF_FIELDS: list[str] = []
for _i in range(1, MAX_NOTIF_SLOTS + 1):
    _NOTIF_FIELDS += [
        f"notif_{_i}_template_id",
        f"notif_{_i}_time",
        f"notif_{_i}_channel",
    ]
SCHEDULE_V2_FIELDNAMES = ["user_id", "segment_id", "lifecycle_day"] + _NOTIF_FIELDS


# ══════════════════════════════════════════════════════════════════════════════
# Lifecycle-Aware Goal Resolution (identical to V1)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_active_goal(lifecycle_stage: str, days_since_signup: float) -> str:
    """
    Map user lifecycle stage + days_since_signup to their active goal milestone.
    (Mirror of schedule_generator.py — immutable V1 contract preserved.)
    """
    stage = str(lifecycle_stage).strip().lower()
    try:
        days = float(days_since_signup)
    except (ValueError, TypeError):
        days = 0.0

    if "trial" in stage or stage in ("free", "freemium"):
        if days <= 1:
            return "Day 1"
        elif days <= 3:
            return "Day 3"
        elif days <= 7:
            return "Week 1"
        else:
            return "Week 2"
    else:
        if days <= 7:
            return "Week 1"
        elif days <= 14:
            return "Week 2"
        else:
            return "Week 4"


def _get_templates_with_fallback(
    templates_by_goal: dict[str, list[tuple[str, float]]],
    active_goal: str,
) -> tuple[list[tuple[str, float]], str]:
    """
    Return (sorted_pool, resolved_goal) for active_goal.
    Each pool entry is (template_id, ucb_score).
    Falls back to nearest earlier milestone if the goal pool is empty.
    Suppressed templates (UCB = -1.0) are excluded.
    """
    def _valid_pool(pool: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return [entry for entry in pool if entry[1] > SUPPRESSED_SCORE]

    if active_goal in templates_by_goal:
        vpool = _valid_pool(templates_by_goal[active_goal])
        if vpool:
            return vpool, active_goal

    try:
        start = GOAL_MILESTONES.index(active_goal)
    except ValueError:
        start = len(GOAL_MILESTONES)

    for i in range(start - 1, -1, -1):
        label = GOAL_MILESTONES[i]
        if label in templates_by_goal:
            vpool = _valid_pool(templates_by_goal[label])
            if vpool:
                return vpool, label

    # Last resort: flatten all valid templates for this segment
    flat = _valid_pool([entry for pool in templates_by_goal.values() for entry in pool])
    return flat, "any"


# ══════════════════════════════════════════════════════════════════════════════
# UCB1-Based Template Selection
# ══════════════════════════════════════════════════════════════════════════════

def _select_top_n_by_ucb(
    pool: list[tuple[str, float]],
    n: int,
) -> list[str]:
    """
    Step 6.3: Sort pool descending by UCB score and take the top N.

    If pool size < n, cycle through the remaining (sorted) entries
    to fill all n slots (preserves UCB ordering).
    """
    if not pool:
        return [""] * n

    # Sort descending by UCB score
    sorted_pool = sorted(pool, key=lambda x: x[1], reverse=True)
    ids_sorted  = [entry[0] for entry in sorted_pool]

    if len(ids_sorted) >= n:
        return ids_sorted[:n]

    # Cycle through sorted entries to fill remaining slots
    cycler = itertools.cycle(ids_sorted)
    return [next(cycler) for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Send-Time Optimization  (Anchoring + Anti-Clumping) — same as V1
# ══════════════════════════════════════════════════════════════════════════════

def _safe_preferred_hour(raw_value) -> int:
    """Parse preferred_hour; returns 0-23 or DEFAULT_PREFERRED_HOUR."""
    try:
        hour = int(float(raw_value))
        if 0 <= hour <= 23:
            return hour
    except (ValueError, TypeError):
        pass
    return DEFAULT_PREFERRED_HOUR


def _hour_to_window(hour: int) -> str:
    """Map an integer hour (0-23) to a standard time-window label."""
    for lo, hi, label in TIME_WINDOWS:
        if lo <= hour <= hi:
            return label
    return "night"


def compute_send_time(preferred_hour: int, slot_index: int) -> str:
    """Return the time-window label for a given notification slot.

    computed_hour = (preferred_hour + (slot_index - 1) * SPACING_HOURS) % 24
    Then maps to its standard time window.
    """
    hour = (preferred_hour + (slot_index - 1) * SPACING_HOURS) % 24
    return _hour_to_window(hour)


def assign_channel(slot_index: int, push_budget: int) -> str:
    """Slots 1..push_budget → Push; rest → In-App."""
    return "Push" if slot_index <= push_budget else "In-App"


# ══════════════════════════════════════════════════════════════════════════════
# Write schedule CSV
# ══════════════════════════════════════════════════════════════════════════════

def _write_schedule(rows: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SCHEDULE_V2_FIELDNAMES,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "user_notification_schedule.csv written → %s  (%d rows)", out, len(rows)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_schedule_generator_v2(
    segments_path: str,
    timing_path: str,
    templates_path: str,
    ucb_scores_path: str,
    output_path: str,
) -> None:
    """
    Full Schedule Generator V2 — UCB1-Aware Routing Engine.

    Identical lifecycle-aware routing as V1 but template selection uses UCB ranking
    instead of random sampling:
      1. Load all inputs
      2. Build nested template dict: segment_id → goal_id → [(template_id, ucb_score)]
         (suppressed BAD templates have UCB = -1.0 and are excluded)
      3. For every user: resolve active_goal → get sorted pool → select top N by UCB
      4. Assemble and write wide-format schedule
    """
    for label, path in [
        ("user_segments.csv",           segments_path),
        ("timing_recommendations.csv", timing_path),
        ("message_templates.csv",    templates_path),
        ("ucb_scores.csv",              ucb_scores_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found at {path}.")

    users_df     = pd.read_csv(segments_path)
    timing_df    = pd.read_csv(timing_path)
    templates_df = pd.read_csv(templates_path)
    ucb_df       = pd.read_csv(ucb_scores_path)

    logger.info(
        "Loaded: %d users | %d timing rows | %d templates | %d UCB scores",
        len(users_df), len(timing_df), len(templates_df), len(ucb_df),
    )

    # Build UCB score lookup
    ucb_lookup: dict[str, float] = {}
    for _, row in ucb_df.iterrows():
        ucb_lookup[str(row["template_id"])] = float(row["ucb_score"])

    # Index timing by segment_id
    timing_index: dict[int, dict] = {
        int(row["segment_id"]): row.to_dict()
        for _, row in timing_df.iterrows()
    }

    # Nested template dict: segment_id → goal_id → [(template_id, ucb_score)]
    tmpl_by_seg_goal: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for _, trow in templates_df.iterrows():
        seg  = int(trow["segment_id"])
        goal = str(trow["goal_id"]).strip()
        tid  = str(trow["template_id"])
        score = ucb_lookup.get(tid, 0.0)
        tmpl_by_seg_goal.setdefault(seg, {}).setdefault(goal, []).append((tid, score))

    rows: list[dict] = []
    skipped = 0
    fallback_used = 0

    for _, user in users_df.iterrows():
        seg_id = int(user["segment_id"])

        if seg_id not in timing_index:
            logger.warning(
                "User %s: segment %d not in timing — skipping.",
                user.get("user_id", "?"), seg_id,
            )
            skipped += 1
            continue

        timing = timing_index[seg_id]

        # Push budget: clamp segment's base_frequency to OS limit
        try:
            base_freq = int(timing.get("base_frequency", 3))
        except (ValueError, TypeError):
            base_freq = 3
        push_budget = min(base_freq, MAX_PUSH_SLOTS)

        # Resolve lifecycle + days
        days = user.get("days_since_signup", 0)
        try:
            days_float    = float(days)
            lifecycle_day = f"Day {int(days_float)}"
        except (ValueError, TypeError):
            days_float    = 0.0
            lifecycle_day = str(days)

        lifecycle_stage = str(user.get("lifecycle_stage", ""))
        active_goal     = _resolve_active_goal(lifecycle_stage, days_float)

        # UCB-sorted pool with fallback
        seg_goal_map = tmpl_by_seg_goal.get(seg_id, {})
        pool, resolved_goal = _get_templates_with_fallback(seg_goal_map, active_goal)

        if resolved_goal != active_goal:
            fallback_used += 1

        # Send-Time Optimization (preferred_hour anchor + 2h spacing)
        raw_hour = user.get("preferred_hour", "")
        pref_hour = _safe_preferred_hour(raw_hour)

        # Step 6.3: Select top 9 by UCB score (descending) — always fill all 9
        selected_ids = _select_top_n_by_ucb(pool, MAX_NOTIF_SLOTS)

        # Assemble wide-format row — ALL 9 slots populated
        row: dict = {
            "user_id":       str(user["user_id"]),
            "segment_id":    seg_id,
            "lifecycle_day": lifecycle_day,
        }
        for i in range(1, MAX_NOTIF_SLOTS + 1):
            row[f"notif_{i}_template_id"] = selected_ids[i - 1]
            row[f"notif_{i}_time"]        = compute_send_time(pref_hour, i)
            row[f"notif_{i}_channel"]     = assign_channel(i, push_budget)

        rows.append(row)

    if skipped:
        logger.warning("Skipped %d users.", skipped)
    if fallback_used:
        logger.info("%d users used fallback goal milestone.", fallback_used)

    _write_schedule(rows, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_push = 0
    total_inapp = 0
    for r in rows:
        for i in range(1, MAX_NOTIF_SLOTS + 1):
            ch = r.get(f"notif_{i}_channel", "")
            if ch == "Push":
                total_push += 1
            elif ch == "In-App":
                total_inapp += 1

    print("\n" + "=" * 70)
    print("  SCHEDULE GENERATOR V2 COMPLETE  (UCB1-Aware Routing Engine)")
    print("=" * 70)
    print(f"  Users scheduled      : {len(rows)}")
    print(f"  Users skipped        : {skipped}")
    print(f"  Fallback goal used   : {fallback_used}")
    print(f"  Slots per user       : {MAX_NOTIF_SLOTS} (all filled)")
    print(f"  Total notifications  : {len(rows) * MAX_NOTIF_SLOTS}")
    print(f"  Output               : {output_path}")
    print()
    print(f"  Channel mix  →  Push: {total_push}  |  In-App: {total_inapp}")
    print(f"  Spacing      →  {SPACING_HOURS}h between consecutive sends")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aurora – Schedule Generator V2 (Task 3, Part 6)")
    p.add_argument("--segments",   "-s", default="iteration_0_before_learning/user_segments.csv")
    p.add_argument("--timing",     "-t", default="iteration_1_after_learning/timing_recommendations.csv")
    p.add_argument("--templates",  "-m", default="iteration_1_after_learning/message_templates.csv")
    p.add_argument("--ucb-scores", "-u", default="codebase/pipeline_cache/ucb_scores.csv")
    p.add_argument("--output",     "-o", default="iteration_1_after_learning/user_notification_schedule.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_schedule_generator_v2(
        segments_path=args.segments,
        timing_path=args.timing,
        templates_path=args.templates,
        ucb_scores_path=args.ucb_scores,
        output_path=args.output,
    )
