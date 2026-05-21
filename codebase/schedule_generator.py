"""
Schedule Generator  –  Project Aurora (SpeakX)
===============================================
Task 2C — Final "Assembly Line"

Reads:
  - iteration_0_before_learning/user_segments.csv         (users with segment assignments)
  - iteration_0_before_learning/timing_recommendations.csv (freq + time windows per segment)
  - iteration_0_before_learning/message_templates.csv      (template library)

Algorithm: Dynamic Individual-Level Pacing
  1. Memory Loading & Indexing
     - Fast-lookup timing dict: segment_id → timing row
     - Nested template dict: segment_id → goal_id → [template_ids]
  2. User State Resolution
     - active_goal derived from lifecycle_stage + days_since_signup
  3. Send-Time Optimization  (Anchoring + Anti-Clumping)
     - Anchor to preferred_hour from user_segments.csv (default 18 if missing)
     - Dynamic spacing: notif_n_time = (preferred_hour + (n-1)*SPACING) % 24
     - SPACING = 24 // 9 = 2 hours → all 9 slots get unique, evenly-spread times
  4. Multi-Channel Frequency Capping
     - First ``base_frequency`` or 3 slots (whichever is smaller) → "Push"
     - Remaining slots → "In-App"  (avoids OS push-throttling)
  5. Template Selection & Shortfall Mitigation
     - Query nested dict by segment + active_goal
     - Round-Robin Cycler (itertools.cycle) when pool < 9
     - Fallback to nearest earlier chronological milestone if goal pool is empty
  6. Wide-Format Matrix Assembly
     - ALL 9 slots always populated — zero NaN / empty rows

Outputs:
  - iteration_0_before_learning/user_notification_schedule.csv

Wide-format schema (27 data columns + 3 id/meta columns):
  user_id, segment_id, lifecycle_day,
  notif_1_template_id, notif_1_time, notif_1_channel,
  ...
  notif_9_template_id, notif_9_time, notif_9_channel

Notes:
  - Every user ALWAYS receives exactly 9 notification slots (no NaN / empty).
  - Push budget is capped by min(base_frequency, 3); overflow routed to In-App.
  - lifecycle_day is derived from days_since_signup (e.g. "Day 849").
  - Template sampling uses random.seed for reproducibility; seed via env var AURORA_SEED.

Usage
-----
    python schedule_generator.py \\
        --segments   iteration_0_before_learning/user_segments.csv \\
        --timing     iteration_0_before_learning/timing_recommendations.csv \\
        --templates  iteration_0_before_learning/message_templates.csv \\
        --output     iteration_0_before_learning/user_notification_schedule.csv
"""

import argparse
import csv
import itertools
import logging
import os
import random
from pathlib import Path

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.schedule_generator")

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_NOTIF_SLOTS: int = 9     # every user always gets 9 slots
MAX_PUSH_SLOTS: int  = 3     # OS push budget — at most 3 Push per user per day
DEFAULT_PREFERRED_HOUR: int = 18   # fallback when preferred_hour is missing/NaN
SPACING_HOURS: int = 24 // MAX_NOTIF_SLOTS   # = 2 h → 9 unique times fit in 24 h

# Deterministic sampling for reproducibility
_SEED: int = int(os.getenv("AURORA_SEED", "42"))
random.seed(_SEED)

# Chronological order of goal milestones for fallback stepping
GOAL_MILESTONES: list[str] = ["Day 1", "Day 3", "Week 1", "Week 2", "Week 4"]

# Standard Time Windows — hour → window label mapping
# early_morning : 06:00 - 08:59  → Morning motivation, habit trigger
# mid_morning   : 09:00 - 11:59  → Work break reminder
# afternoon     : 12:00 - 14:59  → Lunch break engagement
# late_afternoon: 15:00 - 17:59  → Productivity boost
# evening       : 18:00 - 20:59  → Post-work learning
# night         : 21:00 - 05:59  → End-of-day recap, streak save
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

SCHEDULE_CSV_FIELDNAMES: list[str] = [
    "user_id", "segment_id", "lifecycle_day",
] + _NOTIF_FIELDS


# ══════════════════════════════════════════════════════════════════════════════
# Send-Time Optimization  (Anchoring + Anti-Clumping)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_preferred_hour(raw_value) -> int:
    """Parse preferred_hour from a raw CSV / DataFrame value.

    Returns an integer 0-23.  Falls back to DEFAULT_PREFERRED_HOUR (18)
    when the value is missing, empty, NaN, or out of range.
    """
    try:
        hour: int = int(float(raw_value))
        if 0 <= hour <= 23:
            return hour
    except (ValueError, TypeError):
        pass
    return DEFAULT_PREFERRED_HOUR


def _hour_to_window(hour: int) -> str:
    """Map an integer hour (0-23) to a standard time-window label.

    Mapping:
        06-08 → early_morning | 09-11 → mid_morning  | 12-14 → afternoon
        15-17 → late_afternoon | 18-20 → evening      | 21-23 / 00-05 → night
    """
    for lo, hi, label in TIME_WINDOWS:
        if lo <= hour <= hi:
            return label
    return "night"   # fallback (should never reach)


def compute_send_time(preferred_hour: int, slot_index: int) -> str:
    """Return the time-window label for a given notification slot.

    Uses dynamic spacing so all 9 slots spread across different windows:
        computed_hour = (preferred_hour + (slot_index - 1) * SPACING_HOURS) % 24
    Then maps ``computed_hour`` to its standard time window.

    Parameters
    ----------
    preferred_hour : int
        The user's historically most-active hour (0-23).
    slot_index : int
        1-based notification slot number (1 = first notification).

    Returns
    -------
    str
        Time-window label (e.g. ``"evening"``, ``"mid_morning"``).
    """
    hour: int = (preferred_hour + (slot_index - 1) * SPACING_HOURS) % 24
    return _hour_to_window(hour)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Channel Frequency Capping
# ══════════════════════════════════════════════════════════════════════════════

def assign_channel(slot_index: int, push_budget: int) -> str:
    """Return the delivery channel for a given notification slot.

    Slots 1..push_budget → ``"Push"``  (within OS daily push limit).
    Slots push_budget+1..9 → ``"In-App"`` (avoids OS throttling / blocking).

    Parameters
    ----------
    slot_index : int
        1-based notification slot number.
    push_budget : int
        Number of push-eligible slots for this user (min(base_frequency, 3)).

    Returns
    -------
    str
        ``"Push"`` or ``"In-App"``.
    """
    return "Push" if slot_index <= push_budget else "In-App"


# ══════════════════════════════════════════════════════════════════════════════
# User State Resolution
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_active_goal(lifecycle_stage: str, days_since_signup: float) -> str:
    """Derive a user's active goal milestone from lifecycle stage and tenure.

    Mapping rules (from blueprint):
      Trial  Day 0-1  → ``"Day 1"``
      Trial  Day 2-3  → ``"Day 3"``
      Trial  Day 4-7  → ``"Week 1"``
      Trial  Day 8+   → ``"Week 2"``
      Paid   Day 0-7  → ``"Week 1"``
      Paid   Day 8-14 → ``"Week 2"``
      Paid   Day 15+  → ``"Week 4"``
    """
    stage: str = str(lifecycle_stage).strip().lower()
    try:
        days: float = float(days_since_signup)
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
    else:  # Paid / Premium / churned / inactive / any other
        if days <= 7:
            return "Week 1"
        elif days <= 14:
            return "Week 2"
        else:
            return "Week 4"


# ══════════════════════════════════════════════════════════════════════════════
# Template Selection & Shortfall Mitigation
# ══════════════════════════════════════════════════════════════════════════════

def _get_templates_with_fallback(
    templates_by_goal: dict[str, list[str]],
    active_goal: str,
) -> tuple[list[str], str]:
    """Return the template pool for *active_goal* with chronological fallback.

    If the requested goal has no templates, step backward through
    ``GOAL_MILESTONES`` until a non-empty pool is found.  As a last resort
    flatten *all* templates for the segment.

    Returns
    -------
    tuple[list[str], str]
        ``(template_id_list, resolved_goal_label)``
    """
    # Direct hit
    if active_goal in templates_by_goal and templates_by_goal[active_goal]:
        return templates_by_goal[active_goal], active_goal

    # Step backward through GOAL_MILESTONES
    try:
        start: int = GOAL_MILESTONES.index(active_goal)
    except ValueError:
        start = len(GOAL_MILESTONES)

    for i in range(start - 1, -1, -1):
        label: str = GOAL_MILESTONES[i]
        if label in templates_by_goal and templates_by_goal[label]:
            return templates_by_goal[label], label

    # Last resort: flatten all templates for this segment
    flat: list[str] = [t for pool in templates_by_goal.values() for t in pool]
    return flat, "any"


def _select_templates_round_robin(template_pool: list[str], n: int) -> list[str]:
    """Select *n* template_ids from *template_pool*.

    When pool >= *n* → ``random.sample`` (no repeats).
    When pool <  *n* → ``itertools.cycle`` (Round-Robin) to fill all *n* slots.

    Returns a list of length exactly *n*.
    """
    if not template_pool:
        return [""] * n
    if len(template_pool) >= n:
        return random.sample(template_pool, n)
    # Pool smaller than required: cycle (round-robin)
    cycler = itertools.cycle(template_pool)
    return [next(cycler) for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Write schedule CSV
# ══════════════════════════════════════════════════════════════════════════════

def write_schedule(rows: list[dict], output_path: str) -> None:
    """Write ``user_notification_schedule.csv`` with ``csv.QUOTE_ALL``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SCHEDULE_CSV_FIELDNAMES,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "user_notification_schedule.csv written → %s  (%d rows)", out, len(rows),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_schedule_generator(
    segments_path: str,
    timing_path: str,
    templates_path: str,
    output_path: str,
) -> None:
    """Full Schedule Generator pipeline — Dynamic Individual-Level Pacing.

    Every user receives exactly 9 notification slots — zero empty columns.

    Step 1  Memory Loading & Indexing
    Step 2  User State Resolution      (active_goal from lifecycle + days)
    Step 3  Send-Time Optimization     (preferred_hour anchor + 2 h spacing)
    Step 4  Multi-Channel Capping      (Push × push_budget, In-App × rest)
    Step 5  Template Selection         (Round-Robin fills all 9 + fallback)
    Step 6  Wide-Format Matrix Assembly
    """
    for label, path in [
        ("user_segments.csv",           segments_path),
        ("timing_recommendations.csv",  timing_path),
        ("message_templates.csv",       templates_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found at {path}.")

    users_df     = pd.read_csv(segments_path)
    timing_df    = pd.read_csv(timing_path)
    templates_df = pd.read_csv(templates_path)

    logger.info(
        "Loaded: %d users | %d timing rows | %d templates",
        len(users_df), len(timing_df), len(templates_df),
    )

    # ── Step 1: Index timing by segment_id for O(1) lookups ───────────────────
    timing_index: dict[int, dict] = {
        int(row["segment_id"]): row.to_dict()
        for _, row in timing_df.iterrows()
    }

    # ── Step 1: Nested Template Dictionary: segment_id → goal_id → [ids] ─────
    templates_by_segment_goal: dict[int, dict[str, list[str]]] = {}
    for _, trow in templates_df.iterrows():
        seg:  int = int(trow["segment_id"])
        goal: str = str(trow["goal_id"]).strip()
        tid:  str = str(trow["template_id"])
        templates_by_segment_goal.setdefault(seg, {}).setdefault(goal, []).append(tid)

    rows: list[dict] = []
    skipped: int       = 0
    fallback_used: int = 0
    pref_defaulted: int = 0

    for _, user in users_df.iterrows():
        seg_id: int = int(user["segment_id"])

        if seg_id not in timing_index:
            logger.warning(
                "User %s: segment %d not in timing_recommendations — skipping.",
                user.get("user_id", "?"), seg_id,
            )
            skipped += 1
            continue

        timing: dict = timing_index[seg_id]

        # Push budget: clamp segment's base_frequency to OS limit
        try:
            base_freq: int = int(timing.get("base_frequency", 3))
        except (ValueError, TypeError):
            base_freq = 3
        push_budget: int = min(base_freq, MAX_PUSH_SLOTS)

        # ── Step 2: Derive lifecycle_day and active_goal ──────────────────────
        days = user.get("days_since_signup", 0)
        try:
            days_float: float = float(days)
            lifecycle_day: str = f"Day {int(days_float)}"
        except (ValueError, TypeError):
            days_float = 0.0
            lifecycle_day = str(days)

        lifecycle_stage: str = str(user.get("lifecycle_stage", ""))
        active_goal: str = _resolve_active_goal(lifecycle_stage, days_float)

        # ── Step 3: Send-Time Optimization (anchor + spacing) ─────────────────
        raw_hour = user.get("preferred_hour", "")
        pref_hour: int = _safe_preferred_hour(raw_hour)
        if pref_hour == DEFAULT_PREFERRED_HOUR:
            # Only count as defaulted if the raw value wasn't genuinely 18
            try:
                if int(float(raw_hour)) != DEFAULT_PREFERRED_HOUR:
                    pref_defaulted += 1
            except (ValueError, TypeError):
                pref_defaulted += 1

        # ── Step 5: Template selection with fallback (fill all 9) ─────────────
        seg_goal_map: dict[str, list[str]] = templates_by_segment_goal.get(seg_id, {})
        pool: list[str]
        resolved_goal: str
        pool, resolved_goal = _get_templates_with_fallback(seg_goal_map, active_goal)

        if resolved_goal != active_goal:
            fallback_used += 1
            logger.debug(
                "User %s: fallback %s → %s (segment %d)",
                user.get("user_id", "?"), active_goal, resolved_goal, seg_id,
            )

        # Always select 9 templates (round-robin will cycle if pool < 9)
        sampled_ids: list[str] = _select_templates_round_robin(pool, MAX_NOTIF_SLOTS)

        # ── Step 6: Assemble wide-format row (all 9 slots populated) ──────────
        row: dict = {
            "user_id":       str(user["user_id"]),
            "segment_id":    seg_id,
            "lifecycle_day": lifecycle_day,
        }
        for i in range(1, MAX_NOTIF_SLOTS + 1):
            row[f"notif_{i}_template_id"] = sampled_ids[i - 1]
            row[f"notif_{i}_time"]        = compute_send_time(pref_hour, i)
            row[f"notif_{i}_channel"]     = assign_channel(i, push_budget)

        rows.append(row)

    # ── Warnings ──────────────────────────────────────────────────────────────
    if skipped:
        logger.warning("Skipped %d users (missing segment or timing data).", skipped)
    if fallback_used:
        logger.info(
            "%d users used fallback goal milestone (active goal had no templates).",
            fallback_used,
        )
    if pref_defaulted:
        logger.info(
            "%d users defaulted preferred_hour to %d (missing/invalid).",
            pref_defaulted, DEFAULT_PREFERRED_HOUR,
        )

    write_schedule(rows, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_push: int   = 0
    total_inapp: int  = 0
    for r in rows:
        for i in range(1, MAX_NOTIF_SLOTS + 1):
            ch = r.get(f"notif_{i}_channel", "")
            if ch == "Push":
                total_push += 1
            elif ch == "In-App":
                total_inapp += 1

    print("\n" + "=" * 70)
    print("  SCHEDULE GENERATOR COMPLETE  (Dynamic Individual-Level Pacing)")
    print("=" * 70)
    print(f"  Users scheduled      : {len(rows)}")
    print(f"  Users skipped        : {skipped}")
    print(f"  Fallback goal used   : {fallback_used}")
    print(f"  Preferred-hour dflts : {pref_defaulted}")
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
    """Parse command-line arguments for standalone execution."""
    p = argparse.ArgumentParser(
        description="Aurora – Schedule Generator (Task 2C)"
    )
    p.add_argument(
        "--segments", "-s",
        default="iteration_0_before_learning/user_segments.csv",
        help="Path to user_segments.csv",
    )
    p.add_argument(
        "--timing", "-t",
        default="iteration_0_before_learning/timing_recommendations.csv",
        help="Path to timing_recommendations.csv from timing_optimizer.py",
    )
    p.add_argument(
        "--templates", "-m",
        default="iteration_0_before_learning/message_templates.csv",
        help="Path to message_templates.csv from template_generator.py",
    )
    p.add_argument(
        "--output", "-o",
        default="iteration_0_before_learning/user_notification_schedule.csv",
        help="Output path for user_notification_schedule.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_schedule_generator(
        segments_path=args.segments,
        timing_path=args.timing,
        templates_path=args.templates,
        output_path=args.output,
    )
