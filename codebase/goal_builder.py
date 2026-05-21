"""
Goal Builder  –  Project Aurora (SpeakX)
=========================================
Stage 2 of the Knowledge Bank pipeline.

Reads:
  - feature_goal_map.json    (output of kb_ingestion.py Stage 1)
  - segment_summary.csv      (output of segmentation.py)

For each segment, calls the LLM (Growth Strategist role) to:
  1. Infer a semantic segment name   (inferred_segment_name)
  2. Define primary goal + sub-goals
  3. Build a time-unit progression path

Outputs:
  - segment_goals.csv        – one row per (segment × time_unit)
  - feature_goal_map.json    – enriched with 'segment_strategies' key (--enrich-map)

Usage
-----
    python goal_builder.py \
        --feature-map  data/output/feature_goal_map.json \
        --segments     data/output/segment_summary.csv \
        --output       data/output/segment_goals.csv \
        --enrich-map   data/output/feature_goal_map.json
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.goal_builder")

# ── Ollama client (OpenAI-compatible, no rate limits) ─────────────────────────
from llm_client import client, OLLAMA_MODEL

# ── Timeline constraints per lifecycle stage ───────────────────────────────────
TIMELINE_MAP: dict[str, list[str]] = {
    "trial":    ["Day 0", "Day 1", "Day 3", "Day 5", "Day 7"],
    "paid":     ["Week 1", "Week 2", "Week 4"],
    "churned":  ["Day 1", "Day 7", "Day 14", "Day 30"],
    "inactive": ["Day 1", "Day 7", "Day 14", "Day 30"],
}
DEFAULT_TIMELINE = ["Day 1", "Day 3", "Day 7", "Week 2", "Week 4"]


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt  (Growth Strategist — verbatim from feature_goal_and_define_goals.txt)
# ══════════════════════════════════════════════════════════════════════════════

GROWTH_STRATEGIST_SYSTEM = """**ROLE:**
You are an expert User Lifecycle Strategist. Your goal is to analyze a specific user cohort based on their behavioral attributes, name them, and design a high-conversion journey.

CRITICAL RULES:
- Return ONLY valid JSON. No commentary outside JSON.
- Never return empty arrays or empty strings.
- Each rationale must be at least 2 sentences.
- focus_feature_id MUST exactly match a key from the provided feature_map JSON.
- inferred_segment_name must be short (2-3 words, underscore-separated, e.g. Trial_Gamer).
- sub_goals must have 2-3 items."""


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers (used when segment_summary lacks dominant_* columns)
# ══════════════════════════════════════════════════════════════════════════════

def _infer_lifecycle(seg_profile: dict) -> str:
    """Infer dominant lifecycle from avg score columns."""
    churn = float(seg_profile.get("avg_churn_risk", 0))
    act   = float(seg_profile.get("avg_activeness", 0))
    if churn > 0.5:
        return "inactive"
    if act > 0.6:
        return "paid"
    return "trial"


def _infer_drive(seg_profile: dict) -> str:
    """Infer dominant propensity drive as argmax of avg propensity scores."""
    drives = {
        "gamification": float(seg_profile.get("avg_gamification", 0)),
        "learning":     float(seg_profile.get("avg_learning",     0)),
        "achievement":  float(seg_profile.get("avg_achievement",  0)),
        "social":       float(seg_profile.get("avg_social",       0)),
    }
    return max(drives, key=drives.get)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def extract_json_safely(raw: str) -> dict:
    """Strip markdown fences, find first '{', parse JSON."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found:\n{raw[:500]}")
    return json.loads(cleaned[start:])


def call_llm(system_prompt: str, user_prompt: str, retries: int = 5) -> dict:
    """Call Gemini LLM with retry + exponential backoff. Returns parsed JSON dict."""
    for attempt in range(1, retries + 1):
        try:
            logger.info("LLM call (attempt %d/%d)…", attempt, retries)
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=16384,
            )
            raw = response.choices[0].message.content or ""
            return extract_json_safely(raw)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("JSON parse error on attempt %d: %s", attempt, e)
        except Exception as e:
            logger.error("Unexpected error on attempt %d: %s", attempt, e)

        if attempt < retries:
            time.sleep(3)

    raise RuntimeError(f"LLM call failed after {retries} attempts.")


def _time_sort_key(label: str) -> tuple[int, int]:
    """
    Sort progression time labels in chronological order.
    Day N  → (0, N)
    Week N → (1, N)
    Month N → (2, N)
    """
    label = label.strip()
    m = re.match(r"(Day|Week|Month)\s+(\d+)", label, re.IGNORECASE)
    if not m:
        return (99, 0)
    unit = m.group(1).lower()
    n    = int(m.group(2))
    order = {"day": 0, "week": 1, "month": 2}
    return (order.get(unit, 99), n)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Generate segment strategy via LLM
# ══════════════════════════════════════════════════════════════════════════════

def generate_segment_strategy(
    feature_map: dict,
    segment_profile: dict,
) -> dict:
    """
    Call the Growth Strategist LLM for one segment.

    segment_profile keys (from segment_summary.csv):
        segment_id, segment_name, n_users, dominant_lifecycle,
        dominant_drive, avg_activeness, avg_churn_risk,
        avg_gamification, avg_learning, avg_achievement, avg_social,
        decision_rules
    """
    lifecycle    = str(segment_profile.get("dominant_lifecycle") or _infer_lifecycle(segment_profile)).lower()
    timeline     = TIMELINE_MAP.get(lifecycle, DEFAULT_TIMELINE)
    timeline_str = ", ".join(timeline)

    # Build behavioral summary from profile metrics
    seg_id   = segment_profile.get("segment_id", "?")
    n_users  = segment_profile.get("n_users", "?")
    dom_drive = segment_profile.get("dominant_drive") or _infer_drive(segment_profile)
    act       = float(segment_profile.get("avg_activeness",    0))
    churn     = float(segment_profile.get("avg_churn_risk",    0))
    gamif     = float(segment_profile.get("avg_gamification",  0))
    learn     = float(segment_profile.get("avg_learning",      0))
    achiev    = float(segment_profile.get("avg_achievement",   0))
    social    = float(segment_profile.get("avg_social",        0))
    rules     = segment_profile.get("decision_rules", "Not available")

    feature_map_str = json.dumps(feature_map, indent=2)

    user_prompt = f"""**INPUT CONTEXT 1: PRODUCT FEATURES**
{feature_map_str}

**INPUT CONTEXT 2: COHORT DECISION RULES (Behavioral Profile)**
- **Segment ID:** {seg_id}  ({n_users} users)
- **Lifecycle Logic:** Dominant lifecycle stage = "{lifecycle}" → use timeline: {timeline_str}
- **Decision Logic:** {rules}
- **Dominant Propensity Drive:** {dom_drive}
- **Propensity Scores:** gamification={gamif:.3f}, learning={learn:.3f}, achievement={achiev:.3f}, social={social:.3f}
- **Activity Level:** avg_activeness={act:.3f}, avg_churn_risk={churn:.3f}

**TASK:**
1. **Name the Segment:** Generate a semantic Segment Name (2-3 words, underscore_separated, e.g. "Trial_Gamer", "Paid_Scholar").
2. **Define Strategy:**
   - **Primary Goal:** The single macro outcome for this specific profile.
   - **Sub-Goals:** 2-3 specific milestones leading to the primary goal.
3. **Map Progression:**
   - Create a time-unit plan using EXACTLY these time labels: {timeline_str}
   - Each step MUST reference a focus_feature_id that exactly matches a key in the feature_map.

**TIMELINE CONSTRAINTS:**
- Use ONLY these time units: {timeline_str}
- Do NOT add extra time units.

**OUTPUT SCHEMA (JSON ONLY):**
{{
  "inferred_segment_name": "String (e.g., Trial_Gamer)",
  "strategy": {{
    "primary_goal": "String",
    "sub_goals": ["String", "String", "String"]
  }},
  "progression_path": [
    {{
      "time_unit": "Day 0",
      "focus_feature_id": "String (Must match a key in feature_map)",
      "user_task": "String (Actionable Goal)",
      "rationale": "String (Why this feature fits this behavioral rule — 2+ sentences)"
    }}
  ]
}}"""

    return call_llm(GROWTH_STRATEGIST_SYSTEM, user_prompt)


# ══════════════════════════════════════════════════════════════════════════════
# Build segment_goals rows
# ══════════════════════════════════════════════════════════════════════════════

def build_goals_rows(
    feature_goal_map: dict,
    segment_summary_df: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """
    For each segment in segment_summary_df, call the LLM and flatten
    progression_path into one row per (segment × time_unit).

    Returns (rows_list, strategies_dict) where strategies_dict maps
    segment_id → full LLM response (for --enrich-map).
    """
    feature_map   = feature_goal_map.get("feature_map", feature_goal_map)
    rows: list[dict]   = []
    strategies: dict   = {}

    total = len(segment_summary_df)
    for idx, row in segment_summary_df.iterrows():
        seg_profile = row.to_dict()
        seg_id      = int(seg_profile["segment_id"])
        logger.info(
            "Processing segment %d/%d (Segment_%d, n=%s, lifecycle=%s)…",
            idx + 1, total, seg_id,
            seg_profile.get("n_users", "?"),
            seg_profile.get("dominant_lifecycle") or _infer_lifecycle(seg_profile),
        )

        try:
            result = generate_segment_strategy(feature_map, seg_profile)
        except RuntimeError as e:
            logger.error("Segment %d failed: %s — skipping.", seg_id, e)
            continue

        inferred_name  = result.get("inferred_segment_name", f"Segment_{seg_id}")
        strategy       = result.get("strategy", {})
        primary_goal   = strategy.get("primary_goal", "")
        sub_goals_list = strategy.get("sub_goals", [])
        sub_goals_str  = " | ".join(sub_goals_list)
        progression    = result.get("progression_path", [])

        # Sort progression by time label
        progression.sort(key=lambda p: _time_sort_key(p.get("time_unit", "")))

        for step in progression:
            rows.append({
                "segment_id":       seg_id,
                "segment_name":     inferred_name,
                "lifecycle_stage":  str(seg_profile.get("dominant_lifecycle") or _infer_lifecycle(seg_profile)),
                "primary_goal":     primary_goal,
                "sub_goals":        sub_goals_str,
                "day_label":        step.get("time_unit", ""),
                "focus_feature_id": step.get("focus_feature_id", ""),
                "user_task":        step.get("user_task", ""),
                "rationale":        step.get("rationale", ""),
            })

        strategies[str(seg_id)] = result
        logger.info(
            "  → %s | %d steps | primary_goal: %s",
            inferred_name, len(progression), primary_goal[:60],
        )

    return rows, strategies


# ══════════════════════════════════════════════════════════════════════════════
# Write segment_goals.csv
# ══════════════════════════════════════════════════════════════════════════════

GOALS_CSV_FIELDNAMES = [
    "segment_id",
    "segment_name",
    "lifecycle_stage",
    "primary_goal",
    "sub_goals",
    "day_label",
    "focus_feature_id",
    "user_task",
    "rationale",
]


def _atomic_write_csv(
    rows: list[dict],
    fieldnames: list[str],
    output_path: Path,
    label: str,
) -> None:
    """
    Write rows to a .tmp file first, then atomically replace the target.
    If the target is locked (e.g. open in Excel), the .tmp file is kept
    with the data intact and a clear error is raised.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    try:
        tmp_path.replace(output_path)  # atomic on POSIX; overwrites on Windows
    except PermissionError:
        raise PermissionError(
            f"Cannot write to '{output_path}' — it is open in another program (e.g. Excel).\n"
            f"Close the file, then rename '{tmp_path}' → '{output_path}' and re-run."
        )
    logger.info("%s written → %s  (%d rows)", label, output_path, len(rows))


def write_segment_goals(rows: list[dict], output_path: str) -> None:
    """Write segment_goals.csv with csv.QUOTE_ALL to handle commas in text."""
    _atomic_write_csv(rows, GOALS_CSV_FIELDNAMES, Path(output_path), "segment_goals.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Enrich feature_goal_map.json with segment_strategies
# ══════════════════════════════════════════════════════════════════════════════

def enrich_feature_map(
    feature_map_path: str,
    strategies: dict,
    output_path: str,
) -> None:
    """
    Load existing feature_goal_map.json, add 'segment_strategies' key,
    and write back to output_path (may be the same file).
    """
    fm_path = Path(feature_map_path)
    existing = json.loads(fm_path.read_text(encoding="utf-8"))
    existing["segment_strategies"] = strategies
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("feature_goal_map.json enriched with segment_strategies → %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_goal_builder(
    feature_map_path: str,
    segments_path: str,
    output_path: str,
    enrich_map_path: str | None = None,
) -> None:
    """
    Full Goal Builder pipeline.
    1. Load feature_goal_map.json + segment_summary.csv
    2. Call LLM per segment (Growth Strategist)
    3. Write segment_goals.csv
    4. Optionally enrich feature_goal_map.json with segment_strategies
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    fm_path   = Path(feature_map_path)
    seg_path  = Path(segments_path)

    if not fm_path.exists():
        raise FileNotFoundError(
            f"feature_goal_map.json not found at {fm_path}. "
            "Run kb_ingestion.py first."
        )
    if not seg_path.exists():
        raise FileNotFoundError(
            f"segment_summary.csv not found at {seg_path}. "
            "Run segmentation.py with --summary flag first."
        )

    feature_goal_map   = json.loads(fm_path.read_text(encoding="utf-8"))
    segment_summary_df = pd.read_csv(seg_path)

    logger.info(
        "Loaded feature_goal_map: %d features | segment_summary: %d segments",
        len(feature_goal_map.get("feature_map", feature_goal_map)),
        len(segment_summary_df),
    )

    # ── Generate goals per segment ────────────────────────────────────────────
    rows, strategies = build_goals_rows(feature_goal_map, segment_summary_df)

    if not rows:
        logger.error("No rows generated — check LLM errors above.")
        return

    # ── Write segment_goals.csv ───────────────────────────────────────────────
    write_segment_goals(rows, output_path)

    # ── Optionally enrich feature_goal_map.json ───────────────────────────────
    if enrich_map_path:
        enrich_feature_map(feature_map_path, strategies, enrich_map_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    segment_names = {r["segment_id"]: r["segment_name"] for r in rows}
    print("\n" + "=" * 70)
    print("  GOAL BUILDER COMPLETE")
    print("=" * 70)
    print(f"  Segments processed : {len(strategies)}/{len(segment_summary_df)}")
    print(f"  Total rows written : {len(rows)}")
    print(f"  Output             : {output_path}")
    if enrich_map_path:
        print(f"  Enriched map       : {enrich_map_path}")
    print()
    print(f"  {'Seg ID':<8} {'Inferred Name':<30} {'Steps'}")
    print(f"  {'------':<8} {'-------------':<30} {'-----'}")
    step_counts: dict[int, int] = {}
    for r in rows:
        step_counts[r["segment_id"]] = step_counts.get(r["segment_id"], 0) + 1
    for seg_id in sorted(step_counts):
        print(f"  {seg_id:<8} {segment_names[seg_id]:<30} {step_counts[seg_id]}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – Goal Builder (Task 1, Stage 2)"
    )
    p.add_argument(
        "--feature-map", "-f",
        default="data/output/feature_goal_map.json",
        help="Path to feature_goal_map.json from kb_ingestion.py",
    )
    p.add_argument(
        "--segments", "-s",
        default="data/output/segment_summary.csv",
        help="Path to segment_summary.csv from segmentation.py",
    )
    p.add_argument(
        "--output", "-o",
        default="data/output/segment_goals.csv",
        help="Output path for segment_goals.csv",
    )
    p.add_argument(
        "--enrich-map", "-e",
        default=None,
        help="If set, enrich feature_goal_map.json with segment_strategies at this path "
             "(can be the same as --feature-map to overwrite in-place)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_goal_builder(
        feature_map_path=args.feature_map,
        segments_path=args.segments,
        output_path=args.output,
        enrich_map_path=args.enrich_map,
    )
