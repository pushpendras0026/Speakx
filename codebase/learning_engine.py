"""
Learning Engine  –  Project Aurora (SpeakX)
============================================
Task 3 — Iteration 1 Self-Learning Loop

Reads:
  - input/experiment_results.csv                           (CTR feedback from Iteration 0)
  - iteration_0_before_learning/message_templates.csv      (V0 template library)
  - iteration_0_before_learning/timing_recommendations.csv (V0 timing per segment)
  - iteration_0_before_learning/user_segments.csv          (user → segment mapping)
  - iteration_0_before_learning/segment_goals.csv          (segment × goal context)
  - iteration_0_before_learning/communication_themes.csv   (themes per segment)
  - iteration_0_before_learning/feature_goal_map.json      (feature metadata)
  - iteration_0_before_learning/allowed_tone_hook_matrix.json

experiment_results.csv expected schema:
  template_id, segment_id, lifecycle_stage, goal, theme,
  notification_window, total_sends, total_opens, total_engagements,
  ctr, engagement_rate, uninstall_rate, performance_status

Produces (all in iteration_1_after_learning/  unless noted):
  - message_templates.csv          (V0 templates + new contrastive-LLM templates)
  - timing_recommendations.csv     (EMA-updated timing per segment)
  - user_segments.csv                 (copy of iteration_0 user_segments into iteration_1)
  - pipeline_cache/ucb_scores.csv     (UCB1 score per template for schedule_generator_v2)
  - learning_delta_report.csv         (project root — causal trace: what changed and why)

Part 6 (schedule_generator_v2.py) is called at the end to produce:
  - user_notification_schedule.csv

Usage
-----
    python codebase/learning_engine.py \
        --experiment   input/experiment_results.csv \
        --templates    iteration_0_before_learning/message_templates.csv \
        --timing       iteration_0_before_learning/timing_recommendations.csv \
        --segments     iteration_0_before_learning/user_segments.csv \
        --goals        iteration_0_before_learning/segment_goals.csv \
        --themes       iteration_0_before_learning/communication_themes.csv \
        --feature-map  iteration_0_before_learning/feature_goal_map.json \
        --tone-matrix  iteration_0_before_learning/allowed_tone_hook_matrix.json \
        --out-dir      iteration_1_after_learning
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.learning_engine")

# ── Ollama client (OpenAI-compatible, no rate limits) ─────────────────────────
from llm_client import client, OLLAMA_MODEL

# ── Constants ──────────────────────────────────────────────────────────────────
# Part 5: EMA smoothing factor
EMA_ALPHA = 0.70       # 70% empirical weight, 30% prior

# Part 6: UCB1 exploration constant (tuned for low-decimal CTR values)
UCB_C = 0.15

# All 6 standard time windows (same order as timing_optimizer.py)
TIME_WINDOWS = [
    "early_morning",
    "mid_morning",
    "afternoon",
    "late_afternoon",
    "evening",
    "night",
]

# Output CSV fieldnames
TEMPLATES_V2_FIELDNAMES = [
    "template_id", "segment_id", "lifecycle_stage", "goal_id", "theme",
    "message_title_en", "message_title_hi",
    "message_body_en", "message_body_hi",
    "cta_text_en", "cta_text_hi",
    "tone_used", "hook_type", "feature_reference",
    "iteration",   # 0 = Iteration 0, 1 = Iteration 1 (newly generated)
]

TIMING_V2_FIELDNAMES = [
    "segment_id", "base_frequency", "guardrail_applied",
    "primary_window", "secondary_window", "tertiary_window",
    "allocation_distribution", "expected_ctr", "expected_engagement",
]

UCB_FIELDNAMES = ["template_id", "ucb_score", "flag"]

DELTA_REPORT_FIELDNAMES = [
    "trace_type", "segment_id", "entity_id", "change_summary", "causal_reason",
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared LLM utilities
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json_safely(raw: str) -> dict:
    """Strip markdown fences, find first '{', parse JSON."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found:\n{raw[:500]}")
    return json.loads(cleaned[start:])


def _call_llm(system_prompt: str, user_prompt: str, retries: int = 5) -> dict:
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
                temperature=0.7,
                max_tokens=16384,
            )
            raw = response.choices[0].message.content or ""
            return _extract_json_safely(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("JSON parse error on attempt %d: %s", attempt, e)
        except Exception as e:
            logger.error("Unexpected LLM error on attempt %d: %s", attempt, e)
        if attempt < retries:
            time.sleep(3)
    raise RuntimeError(f"LLM call failed after {retries} attempts.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Feedback Ingestion & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(exp_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Step 3.1: Aggregate experiment_results.csv at three levels.

    Handles the updated experiment_results.csv schema:
      template_id, segment_id, lifecycle_stage, goal, theme,
      notification_window, total_sends, total_opens, total_engagements,
      ctr, engagement_rate, uninstall_rate, performance_status

    Returns:
      template_metrics:   {template_id: {sends, clicks, ctr, engagement_rate, uninstall_rate}}
      timing_metrics:     {(segment_id, time_window): {sends, clicks, ctr}}
      segment_engagement: {segment_id: float}  — sends-weighted engagement_rate per segment
    """
    # Template-level
    tmpl_grp = exp_df.groupby("template_id").agg({
        "total_sends":        "sum",
        "total_opens":        "sum",
        "total_engagements":  "sum",
        "ctr":                "mean",
        "engagement_rate":    "mean",
        "uninstall_rate":     "mean",
    })
    template_metrics: dict = {}
    for tid, row in tmpl_grp.iterrows():
        sends  = int(row["total_sends"])
        clicks = int(row["total_opens"])
        ctr    = clicks / sends if sends > 0 else 0.0
        template_metrics[str(tid)] = {
            "sends": sends,
            "clicks": clicks,
            "ctr": ctr,
            "engagement_rate": float(row["engagement_rate"]),
            "uninstall_rate":  float(row["uninstall_rate"]),
        }

    # Timing-level (segment × notification_window)
    timing_grp = exp_df.groupby(["segment_id", "notification_window"])[["total_sends", "total_opens"]].sum()
    timing_metrics: dict = {}
    for (seg_id, tw), row in timing_grp.iterrows():
        sends  = int(row["total_sends"])
        clicks = int(row["total_opens"])
        ctr    = clicks / sends if sends > 0 else 0.0
        timing_metrics[(int(seg_id), str(tw))] = {"sends": sends, "clicks": clicks, "ctr": ctr}

    # Segment-level engagement (sends-weighted average of total_engagements / total_sends)
    seg_grp = exp_df.groupby("segment_id")[["total_sends", "total_engagements"]].sum()
    segment_engagement: dict[int, float] = {}
    for seg_id, row in seg_grp.iterrows():
        sends       = int(row["total_sends"])
        engagements = int(row["total_engagements"])
        segment_engagement[int(seg_id)] = engagements / sends if sends > 0 else 0.0

    logger.info(
        "Aggregated: %d template CTRs | %d (segment, window) CTRs | %d segment engagement rates",
        len(template_metrics), len(timing_metrics), len(segment_engagement),
    )
    return template_metrics, timing_metrics, segment_engagement


def _normalize_status(raw_status: str) -> str:
    """Normalize performance_status values (case + common typos)."""
    s = str(raw_status or "").strip().lower()
    if s in {"good", "god", "gud"}:
        return "GOOD"
    if s in {"bad", "bsd"}:
        return "BAD"
    if s in {"neutral", "neutal", "neutrl", "ok"}:
        return "NEUTRAL"
    return "NEUTRAL"


def build_exp_lookup(exp_df: pd.DataFrame) -> dict[str, dict]:
    """
    Step 3.2: Read performance_status DIRECTLY from experiment_results.csv.

    Uses the performance_status column that SpeakX provides — no re-computation.
    Returns dict: { template_id → {segment_id, performance_status, ctr,
                                    engagement_rate, sends, goal, theme,
                                    lifecycle_stage} }
    """
    lookup: dict[str, dict] = {}
    for _, row in exp_df.iterrows():
        tid = str(row["template_id"])
        lookup[tid] = {
            "segment_id":        int(row["segment_id"]),
            "performance_status": _normalize_status(row.get("performance_status", "NEUTRAL")),
            "ctr":               float(row.get("ctr", 0.0)),
            "engagement_rate":   float(row.get("engagement_rate", 0.0)),
            "sends":             int(row.get("total_sends", 0)),
            "opens":             int(row.get("total_opens", 0)),
            "goal":              str(row.get("goal", "")),
            "theme":             str(row.get("theme", "")),
            "lifecycle_stage":   str(row.get("lifecycle_stage", "")),
        }
    good_n    = sum(1 for v in lookup.values() if v["performance_status"] == "GOOD")
    neutral_n = sum(1 for v in lookup.values() if v["performance_status"] == "NEUTRAL")
    bad_n     = sum(1 for v in lookup.values() if v["performance_status"] == "BAD")
    logger.info(
        "Experiment lookup: %d templates  →  GOOD=%d  NEUTRAL=%d  BAD=%d",
        len(lookup), good_n, neutral_n, bad_n,
    )
    return lookup


def _extract_t_num(template_id: str) -> str:
    """Extract T-number suffix from template_id. e.g. 'Foo_Bar_T3' → 'T3'."""
    m = re.search(r"_(T\d+)$", str(template_id), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _extract_goal_period(template_id: str) -> str:
    """
    Extract the time-period label embedded in an experiment template_id.
    e.g. 'Social_Achievers_Week_1_Social_Influence_T1' → 'Week_1'
         'Achievement_Seeker_Day_1_Unpredictability_T1'  → 'Day_1'
    """
    m = re.search(r"_(Week_\d+|Day_\d+|Month_\d+)", str(template_id), re.IGNORECASE)
    return m.group(1) if m else ""


# ════════════════════════════════════════════════════════════════════════════════
# PART 4: Template Generation — per-experiment-template-id
# ════════════════════════════════════════════════════════════════════════════════
#
# Logic (reads performance_status DIRECTLY from experiment_results.csv):
#   GOOD    -> keep as-is: copy best same-segment same-T-num template from iter-0
#              under the experiment template_id  (no LLM call)
#   NEUTRAL -> LLM IMPROVE: use same-segment same-T-num iter-0 template as
#              reference -> generate ONE improved replacement under the same
#              experiment template_id
#   BAD     -> LLM REDESIGN: generate ONE entirely new template under the same
#              experiment template_id (iter-0 reference shown as anti-pattern)
#
#   All iter-0 rows from message_templates.csv NOT matched by any experiment
#   template_id are copied through unchanged (iteration=0).
# ════════════════════════════════════════════════════════════════════════════════

_IMPROVE_SYSTEM = """ROLE: Elite Behavioral Copywriter for Indian EdTech push notifications (SpeakX).

TASK: A template performed NEUTRAL (CTR 5-15%, Engagement 20-40%). Improve it to hit GOOD thresholds (CTR >15%, Engagement >40%).

RULES:
- Keep the SAME segment goal, theme, and lifecycle stage.
- Improve specificity, urgency, psychological hook, and Hinglish naturalness.
- message_title_en: max 8 words. message_body_en: max 15 words. cta_text_en: max 5 words.
- message_title_hi / message_body_hi / cta_text_hi: natural Hinglish (NOT literal translation).
- tone_used MUST exactly match one allowed tone from the tone_hook_matrix.
- hook_type MUST exactly match one allowed hook from the tone_hook_matrix.
- feature_reference: snake_case feature name (e.g. "streak_counter", "leaderboard").
- Return ONLY valid JSON. No commentary outside JSON.

OUTPUT SCHEMA:
{
  "template": {
    "message_title_en": "...",
    "message_title_hi": "...",
    "message_body_en": "...",
    "message_body_hi": "...",
    "cta_text_en": "...",
    "cta_text_hi": "...",
    "tone_used": "...",
    "hook_type": "...",
    "feature_reference": "..."
  }
}"""


_REDESIGN_SYSTEM = """ROLE: Elite Behavioral Copywriter for Indian EdTech push notifications (SpeakX).

TASK: A template FAILED (CTR <5%, Engagement <20%). Generate a completely NEW replacement. Do NOT copy any phrasing, emotional angle, or structure from the failed template.

RULES:
- Target CTR >15% and Engagement >40%. Start completely fresh.
- message_title_en: max 8 words. message_body_en: max 15 words. cta_text_en: max 5 words.
- message_title_hi / message_body_hi / cta_text_hi: natural Hinglish (NOT literal translation).
- tone_used MUST exactly match one allowed tone from the tone_hook_matrix.
- hook_type MUST exactly match one allowed hook from the tone_hook_matrix.
- feature_reference: snake_case feature name (e.g. "streak_counter", "leaderboard").
- Return ONLY valid JSON. No commentary outside JSON.

OUTPUT SCHEMA:
{
  "template": {
    "message_title_en": "...",
    "message_title_hi": "...",
    "message_body_en": "...",
    "message_body_hi": "...",
    "cta_text_en": "...",
    "cta_text_hi": "...",
    "tone_used": "...",
    "hook_type": "...",
    "feature_reference": "..."
  }
}"""


def _seg_templates_by_t_num(templates_df: "pd.DataFrame") -> "dict[tuple[int,str], list[dict]]":
    """
    Build index: (segment_id, T-num) -> [template dicts ...]
    Used to find the best reference template from iteration-0 library
    for a given experiment template_id (matched on segment + T-number).
    """
    index: dict = {}
    for _, row in templates_df.iterrows():
        seg_id = int(row["segment_id"])
        t_num  = _extract_t_num(str(row["template_id"]))
        key    = (seg_id, t_num)
        index.setdefault(key, []).append(row.to_dict())
    return index


def _seg_templates_by_seg(templates_df: "pd.DataFrame") -> "dict[int, list[dict]]":
    """Build index: segment_id -> [all template dicts]. Fallback when T-num match fails."""
    index: dict = {}
    for _, row in templates_df.iterrows():
        seg_id = int(row["segment_id"])
        index.setdefault(seg_id, []).append(row.to_dict())
    return index


def _find_ref_template(
    exp_tid: str,
    exp_seg_id: int,
    by_t_num: dict,
    by_seg: dict,
) -> "dict | None":
    """
    Find best reference template from iteration-0 library for an experiment template_id.
    Tries (segment_id, T-num) match first, then falls back to first template in segment.
    """
    t_num = _extract_t_num(exp_tid)
    candidates = by_t_num.get((exp_seg_id, t_num), [])
    if candidates:
        return candidates[0]
    seg_candidates = by_seg.get(exp_seg_id, [])
    return seg_candidates[0] if seg_candidates else None


def _s(v) -> str:
    """Convert any value (including pandas NaN floats) to a safe string."""
    if v is None:
        return ""
    s = str(v)
    return "" if s.lower() == "nan" else s


def _format_ref_template(t: dict) -> str:
    """Format a reference template as readable text for the LLM prompt."""
    return (
        "  Title EN : " + _s(t.get("message_title_en")) + "\n"
        "  Body EN  : " + _s(t.get("message_body_en")) + "\n"
        "  CTA EN   : " + _s(t.get("cta_text_en")) + "\n"
        "  Title HI : " + _s(t.get("message_title_hi")) + "\n"
        "  Body HI  : " + _s(t.get("message_body_hi")) + "\n"
        "  CTA HI   : " + _s(t.get("cta_text_hi")) + "\n"
        "  Tone     : " + _s(t.get("tone_used")) + "  |  Hook: " + _s(t.get("hook_type")) + "\n"
        "  Feature  : " + _s(t.get("feature_reference"))
    )


def _build_improve_prompt(
    exp_tid: str,
    exp_row: dict,
    ref_template: "dict | None",
    tone_matrix: dict,
) -> str:
    ref_block = _format_ref_template(ref_template) if ref_template else "  (no reference available)"
    return (
        "EXPERIMENT RESULT:\n"
        f"  Template ID     : {exp_tid}\n"
        f"  Segment         : {exp_row['segment_id']}\n"
        f"  Lifecycle       : {exp_row['lifecycle_stage']}\n"
        f"  Goal            : {exp_row['goal']}\n"
        f"  Theme           : {exp_row['theme']}\n"
        f"  Measured CTR    : {exp_row['ctr']:.1%}   (target: >15%)\n"
        f"  Measured Eng    : {exp_row['engagement_rate']:.1%}   (target: >40%)\n"
        "  Performance     : NEUTRAL -- improve it\n\n"
        "REFERENCE TEMPLATE (use as starting point, improve upon it):\n"
        + ref_block + "\n\n"
        "ALLOWED TONE & HOOK MATRIX:\n"
        + json.dumps(tone_matrix, indent=2) + "\n\n"
        "Generate ONE improved template using the schema in the system prompt."
    )


def _build_redesign_prompt(
    exp_tid: str,
    exp_row: dict,
    ref_template: "dict | None",
    tone_matrix: dict,
) -> str:
    ref_block = _format_ref_template(ref_template) if ref_template else "  (no reference available)"
    return (
        "EXPERIMENT RESULT:\n"
        f"  Template ID     : {exp_tid}\n"
        f"  Segment         : {exp_row['segment_id']}\n"
        f"  Lifecycle       : {exp_row['lifecycle_stage']}\n"
        f"  Goal            : {exp_row['goal']}\n"
        f"  Theme           : {exp_row['theme']}\n"
        f"  Measured CTR    : {exp_row['ctr']:.1%}   (target: >15%)\n"
        f"  Measured Eng    : {exp_row['engagement_rate']:.1%}   (target: >40%)\n"
        "  Performance     : BAD -- completely redesign\n\n"
        "FAILED TEMPLATE (what NOT to do -- do not copy its phrasing or structure):\n"
        + ref_block + "\n\n"
        "ALLOWED TONE & HOOK MATRIX:\n"
        + json.dumps(tone_matrix, indent=2) + "\n\n"
        "Generate ONE completely new template using the schema in the system prompt."
    )


def _call_llm_single(system_prompt: str, user_prompt: str) -> dict:
    """
    Like _call_llm but expects a single template response.
    Handles both {template: {...}} and {templates: [{...}]} response shapes.
    """
    result = _call_llm(system_prompt, user_prompt)
    if "template" in result and isinstance(result["template"], dict):
        return result["template"]
    if "templates" in result and isinstance(result["templates"], list) and result["templates"]:
        return result["templates"][0]
    return result


def generate_v2_library(
    templates_df: "pd.DataFrame",
    exp_lookup: "dict[str, dict]",
    tone_matrix: dict,
    themes_df: "pd.DataFrame",
) -> "tuple[list[dict], list[dict]]":
    """
    Step 4.1-4.3: Build the complete iteration-1 template library.

    For EVERY experiment template_id:
      GOOD    -> copy content from best same-segment/T-num iter-0 template,
                 output under experiment template_id (no LLM call needed)
      NEUTRAL -> LLM improve the reference template -> output under exp template_id
      BAD     -> LLM redesign -> fresh template under exp template_id

    All iter-0 templates NOT covered by any experiment template_id are copied as-is.

    Returns:
      all_rows       -- complete output library (iter-0 originals + iter-1 generated)
      generated_rows -- only the newly generated / promoted rows (for delta report)
    """
    by_t_num = _seg_templates_by_t_num(templates_df)
    by_seg   = _seg_templates_by_seg(templates_df)

    generated_rows: list = []
    good_n = neutral_n = bad_n = 0

    for exp_idx, (exp_tid, exp_row) in enumerate(exp_lookup.items(), 1):
        status   = exp_row["performance_status"]    # GOOD / NEUTRAL / BAD
        seg_id   = exp_row["segment_id"]
        theme    = exp_row["theme"]
        lc_stage = exp_row["lifecycle_stage"]
        goal_lbl = _extract_goal_period(exp_tid) or exp_row["goal"][:40]

        ref_template = _find_ref_template(exp_tid, seg_id, by_t_num, by_seg)

        if status == "GOOD":
            # Keep: copy reference content under experiment template_id
            if ref_template:
                row = {
                    "template_id":       exp_tid,
                    "segment_id":        seg_id,
                    "lifecycle_stage":   lc_stage,
                    "goal_id":           goal_lbl,
                    "theme":             theme,
                    "message_title_en":  ref_template.get("message_title_en", ""),
                    "message_title_hi":  ref_template.get("message_title_hi", ""),
                    "message_body_en":   ref_template.get("message_body_en", ""),
                    "message_body_hi":   ref_template.get("message_body_hi", ""),
                    "cta_text_en":       ref_template.get("cta_text_en", ""),
                    "cta_text_hi":       ref_template.get("cta_text_hi", ""),
                    "tone_used":         ref_template.get("tone_used", ""),
                    "hook_type":         ref_template.get("hook_type", ""),
                    "feature_reference": ref_template.get("feature_reference", ""),
                    "iteration":         0,
                    "_action":           "GOOD_KEPT",
                }
                generated_rows.append(row)
                good_n += 1
                logger.debug("GOOD kept  [%d/%d] %s", exp_idx, len(exp_lookup), exp_tid)
            continue

        # NEUTRAL or BAD -> LLM call
        system_prompt = _IMPROVE_SYSTEM  if status == "NEUTRAL" else _REDESIGN_SYSTEM
        user_prompt   = (
            _build_improve_prompt(exp_tid, exp_row, ref_template, tone_matrix)
            if status == "NEUTRAL" else
            _build_redesign_prompt(exp_tid, exp_row, ref_template, tone_matrix)
        )

        logger.info(
            "LLM gen [%d/%d] %s -- status=%s  (ref=%s)",
            exp_idx, len(exp_lookup), exp_tid, status,
            ref_template.get("template_id", "none") if ref_template else "none",
        )

        try:
            item = _call_llm_single(system_prompt, user_prompt)
        except RuntimeError as e:
            logger.error("LLM failed for %s: %s -- copying reference as fallback.", exp_tid, e)
            item = {}

        ref = ref_template or {}
        new_row = {
            "template_id":       exp_tid,
            "segment_id":        seg_id,
            "lifecycle_stage":   lc_stage,
            "goal_id":           goal_lbl,
            "theme":             theme,
            "message_title_en":  item.get("message_title_en",  ref.get("message_title_en",  "")),
            "message_title_hi":  item.get("message_title_hi",  ref.get("message_title_hi",  "")),
            "message_body_en":   item.get("message_body_en",   ref.get("message_body_en",   "")),
            "message_body_hi":   item.get("message_body_hi",   ref.get("message_body_hi",   "")),
            "cta_text_en":       item.get("cta_text_en",       ref.get("cta_text_en",       "")),
            "cta_text_hi":       item.get("cta_text_hi",       ref.get("cta_text_hi",       "")),
            "tone_used":         item.get("tone_used",         ref.get("tone_used",         "")),
            "hook_type":         item.get("hook_type",         ref.get("hook_type",         "")),
            "feature_reference": item.get("feature_reference", ref.get("feature_reference", "")),
            "iteration":         1,
            "_action":           status,
        }
        generated_rows.append(new_row)
        if status == "NEUTRAL":
            neutral_n += 1
        else:
            bad_n += 1

    # Build the complete library:
    # 1. All iteration-0 originals from message_templates.csv (unchanged)
    v0_rows: list = []
    for _, row in templates_df.iterrows():
        r = row.to_dict()
        r["iteration"] = 0
        v0_rows.append(r)

    # 2. Append generated rows (GOOD_KEPT + NEUTRAL improved + BAD redesigned)
    all_rows = v0_rows + generated_rows

    logger.info(
        "V2 library: %d iter-0 originals + %d generated "
        "(GOOD_KEPT=%d  NEUTRAL_improved=%d  BAD_redesigned=%d) = %d total",
        len(v0_rows), len(generated_rows), good_n, neutral_n, bad_n, len(all_rows),
    )
    return all_rows, generated_rows

# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Timing Update (EMA)
# ══════════════════════════════════════════════════════════════════════════════

def _largest_remainder_method(probs: np.ndarray, total: int) -> list[int]:
    """Distribute `total` across len(probs) slots using Largest Remainder Method."""
    fractional   = probs * total
    integer_part = np.floor(fractional).astype(int)
    remainder    = total - int(integer_part.sum())
    frac_parts   = fractional - integer_part
    order        = np.argsort(frac_parts)[::-1]
    for i in range(remainder):
        integer_part[order[i]] += 1
    return integer_part.tolist()


def _reconstruct_prior_probs(timing_row: dict) -> np.ndarray:
    """
    Step 5.2: Reconstruct P_prior (6-element vector over TIME_WINDOWS)
    from the stored allocation_distribution in timing_recommendations.csv.

    The allocation proportions (alloc / base_freq) are used as valid proxies
    for the pre-rounded Softmax probabilities.
    """
    base_freq = int(timing_row.get("base_frequency", 3))
    windows   = [
        str(timing_row.get("primary_window",   "")),
        str(timing_row.get("secondary_window",  "")),
        str(timing_row.get("tertiary_window",   "")),
    ]
    raw = str(timing_row.get("allocation_distribution", "1:1:1"))
    try:
        alloc = [int(x) for x in raw.split(":")]
    except ValueError:
        alloc = [1, 1, 1]
    while len(alloc) < 3:
        alloc.append(0)
    alloc = alloc[:3]

    prior = np.zeros(len(TIME_WINDOWS))
    for tw, count in zip(windows, alloc):
        if tw in TIME_WINDOWS:
            prior[TIME_WINDOWS.index(tw)] = count / max(base_freq, 1)

    total = prior.sum()
    if total > 0:
        prior /= total
    else:
        prior = np.ones(len(TIME_WINDOWS)) / len(TIME_WINDOWS)
    return prior


def compute_ema_timing(
    timing_df: pd.DataFrame,
    timing_metrics: dict,
    segment_engagement: dict | None = None,
) -> list[dict]: 
    """
    Steps 5.1–5.4: EMA-based timing update per segment.

    For each segment:
    1. Build P_obs  — normalize empirical CTR across 6 time windows
    2. Fetch P_prior — from allocation_distribution proportions
    3. Blend: P_new = 0.70 * P_obs + 0.30 * P_prior
    4. LRM re-allocation with same base_frequency
    5. Compute expected_ctr (weighted avg CTR) and expected_engagement (EMA of observed)
    """
    if segment_engagement is None:
        segment_engagement = {}

    rows_out: list[dict] = []

    for _, timing_row in timing_df.iterrows():
        seg_id    = int(timing_row["segment_id"])
        base_freq = int(timing_row["base_frequency"])

        # ── Step 5.1: P_obs ───────────────────────────────────────────────────
        ctr_vector = np.array([
            timing_metrics.get((seg_id, tw), {}).get("ctr", 0.0)
            for tw in TIME_WINDOWS
        ])
        ctr_sum = ctr_vector.sum()
        if ctr_sum > 0:
            p_obs = ctr_vector / ctr_sum
        else:
            # No empirical data for this segment: fall back to uniform
            p_obs = np.ones(len(TIME_WINDOWS)) / len(TIME_WINDOWS)

        # ── Step 5.2: P_prior ─────────────────────────────────────────────────
        p_prior = _reconstruct_prior_probs(timing_row.to_dict())

        # ── Step 5.3: Bayesian Blend ──────────────────────────────────────────
        p_new = EMA_ALPHA * p_obs + (1 - EMA_ALPHA) * p_prior
        p_new /= p_new.sum()   # re-normalise for floating-point safety

        # ── Step 5.4: LRM Re-Allocation ───────────────────────────────────────
        alloc     = _largest_remainder_method(p_new, base_freq)
        top3_idx  = np.argsort(p_new)[::-1][:3]
        top3_wins = [TIME_WINDOWS[i] for i in top3_idx]
        top3_alloc= [alloc[i] for i in top3_idx]

        # Compute updated expected_ctr as weighted average across top-3 windows
        weighted_ctr = sum(
            p_new[i] * ctr_vector[i] for i in top3_idx
        )

        # Compute expected_engagement via EMA blend of observed and prior
        prior_eng = float(str(timing_row.get("expected_engagement", "0.20")).strip() or "0.20")
        obs_eng   = segment_engagement.get(seg_id, None)
        if obs_eng is not None:
            blended_eng = EMA_ALPHA * obs_eng + (1 - EMA_ALPHA) * prior_eng
        else:
            blended_eng = prior_eng  # no experiment data for segment, keep prior
        blended_eng = round(max(0.02, min(0.95, blended_eng)), 4)

        rows_out.append({
            "segment_id":             seg_id,
            "base_frequency":         base_freq,
            "guardrail_applied":      str(timing_row.get("guardrail_applied", "False")),
            "primary_window":         top3_wins[0],
            "secondary_window":       top3_wins[1] if len(top3_wins) > 1 else "",
            "tertiary_window":        top3_wins[2] if len(top3_wins) > 2 else "",
            "allocation_distribution": ":".join(str(n) for n in top3_alloc),
            "expected_ctr":           f"{weighted_ctr:.4f}",
            "expected_engagement":    f"{blended_eng:.4f}",
        })

        logger.info(
            "  Segment %d  EMA update: %s → %s  (alpha=%.2f)",
            seg_id,
            timing_row.get("allocation_distribution", "?"),
            ":".join(str(n) for n in top3_alloc),
            EMA_ALPHA,
        )

    return rows_out

# ════════════════════════════════════════════════════════════════════════════════
# PART 6: UCB1 Score Calculation
# ════════════════════════════════════════════════════════════════════════════════

def compute_ucb_scores(
    templates_v2: list,
    exp_lookup: dict,
) -> dict:
    """
    Step 6.1-6.2: Compute UCB1 score for every template in the V2 library.

      UCB_i = ctr + C * sqrt( ln(N_segment) / sends_i )

    Rules:
      BAD template (from experiment)  -> UCB = -1.0  (suppressed)
      New template (iteration=1)       -> UCB = 999.0 (cold-start, force exploration)
      GOOD / NEUTRAL from experiment  -> standard UCB formula using exp CTR/sends
      iter-0 templates not in exp     -> UCB = 999.0 (no sends data, explore them)
    """
    # Compute segment-level total sends from experiment data
    seg_total_sends: dict = {}
    for exp_data in exp_lookup.values():
        seg = exp_data["segment_id"]
        seg_total_sends[seg] = seg_total_sends.get(seg, 0) + exp_data["sends"]

    ucb_scores: dict = {}
    for t in templates_v2:
        tid      = str(t["template_id"])
        seg      = int(t["segment_id"])
        iter_num = int(t.get("iteration", 0))
        action   = str(t.get("_action", ""))
        n_seg    = max(seg_total_sends.get(seg, 1), 1)

        if action == "BAD":
            # Suppressed: was BAD in experiment, do not route users to it
            ucb_scores[tid] = -1.0
        elif iter_num == 1:
            # Newly generated (NEUTRAL improved or BAD redesigned) -> cold start
            ucb_scores[tid] = 999.0
        elif tid in exp_lookup:
            # Experiment template_id with known sends/CTR
            exp_data = exp_lookup[tid]
            sends_i  = exp_data["sends"]
            ctr_i    = exp_data["ctr"]
            if sends_i == 0:
                ucb_scores[tid] = 999.0
            else:
                exploration = UCB_C * math.sqrt(math.log(n_seg) / sends_i)
                ucb_scores[tid] = ctr_i + exploration
        else:
            # Original iter-0 template not in experiment -> explore
            ucb_scores[tid] = 999.0

    return ucb_scores


# ════════════════════════════════════════════════════════════════════════════════
# PART 7: Delta Reporter
# ════════════════════════════════════════════════════════════════════════════════

def generate_delta_report(
    timing_df: "pd.DataFrame",
    timing_v2_rows: list,
    generated_rows: list,
    exp_lookup: dict,
    timing_metrics: dict,
) -> list:
    """
    Step 7.1-7.3: Generate causal trace comparing Iteration 0 and Iteration 1.

    Timing Traces:
      - Compare allocation_distribution per segment before/after EMA.

    Content Traces (one per processed experiment template):
      GOOD    -> "Retained" trace (CTR >= 15%, Eng >= 40%)
      NEUTRAL -> "Improved" trace (reference used, LLM generated improved variant)
      BAD     -> "Redesigned" trace (fresh generation, original suppressed)
    """
    traces: list = []

    # -- Timing Traces ---------------------------------------------------------
    timing_v0_index = {int(r["segment_id"]): r.to_dict() for _, r in timing_df.iterrows()}
    timing_v2_index = {int(r["segment_id"]): r for r in timing_v2_rows}

    for seg_id, v0 in timing_v0_index.items():
        v2 = timing_v2_index.get(seg_id)
        if v2 is None:
            continue
        v0_alloc = str(v0.get("allocation_distribution", "?"))
        v2_alloc = str(v2.get("allocation_distribution", "?"))
        if v0_alloc == v2_alloc:
            continue

        primary_v0 = v0.get("primary_window", "")
        primary_v2 = v2.get("primary_window", "")
        ctr_v0 = timing_metrics.get((seg_id, primary_v0), {}).get("ctr", 0.0)
        ctr_v2 = timing_metrics.get((seg_id, primary_v2), {}).get("ctr", 0.0)

        traces.append({
            "trace_type":     "timing",
            "segment_id":     seg_id,
            "entity_id":      f"Segment_{seg_id}_timing",
            "change_summary": (
                f"Window allocation shifted from [{v0_alloc}] to [{v2_alloc}] "
                f"(primary: {primary_v0} -> {primary_v2})"
            ),
            "causal_reason": (
                f"EMA update (alpha={EMA_ALPHA}): '{primary_v2}' window "
                f"empirical CTR {ctr_v2:.1%} vs '{primary_v0}' {ctr_v0:.1%} "
                f"from Iteration 0 experiment."
            ),
        })

    # -- Content Traces (one per generated/processed experiment template) ------
    for row in generated_rows:
        tid    = str(row["template_id"])
        seg_id = int(row["segment_id"])
        action = str(row.get("_action", ""))
        exp    = exp_lookup.get(tid, {})
        ctr    = exp.get("ctr", 0.0)
        eng    = exp.get("engagement_rate", 0.0)
        goal   = row.get("goal_id", "")

        if action == "GOOD_KEPT":
            traces.append({
                "trace_type":     "content",
                "segment_id":     seg_id,
                "entity_id":      tid,
                "change_summary": "Template retained (GOOD performance, no change).",
                "causal_reason":  (
                    f"Experiment CTR {ctr:.1%} (>15%) and Engagement {eng:.1%} "
                    f"(>40%) — both thresholds met. Template content kept unchanged."
                ),
            })
        elif action == "NEUTRAL":
            traces.append({
                "trace_type":     "content",
                "segment_id":     seg_id,
                "entity_id":      tid,
                "change_summary": (
                    f"Template improved (NEUTRAL -> Iteration 1 variant) "
                    f"for goal '{goal}'."
                ),
                "causal_reason":  (
                    f"Experiment CTR {ctr:.1%} (5-15%) and Engagement {eng:.1%} "
                    f"(20-40%) were mid-range. LLM generated improved variant "
                    f"using original as reference, targeting CTR >15% and Eng >40%."
                ),
            })
        elif action == "BAD":
            traces.append({
                "trace_type":     "content",
                "segment_id":     seg_id,
                "entity_id":      tid,
                "change_summary": (
                    f"Template redesigned (BAD -> Iteration 1 fresh generation) "
                    f"for goal '{goal}'."
                ),
                "causal_reason":  (
                    f"Experiment CTR {ctr:.1%} (<5%) and Engagement {eng:.1%} "
                    f"(<20%) — both below thresholds. Original suppressed; "
                    f"LLM generated entirely new template (original shown as anti-pattern)."
                ),
            })

    timing_count  = sum(1 for t in traces if t["trace_type"] == "timing")
    content_count = sum(1 for t in traces if t["trace_type"] == "content")
    logger.info(
        "Delta report: %d timing traces + %d content traces = %d total.",
        timing_count, content_count, len(traces),
    )
    return traces

def _write_csv(rows: list[dict], fieldnames: list[str], output_path: str, label: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%s written → %s  (%d rows)", label, out, len(rows))


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_learning_engine(
    experiment_path: str,
    templates_path: str,
    timing_path: str,
    segments_path: str,
    goals_path: str,
    themes_path: str,
    feature_map_path: str,
    tone_matrix_path: str,
    out_dir: str,
) -> None:
    """
    Full Iteration 1 self-learning pipeline.

    Parts 3 → 4 → 5 → 6 → 7
    """
    # ── Input validation ──────────────────────────────────────────────────────
    required = [
        ("experiment_results.csv",           experiment_path),
        ("message_templates.csv",            templates_path),
        ("timing_recommendations.csv",       timing_path),
        ("user_segments.csv",                segments_path),
        ("segment_goals.csv",                goals_path),
        ("communication_themes.csv",         themes_path),
        ("feature_goal_map.json",            feature_map_path),
        ("allowed_tone_hook_matrix.json",    tone_matrix_path),
    ]
    for label, p in required:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found at {p}.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    exp_df       = pd.read_csv(experiment_path)
    templates_df = pd.read_csv(templates_path)
    timing_df    = pd.read_csv(timing_path)
    segments_df  = pd.read_csv(segments_path)
    goals_df     = pd.read_csv(goals_path)
    themes_df    = pd.read_csv(themes_path)
    tone_matrix  = json.loads(Path(tone_matrix_path).read_text(encoding="utf-8"))

    logger.info(
        "Loaded: %d experiment rows | %d templates | %d timing rows | %d users",
        len(exp_df), len(templates_df), len(timing_df), len(segments_df),
    )

    # ── PART 3: Feedback Ingestion & Evaluation ───────────────────────────────
    print("\n" + "─" * 68)
    print("  PART 3 — Feedback Ingestion & Evaluation")
    print("─" * 68)
    template_metrics, timing_metrics, segment_engagement = aggregate_metrics(exp_df)
    exp_lookup = build_exp_lookup(exp_df)

    good_count    = sum(1 for e in exp_lookup.values() if e["performance_status"] == "GOOD")
    neutral_count = sum(1 for e in exp_lookup.values() if e["performance_status"] == "NEUTRAL")
    bad_count     = sum(1 for e in exp_lookup.values() if e["performance_status"] == "BAD")
    print(f"  Experiment status  →  GOOD: {good_count}  NEUTRAL: {neutral_count}  BAD: {bad_count}")

    # ── PART 4: Adaptive LLM Generation → message_templates.csv ─────────────
    print("\n" + "─" * 68)
    print("  PART 4 — Adaptive Template Generation (GOOD=keep / NEUTRAL=improve / BAD=redesign)")
    print("─" * 68)
    all_templates_v2, generated_rows = generate_v2_library(
        templates_df, exp_lookup, tone_matrix, themes_df,
    )
    templates_v2_path = str(out / "message_templates.csv")
    _write_csv(all_templates_v2, TEMPLATES_V2_FIELDNAMES, templates_v2_path, "message_templates.csv")
    good_kept   = sum(1 for r in generated_rows if r.get("_action") == "GOOD_KEPT")
    neutral_gen = sum(1 for r in generated_rows if r.get("_action") == "NEUTRAL")
    bad_gen     = sum(1 for r in generated_rows if r.get("_action") == "BAD")
    print(f"  GOOD retained (no LLM)   : {good_kept}")
    print(f"  NEUTRAL improved via LLM : {neutral_gen}")
    print(f"  BAD redesigned via LLM   : {bad_gen}")
    print(f"  Iter-0 rows copied       : {len(templates_df)}")
    print(f"  Total V2 library         : {len(all_templates_v2)}")

    # ── PART 5: EMA Timing Update → timing_recommendations.csv ────────────
    print("\n" + "─" * 68)
    print("  PART 5 — EMA Timing Update (alpha=0.70)")
    print("─" * 68)
    timing_v2_rows    = compute_ema_timing(timing_df, timing_metrics, segment_engagement)
    timing_v2_path    = str(out / "timing_recommendations.csv")
    _write_csv(timing_v2_rows, TIMING_V2_FIELDNAMES, timing_v2_path, "timing_recommendations.csv")

    # ── PART 6: UCB1 Score Calculation → ucb_scores.csv ──────────────────────
    print("\n" + "─" * 68)
    print("  PART 6 — UCB1 Score Calculation")
    print("─" * 68)
    ucb_scores = compute_ucb_scores(all_templates_v2, exp_lookup)

    ucb_rows = []
    for t in all_templates_v2:
        tid    = str(t["template_id"])
        action = str(t.get("_action", ""))
        if action == "BAD":
            flag = "BAD"
        elif int(t.get("iteration", 0)) == 1:
            flag = "NEW"
        elif action == "GOOD_KEPT":
            flag = "GOOD"
        elif action == "NEUTRAL":
            flag = "NEUTRAL"
        else:
            flag = "V0"  # original iter-0 rows not in experiment
        ucb_rows.append({
            "template_id": tid,
            "ucb_score":   f"{ucb_scores.get(tid, 0.0):.6f}",
            "flag":        flag,
        })
    ucb_dir = Path("codebase/pipeline_cache")
    ucb_dir.mkdir(parents=True, exist_ok=True)
    ucb_path = str(ucb_dir / "ucb_scores.csv")
    _write_csv(ucb_rows, UCB_FIELDNAMES, ucb_path, "ucb_scores.csv")

    suppressed = sum(1 for r in ucb_rows if r["flag"] == "BAD")
    cold_start = sum(1 for r in ucb_rows if r["flag"] == "NEW")
    print(f"  Suppressed (BAD)  : {suppressed}")
    print(f"  Cold-start (NEW)  : {cold_start}")
    print(f"  Total scored      : {len(ucb_rows)}")

    # Call schedule_generator_v2.py to produce user_notification_schedule.csv
    try:
        from schedule_generator_v2 import run_schedule_generator_v2
        schedule_v2_path = str(out / "user_notification_schedule.csv")
        run_schedule_generator_v2(
            segments_path=segments_path,
            timing_path=timing_v2_path,
            templates_path=templates_v2_path,
            ucb_scores_path=ucb_path,
            output_path=schedule_v2_path,
        )
    except ImportError:
        logger.warning(
            "schedule_generator_v2.py not found on path — "
            "run it manually to produce user_notification_schedule.csv."
        )

    # ── PART 7: Delta Reporter → learning_delta_report.csv ───────────────────
    print("\n" + "─" * 68)
    print("  PART 7 — Delta Reporter (causal tracing)")
    print("─" * 68)
    delta_rows = generate_delta_report(
        timing_df, timing_v2_rows,
        generated_rows, exp_lookup, timing_metrics,
    )
    delta_path = "learning_delta_report.csv"
    _write_csv(delta_rows, DELTA_REPORT_FIELDNAMES, delta_path, "learning_delta_report.csv")

    # ── Copy user_segments.csv into iteration_1 folder ─────────────────────
    seg_v2_path = str(out / "user_segments.csv")
    import shutil
    shutil.copy2(segments_path, seg_v2_path)
    logger.info("user_segments.csv copied → %s", seg_v2_path)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  LEARNING ENGINE COMPLETE")
    print("=" * 68)
    print(f"  Output directory       : {out}")
    print()
    outputs = [
        ("message_templates.csv",        templates_v2_path),
        ("timing_recommendations.csv",   timing_v2_path),
        ("ucb_scores.csv",                  ucb_path),
        ("user_segments.csv",               seg_v2_path),
        ("learning_delta_report.csv",       delta_path),
    ]
    for label, p in outputs:
        exists = "✓" if Path(p).exists() else "✗"
        print(f"  {exists}  {label:<42} {p}")
    print("=" * 68 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – Learning Engine (Task 3)",
        epilog="All paths are read from codebase/config.yaml by default.",
    )
    p.add_argument(
        "--config", "-c",
        default="codebase/config.yaml",
        help="Path to config YAML (default: codebase/config.yaml)",
    )
    # Optional overrides — if not supplied, values come from the YAML
    p.add_argument("--experiment",  "-e", default=None)
    p.add_argument("--templates",   "-m", default=None)
    p.add_argument("--timing",      "-t", default=None)
    p.add_argument("--segments",    "-s", default=None)
    p.add_argument("--goals",       "-g", default=None)
    p.add_argument("--themes",            default=None)
    p.add_argument("--feature-map", "-f", default=None)
    p.add_argument("--tone-matrix",       default=None)
    p.add_argument("--out-dir",     "-o", default=None)
    return p.parse_args()


if __name__ == "__main__":
    import os

    args = _parse_args()

    # Resolve project root (one level up from codebase/)
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Load config YAML
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    with config_path.open(encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)

    _p = _cfg.get("paths", {})

    # CLI arg > YAML value > hardcoded fallback
    def _get(cli_val, yaml_key, fallback):
        return cli_val if cli_val is not None else _p.get(yaml_key, fallback)

    run_learning_engine(
        experiment_path  = _get(args.experiment,  "experiment_results",      "experiment_results.csv"),
        templates_path   = _get(args.templates,   "message_templates",       "iteration_0_before_learning/message_templates.csv"),
        timing_path      = _get(args.timing,       "timing_recommendations",  "iteration_0_before_learning/timing_recommendations.csv"),
        segments_path    = _get(args.segments,     "user_segments",           "iteration_0_before_learning/user_segments.csv"),
        goals_path       = _get(args.goals,        "segment_goals",           "iteration_0_before_learning/segment_goals.csv"),
        themes_path      = _get(args.themes,       "communication_themes",    "iteration_0_before_learning/communication_themes.csv"),
        feature_map_path = _get(args.feature_map,  "feature_map",             "iteration_0_before_learning/feature_goal_map.json"),
        tone_matrix_path = _get(args.tone_matrix,  "tone_matrix",             "iteration_0_before_learning/allowed_tone_hook_matrix.json"),
        out_dir          = _get(args.out_dir,      "iteration_1_dir",         "iteration_1_after_learning"),
    )
