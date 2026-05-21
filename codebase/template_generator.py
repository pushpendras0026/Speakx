"""
Template Generator  –  Project Aurora (SpeakX)
===============================================
Task 2A, Part 2 — LLM-based bilingual push notification template generation.

Reads:
  - iteration_0_before_learning/segment_goals.csv       (Task 1 output)
  - iteration_0_before_learning/communication_themes.csv (Task 2A Part 1 output)
  - iteration_0_before_learning/feature_goal_map.json   (Task 1 output)
  - iteration_0_before_learning/allowed_tone_hook_matrix.json (Task 1 output)

For each row in segment_goals.csv (segment × day_label), calls the LLM
(Expert Behavioral Copywriter role) to generate 5 message template variants
in bilingual format (English + Hindi/Hinglish, max 15 words each):
  T1 — Direct Benefit       T4 — Social Proof / FOMO
  T2 — Emotional Hook       T5 — Short & Punchy
  T3 — Curiosity / Question

Outputs:
  - iteration_0_before_learning/message_templates.csv

Usage
-----
    python template_generator.py \
        --goals       iteration_0_before_learning/segment_goals.csv \
        --themes      iteration_0_before_learning/communication_themes.csv \
        --feature-map iteration_0_before_learning/feature_goal_map.json \
        --tone-matrix iteration_0_before_learning/allowed_tone_hook_matrix.json \
        --output      iteration_0_before_learning/message_templates.csv
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
logger = logging.getLogger("aurora.template_generator")

# ── Ollama client (OpenAI-compatible, no rate limits) ─────────────────────────
from llm_client import client, OLLAMA_MODEL

# ── Template variant metadata ──────────────────────────────────────────────────
VARIANT_METADATA = [
    ("T1", "Direct_Benefit"),
    ("T2", "Emotional_Hook"),
    ("T3", "Curiosity_Question"),
    ("T4", "Social_Proof_FOMO"),
    ("T5", "Short_Punchy"),
]

TEMPLATES_CSV_FIELDNAMES = [
    "template_id",
    "segment_id",
    "lifecycle_stage",
    "goal_id",
    "theme",
    "message_title_en",
    "message_title_hi",
    "message_body_en",
    "message_body_hi",
    "cta_text_en",
    "cta_text_hi",
    "tone_used",
    "hook_type",
    "feature_reference",  # snake_case app feature identifier (e.g. "streak_counter")
]


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt  (Expert Behavioral Copywriter)
# ══════════════════════════════════════════════════════════════════════════════

COPYWRITER_SYSTEM = """ROLE: You are an Expert Behavioral Copywriter and Linguist specialising in EdTech push notifications and in-app messages for Indian users. You deeply understand Octalysis gamification psychology and how each Core Drive motivates different behavioral responses.

CRITICAL RULES:
- Return ONLY valid JSON matching the exact output schema. No commentary outside JSON.
- Never return empty strings. Produce exactly 5 variants — never fewer.
- Each variant MUST include structured bilingual content with 3 parts: title, body, CTA.
- English fields (message_title_en, message_body_en, cta_text_en): natural modern English.
  - message_title_en: max 8 words, attention-grabbing headline.
  - message_body_en: max 15 words, action-oriented supporting text that references the user goal or focus feature.
  - cta_text_en: max 5 words, clear call-to-action button text (e.g. "Start Practicing", "Claim Reward").
- Hindi/Hinglish fields (message_title_hi, message_body_hi, cta_text_hi): natural Hinglish (mix of Hindi and English as Indians actually speak daily, NOT a literal translation). Example style: "Kal ka streak mat todna — abhi practice karo!"
  - message_title_hi: max 8 words.
  - message_body_hi: max 15 words.
  - cta_text_hi: max 5 words (e.g. "Abhi Shuru Karo", "Reward Claim Karo").
- tone_used MUST be EXACTLY one of the tone names listed in the ALLOWED TONES section below. Do NOT use variant names (T1, Direct_Benefit, etc.) here.
- hook_type MUST be EXACTLY one of the snake_case feature names listed in the ALLOWED HOOKS section below (e.g. "leaderboards_social", "streak_system"). Do NOT use variant names (T1, Direct_Benefit, Short_Punchy, etc.) here — those are template variant types, not hooks.
- Each of the 5 variants MUST use a different combination of tone and hook.
- feature_reference: a short snake_case identifier for the app feature this message highlights (e.g. "streak_system", "leaderboards_social", "ai_tutor_sia", "coins_economy", "progress_analytics")."""


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions  (same pattern as goal_builder.py)
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
                temperature=0.7,    # higher than goal_builder for creative diversity
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


# ══════════════════════════════════════════════════════════════════════════════
# Template ID generation
# ══════════════════════════════════════════════════════════════════════════════

def build_template_id(
    segment_name: str,
    day_label: str,
    primary_theme: str,
    variant_id: str,
) -> str:
    """
    Build a unique, file-safe template ID.
    Format: {segment_name}_{day_label_underscored}_{primary_theme}_{variant_id}
    Example: Trial_Gamer_Day_1_Loss_and_Avoidance_T1
    """
    safe_day = re.sub(r"[^A-Za-z0-9]+", "_", day_label).strip("_")
    safe_seg = re.sub(r"[^A-Za-z0-9_]+", "_", segment_name).strip("_")
    return f"{safe_seg}_{safe_day}_{primary_theme}_{variant_id}"


# ══════════════════════════════════════════════════════════════════════════════
# User prompt construction
# ══════════════════════════════════════════════════════════════════════════════

def build_user_prompt(
    goal_row: dict,
    theme_row: dict,
    feature_map: dict,
    tone_matrix: dict,
) -> str:
    """Build the structured user prompt for one segment × day_label goal step."""
    seg_id        = goal_row.get("segment_id", "?")
    seg_name      = goal_row.get("segment_name", f"Segment_{seg_id}")
    lifecycle     = goal_row.get("lifecycle_stage", "")
    day_label     = goal_row.get("day_label", "")
    primary_goal  = goal_row.get("primary_goal", "")
    sub_goals     = goal_row.get("sub_goals", "")
    focus_feat    = goal_row.get("focus_feature_id", "")
    user_task     = goal_row.get("user_task", "")
    rationale     = goal_row.get("rationale", "")

    primary_theme   = theme_row.get("primary_theme",   "")
    secondary_theme = theme_row.get("secondary_theme", "")
    tertiary_theme  = theme_row.get("tertiary_theme",  "")

    # Pass the most relevant CD scores to help the LLM calibrate intensity
    cd_summary = (
        f"CD1(Epic_Meaning)={theme_row.get('cd1_score', '?')}, "
        f"CD2(Accomplishment)={theme_row.get('cd2_score', '?')}, "
        f"CD5(Social_Influence)={theme_row.get('cd5_score', '?')}, "
        f"CD8(Loss_and_Avoidance)={theme_row.get('cd8_score', '?')}"
    )

    feature_detail = feature_map.get(focus_feat, {})
    feature_str    = json.dumps(feature_detail, indent=2) if feature_detail else f'"{focus_feat}" (no detail available)'

    # ── Build focused tone / hook context from pre-computed theme data ──────────
    # tone_preferences and hooks were calculated by theme_engine.py specifically
    # for this segment's Octalysis drives — use them directly instead of dumping
    # the full matrix, which overwhelms the LLM and causes it to copy variant
    # names (T1, Direct_Benefit) into hook_type.
    _raw_tones = theme_row.get("tone_preferences", "") or ""
    tone_names = [t.strip() for t in str(_raw_tones).split("|") if t.strip()]
    _raw_hooks = theme_row.get("hooks", "") or ""
    hook_names = [h.strip() for h in str(_raw_hooks).split("|") if h.strip()]

    # Build a lookup: snake_case feature name → example_hook_copy (for LLM inspiration)
    hook_example_lookup: dict[str, str] = {}
    for entry in tone_matrix.get("hook_taxonomy", []):
        feature_label = entry.get("feature", "")
        if feature_label:
            snake = re.sub(r"[^A-Za-z0-9]+", "_", feature_label).strip("_").lower()
            hook_example_lookup[snake] = entry.get("example_hook_copy", "")

    disallowed_tones = [
        e.get("tone", "") for e in tone_matrix.get("disallowed_tones", []) if e.get("tone")
    ]

    allowed_tones_block = "ALLOWED TONES (tone_used must be exactly one of these):\n" + \
        "\n".join(f"  - {t}" for t in tone_names)

    allowed_hooks_block = "ALLOWED HOOKS (hook_type must be exactly one of these snake_case names):\n"
    for h in hook_names:
        example = hook_example_lookup.get(h, "")
        allowed_hooks_block += f"  - {h}" + (f'  →  inspiration example: "{example}"' if example else "") + "\n"

    disallowed_block = "DISALLOWED TONES (never use):\n" + \
        "\n".join(f"  - {t}" for t in disallowed_tones)

    return f"""**SEGMENT CONTEXT**
- Segment ID       : {seg_id}
- Segment Name     : {seg_name}
- Lifecycle Stage  : {lifecycle}
- Octalysis Themes : Primary={primary_theme}, Secondary={secondary_theme}, Tertiary={tertiary_theme}
- CD Scores (selected top 3 from 8 drives): {cd_summary}

**GOAL CONTEXT (Journey Step)**
- Day/Week Label   : {day_label}
- Primary Goal     : {primary_goal}
- Sub-Goals        : {sub_goals}
- Focus Feature    : {focus_feat}
- User Task        : {user_task}
- Strategic Rationale: {rationale}

**FEATURE DETAIL**
{feature_str}

**COMMUNICATION CONSTRAINTS**
{allowed_tones_block}

{allowed_hooks_block}
{disallowed_block}

**TASK**
Generate exactly 5 push notification / in-app message variants for users in segment "{seg_name}" at journey step "{day_label}".

Requirements per variant:
- Activate the PRIMARY theme ({primary_theme}) through the message's emotional angle
- Address the focus feature ({focus_feat}) and the specific user task
- Each variant uses a DISTINCT tone/hook combination from the matrix above
- T1 = Direct Benefit (state the gain clearly)
- T2 = Emotional Hook (connect to a feeling or aspiration)
- T3 = Curiosity / Question (make the user wonder or want to find out)
- T4 = Social Proof / FOMO (reference others, competition, or urgency)
- T5 = Short & Punchy (max 8 words each language, maximum impact)

**OUTPUT SCHEMA (JSON ONLY)**
{{
  "templates": [
    {{
      "variant_id": "T1",
      "variant_name": "Direct_Benefit",
      "message_title_en": "...",
      "message_title_hi": "...",
      "message_body_en": "...",
      "message_body_hi": "...",
      "cta_text_en": "...",
      "cta_text_hi": "...",
      "tone_used": "...",
      "hook_type": "...",
      "feature_reference": "..."
    }}
  ]
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# Generate templates for one goal row
# ══════════════════════════════════════════════════════════════════════════════

def generate_templates_for_row(
    goal_row: dict,
    theme_row: dict,
    feature_map: dict,
    tone_matrix: dict,
) -> list[dict]:
    """
    Call LLM for one segment × day_label step and return 5 flattened output rows.
    """
    user_prompt   = build_user_prompt(goal_row, theme_row, feature_map, tone_matrix)
    result        = call_llm(COPYWRITER_SYSTEM, user_prompt)
    templates     = result.get("templates", [])

    seg_name      = goal_row.get("segment_name", f"Segment_{goal_row.get('segment_id', '')}")
    day_label     = goal_row.get("day_label", "")
    primary_theme = theme_row.get("primary_theme", "")

    rows: list[dict] = []
    for item in templates:
        variant_id = item.get("variant_id", "T?")
        rows.append({
            "template_id":      build_template_id(seg_name, day_label, primary_theme, variant_id),
            "segment_id":       goal_row.get("segment_id", ""),
            "lifecycle_stage":  goal_row.get("lifecycle_stage", ""),
            "goal_id":          day_label,
            "theme":            primary_theme,
            "message_title_en":  item.get("message_title_en", ""),
            "message_title_hi":  item.get("message_title_hi", ""),
            "message_body_en":   item.get("message_body_en", ""),
            "message_body_hi":   item.get("message_body_hi", ""),
            "cta_text_en":       item.get("cta_text_en", ""),
            "cta_text_hi":       item.get("cta_text_hi", ""),
            "tone_used":        item.get("tone_used", ""),
            "hook_type":        item.get("hook_type", ""),
            "feature_reference": item.get("feature_reference", ""),
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Write message_templates.csv
# ══════════════════════════════════════════════════════════════════════════════

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


def write_templates(rows: list[dict], output_path: str) -> None:
    """Write message_templates.csv with csv.QUOTE_ALL."""
    _atomic_write_csv(rows, TEMPLATES_CSV_FIELDNAMES, Path(output_path), "message_templates.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_template_generator(
    goals_path: str,
    themes_path: str,
    feature_map_path: str,
    tone_matrix_path: str,
    output_path: str,
) -> None:
    """
    Full Template Generator pipeline.
    1. Load all inputs
    2. Join themes into goals on segment_id
    3. For each row, call LLM → 5 templates
    4. Write message_templates.csv
    """
    for label, path in [
        ("segment_goals.csv",            goals_path),
        ("communication_themes.csv",     themes_path),
        ("feature_goal_map.json",        feature_map_path),
        ("allowed_tone_hook_matrix.json", tone_matrix_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found at {path}.")

    goals_df  = pd.read_csv(goals_path)
    themes_df = pd.read_csv(themes_path)

    raw_feature_map = json.loads(Path(feature_map_path).read_text(encoding="utf-8"))
    # Support both {"feature_map": {...}} envelope and flat dict
    feature_map = raw_feature_map.get("feature_map", raw_feature_map)

    tone_matrix = json.loads(Path(tone_matrix_path).read_text(encoding="utf-8"))

    # Join themes (one row per segment) into goals (one row per segment × day_label)
    context_df = goals_df.merge(themes_df, on="segment_id", how="left")

    logger.info(
        "Loaded: %d goal rows × 5 variants = %d expected templates",
        len(context_df), len(context_df) * 5,
    )

    all_rows: list[dict] = []
    total = len(context_df)

    for idx, row in context_df.iterrows():
        goal_row  = row.to_dict()
        theme_row = themes_df[themes_df["segment_id"] == goal_row["segment_id"]].iloc[0].to_dict() \
                    if not themes_df[themes_df["segment_id"] == goal_row["segment_id"]].empty \
                    else {}

        seg_name  = goal_row.get("segment_name", f"Segment_{goal_row.get('segment_id', '')}")
        day_label = goal_row.get("day_label", "")

        logger.info(
            "Generating templates %d/%d — %s | %s …",
            idx + 1, total, seg_name, day_label,
        )

        try:
            rows = generate_templates_for_row(goal_row, theme_row, feature_map, tone_matrix)
        except RuntimeError as e:
            logger.error("Row %d failed: %s — skipping.", idx + 1, e)
            continue

        all_rows.extend(rows)

    if not all_rows:
        logger.error("No templates generated — check LLM errors above.")
        return

    write_templates(all_rows, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    seg_counts: dict = {}
    for r in all_rows:
        seg_id = r["segment_id"]
        seg_counts[seg_id] = seg_counts.get(seg_id, 0) + 1

    print("\n" + "=" * 70)
    print("  TEMPLATE GENERATOR COMPLETE")
    print("=" * 70)
    print(f"  Goal rows processed : {total}")
    print(f"  Total rows written  : {len(all_rows)}")
    print(f"  Output              : {output_path}")
    print()
    print(f"  {'Seg ID':<8} {'Templates'}")
    print(f"  {'------':<8} {'---------'}")
    for seg_id in sorted(seg_counts):
        print(f"  {seg_id:<8} {seg_counts[seg_id]}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – Template Generator (Task 2A, Part 2)"
    )
    p.add_argument(
        "--goals", "-g",
        default="iteration_0_before_learning/segment_goals.csv",
        help="Path to segment_goals.csv from goal_builder.py",
    )
    p.add_argument(
        "--themes", "-t",
        default="iteration_0_before_learning/communication_themes.csv",
        help="Path to communication_themes.csv from theme_engine.py",
    )
    p.add_argument(
        "--feature-map", "-f",
        default="iteration_0_before_learning/feature_goal_map.json",
        help="Path to feature_goal_map.json from kb_ingestion.py",
    )
    p.add_argument(
        "--tone-matrix", "-m",
        default="iteration_0_before_learning/allowed_tone_hook_matrix.json",
        help="Path to allowed_tone_hook_matrix.json from kb_ingestion.py",
    )
    p.add_argument(
        "--output", "-o",
        default="iteration_0_before_learning/message_templates.csv",
        help="Output path for message_templates.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_template_generator(
        goals_path=args.goals,
        themes_path=args.themes,
        feature_map_path=args.feature_map,
        tone_matrix_path=args.tone_matrix,
        output_path=args.output,
    )
