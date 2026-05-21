"""
KB Ingestion  –  Project Aurora (SpeakX)
=========================================
Stage 1 of the Knowledge Bank pipeline.

Reads a concise KB markdown file and produces three JSON deliverables:
  1. company_north_star.json       – North Star Metric analysis
  2. allowed_tone_hook_matrix.json – Tone/Hook taxonomy (Octalysis-aligned)
  3. feature_goal_map.json         – Feature ↔ Business/User goal lookup table
                                     (Stage 2 segment strategies added by goal_builder.py)

This script is fully INDEPENDENT of the user ingestion pipeline.
It reads only the KB markdown file; no user CSV is required.

Usage
-----
    python kb_ingestion.py \
        --kb     data/input/knowledge_bank.md \
        --output data/output/
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.kb_ingestion")

# ── Ollama client (OpenAI-compatible, no rate limits) ─────────────────────────
from llm_client import client, OLLAMA_MODEL

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (verbatim from nsm_&_allowed_tone.md)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior product strategist, growth scientist, and behavioral economist.

The Knowledge Bank is small (1–2 pages), but you must extract maximum structured intelligence.

CRITICAL RULES:
- Never return empty arrays.
- Never return empty strings.
- Never give shallow answers.
- Each explanation must be at least 2–3 sentences.
- Every mapping must include reasoning.
- Infer implicit goals and mechanisms.
- Return ONLY valid JSON.
- Do not include commentary outside JSON.

DEFINITIONS:

1. Product Feature:
A user-facing capability that directly changes user behavior or engagement loop.
Ignore pricing or metrics as features.

2. User Goal:
Transformation the user is trying to achieve (skill, identity, discipline, confidence, mastery).

3. Business Outcome:
Metric-level outcome such as:
- Conversion uplift
- Retention improvement
- Increased session frequency
- Reduced churn probability

4. Lifecycle Stages:
- Trial (Day 0–7)
- Paid (Day 8–30)
- Retention (Day 31+)
- Churned/Inactive

5. North Star Metric:
Must:
- Represent repeated value realization
- Correlate strongly with retention
- Capture user transformation
- Include measurable proxy with explicit logical formula
- Include quantitative threshold when possible

6. Octalysis Core Drives:
Use mainly these 8:
1. Epic Meaning
2. Accomplishment
3. Empowerment
4. Ownership
5. Social Influence
6. Scarcity
7. Unpredictability
8. Loss Avoidance

For every feature:
- Identify primary drive
- Identify secondary drive if applicable
- Explain behavioral mechanism clearly

DEPTH REQUIREMENTS:

North Star must include:
- Strategic reasoning
- Why this metric over others
- Leading vs lagging explanation

Feature Map must include:
- Why lifecycle mapping makes sense
- Behavioral trigger explanation
- Expected measurable business impact 

Tone Mapping must include:
- Psychological effect
- Risk mitigation reasoning
- Alignment with ethical guidelines

Think step-by-step internally before responding.
Return final JSON only."""

# Product Architect system prompt for feature-goal map (Stage 1)
PRODUCT_ARCHITECT_PROMPT = """**ROLE:**
You are the Lead System Architect for 'SpeakX'. Your sole responsibility is to parse unstructured product documentation into a structured 'Feature-Goal Map'.

**TASK:**
Analyze the specific product features mentioned in the Knowledge Bank and output a strictly formatted JSON object.

**REQUIREMENTS:**
1. **Feature Extraction:** Identify every distinct feature (e.g., AI Tutor, Streaks, Leaderboard, Wallet).
2. **Goal Mapping:** For each feature, strictly define:
   - **Business Goal:** The specific macro outcome this feature drives (e.g., Activation, Retention, Monetization).
   - **User Utility:** The tangible value the user receives (e.g., "Track progress", "Practice speaking"). Do not exceed more than 4.
3. **Feature IDs:** Use snake_case for all feature_id keys (e.g., "ai_tutor", "streak_system").

**OUTPUT SCHEMA (JSON ONLY):**
{
  "feature_map": {
    "feature_id_1": {
      "name": "String",
      "business_goal": "String",
      "user_utility": ["String", "String", ...]
    }
  }
}

Return ONLY valid JSON. No commentary outside JSON."""


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_markdown_sections(text: str) -> dict[str, str]:
    """
    Split KB markdown on '## ' headings.
    Returns {heading_lower_snake: body_text}.
    """
    sections: dict[str, str] = {}
    # Split on lines starting with "## "
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    # parts = [pre_text, heading1, body1, heading2, body2, ...]
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lower().replace(" ", "_").replace("&", "and")
        body    = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[heading] = body
    return sections


def extract_json_safely(raw: str) -> dict:
    """
    Strip markdown fences, find first '{', parse JSON.
    Raises ValueError on failure.
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    # Find first opening brace
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM output:\n{raw[:500]}")
    json_str = cleaned[start:]
    return json.loads(json_str)


def call_llm(system_prompt: str, user_prompt: str, retries: int = 5) -> dict:
    """
    Call Gemini LLM with retry + exponential backoff.
    Returns parsed JSON dict.
    """
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
            logger.error("Unexpected LLM error on attempt %d: %s", attempt, e)

        if attempt < retries:
            time.sleep(3)

    raise RuntimeError(f"LLM call failed after {retries} attempts.")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1a: North Star Metric
# ══════════════════════════════════════════════════════════════════════════════

def generate_north_star(sections: dict[str, str]) -> dict:
    """
    Generate North Star Metric analysis from KB sections.
    Outputs: company_north_star.json
    """
    # Pull relevant sections for context
    context_keys = [k for k in sections if any(
        kw in k for kw in ["vision", "north_star", "nsm", "success", "lifecycle", "mission", "overview"]
    )]
    context = "\n\n".join(
        f"### {k.replace('_', ' ').title()}\n{sections[k]}"
        for k in context_keys
    ) or "\n\n".join(f"### {k}\n{v}" for k, v in list(sections.items())[:4])

    user_prompt = f"""Based on the following Knowledge Bank sections, infer and analyze the North Star Metric for this product.

{context}

Return a JSON object with this exact schema:
{{
  "north_star_metric": {{
    "name": "String — metric name",
    "definition": "String — 2-3 sentence definition of what this metric measures",
    "strategic_reasoning": "String — why this metric over revenue, DAU, or other common choices (2-3 sentences)",
    "leading_vs_lagging": {{
      "classification": "leading | lagging",
      "explanation": "String — 2-3 sentences explaining why"
    }},
    "measurable_proxy": {{
      "variable": "String — column name or observable signal",
      "logic": "String — formula or scoring logic",
      "threshold": "String — quantitative threshold (e.g. ≥3 sessions/week for 4 weeks)",
      "window": "String — observation window (e.g. rolling 30 days)"
    }},
    "business_impact": "String — how improving this metric translates to revenue / retention / growth (2-3 sentences)"
  }}
}}"""

    logger.info("Generating North Star Metric…")
    result = call_llm(SYSTEM_PROMPT, user_prompt)
    logger.info("North Star Metric generated ✓")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1b: Tone & Hook Matrix
# ══════════════════════════════════════════════════════════════════════════════

def generate_tone_hook_matrix(sections: dict[str, str]) -> dict:
    """
    Generate allowed/disallowed tones and hook taxonomy.
    Outputs: allowed_tone_hook_matrix.json
    """
    context_keys = [k for k in sections if any(
        kw in k for kw in ["ethical", "tone", "feature", "hook", "guideline", "communication"]
    )]
    context = "\n\n".join(
        f"### {k.replace('_', ' ').title()}\n{sections[k]}"
        for k in context_keys
    ) or "\n\n".join(f"### {k}\n{v}" for k, v in list(sections.items())[:5])

    user_prompt = f"""Based on the following Knowledge Bank sections, generate a comprehensive Tone & Hook Matrix.

{context}

Return a JSON object with this exact schema:
{{
  "allowed_tones": [
    {{
      "tone": "String — tone name (e.g. Encouraging, Celebratory)",
      "psychological_effect": "String — 2-3 sentences on how this tone affects user psychology",
      "use_case": "String — when/where to apply this tone",
      "example": "String — example notification copy using this tone"
    }}
  ],
  "disallowed_tones": [
    {{
      "tone": "String — tone name (e.g. Guilt-Tripping, Alarming)",
      "risk": "String — 2-3 sentences explaining psychological harm and ethical concern",
      "alternative": "String — what allowed tone to use instead"
    }}
  ],
  "hook_taxonomy": [
    {{
      "feature": "String — product feature name",
      "primary_octalysis_drive": "Integer 1-8",
      "secondary_octalysis_drive": "Integer 1-8 or null",
      "drive_names": {{
        "primary": "String — drive name",
        "secondary": "String or null"
      }},
      "behavioral_mechanism": "String — 2-3 sentences explaining the psychological loop",
      "example_hook_copy": "String — sample notification or in-app message",
      "ethical_alignment": "String — why this hook is ethical and user-respecting"
    }}
  ]
}}"""

    logger.info("Generating Tone & Hook Matrix…")
    result = call_llm(SYSTEM_PROMPT, user_prompt)
    logger.info("Tone & Hook Matrix generated ✓")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1c: Feature-Goal Map (Product Architect — Stage 1 only)
# ══════════════════════════════════════════════════════════════════════════════

def generate_feature_map(kb_text: str) -> dict:
    """
    Generate Feature ↔ Business/User goal lookup table.
    Stage 2 (segment strategies) is added later by goal_builder.py.
    Outputs: feature_goal_map.json (Stage 1 only)
    """
    user_prompt = f"""Here is the full Knowledge Bank:

---
{kb_text}
---

Parse all product features and output the Feature-Goal Map JSON as specified."""

    logger.info("Generating Feature-Goal Map (Stage 1)…")
    result = call_llm(PRODUCT_ARCHITECT_PROMPT, user_prompt)
    logger.info("Feature-Goal Map generated ✓ | features: %d",
                len(result.get("feature_map", {})))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_kb_ingestion(kb_path: str, output_dir: str) -> None:
    """
    Full KB ingestion pipeline.
    Reads KB markdown → generates 3 JSON deliverables → writes to output_dir.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load KB ───────────────────────────────────────────────────────────────
    kb_text = Path(kb_path).read_text(encoding="utf-8")
    logger.info("Loaded KB: %s (%d chars)", kb_path, len(kb_text))

    sections = parse_markdown_sections(kb_text)
    logger.info("Parsed %d KB sections: %s", len(sections), list(sections.keys()))

    # ── 1. North Star Metric ──────────────────────────────────────────────────
    north_star = generate_north_star(sections)
    ns_path = out / "company_north_star.json"
    ns_path.write_text(json.dumps(north_star, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved → %s", ns_path)

    # ── 2. Tone & Hook Matrix ─────────────────────────────────────────────────
    tone_matrix = generate_tone_hook_matrix(sections)
    tm_path = out / "allowed_tone_hook_matrix.json"
    tm_path.write_text(json.dumps(tone_matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved → %s", tm_path)

    # ── 3. Feature-Goal Map (Stage 1) ─────────────────────────────────────────
    feature_map = generate_feature_map(kb_text)
    fm_path = out / "feature_goal_map.json"
    fm_path.write_text(json.dumps(feature_map, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved → %s", fm_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KB INGESTION COMPLETE")
    print("=" * 60)
    print(f"  KB file   : {kb_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  ✓ company_north_star.json")
    print(f"    → NSM: {north_star.get('north_star_metric', {}).get('name', 'N/A')}")
    print(f"  ✓ allowed_tone_hook_matrix.json")
    print(f"    → Allowed tones : {len(tone_matrix.get('allowed_tones', []))}")
    print(f"    → Hook entries  : {len(tone_matrix.get('hook_taxonomy', []))}")
    print(f"  ✓ feature_goal_map.json  (Stage 1)")
    print(f"    → Features mapped: {len(feature_map.get('feature_map', {}))}")
    print(f"    → Run goal_builder.py to add segment strategies (Stage 2)")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – KB Ingestion Pipeline (Task 1, Stage 1)"
    )
    p.add_argument(
        "--kb", "-k",
        default="data/input/knowledge_bank.md",
        help="Path to Knowledge Bank markdown file (default: data/input/knowledge_bank.md)",
    )
    p.add_argument(
        "--output", "-o",
        default="data/output/",
        help="Output directory for JSON deliverables (default: data/output/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_kb_ingestion(args.kb, args.output)
