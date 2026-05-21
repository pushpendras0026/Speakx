"""
Generate a synthetic experiment_results.csv for demo/testing.
Reads message_templates.csv and writes a valid experiment_results.csv
with CTR/engagement ranges aligned to GOOD/NEUTRAL/BAD rules.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd
import yaml


WINDOWS = ["morning", "mid_morning", "afternoon", "evening", "night"]


def _sample_status(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.70:
        return "GOOD"
    if r < 0.90:
        return "NEUTRAL"
    return "BAD"


def _ranges_for(status: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if status == "GOOD":
        return (0.16, 0.30), (0.41, 0.70)
    if status == "NEUTRAL":
        return (0.05, 0.15), (0.20, 0.40)
    return (0.00, 0.049), (0.00, 0.19)


def _pick_in_range(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def _load_config(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: Config not found at {path}")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic experiment_results.csv")
    p.add_argument("--config", "-c", default="codebase/config.yaml")
    p.add_argument("--out", "-o", default=None, help="Override output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    cfg = _load_config(project_root / args.config)

    paths = cfg.get("paths", {})
    templates_path = project_root / paths.get(
        "message_templates", "iteration_0_before_learning/message_templates.csv"
    )
    output_path = project_root / (
        args.out or paths.get("experiment_results", "experiment_results.csv")
    )

    if not templates_path.exists():
        print(f"ERROR: message_templates.csv not found at {templates_path}")
        sys.exit(1)

    df = pd.read_csv(templates_path)
    rng = random.Random(args.seed)

    rows = []
    for _, t in df.iterrows():
        status = _sample_status(rng)
        ctr_range, eng_range = _ranges_for(status)

        sends = rng.randint(60, 400)
        ctr = _pick_in_range(rng, *ctr_range)
        eng = _pick_in_range(rng, *eng_range)

        opens = max(0, int(round(ctr * sends)))
        engagements = max(0, int(round(eng * sends)))
        uninstall_rate = round(rng.uniform(0.0, 0.02), 4)

        rows.append({
            "template_id": str(t.get("template_id", "")),
            "segment_id": int(t.get("segment_id", 0)),
            "lifecycle_stage": str(t.get("lifecycle_stage", "")),
            "goal": str(t.get("goal_id", "")),
            "theme": str(t.get("theme", "")),
            "notification_window": rng.choice(WINDOWS),
            "total_sends": sends,
            "total_opens": opens,
            "total_engagements": engagements,
            "ctr": ctr,
            "engagement_rate": eng,
            "uninstall_rate": uninstall_rate,
            "performance_status": status,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    status_counts = {"GOOD": 0, "NEUTRAL": 0, "BAD": 0}
    for r in rows:
        status_counts[r["performance_status"]] += 1

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
