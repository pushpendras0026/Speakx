"""
Pipeline Runner  –  Project Aurora (SpeakX)
============================================
Single entry point for the full pipeline.

Reads config.yaml from the project root (or a path you specify with --config)
and runs all stages in sequence without ever typing flags again:

    python codebase/pipeline.py                           # run all enabled stages
    python codebase/pipeline.py --stages ingest,segment   # run subset (overrides yaml)
    python codebase/pipeline.py --skip kb,goals           # skip specific stages
    python codebase/pipeline.py --config my_config.yaml   # use alternate config

Stage order:
    ingest  →  kb  →  segment  →  goals  →  theme  →  template
                                         ↘
                                           timing   (independent of theme/template)

Each stage calls the existing module's entry function directly (no subprocess).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.pipeline")

# ── All valid stage names in execution order ───────────────────────────────────
STAGE_ORDER = ["ingest", "kb", "segment", "goals", "theme", "template", "timing", "schedule", "learn"]


# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load and return config.yaml as a dict."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is not installed.  Run:  pip install pyyaml")
        sys.exit(1)

    p = Path(config_path)
    if not p.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Config loaded from %s", p)
    return cfg


def _p(cfg: dict, key: str) -> str:
    """Shorthand: get a path string from cfg['paths'][key]."""
    return cfg["paths"][key]


# ══════════════════════════════════════════════════════════════════════════════
# Stage resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_stages(
    cfg: dict,
    cli_stages: list[str] | None,
    cli_skip: list[str] | None,
) -> dict[str, bool]:
    """
    Determine which stages to run.

    Priority:
    1. If --stages is given, run ONLY those stages (ignore yaml toggles).
    2. If --skip  is given, start from yaml toggles and disable those stages.
    3. Otherwise, use yaml toggles as-is.
    """
    yaml_toggles = cfg.get("stages", {})
    # Start with yaml defaults (unknown stages default to True)
    active = {s: bool(yaml_toggles.get(s, True)) for s in STAGE_ORDER}

    if cli_stages:
        # --stages overrides everything: only run the listed stages
        for s in STAGE_ORDER:
            active[s] = s in cli_stages
    elif cli_skip:
        # --skip disables specific stages, rest stay as yaml says
        for s in cli_skip:
            if s in active:
                active[s] = False
            else:
                print(f"WARNING: Unknown stage in --skip: '{s}'. "
                      f"Valid stages: {', '.join(STAGE_ORDER)}")

    return active


# ══════════════════════════════════════════════════════════════════════════════
# Dependency checks
# ══════════════════════════════════════════════════════════════════════════════

def _check_dependencies(cfg: dict, stages: dict[str, bool]) -> None:
    """
    Warn (and auto-enable) if a stage needs an input that doesn't exist
    and the stage that produces it is disabled.
    """
    profiles_path = Path(_p(cfg, "user_profiles"))
    summary_path  = Path(_p(cfg, "segment_summary"))
    fmap_path     = Path(_p(cfg, "feature_map"))

    # segment needs user_profiles.csv
    if stages.get("segment") and not stages.get("ingest") and not profiles_path.exists():
        logger.warning(
            "'segment' is enabled but user_profiles.csv not found and 'ingest' is disabled. "
            "Auto-enabling 'ingest'."
        )
        stages["ingest"] = True

    # goals needs segment_summary.csv
    if stages.get("goals") and not stages.get("segment") and not summary_path.exists():
        logger.warning(
            "'goals' is enabled but segment_summary.csv not found and 'segment' is disabled. "
            "Auto-enabling 'segment'."
        )
        stages["segment"] = True
        # segment itself needs profiles
        if not profiles_path.exists() and not stages.get("ingest"):
            logger.warning("Auto-enabling 'ingest' (needed by 'segment').")
            stages["ingest"] = True

    # goals needs feature_goal_map.json
    if stages.get("goals") and not stages.get("kb") and not fmap_path.exists():
        logger.warning(
            "'goals' is enabled but feature_goal_map.json not found and 'kb' is disabled. "
            "Auto-enabling 'kb'."
        )
        stages["kb"] = True

    segments_path = Path(_p(cfg, "user_segments"))
    goals_path    = Path(_p(cfg, "segment_goals"))
    themes_path   = Path(_p(cfg, "communication_themes"))

    # theme needs user_segments.csv
    if stages.get("theme") and not stages.get("segment") and not segments_path.exists():
        logger.warning(
            "'theme' needs user_segments.csv — auto-enabling 'segment'."
        )
        stages["segment"] = True
        if not profiles_path.exists() and not stages.get("ingest"):
            logger.warning("Auto-enabling 'ingest' (needed by 'segment').")
            stages["ingest"] = True

    # template needs segment_goals.csv + communication_themes.csv + kb outputs
    if stages.get("template"):
        if not stages.get("goals") and not goals_path.exists():
            logger.warning(
                "'template' needs segment_goals.csv — auto-enabling 'goals'."
            )
            stages["goals"] = True
        if not stages.get("theme") and not themes_path.exists():
            logger.warning(
                "'template' needs communication_themes.csv — auto-enabling 'theme'."
            )
            stages["theme"] = True
        if not stages.get("kb") and not fmap_path.exists():
            logger.warning(
                "'template' needs feature_goal_map.json — auto-enabling 'kb'."
            )
            stages["kb"] = True
        tone_path = Path(_p(cfg, "tone_matrix"))
        if not stages.get("kb") and not tone_path.exists():
            logger.warning(
                "'template' needs allowed_tone_hook_matrix.json — auto-enabling 'kb'."
            )
            stages["kb"] = True

    # timing needs segment_summary.csv + user_segments.csv
    if stages.get("timing") and not stages.get("segment"):
        if not summary_path.exists() or not segments_path.exists():
            logger.warning(
                "'timing' needs segment data — auto-enabling 'segment'."
            )
            stages["segment"] = True
            if not profiles_path.exists() and not stages.get("ingest"):
                logger.warning("Auto-enabling 'ingest' (needed by 'segment').")
                stages["ingest"] = True

    # schedule needs user_segments.csv, timing_recommendations.csv, message_templates.csv
    if stages.get("schedule"):
        templates_path = Path(_p(cfg, "message_templates"))
        timing_path    = Path(_p(cfg, "timing_recommendations"))
        if not stages.get("template") and not templates_path.exists():
            logger.warning(
                "'schedule' needs message_templates.csv — auto-enabling 'template'."
            )
            stages["template"] = True
        if not stages.get("timing") and not timing_path.exists():
            logger.warning(
                "'schedule' needs timing_recommendations.csv — auto-enabling 'timing'."
            )
            stages["timing"] = True
        if not stages.get("segment") and not segments_path.exists():
            logger.warning(
                "'schedule' needs user_segments.csv — auto-enabling 'segment'."
            )
            stages["segment"] = True
            if not profiles_path.exists() and not stages.get("ingest"):
                logger.warning("Auto-enabling 'ingest' (needed by 'segment').")
                stages["ingest"] = True

    # learn needs all Iteration 0 outputs + experiment_results.csv
    if stages.get("learn"):
        exp_path       = Path(_p(cfg, "experiment_results"))
        templates_path = Path(_p(cfg, "message_templates"))
        timing_path    = Path(_p(cfg, "timing_recommendations"))
        if not exp_path.exists():
            logger.warning(
                "'learn' requires %s — create it before running the learn stage.",
                exp_path,
            )
        if not stages.get("schedule") and not Path(_p(cfg, "user_notification_schedule")).exists():
            logger.warning(
                "'learn' benefits from completed Iteration 0 schedule — "
                "consider running 'schedule' first."
            )
        if not stages.get("template") and not templates_path.exists():
            logger.warning(
                "'learn' needs message_templates.csv — auto-enabling 'template'."
            )
            stages["template"] = True
        if not stages.get("timing") and not timing_path.exists():
            logger.warning(
                "'learn' needs timing_recommendations.csv — auto-enabling 'timing'."
            )
            stages["timing"] = True


# ══════════════════════════════════════════════════════════════════════════════
# Stage runners  (each calls the module's entry function directly)
# ══════════════════════════════════════════════════════════════════════════════

def run_ingest(cfg: dict) -> None:
    from user_data_ingestion import run_pipeline
    run_pipeline(
        input_path=_p(cfg, "raw_input"),
        output_path=_p(cfg, "user_profiles"),
    )


def run_kb(cfg: dict) -> None:
    from kb_ingestion import run_kb_ingestion
    run_kb_ingestion(
        kb_path=_p(cfg, "knowledge_bank"),
        output_dir=_p(cfg, "deliverables_dir"),
    )


def run_segment(cfg: dict) -> None:
    from segmentation import run_segmentation
    run_segmentation(
        input_path=_p(cfg, "user_profiles"),
        output_path=_p(cfg, "user_segments"),
        k=int(cfg.get("segmentation", {}).get("k", 8)),
        plots_path=_p(cfg, "segment_plots"),
        summary_path=_p(cfg, "segment_summary"),
    )


def run_goals(cfg: dict) -> None:
    from goal_builder import run_goal_builder
    run_goal_builder(
        feature_map_path=_p(cfg, "feature_map"),
        segments_path=_p(cfg, "segment_summary"),
        output_path=_p(cfg, "segment_goals"),
        enrich_map_path=_p(cfg, "feature_map"),   # enrich in-place
    )


def run_theme(cfg: dict) -> None:
    from theme_engine import run_theme_engine
    run_theme_engine(
        segments_path=_p(cfg, "user_segments"),
        output_path=_p(cfg, "communication_themes"),
        tone_matrix_path=_p(cfg, "tone_matrix"),
    )


def run_template(cfg: dict) -> None:
    from template_generator import run_template_generator
    run_template_generator(
        goals_path=_p(cfg, "segment_goals"),
        themes_path=_p(cfg, "communication_themes"),
        feature_map_path=_p(cfg, "feature_map"),
        tone_matrix_path=_p(cfg, "tone_matrix"),
        output_path=_p(cfg, "message_templates"),
    )


def run_timing(cfg: dict) -> None:
    from timing_optimizer import run_timing_optimizer
    run_timing_optimizer(
        summary_path=_p(cfg, "segment_summary"),
        segments_path=_p(cfg, "user_segments"),
        output_path=_p(cfg, "timing_recommendations"),
    )


def run_schedule(cfg: dict) -> None:
    from schedule_generator import run_schedule_generator
    run_schedule_generator(
        segments_path=_p(cfg, "user_segments"),
        timing_path=_p(cfg, "timing_recommendations"),
        templates_path=_p(cfg, "message_templates"),
        output_path=_p(cfg, "user_notification_schedule"),
    )


def run_learn(cfg: dict) -> None:
    from learning_engine import run_learning_engine
    run_learning_engine(
        experiment_path=_p(cfg, "experiment_results"),
        templates_path=_p(cfg, "message_templates"),
        timing_path=_p(cfg, "timing_recommendations"),
        segments_path=_p(cfg, "user_segments"),
        goals_path=_p(cfg, "segment_goals"),
        themes_path=_p(cfg, "communication_themes"),
        feature_map_path=_p(cfg, "feature_map"),
        tone_matrix_path=_p(cfg, "tone_matrix"),
        out_dir=_p(cfg, "iteration_1_dir"),
    )


# Stage function registry (in execution order)
STAGE_RUNNERS = {
    "ingest":   run_ingest,
    "kb":       run_kb,
    "segment":  run_segment,
    "goals":    run_goals,
    "theme":    run_theme,
    "template": run_template,
    "timing":   run_timing,
    "schedule": run_schedule,
    "learn":    run_learn,
}

STAGE_LABELS = {
    "ingest":   "USER INGESTION   →  codebase/pipeline_cache/user_profiles.csv",
    "kb":       "KB INGESTION     →  iteration_0_before_learning/ (3 JSONs)",
    "segment":  "SEGMENTATION     →  iteration_0_before_learning/user_segments.csv",
    "goals":    "GOAL BUILDER     →  iteration_0_before_learning/segment_goals.csv",
    "theme":    "THEME ENGINE     →  iteration_0_before_learning/communication_themes.csv",
    "template": "TEMPLATE GEN     →  iteration_0_before_learning/message_templates.csv",
    "timing":   "TIMING OPTIM.    →  iteration_0_before_learning/timing_recommendations.csv",
    "schedule": "SCHEDULE GEN     →  iteration_0_before_learning/user_notification_schedule.csv",
    "learn":    "LEARNING ENGINE  →  iteration_1_after_learning/ (4 outputs + schedule_v2)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 68  # banner width


def _banner(cfg_path: str, stages: dict[str, bool]) -> None:
    enabled  = [s for s in STAGE_ORDER if stages.get(s)]
    disabled = [s for s in STAGE_ORDER if not stages.get(s)]
    ticks    = "  ".join(f"{s} ✓" for s in enabled)
    print("\n" + "═" * W)
    print("  PROJECT AURORA  –  Pipeline Runner")
    print(f"  Config : {cfg_path}")
    print(f"  Running: {ticks}")
    if disabled:
        print(f"  Skipped: {', '.join(disabled)}")
    print("═" * W)


def _stage_header(n: int, total: int, name: str) -> None:
    label = STAGE_LABELS.get(name, name.upper())
    print(f"\n{'─' * W}")
    print(f"  STAGE {n}/{total}: {label}")
    print("─" * W)


def _stage_done(name: str, elapsed: float) -> None:
    print(f"  ✓ Stage '{name}' complete in {elapsed:.1f}s")


def _final_banner(stages: dict[str, bool], cfg: dict, total_elapsed: float) -> None:
    DELIVERABLES = [
        ("kb",      "north_star",      "company_north_star.json"),
        ("kb",      "tone_matrix",     "allowed_tone_hook_matrix.json"),
        ("kb",      "feature_map",     "feature_goal_map.json"),
        ("segment", "user_segments",   "user_segments.csv"),
        ("goals",   "segment_goals",   "segment_goals.csv"),
    ]
    TASK2_DELIVERABLES = [
        ("theme",    "communication_themes",      "communication_themes.csv"),
        ("template", "message_templates",         "message_templates.csv"),
        ("timing",   "timing_recommendations",    "timing_recommendations.csv"),
        ("schedule", "user_notification_schedule","user_notification_schedule.csv"),
    ]
    TASK3_DELIVERABLES = [
        ("learn", "message_templates",          "message_templates.csv"),
        ("learn", "timing_recommendations",     "timing_recommendations.csv"),
        ("learn", "ucb_scores",                    "ucb_scores.csv"),
        ("learn", "user_notification_schedule", "user_notification_schedule.csv"),
        ("learn", "user_segments_v2",              "user_segments.csv"),
        ("learn", "learning_delta_report",         "learning_delta_report.csv"),
    ]
    INTERMEDIATES = [
        ("ingest",  "user_profiles",   "user_profiles.csv"),
        ("segment", "segment_summary", "segment_summary.csv"),
        ("segment", "segment_plots",   "segmentation_plots.png"),
    ]

    def _print_group(items):
        for stage, path_key, label in items:
            if stages.get(stage):
                path = _p(cfg, path_key)
                exists = "✓" if Path(path).exists() else "✗"
                print(f"    {exists}  {label:<40}  {path}")

    print("\n" + "═" * W)
    print(f"  PIPELINE COMPLETE  |  total: {total_elapsed:.1f}s")
    print()
    print(f"  {'─' * (W - 4)}  Task 1 Deliverables")
    _print_group(DELIVERABLES)
    print()
    print(f"  {'─' * (W - 4)}  Task 2 Deliverables")
    _print_group(TASK2_DELIVERABLES)
    if stages.get("learn"):
        print()
        print(f"  {'─' * (W - 4)}  Task 3 Deliverables (Iteration 1)")
        _print_group(TASK3_DELIVERABLES)
    print()
    print(f"  {'─' * (W - 4)}  Intermediate Outputs")
    _print_group(INTERMEDIATES)
    print("═" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # Resolve config path relative to project root (one level up from codebase/)
    script_dir  = Path(__file__).parent           # codebase/
    project_root = script_dir.parent              # project root
    config_path  = (project_root / args.config).resolve()

    # Change working directory to project root so all relative paths in config work
    import os
    os.chdir(project_root)

    cfg = load_config(str(config_path))

    # Validate k
    k = int(cfg.get("segmentation", {}).get("k", 8))
    if not (6 <= k <= 12):
        print(f"ERROR: segmentation.k in config must be 6-12, got {k}")
        sys.exit(1)

    # Parse CLI overrides
    cli_stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    cli_skip   = [s.strip() for s in args.skip.split(",")]   if args.skip   else None

    # Validate stage names
    for name_list, flag in [(cli_stages, "--stages"), (cli_skip, "--skip")]:
        if name_list:
            invalid = [s for s in name_list if s not in STAGE_ORDER]
            if invalid:
                print(f"ERROR: Unknown stage(s) in {flag}: {invalid}")
                print(f"       Valid stages: {', '.join(STAGE_ORDER)}")
                sys.exit(1)

    stages = resolve_stages(cfg, cli_stages, cli_skip)
    _check_dependencies(cfg, stages)

    active_stages = [s for s in STAGE_ORDER if stages[s]]
    if not active_stages:
        print("No stages enabled. Check config.yaml stages: section or --stages flag.")
        sys.exit(0)

    _banner(str(config_path), stages)

    total_start = time.time()
    total_count = len(active_stages)

    for i, stage in enumerate(active_stages, 1):
        _stage_header(i, total_count, stage)
        t0 = time.time()
        try:
            STAGE_RUNNERS[stage](cfg)
        except Exception as e:
            print(f"\n  ✗ Stage '{stage}' FAILED: {e}")
            logger.exception("Stage '%s' raised an exception", stage)
            sys.exit(1)
        _stage_done(stage, time.time() - t0)

    _final_banner(stages, cfg, time.time() - total_start)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project Aurora – single-command pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python codebase/pipeline.py                          # run all enabled stages
  python codebase/pipeline.py --stages ingest,segment  # run only these stages
  python codebase/pipeline.py --skip kb,goals          # skip these stages
  python codebase/pipeline.py --config alt_config.yaml # use a different config
        """,
    )
    p.add_argument(
        "--config", "-c",
        default="codebase/config.yaml",
        help="Path to config YAML (relative to project root, default: codebase/config.yaml)",
    )
    p.add_argument(
        "--stages",
        default=None,
        metavar="STAGES",
        help="Comma-separated list of stages to run, e.g. 'ingest,segment' "
             "(overrides yaml stages: toggles). "
             f"Valid: {', '.join(STAGE_ORDER)}",
    )
    p.add_argument(
        "--skip",
        default=None,
        metavar="STAGES",
        help="Comma-separated list of stages to skip, e.g. 'kb,goals'. "
             "Applied on top of yaml stage toggles.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
