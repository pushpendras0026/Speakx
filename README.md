Project Aurora - SpeakX Intelligent Notification System
=========================================================

A self-learning push-notification pipeline that segments users, generates
bilingual (EN/HI) personalised messages, optimises delivery timing, and
adapts from experiment feedback using Gemini 2.5 Flash + UCB1 exploration.

Architecture diagram: codebase/architecture.png


SETUP
-----
Requirements: Python 3.11+, internet access, Gemini API key.

Install dependencies:
  pip install pandas numpy scikit-learn matplotlib pyyaml openai python-dotenv tqdm

Add your API key to codebase/.env:
  GEMINI_API_KEY=AIzaSy...YOUR_KEY_HERE...

All pipeline settings (input paths, output paths, segmentation k, stage
toggles) live in codebase/config.yaml. Edit that file to change any path
before running.


RUN
---

Command 1 - Tasks 1 & 2 (Ingestion > Segmentation > Content > Timing > Schedule)

  python codebase/pipeline.py

Reads codebase/config.yaml and runs all enabled stages in order.
Runtime: 10-20 min (~57 Gemini calls).

  Stage      What it does
  --------   ------------------------------------------------------------
  ingest     Validate raw users, KNN-impute, feature-engineer, score
             activeness / churn / propensity
  kb         Parse knowledge_bank.md -> company_north_star.json,
             allowed_tone_hook_matrix.json, feature_goal_map.json
  segment    KMeans (k=8) -> user_segments.csv + Decision-Tree rules
  goals      LLM per segment -> segment_goals.csv
  theme      Octalysis-aligned theme assignment -> communication_themes.csv
  template   5 bilingual variants (T1-T5) per segment x time-unit x theme
             -> message_templates.csv
  timing     Window allocations + expected CTR -> timing_recommendations.csv
  schedule   Individual-level paced schedule -> user_notification_schedule.csv

Outputs land in iteration_0_before_learning/


Command 2 - Task 3 (Self-Learning Loop, Iteration 1)

  python codebase/learning_engine.py

Requires experiment_results.csv at project root with columns:
  template_id, segment_id, lifecycle_stage, goal, theme,
  notification_window, total_sends, total_opens, total_engagements,
  ctr, engagement_rate, uninstall_rate, performance_status (GOOD/NEUTRAL/BAD)

All paths are read from codebase/config.yaml.

  Status    Action
  --------  ---------------------------------------------------------------
  GOOD      Keep original template, no LLM call
  NEUTRAL   Gemini improves it using original as reference
  BAD       Gemini redesigns it fresh; original suppressed (UCB = -1)

Also applies EMA timing update (alpha=0.70) and UCB1 scoring (c=0.15).
Outputs land in iteration_1_after_learning/ + learning_delta_report.csv


PROJECT LAYOUT
--------------
speakx/
  codebase/
    config.yaml           <- all pipeline settings
    .env                  <- GEMINI_API_KEY (never commit)
    pipeline.py           <- Command 1 entry point
    learning_engine.py    <- Command 2 entry point
    architecture.png               <- architecture diagram
    input/                <- raw user CSV + knowledge_bank.md
    pipeline_cache/       <- intermediate outputs (not deliverables)
  experiment_results.csv  <- Iteration 0 feedback
  iteration_0_before_learning/
  iteration_1_after_learning/
