"""
MECE Segmentation Engine  –  Project Aurora (SpeakX)
======================================================
Clusters users into 6-12 segments using KMeans on a carefully chosen
feature set:

  ① Normalized behavioural features  (*_norm columns)
  ② Encoded categorical / unchanged features
     (lifecycle_stage, age_band, region, time_window, feature_* bools)
  ③ Derived intelligence scores
     (activeness_score, churn_risk, propensity_*)

Decision-Tree boundaries are extracted to produce human-readable rules
per segment.  All 9 diagnostic plots are saved to a single PNG.

Usage
-----
    python segmentation.py \
        --input   data/output/user_profiles.csv \
        --output  data/output/user_segments.csv \
        --summary data/output/segment_summary.csv \
        --k       8          # 6-12, default 8
        --plots   data/output/segmentation_plots.png
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aurora.segmentation")

RANDOM_STATE = 42
MAX_TREE_DEPTH = 4

# ── Feature selection ──────────────────────────────────────────────────────────
# Only normalized / engineered / score columns feed the clusterer.
# Raw originals and intermediate log-scaled cols are excluded.
CATEGORICAL_FEATURES = [
    "lifecycle_stage",
    "age_band",
    "region",
    "time_window",
]
BOOL_FEATURES = [
    "feature_ai_tutor_used",
    "feature_leaderboard_viewed",
    "feature_progress_checked",
]
NORM_FEATURES = [
    "sessions_last_7d_norm",
    "exercises_completed_7d_norm",
    "days_since_signup_norm",
    "coins_balance_scaled_norm",
    "streak_scaled_norm",
    "notif_open_rate_30d_norm",
    "motivation_score_norm",
]
SCORE_FEATURES = [
    "activeness_score",
    "churn_risk",
    "propensity_gamification",
    "propensity_learning",
    "propensity_achievement",
    "propensity_social",
]

# Segment names are placeholder (Segment_N) here.
# The authoritative semantic name (inferred_segment_name) is produced by the
# LLM in goal_builder.py Stage 2 and written into segment_goals.csv.


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode categoricals, cast bools to int, keep norm+score features.
    Returns (feature_df, feature_names) — no user_id, no raw cols.
    """
    feat = pd.DataFrame(index=df.index)
    label_encoders: dict = {}

    # Categorical → integer label encoding
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        feat[col] = le.fit_transform(df[col].astype(str).str.lower())
        label_encoders[col] = le

    # Boolean → 0/1
    for col in BOOL_FEATURES:
        feat[col] = df[col].astype(int)

    # Normalized numerics + scores (already in [0,1], included as-is)
    for col in NORM_FEATURES + SCORE_FEATURES:
        feat[col] = df[col].astype(float)

    return feat, list(feat.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Clustering
# ══════════════════════════════════════════════════════════════════════════════

def _run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    logger.info("KMeans converged | k=%d | inertia=%.2f", k, km.inertia_)
    return labels


def _elbow_inertias(X_scaled: np.ndarray, k_range=range(2, 13)) -> dict:
    return {
        k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            .fit(X_scaled).inertia_
        for k in k_range
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Decision-Tree rule extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rules(tree, feature_names: list[str], cluster_id: int) -> list[str]:
    """Return all leaf-level rule paths that predict cluster_id."""
    rules: list[str] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree.children_left[node] == -1:          # leaf
            if np.argmax(tree.value[node]) == cluster_id:
                rules.append(
                    " AND\n    ".join(conditions) if conditions else "ALL samples"
                )
            return
        feat = feature_names[tree.feature[node]]
        thr  = tree.threshold[node]
        recurse(tree.children_left[node],  conditions + [f"{feat} <= {thr:.4f}"])
        recurse(tree.children_right[node], conditions + [f"{feat} >  {thr:.4f}"])

    recurse(0, [])
    return rules


def _print_rules(tree, feature_names: list[str], k: int,
                 labels: np.ndarray) -> None:
    print("\n" + "=" * 70)
    print(f"  DECISION BOUNDARY RULES  (K={k})")
    print("=" * 70)
    for c in range(k):
        rules  = _extract_rules(tree, feature_names, c)
        n_users = int(np.sum(labels == c))
        print(f"\n{'─'*70}")
        print(f"  Segment {c}  ({n_users} users)")
        print(f"{'─'*70}")
        if rules:
            for i, rule in enumerate(rules, 1):
                print(f"  Rule {i}:\n    {rule}\n")
        else:
            print("  No pure leaf predicts this segment.\n")
    print("=" * 70 + "\n")


def _collect_rules(tree, feature_names: list[str], k: int) -> dict[int, str]:
    """
    Collect decision-boundary rules for all segments into a dict.
    Returns {segment_id: "rule_string"} where multiple paths are joined with ' OR '.
    Does NOT print anything — printing is handled by _print_rules().
    """
    result: dict[int, str] = {}
    for c in range(k):
        paths = _extract_rules(tree, feature_names, c)
        if paths:
            combined = " OR ".join(
                f"({p.replace(chr(10), ' ').replace('    ', ' ')})" for p in paths
            )
        else:
            combined = "No pure leaf — majority-class assignment"
        result[c] = combined
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Output CSV
# ══════════════════════════════════════════════════════════════════════════════

def _build_output(df: pd.DataFrame, labels: np.ndarray,
                  feat_df: pd.DataFrame, k: int,
                  tree, feature_names: list[str],
                  summary_path: str | None = None) -> pd.DataFrame:
    """
    Attach segment_id, segment_name (placeholder), and decision_rules to df.
    Optionally writes a segment-level summary CSV to summary_path.

    segment_name here is 'Segment_N' — a data-driven placeholder only.
    The authoritative semantic name (inferred_segment_name) is produced by
    the LLM in goal_builder.py and stored in segment_goals.csv.
    """
    out = df.copy()
    out["segment_id"] = labels

    # ── Placeholder naming (Segment_0, Segment_1, …) ────────────────────────
    out["segment_name"] = out["segment_id"].map(lambda sid: f"Segment_{sid}")

    # ── Decision rules (only for segment_summary — NOT added to user_segments) ─
    rules_map = _collect_rules(tree, feature_names, k)

    # ── Segment summary CSV ───────────────────────────────────────────────────
    if summary_path:
        agg = (
            out.groupby("segment_id")
            .agg(
                segment_name=("segment_name", "first"),
                n_users=("user_id", "count"),
                avg_activeness=("activeness_score", "mean"),
                avg_churn_risk=("churn_risk", "mean"),
                avg_gamification=("propensity_gamification", "mean"),
                avg_learning=("propensity_learning", "mean"),
                avg_achievement=("propensity_achievement", "mean"),
                avg_social=("propensity_social", "mean"),
            )
            .round(4)
            .reset_index()
        )
        agg["decision_rules"] = agg["segment_id"].map(rules_map)

        summ_path = Path(summary_path)
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(summ_path, index=False)
        logger.info("Segment summary written → %s", summ_path)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Plots (9-panel)
# ══════════════════════════════════════════════════════════════════════════════

def _make_plots(df_plot: pd.DataFrame, feat_df: pd.DataFrame,
                X_scaled: np.ndarray, labels: np.ndarray,
                dt: DecisionTreeClassifier, feature_names: list[str],
                k: int, output_path: str) -> None:

    COLORS = plt.cm.tab10.colors
    cluster_colors = [COLORS[i % len(COLORS)] for i in range(k)]
    cluster_names  = [f"Seg {i}" for i in range(k)]
    unique         = np.arange(k)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(24, 22))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Panel 1: Cluster distribution ─────────────────────────────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    counts = [int(np.sum(labels == c)) for c in unique]
    bars = ax1.bar(cluster_names, counts, color=cluster_colors,
                   edgecolor="white", linewidth=1.5, width=0.6)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2, str(cnt),
                 ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_title("Segment Distribution", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Users")
    ax1.set_ylim(0, max(counts) + 4)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: PCA 2-D projection ───────────────────────────────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    for c in unique:
        mask = labels == c
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[cluster_colors[c]], s=90, alpha=0.85,
                    edgecolors="white", linewidth=0.7, label=f"Seg {c}")
    ax2.set_title("PCA 2-D Projection", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax2.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: Feature importances ──────────────────────────────────────────
    ax3 = fig.add_subplot(3, 3, 3)
    importances = pd.Series(dt.feature_importances_, index=feature_names)
    importances = importances[importances > 0].sort_values()
    bar_colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(importances)))
    importances.plot(kind="barh", ax=ax3, color=bar_colors,
                     edgecolor="white", linewidth=0.8)
    ax3.set_title("Decision Tree Feature Importances",
                  fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Importance Score")
    for i, v in enumerate(importances):
        ax3.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: Key feature profiles per segment ─────────────────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    key_features = ["activeness_score", "churn_risk",
                    "propensity_gamification", "propensity_social"]
    tmp = feat_df.copy()
    tmp["segment_id"] = labels
    seg_means = tmp.groupby("segment_id")[key_features].mean()
    x     = np.arange(len(key_features))
    width = 0.8 / k
    for i, c in enumerate(unique):
        ax4.bar(x + i * width, seg_means.loc[c], width=width,
                color=cluster_colors[c], label=f"Seg {c}",
                edgecolor="white", linewidth=0.7)
    ax4.set_xticks(x + width * (k - 1) / 2)
    ax4.set_xticklabels(["Activeness", "Churn Risk", "Gamif.", "Social"],
                        fontsize=9)
    ax4.set_title("Segment Profiles — Intelligence Scores",
                  fontsize=13, fontweight="bold", pad=10)
    ax4.set_ylabel("Mean Value")
    ax4.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax4.spines[["top", "right"]].set_visible(False)

    # ── Panel 5: Motivation score distribution ────────────────────────────────
    ax5 = fig.add_subplot(3, 3, 5)
    for c in unique:
        vals = tmp.loc[tmp["segment_id"] == c, "motivation_score_norm"]
        ax5.hist(vals, bins=8, alpha=0.60, color=cluster_colors[c],
                 label=f"Seg {c}", edgecolor="white", linewidth=0.5)
    ax5.set_title("Motivation Score Distribution (norm)",
                  fontsize=13, fontweight="bold", pad=10)
    ax5.set_xlabel("Motivation Score (normalized)")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax5.spines[["top", "right"]].set_visible(False)

    # ── Panel 6: Days since signup vs activeness ──────────────────────────────
    ax6 = fig.add_subplot(3, 3, 6)
    for c in unique:
        mask = labels == c
        ax6.scatter(
            df_plot.loc[mask, "days_since_signup"],
            df_plot.loc[mask, "activeness_score"],
            c=[cluster_colors[c]], s=80, alpha=0.8,
            label=f"Seg {c}", edgecolors="white", linewidth=0.6,
        )
    ax6.set_title("Days Since Signup vs Activeness Score",
                  fontsize=13, fontweight="bold", pad=10)
    ax6.set_xlabel("Days Since Signup")
    ax6.set_ylabel("Activeness Score")
    ax6.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax6.spines[["top", "right"]].set_visible(False)

    # ── Panel 7: Radar chart ──────────────────────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 7, polar=True)
    radar_features = [
        "motivation_score_norm", "notif_open_rate_30d_norm",
        "sessions_last_7d_norm", "exercises_completed_7d_norm",
        "streak_scaled_norm",    "coins_balance_scaled_norm",
    ]
    seg_radar = tmp.groupby("segment_id")[radar_features].mean()
    seg_radar_norm = (
        (seg_radar - seg_radar.min()) /
        (seg_radar.max() - seg_radar.min() + 1e-9)
    )
    N      = len(radar_features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    labels_radar = ["Motivation", "Notif Open", "Sessions",
                    "Exercises", "Streak", "Coins"]
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(labels_radar, fontsize=8)
    ax7.set_yticklabels([])
    for c in unique:
        vals = seg_radar_norm.loc[c].tolist() + seg_radar_norm.loc[c].tolist()[:1]
        ax7.plot(angles, vals, color=cluster_colors[c], linewidth=2,
                 label=f"Seg {c}")
        ax7.fill(angles, vals, color=cluster_colors[c], alpha=0.10)
    ax7.set_title("Segment Radar Profiles\n(Normalized)",
                  fontsize=13, fontweight="bold", pad=18)
    ax7.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
               fontsize=8, ncol=2)

    # ── Panel 8: Elbow plot ───────────────────────────────────────────────────
    ax8 = fig.add_subplot(3, 3, 8)
    k_range  = range(2, 13)
    inertias = _elbow_inertias(X_scaled, k_range)
    ax8.plot(list(inertias.keys()), list(inertias.values()),
             "o-", color="#4C72B0", linewidth=2.5,
             markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax8.axvline(k, color="#C44E52", linestyle="--",
                linewidth=1.5, alpha=0.8, label=f"Chosen K={k}")
    ax8.set_title("Elbow Plot — Optimal K",
                  fontsize=13, fontweight="bold", pad=10)
    ax8.set_xlabel("Number of Segments (K)")
    ax8.set_ylabel("Inertia")
    ax8.legend(fontsize=9)
    ax8.spines[["top", "right"]].set_visible(False)

    # ── Panel 9: Decision tree structure ──────────────────────────────────────
    ax9 = fig.add_subplot(3, 3, 9)
    plot_tree(dt, feature_names=feature_names,
              class_names=[f"Seg{i}" for i in range(k)],
              filled=True, rounded=True, fontsize=5, ax=ax9,
              impurity=False, proportion=False,
              node_ids=False, precision=2)
    ax9.set_title("Decision Tree Structure",
                  fontsize=13, fontweight="bold", pad=10)

    plt.suptitle(
        f"K-Means Segmentation (K={k}) + Decision Tree Boundaries\n"
        "(Project Aurora – SpeakX User Behavioral Data)",
        fontsize=16, fontweight="bold", y=1.01, color="#2d2d2d",
    )
    plt.tight_layout(pad=2.5)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Plots saved → %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

# Sentinel column that only exists in processed (post-ingestion) files
_PROCESSED_SENTINEL = "activeness_score"


def _is_raw_input(df: pd.DataFrame) -> bool:
    """Return True if the CSV hasn't been through the ingestion pipeline yet."""
    return _PROCESSED_SENTINEL not in df.columns


def _run_ingestion(raw_path: str) -> pd.DataFrame:
    """Run the full ingestion pipeline on a raw CSV and return the result."""
    # Import here to avoid circular deps at module level
    from user_ingestion.validator import validate_and_coerce
    from user_ingestion.imputer import impute_missing
    from user_ingestion.feature_engineer import engineer_features
    from user_ingestion.scorer import score_users

    logger.info("Raw input detected — running full ingestion pipeline first...")
    raw_df = pd.read_csv(raw_path, dtype=str)

    validated_df, warnings = validate_and_coerce(raw_df)
    for w in warnings:
        logger.warning("  [VALIDATION] %s", w)

    imputed_df  = impute_missing(validated_df)
    engineered_df = engineer_features(imputed_df)
    scored_df   = score_users(engineered_df)

    logger.info(
        "Ingestion complete: %d users | activeness mean=%.3f | churn mean=%.3f",
        len(scored_df),
        scored_df["activeness_score"].mean(),
        scored_df["churn_risk"].mean(),
    )
    return scored_df


def run_segmentation(input_path: str, output_path: str,
                     k: int, plots_path: str,
                     summary_path: str | None = None) -> pd.DataFrame:

    # Load — auto-detect raw vs processed
    df = pd.read_csv(input_path)
    logger.info("Loaded %d users × %d cols from %s", *df.shape, input_path)

    if _is_raw_input(df):
        df = _run_ingestion(input_path)

    # Prepare feature matrix
    feat_df, feature_names = _prepare_features(df)
    logger.info("Feature matrix: %d users × %d features", *feat_df.shape)
    logger.info("Features used: %s", feature_names)

    # Scale (StandardScaler so KMeans treats all dims equally)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    # KMeans clustering
    labels = _run_kmeans(X_scaled, k)

    # Decision Tree for boundary rules (fit on unscaled feat_df so thresholds
    # are in natural / [0,1] units — easier to interpret)
    dt = DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH,
                                random_state=RANDOM_STATE)
    dt.fit(feat_df.values, labels)
    logger.info("Decision tree DT accuracy (self): %.3f",
                (dt.predict(feat_df.values) == labels).mean())

    # Print rules to stdout
    _print_rules(dt.tree_, feature_names, k, labels)

    # Build output CSV (placeholder names + decision_rules + optional summary)
    out_df = _build_output(df, labels, feat_df, k,
                           tree=dt.tree_, feature_names=feature_names,
                           summary_path=summary_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Segmented profiles written → %s", out_path)

    # Print segment summary to stdout
    summary_cols = ["segment_id", "segment_name", "n_users"]
    tmp_summary = (
        out_df.groupby(["segment_id", "segment_name"])
        .agg(
            n_users=("user_id", "count"),
            avg_activeness=("activeness_score", "mean"),
            avg_churn_risk=("churn_risk", "mean"),
        )
        .round(3)
        .reset_index()
    )
    print("\n" + "=" * 80)
    print("  SEGMENT SUMMARY")
    print("=" * 80)
    print(tmp_summary.to_string(index=False))
    print("=" * 80 + "\n")

    # Plots
    _make_plots(out_df, feat_df, X_scaled, labels,
                dt, feature_names, k, plots_path)

    return out_df


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aurora – MECE Segmentation Engine"
    )
    p.add_argument("--input",  "-i",
                   default="data/output/user_profiles.csv")
    p.add_argument("--output", "-o",
                   default="data/output/user_segments.csv")
    p.add_argument("--k",      "-k",
                   type=int, default=8,
                   help="Number of segments (6-12, default 8)")
    p.add_argument("--plots",  "-p",
                   default="data/output/segmentation_plots.png")
    p.add_argument("--summary", "-s",
                   default="data/output/segment_summary.csv",
                   help="Path for segment-level summary CSV (default: data/output/segment_summary.csv)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not (6 <= args.k <= 12):
        print(f"ERROR: K must be between 6 and 12, got {args.k}")
        sys.exit(1)
    run_segmentation(args.input, args.output, args.k, args.plots,
                     summary_path=args.summary)
