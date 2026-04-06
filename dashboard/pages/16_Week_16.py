"""
Week 16 — Eumaeus
Corpus-wide metrics dashboard: compute stylistic metrics for every episode,
visualise distributions, correlations, and outliers.
"""

import contextlib
import io
import math
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk

for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "vader_lexicon",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "cmudict",
]:
    nltk.download(resource, quiet=True)

from week16.week16_eumaeus import (
    compute_all_metrics,
    EPISODES,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 16 — Eumaeus", page_icon="📖", layout="wide")
st.title("Week 16 — Eumaeus")
st.caption("Corpus-Wide Metrics Dashboard")

# Extend EPISODES to cover all 18 episodes
ALL_EPISODES = list(EPISODES) + [
    ("17", "Ithaca", "17ithaca.txt"),
    ("18", "Penelope", "18penelope.txt"),
]

METRIC_NAMES = [
    "total_tokens",
    "total_types",
    "ttr",
    "hapax_ratio",
    "avg_sent_len",
    "median_sent_len",
    "sent_len_std",
    "noun_verb_ratio",
    "adj_density",
    "vader_mean",
    "vader_var",
    "avg_word_len",
    "flesch_kincaid",
    "exclamation_rate",
    "comma_rate",
    "entity_density",
]

METRIC_LABELS = {
    "total_tokens": "Total Tokens",
    "total_types": "Total Types",
    "ttr": "Type-Token Ratio",
    "hapax_ratio": "Hapax Ratio",
    "avg_sent_len": "Avg Sentence Length",
    "median_sent_len": "Median Sentence Length",
    "sent_len_std": "Sentence Length Std Dev",
    "noun_verb_ratio": "Noun/Verb Ratio",
    "adj_density": "Adjective Density",
    "vader_mean": "VADER Sentiment Mean",
    "vader_var": "VADER Sentiment Variance",
    "avg_word_len": "Avg Word Length",
    "flesch_kincaid": "Flesch-Kincaid Grade",
    "exclamation_rate": "Exclamation Rate",
    "comma_rate": "Comma Rate",
    "entity_density": "Entity Density",
}


# ============================================================================
# Helpers
# ============================================================================


def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_compute_all_metrics(episode_file):
    """Compute all metrics for a single episode."""
    text = cached_load_episode(episode_file)
    return suppress_stdout(compute_all_metrics, text)


@st.cache_data
def compute_master_table():
    """Compute metrics for all 18 episodes. Returns a DataFrame."""
    rows = []
    for ep_num, ep_name, filename in ALL_EPISODES:
        metrics = cached_compute_all_metrics(filename)
        row = {"Episode": f"{ep_num} — {ep_name}", "Filename": filename}
        for key in METRIC_NAMES:
            row[key] = metrics.get(key, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=15,  # Eumaeus
    caption="Week 16: Corpus-Wide Metrics Dashboard",
    description=(
        "*Eumaeus is the episode of exhaustion — Bloom and Stephen sit in a cabman's "
        "shelter while Bloom talks in tired, cliche-ridden prose. This week we step "
        "back and treat the entire novel as a structured dataset, computing stylistic "
        "metrics for every episode and visualising patterns across the corpus.*"
    ),
)


# ============================================================================
# Section 1: The Master Table
# ============================================================================

st.header("1. The Master Table")

st.markdown(
    "Compute 16 stylistic metrics for all 18 episodes of *Ulysses*. This is "
    "computationally expensive (sentiment analysis, POS tagging, readability scoring "
    "for every episode) so results are cached after the first run."
)

if st.button("Compute All Episodes", key="compute_all_button"):
    progress = st.progress(0, text="Computing metrics for all episodes...")
    rows = []
    for i, (ep_num, ep_name, filename) in enumerate(ALL_EPISODES):
        progress.progress(
            (i + 1) / len(ALL_EPISODES),
            text=f"Computing {ep_num} — {ep_name}...",
        )
        metrics = cached_compute_all_metrics(filename)
        row = {"Episode": f"{ep_num} — {ep_name}", "Filename": filename}
        for key in METRIC_NAMES:
            row[key] = metrics.get(key, 0.0)
        rows.append(row)
    progress.empty()
    df = pd.DataFrame(rows)
    st.session_state["master_table"] = df
    st.success("All episodes computed.")

if "master_table" in st.session_state:
    df = st.session_state["master_table"]

    # --- Master metrics table ---
    st.subheader("All Episodes x All Metrics")

    display_cols = ["Episode"] + METRIC_NAMES
    df_display = df[display_cols].copy()

    # Format numeric columns
    for col in METRIC_NAMES:
        if col in ("total_tokens", "total_types"):
            df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}")
        elif col in ("ttr", "hapax_ratio", "adj_density", "exclamation_rate", "comma_rate", "entity_density"):
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        elif col in ("vader_mean", "vader_var"):
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        else:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")

    # Rename columns to friendly labels
    rename_map = {k: METRIC_LABELS[k] for k in METRIC_NAMES}
    df_display = df_display.rename(columns=rename_map)

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # --- Eumaeus ranking ---
    st.subheader("Eumaeus Rankings")

    st.markdown(
        "Where does Eumaeus rank among all 18 episodes for key stylistic metrics? "
        "Joyce's deliberate use of exhausted, cliche-ridden prose should surface here."
    )

    ranking_metrics = ["avg_sent_len", "ttr", "flesch_kincaid"]
    rank_cols = st.columns(len(ranking_metrics))

    for col, metric in zip(rank_cols, ranking_metrics):
        sorted_df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        rank = sorted_df[sorted_df["Episode"].str.contains("Eumaeus")].index
        if len(rank) > 0:
            rank_val = rank[0] + 1
        else:
            rank_val = "N/A"
        eumaeus_val = df[df["Episode"].str.contains("Eumaeus")][metric].values
        val_str = f"{eumaeus_val[0]:.2f}" if len(eumaeus_val) > 0 else "N/A"
        col.metric(
            METRIC_LABELS[metric],
            val_str,
            delta=f"Rank {rank_val}/18",
            delta_color="off",
        )

    # --- Metric distribution bar chart ---
    st.subheader("Metric Distribution")

    selected_metric = st.selectbox(
        "Select a metric to visualise across all episodes",
        METRIC_NAMES,
        format_func=lambda x: METRIC_LABELS[x],
        key="metric_dist_select",
    )

    sorted_by_metric = df.sort_values(selected_metric, ascending=True).reset_index(drop=True)
    labels = sorted_by_metric["Episode"].tolist()
    values = sorted_by_metric[selected_metric].tolist()
    short_labels = [ep.split(" — ")[1] for ep in labels]

    # Highlight Eumaeus
    colors = [
        "#E07A5F" if "Eumaeus" in lbl else "#4A90D9"
        for lbl in labels
    ]

    fig_dist, ax_dist = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
    ax_dist.barh(range(len(short_labels)), values, color=colors)
    ax_dist.set_yticks(range(len(short_labels)))
    ax_dist.set_yticklabels(short_labels, fontsize=9)
    ax_dist.set_xlabel(METRIC_LABELS[selected_metric])
    ax_dist.set_title(f"{METRIC_LABELS[selected_metric]} — All Episodes (sorted)")

    ax_dist.legend(
        handles=[
            Patch(facecolor="#E07A5F", label="Eumaeus"),
            Patch(facecolor="#4A90D9", label="Other episodes"),
        ],
        loc="lower right",
        fontsize=8,
    )
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)


# ============================================================================
# Section 2: Visual Dashboard
# ============================================================================

st.header("2. Visual Dashboard")

if "master_table" not in st.session_state:
    st.info("Compute the master table above to unlock the visual dashboard.")
else:
    df = st.session_state["master_table"]
    numeric_df = df[METRIC_NAMES].astype(float)
    episode_names = df["Episode"].tolist()
    short_names = [ep.split(" — ")[1] for ep in episode_names]

    # --- Z-score heatmap ---
    st.subheader("Z-Score Heatmap")

    st.markdown(
        "Each cell shows how many standard deviations an episode's metric is from "
        "the corpus mean. Red = above average, blue = below average."
    )

    zscores = numeric_df.apply(lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col * 0)

    fig_heat, ax_heat = plt.subplots(figsize=(16, 10))
    im = ax_heat.imshow(zscores.values, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)
    ax_heat.set_xticks(range(len(METRIC_NAMES)))
    ax_heat.set_xticklabels(
        [METRIC_LABELS[m] for m in METRIC_NAMES], rotation=45, ha="right", fontsize=7
    )
    ax_heat.set_yticks(range(len(short_names)))
    ax_heat.set_yticklabels(short_names, fontsize=8)
    ax_heat.set_title("Stylistic Z-Scores: Episodes x Metrics")
    fig_heat.colorbar(im, ax=ax_heat, label="Z-score", shrink=0.8)

    # Annotate cells
    for i in range(zscores.shape[0]):
        for j in range(zscores.shape[1]):
            val = zscores.values[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax_heat.text(
                j, i, f"{val:.1f}", ha="center", va="center", fontsize=5, color=color
            )

    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)

    # --- Sparklines ---
    st.subheader("Metric Sparklines")

    st.markdown(
        "One small plot per metric showing values across all 18 episodes in episode order."
    )

    n_metrics = len(METRIC_NAMES)
    ncols = 4
    nrows = math.ceil(n_metrics / ncols)

    fig_spark, axes_spark = plt.subplots(nrows, ncols, figsize=(16, nrows * 2.5))
    axes_flat = axes_spark.flatten()

    for idx, metric in enumerate(METRIC_NAMES):
        ax = axes_flat[idx]
        vals = numeric_df[metric].values
        ax.plot(range(len(vals)), vals, color="#4A90D9", linewidth=1.5)
        ax.fill_between(range(len(vals)), vals, alpha=0.15, color="#4A90D9")

        # Highlight Eumaeus (index 15)
        if len(vals) > 15:
            ax.plot(15, vals[15], "o", color="#E07A5F", markersize=6, zorder=5)

        ax.set_title(METRIC_LABELS[metric], fontsize=7, pad=3)
        ax.set_xticks([0, 5, 10, 15, 17])
        ax.set_xticklabels(["01", "06", "11", "16", "18"], fontsize=5)
        ax.tick_params(axis="y", labelsize=5)

    # Hide unused axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig_spark.suptitle("Metric Sparklines (orange dot = Eumaeus)", fontsize=10, y=1.01)
    plt.tight_layout()
    st.pyplot(fig_spark)
    plt.close(fig_spark)

    # --- Correlation matrix ---
    st.subheader("Metric Correlation Matrix")

    st.markdown(
        "Which metrics move together across episodes? Strong positive correlations "
        "suggest metrics that capture the same underlying stylistic dimension."
    )

    corr = numeric_df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
    im_corr = ax_corr.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    metric_labels_short = [METRIC_LABELS[m][:18] for m in METRIC_NAMES]
    ax_corr.set_xticks(range(len(metric_labels_short)))
    ax_corr.set_xticklabels(metric_labels_short, rotation=45, ha="right", fontsize=7)
    ax_corr.set_yticks(range(len(metric_labels_short)))
    ax_corr.set_yticklabels(metric_labels_short, fontsize=7)
    ax_corr.set_title("Metric x Metric Correlation")
    fig_corr.colorbar(im_corr, ax=ax_corr, label="Pearson r", shrink=0.8)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax_corr.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color
            )

    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # --- Radar chart ---
    st.subheader("Episode Radar Chart")

    st.markdown(
        "Compare the stylistic profile of selected episodes. Metrics are normalised "
        "to [0, 1] across the corpus so all dimensions are comparable."
    )

    default_radar = []
    for name in ["Telemachus", "Sirens", "Cyclops", "Eumaeus"]:
        for ep in episode_names:
            if name in ep:
                default_radar.append(ep)
                break

    radar_episodes = st.multiselect(
        "Select 2-5 episodes for radar comparison",
        episode_names,
        default=default_radar,
        key="radar_select",
    )

    if len(radar_episodes) < 2:
        st.warning("Select at least 2 episodes for the radar chart.")
    elif len(radar_episodes) > 5:
        st.warning("Select at most 5 episodes for readability.")
    else:
        # Normalise metrics to [0, 1]
        normed = numeric_df.copy()
        for col in METRIC_NAMES:
            cmin = normed[col].min()
            cmax = normed[col].max()
            if cmax > cmin:
                normed[col] = (normed[col] - cmin) / (cmax - cmin)
            else:
                normed[col] = 0.5

        # Build radar
        angles = np.linspace(0, 2 * np.pi, len(METRIC_NAMES), endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        radar_colors = ["#4A90D9", "#E07A5F", "#81B29A", "#DAA520", "#9B59B6"]

        for i, ep in enumerate(radar_episodes):
            ep_idx = episode_names.index(ep)
            vals = normed.iloc[ep_idx][METRIC_NAMES].values.tolist()
            vals += vals[:1]
            short = ep.split(" — ")[1]
            color = radar_colors[i % len(radar_colors)]
            ax_radar.plot(angles, vals, "o-", linewidth=1.5, label=short, color=color)
            ax_radar.fill(angles, vals, alpha=0.08, color=color)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(
            [METRIC_LABELS[m][:15] for m in METRIC_NAMES], fontsize=6
        )
        ax_radar.set_title("Normalised Stylistic Profiles", pad=20)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_radar)
        plt.close(fig_radar)


# ============================================================================
# Section 3: Outlier Detection
# ============================================================================

st.header("3. Outlier Detection")

if "master_table" not in st.session_state:
    st.info("Compute the master table above to unlock outlier detection.")
else:
    df = st.session_state["master_table"]
    numeric_df = df[METRIC_NAMES].astype(float)
    episode_names = df["Episode"].tolist()
    short_names = [ep.split(" — ")[1] for ep in episode_names]

    # Z-score computation
    zscores = numeric_df.apply(
        lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col * 0
    )

    # --- Threshold slider ---
    z_threshold = st.slider(
        "Z-score threshold for outlier detection",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        key="z_threshold",
    )

    # --- Build outlier table ---
    outlier_rows = []
    for i, ep in enumerate(episode_names):
        for j, metric in enumerate(METRIC_NAMES):
            z = zscores.iloc[i, j]
            if abs(z) >= z_threshold:
                outlier_rows.append(
                    {
                        "Episode": ep,
                        "Metric": METRIC_LABELS[metric],
                        "Value": f"{numeric_df.iloc[i, j]:.4f}",
                        "Z-score": f"{z:.2f}",
                        "Direction": "high" if z > 0 else "low",
                    }
                )

    # --- Outlier count metrics ---
    total_outliers = len(outlier_rows)
    episodes_with_outliers = len(set(r["Episode"] for r in outlier_rows))
    metrics_with_outliers = len(set(r["Metric"] for r in outlier_rows))

    oc1, oc2, oc3 = st.columns(3)
    oc1.metric("Total Outlier Cells", total_outliers)
    oc2.metric("Episodes with Outliers", f"{episodes_with_outliers}/18")
    oc3.metric("Metrics with Outliers", f"{metrics_with_outliers}/{len(METRIC_NAMES)}")

    # --- Outlier table ---
    st.subheader("Outlier Table")

    if outlier_rows:
        df_outliers = pd.DataFrame(outlier_rows)
        df_outliers = df_outliers.sort_values("Z-score", key=lambda x: x.astype(float).abs(), ascending=False)
        st.dataframe(df_outliers, use_container_width=True, hide_index=True)
    else:
        st.info(f"No outliers detected at z-score threshold {z_threshold:.1f}.")

    # --- Outlier episode profile ---
    st.subheader("Episode Outlier Profile")

    selected_ep = st.selectbox(
        "Select an episode to view its z-score profile",
        episode_names,
        index=15,  # default to Eumaeus
        key="outlier_ep_select",
    )

    ep_idx = episode_names.index(selected_ep)
    ep_zscores = zscores.iloc[ep_idx].values
    ep_short = selected_ep.split(" — ")[1]

    # Bar chart of z-scores for this episode
    bar_colors = []
    for z in ep_zscores:
        if abs(z) >= z_threshold:
            bar_colors.append("#E07A5F" if z > 0 else "#4A90D9")
        else:
            bar_colors.append("#A0A0A0")

    fig_profile, ax_profile = plt.subplots(figsize=(12, max(6, len(METRIC_NAMES) * 0.4)))
    y_pos = range(len(METRIC_NAMES))
    ax_profile.barh(y_pos, ep_zscores, color=bar_colors)
    ax_profile.set_yticks(y_pos)
    ax_profile.set_yticklabels([METRIC_LABELS[m] for m in METRIC_NAMES], fontsize=8)
    ax_profile.axvline(x=0, color="black", linewidth=0.5)
    ax_profile.axvline(x=z_threshold, color="#E07A5F", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_profile.axvline(x=-z_threshold, color="#4A90D9", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_profile.set_xlabel("Z-score")
    ax_profile.set_title(f"Z-Score Profile: {ep_short}")

    ax_profile.legend(
        handles=[
            Patch(facecolor="#E07A5F", label=f"High outlier (z >= {z_threshold:.1f})"),
            Patch(facecolor="#4A90D9", label=f"Low outlier (z <= -{z_threshold:.1f})"),
            Patch(facecolor="#A0A0A0", label="Within threshold"),
        ],
        loc="lower right",
        fontsize=7,
    )
    plt.tight_layout()
    st.pyplot(fig_profile)
    plt.close(fig_profile)


# --- Footer ---
st.markdown("""
---

**What this week reveals:** Treating *Ulysses* as a structured dataset of 18 episodes
and 16 metrics reveals the novel's architecture from a new angle. Every episode has a
distinct stylistic fingerprint — Penelope's unpunctuated monologue, Circe's dramatic
sprawl, Eumaeus's exhausted cliches — and these differences are measurable. The z-score
heatmap shows which episodes are genuinely anomalous and which are quietly typical. The
correlation matrix reveals which stylistic dimensions move together (vocabulary richness
and readability, sentence length and sentiment variance) and which are independent. Outlier
detection surfaces the episodes Joyce pushed hardest — not just the famous experimental
ones, but quiet anomalies that close reading might miss. The novel that seemed to resist
systematisation turns out to be a precisely engineered dataset.
""")
