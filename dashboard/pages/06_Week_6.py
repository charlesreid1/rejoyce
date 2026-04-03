"""
Week 06 — Hades
Sentiment analysis, narrative voice registers, and affective lexicons.
"""

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag

for resource in [
    "punkt",
    "punkt_tab",
    "vader_lexicon",
    "sentiwordnet",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

from week06.week06_hades import (
    split_interior_exterior,
    is_bloom_interior,
    DEATH_WORDS,
    PROXIMITY_WORDS,
    get_wordnet_pos,
)
from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

# --- Register colors ---
REGISTER_COLORS = {
    "Dialogue": "#E07A5F",
    "Interior": "#5F9EA0",
    "Narration": "#708090",
}

st.set_page_config(page_title="Week 06 — Hades", page_icon="📖", layout="wide")
st.title("Week 06 — Hades")
st.caption("Sentiment, Voice Registers & Death Lexicon")

# --- Sidebar ---
episode_file, episode_label = episode_sidebar(
    default_index=5,  # Hades
    caption="Week 6: Sentiment, Voice Registers & Death Lexicon",
    description=(
        "**Week 6** examines how VADER sentiment analysis handles a funeral episode, "
        "the challenge of classifying narrative voice into dialogue, interior monologue, "
        "and external narration, and SentiWordNet's context blindness when scoring "
        "death-related and proximity vocabulary."
    ),
)

with st.sidebar:
    window_size = st.slider("Window size", 10, 100, 50, key="window_size")

episode_text = cached_load_episode(episode_file)


# ============================================================================
# Cached helper functions
# ============================================================================


@st.cache_data
def cached_sentiment_scores(text):
    """Score each sentence with VADER. Returns list of (sentence, compound)."""
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    return [(sent, sia.polarity_scores(sent)["compound"]) for sent in sentences]


@st.cache_data
def cached_smoothed_trajectory(scores, window_size):
    """Sliding-window smoothed trajectory over compound scores."""
    trajectory = []
    for i in range(0, len(scores), window_size // 2):
        window = scores[i : i + window_size]
        if window:
            avg = sum(window) / len(window)
            trajectory.append((i + len(window) // 2, avg))
    return trajectory


@st.cache_data
def cached_register_split(text):
    """Split text into registers. Returns dict of register_name -> list of lines."""
    lines = text.split("\n")
    registers = {"Dialogue": [], "Interior": [], "Narration": []}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("\u2014") or stripped.startswith("--"):
            registers["Dialogue"].append(stripped)
        elif is_bloom_interior(stripped):
            registers["Interior"].append(stripped)
        else:
            registers["Narration"].append(stripped)
    return registers


@st.cache_data
def cached_register_sentiment(text):
    """VADER sentiment per register. Returns dict of register -> list of compound scores."""
    registers = cached_register_split(text)
    sia = SentimentIntensityAnalyzer()
    result = {}
    for name, lines in registers.items():
        joined = " ".join(lines)
        if not joined.strip():
            result[name] = []
            continue
        sents = sent_tokenize(joined)
        result[name] = [sia.polarity_scores(s)["compound"] for s in sents]
    return result


@st.cache_data
def cached_sentiwordnet_lookup(word):
    """Look up a word in SentiWordNet. Returns (pos, neg, obj, synset_name) or None."""
    tagged = pos_tag([word])
    wordnet_pos = get_wordnet_pos(tagged[0][1])
    synsets = list(swn.senti_synsets(word))
    if not synsets:
        return None
    if wordnet_pos:
        pos_filtered = [ss for ss in synsets if str(ss.synset.pos()) == wordnet_pos]
        if pos_filtered:
            synsets = pos_filtered
    best = max(synsets, key=lambda ss: ss.pos_score() + ss.neg_score())
    return (best.pos_score(), best.neg_score(), best.obj_score(), best.synset.name())


@st.cache_data
def cached_lexicon_table(word_list):
    """SentiWordNet scores for a list of words. Returns list of dicts."""
    rows = []
    for word in word_list:
        result = cached_sentiwordnet_lookup(word)
        if result:
            pos_s, neg_s, obj_s, syn_name = result
            rows.append({
                "Word": word,
                "Pos Score": pos_s,
                "Neg Score": neg_s,
                "Obj Score": obj_s,
                "Best Synset": syn_name,
            })
        else:
            rows.append({
                "Word": word,
                "Pos Score": 0.0,
                "Neg Score": 0.0,
                "Obj Score": 1.0,
                "Best Synset": "(not found)",
            })
    return rows


@st.cache_data
def cached_lexicon_density(text, word_list_tuple, window_size):
    """Sliding-window density of lexicon words through the text.

    Returns list of (midpoint, density) tuples.
    """
    word_list = list(word_list_tuple)
    words = word_tokenize(text.lower())
    word_set = set(w.lower() for w in word_list)
    density = []
    step = max(1, window_size // 2)
    for i in range(0, len(words), step):
        window = words[i : i + window_size]
        if window:
            count = sum(1 for w in window if w in word_set)
            density.append((i + len(window) // 2, count / len(window)))
    return density


# ============================================================================
# Section 1: Sentiment Trajectory
# ============================================================================

st.header("1. Sentiment Trajectory")

st.markdown(
    "VADER compound sentiment per sentence with sliding-window smoothing. "
    "Adjust the window size in the sidebar to control smoothing, toggle the raw scatter, "
    "and compare trajectories across episodes."
)

scored_sentences = cached_sentiment_scores(episode_text)
compound_scores = [score for _, score in scored_sentences]
trajectory = cached_smoothed_trajectory(compound_scores, window_size)

# --- Metrics row ---
n_sents = len(compound_scores)
mean_compound = sum(compound_scores) / n_sents if n_sents else 0
variance = sum((s - mean_compound) ** 2 for s in compound_scores) / n_sents if n_sents else 0
pos_count = sum(1 for s in compound_scores if s > 0.05)
neg_count = sum(1 for s in compound_scores if s < -0.05)
neu_count = n_sents - pos_count - neg_count

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Sentences", n_sents)
m2.metric("Mean Compound", f"{mean_compound:.4f}")
m3.metric("Variance", f"{variance:.4f}")
pct_pos = 100 * pos_count / n_sents if n_sents else 0
pct_neg = 100 * neg_count / n_sents if n_sents else 0
pct_neu = 100 * neu_count / n_sents if n_sents else 0
m4.metric("Pos / Neg / Neu", f"{pct_pos:.0f}% / {pct_neg:.0f}% / {pct_neu:.0f}%")

# --- Raw scatter toggle ---
show_raw = st.toggle("Show raw sentence scores", value=True, key="show_raw")

# --- Sentiment arc plot ---
# Hades event annotations
HADES_EVENTS = [
    (300, "Child's murder\nhouse"),
    (900, "Cemetery\narrival"),
    (1200, "Burial"),
    (1350, "Rudy\nmemory"),
]

fig_sent, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top panel: scatter + smoothed line
if show_raw:
    axes[0].scatter(
        range(n_sents), compound_scores, alpha=0.3, s=4, color="#4A90D9", zorder=1
    )
if trajectory:
    xs, ys = zip(*trajectory)
    axes[0].plot(xs, ys, color="#E07A5F", linewidth=2, zorder=2)
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_title(f"VADER Compound Score per Sentence: {episode_label}")
axes[0].set_ylabel("Compound Score")

# Bottom panel: smoothed trajectory with fill_between and annotations
if trajectory:
    xs, ys = zip(*trajectory)
    axes[1].plot(xs, ys, color="#E07A5F", linewidth=2)
    axes[1].fill_between(xs, ys, alpha=0.2, color="#E07A5F")
axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_title(f"Sentiment Trajectory (window={window_size}): {episode_label}")
axes[1].set_xlabel("Sentence Index")
axes[1].set_ylabel("Mean Compound Score")

# Annotations only for Hades
if episode_file == "06hades.txt":
    for idx, label_text in HADES_EVENTS:
        if idx < n_sents:
            for ax in axes:
                ax.axvline(x=idx, color="orange", linestyle=":", alpha=0.7)
            axes[0].text(
                idx, 0.8, label_text, rotation=90, verticalalignment="bottom",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

plt.tight_layout()
st.pyplot(fig_sent)
plt.close(fig_sent)

# --- VADER misfires table ---
st.subheader("VADER Misfires & Extremes")

scored_sorted_pos = sorted(scored_sentences, key=lambda x: -x[1])[:5]
scored_sorted_neg = sorted(scored_sentences, key=lambda x: x[1])[:5]

misfire_rows = []
for sent, score in scored_sorted_pos:
    misfire_rows.append({"Score": f"{score:+.3f}", "Sentence": sent[:120], "Type": "Most Positive"})
for sent, score in scored_sorted_neg:
    misfire_rows.append({"Score": f"{score:+.3f}", "Sentence": sent[:120], "Type": "Most Negative"})

st.dataframe(pd.DataFrame(misfire_rows), use_container_width=True, hide_index=True)
st.caption(
    "Positive scores in a funeral chapter often indicate VADER misfires -- "
    "literary language, irony, and euphemism confound a tool trained on social media."
)

# --- Cross-episode comparison ---
with st.expander("Compare sentiment trajectories across episodes"):
    default_compare = ["06 \u2014 Hades", "01 \u2014 Telemachus"]
    compare_episodes = st.multiselect(
        "Select episodes to compare (2-5)",
        EPISODE_LABELS,
        default=[l for l in default_compare if l in EPISODE_LABELS],
        max_selections=5,
        key="compare_episodes",
    )

    if len(compare_episodes) >= 2:
        fig_cmp, ax_cmp = plt.subplots(figsize=(14, 6))
        cmap = plt.cm.Set2
        summary_rows = []

        for i, label in enumerate(compare_episodes):
            file = EPISODE_FILES[EPISODE_LABELS.index(label)]
            text = cached_load_episode(file)
            scores = cached_sentiment_scores(text)
            compounds = [s for _, s in scores]
            traj = cached_smoothed_trajectory(compounds, window_size)

            if traj:
                xs, ys = zip(*traj)
                # Normalize x-axis to 0-100%
                max_x = max(xs) if xs else 1
                xs_norm = [x / max_x * 100 for x in xs]
                mean_c = sum(compounds) / len(compounds)
                ax_cmp.plot(xs_norm, ys, linewidth=2, color=cmap(i), label=f"{label} (mean={mean_c:.3f})")

                var_c = sum((s - mean_c) ** 2 for s in compounds) / len(compounds)
                pos_pct = sum(1 for s in compounds if s > 0.05) / len(compounds) * 100
                neg_pct = sum(1 for s in compounds if s < -0.05) / len(compounds) * 100
                summary_rows.append({
                    "Episode": label,
                    "Mean Compound": f"{mean_c:.4f}",
                    "Variance": f"{var_c:.4f}",
                    "% Positive": f"{pos_pct:.1f}%",
                    "% Negative": f"{neg_pct:.1f}%",
                })

        ax_cmp.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_cmp.set_xlabel("Position in episode (%)")
        ax_cmp.set_ylabel("Mean Compound Score")
        ax_cmp.set_title(f"Cross-Episode Sentiment Comparison (window={window_size})")
        ax_cmp.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Select at least 2 episodes to compare.")


# ============================================================================
# Section 2: Narrative Voice & Register Analysis
# ============================================================================

st.header("2. Narrative Voice & Register Analysis")

st.markdown(
    "Text is split into three registers (dialogue, interior monologue, external narration) "
    "using heuristics, then VADER sentiment is compared across registers. "
    "Interior monologue tends to cluster near neutral while dialogue has wider swings."
)

registers = cached_register_split(episode_text)
register_sentiment = cached_register_sentiment(episode_text)

total_lines = sum(len(lines) for lines in registers.values())

# --- Register proportion metrics ---
rm1, rm2, rm3 = st.columns(3)
for col, name in zip([rm1, rm2, rm3], ["Dialogue", "Interior", "Narration"]):
    count = len(registers[name])
    pct = 100 * count / total_lines if total_lines else 0
    col.metric(name, f"{count} ({pct:.1f}%)")

# --- Donut chart ---
fig_donut, ax_donut = plt.subplots(figsize=(6, 6))
sizes = [len(registers[r]) for r in ["Dialogue", "Interior", "Narration"]]
colors = [REGISTER_COLORS[r] for r in ["Dialogue", "Interior", "Narration"]]
wedges, texts, autotexts = ax_donut.pie(
    sizes, labels=["Dialogue", "Interior", "Narration"],
    colors=colors, autopct="%1.1f%%", wedgeprops=dict(width=0.4),
    pctdistance=0.8,
)
ax_donut.text(0, 0, f"{total_lines}\nlines", ha="center", va="center", fontsize=14, fontweight="bold")
ax_donut.set_title(f"Register Proportions: {episode_label}")
st.pyplot(fig_donut)
plt.close(fig_donut)

# --- Register sentiment grouped bar chart ---
st.subheader("Register Sentiment Comparison")

reg_names = ["Dialogue", "Interior", "Narration"]
reg_means = []
reg_vars = []
reg_stds = []
for name in reg_names:
    scores = register_sentiment[name]
    if scores:
        m = sum(scores) / len(scores)
        v = sum((s - m) ** 2 for s in scores) / len(scores)
        reg_means.append(m)
        reg_vars.append(v)
        reg_stds.append(v ** 0.5)
    else:
        reg_means.append(0)
        reg_vars.append(0)
        reg_stds.append(0)

fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
y = np.arange(len(reg_names))
bar_h = 0.6
bars = ax_bar.barh(
    y, reg_means, bar_h,
    xerr=reg_stds,
    color=[REGISTER_COLORS[r] for r in reg_names],
    capsize=5,
)
ax_bar.set_yticks(y)
ax_bar.set_yticklabels(reg_names)
ax_bar.set_xlabel("Mean Compound Score")
ax_bar.set_title("Mean Sentiment by Register (whiskers = 1 std dev)")
ax_bar.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
for i, (m, v) in enumerate(zip(reg_means, reg_vars)):
    ax_bar.text(m + reg_stds[i] + 0.01, i, f"var={v:.4f}", va="center", fontsize=8)
plt.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

# --- Register sentiment distributions ---
st.subheader("Sentiment Distributions by Register")

use_violin = st.toggle("Violin plot", value=False, key="use_violin")

fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
plot_data = [register_sentiment[r] for r in reg_names]
# Filter out empty registers for plotting
non_empty = [(r, d) for r, d in zip(reg_names, plot_data) if d]
if non_empty:
    ne_names, ne_data = zip(*non_empty)
    if use_violin:
        parts = ax_dist.violinplot(ne_data, positions=range(len(ne_names)), showmeans=True, showmedians=True)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(REGISTER_COLORS[ne_names[i]])
            pc.set_alpha(0.7)
    else:
        bp = ax_dist.boxplot(
            ne_data, positions=range(len(ne_names)), patch_artist=True,
            widths=0.5, showmeans=True,
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(REGISTER_COLORS[ne_names[i]])
            patch.set_alpha(0.7)

    ax_dist.set_xticks(range(len(ne_names)))
    ax_dist.set_xticklabels(ne_names)
    ax_dist.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_dist.set_ylabel("Compound Score")
    ax_dist.set_title(f"Sentiment Distribution by Register: {episode_label}")
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)
else:
    st.info("No register data available for this episode.")
    plt.close(fig_dist)

# --- Cross-episode voice fingerprint ---
with st.expander("Voice fingerprint across episodes"):
    st.caption(
        "The interior monologue heuristic is rough and works best on Bloom-centric episodes. "
        "Click 'Compute all' to see register proportions for all 18 episodes."
    )
    if st.button("Compute all", key="compute_fingerprint"):
        progress = st.progress(0)
        all_registers = {}
        for i, (file, label) in enumerate(EPISODE_MAP.items()):
            text = cached_load_episode(file)
            regs = cached_register_split(text)
            all_registers[label] = {r: len(lines) for r, lines in regs.items()}
            progress.progress((i + 1) / len(EPISODE_MAP))

        # Stacked horizontal bar chart
        fig_fp, ax_fp = plt.subplots(figsize=(12, 10))
        labels = list(all_registers.keys())
        y_pos = np.arange(len(labels))

        dialogue_vals = [all_registers[l].get("Dialogue", 0) for l in labels]
        interior_vals = [all_registers[l].get("Interior", 0) for l in labels]
        narration_vals = [all_registers[l].get("Narration", 0) for l in labels]
        totals = [d + i + n for d, i, n in zip(dialogue_vals, interior_vals, narration_vals)]
        # Normalize to percentages
        d_pct = [d / t * 100 if t else 0 for d, t in zip(dialogue_vals, totals)]
        i_pct = [i / t * 100 if t else 0 for i, t in zip(interior_vals, totals)]
        n_pct = [n / t * 100 if t else 0 for n, t in zip(narration_vals, totals)]

        bars_d = ax_fp.barh(y_pos, d_pct, color=REGISTER_COLORS["Dialogue"], label="Dialogue")
        bars_i = ax_fp.barh(y_pos, i_pct, left=d_pct, color=REGISTER_COLORS["Interior"], label="Interior")
        bars_n = ax_fp.barh(
            y_pos, n_pct,
            left=[d + i for d, i in zip(d_pct, i_pct)],
            color=REGISTER_COLORS["Narration"], label="Narration",
        )

        # Highlight selected episode
        sel_idx = labels.index(episode_label) if episode_label in labels else -1
        if sel_idx >= 0:
            for bars in [bars_d, bars_i, bars_n]:
                bars[sel_idx].set_edgecolor("black")
                bars[sel_idx].set_linewidth(2)

        ax_fp.set_yticks(y_pos)
        ax_fp.set_yticklabels(labels, fontsize=9)
        ax_fp.invert_yaxis()
        ax_fp.set_xlabel("Percentage of lines")
        ax_fp.set_title("Voice Fingerprint Across Episodes")
        ax_fp.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_fp)
        plt.close(fig_fp)


# ============================================================================
# Section 3: Death Lexicon & Affective Analysis
# ============================================================================

st.header("3. Death Lexicon & Affective Analysis")

st.markdown(
    "SentiWordNet scores for death-related and proximity words expose the gap between "
    "context-free sentiment and contextual meaning. Look up any word, build custom lexicons, "
    "and trace lexicon density through the episode."
)

# --- Lexicon sentiment scorecard ---
death_table = cached_lexicon_table(list(DEATH_WORDS))
prox_table = cached_lexicon_table(list(PROXIMITY_WORDS))

death_df = pd.DataFrame(death_table)
prox_df = pd.DataFrame(prox_table)

col_death, col_prox = st.columns(2)

with col_death:
    st.subheader("Death Words")
    avg_neg = death_df["Neg Score"].mean()
    st.metric("Average Negativity", f"{avg_neg:.3f}")
    st.dataframe(
        death_df.style.background_gradient(subset=["Neg Score"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )

with col_prox:
    st.subheader("Proximity Words")
    avg_pos = prox_df["Pos Score"].mean()
    zero_count = sum(1 for _, row in prox_df.iterrows() if row["Pos Score"] == 0.0)
    st.metric("Average Positivity", f"{avg_pos:.3f}")
    st.metric("Words scoring 0.000", f"{zero_count} / {len(prox_df)}")
    st.dataframe(
        prox_df.style.background_gradient(subset=["Pos Score"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )

# --- Side-by-side bar charts ---
st.subheader("Lexicon Sentiment Scores")

fig_lex, (ax_death, ax_prox) = plt.subplots(1, 2, figsize=(14, max(6, len(DEATH_WORDS) * 0.3)))

# Death words sorted by neg score
death_sorted = sorted(death_table, key=lambda r: r["Neg Score"], reverse=True)
d_words = [r["Word"] for r in death_sorted]
d_neg = [r["Neg Score"] for r in death_sorted]
d_colors = ["#B22222" if v > 0 else "#B0B0B0" for v in d_neg]
ax_death.barh(range(len(d_words)), d_neg, color=d_colors)
ax_death.set_yticks(range(len(d_words)))
ax_death.set_yticklabels(d_words, fontsize=8)
ax_death.invert_yaxis()
ax_death.set_xlabel("Negative Score")
ax_death.set_title("Death Words (by neg score)")
ax_death.set_xlim(0, 1)

# Proximity words sorted by pos score
prox_sorted = sorted(prox_table, key=lambda r: r["Pos Score"], reverse=True)
p_words = [r["Word"] for r in prox_sorted]
p_pos = [r["Pos Score"] for r in prox_sorted]
p_colors = ["#228B22" if v > 0 else "#B0B0B0" for v in p_pos]
ax_prox.barh(range(len(p_words)), p_pos, color=p_colors)
ax_prox.set_yticks(range(len(p_words)))
ax_prox.set_yticklabels(p_words, fontsize=8)
ax_prox.invert_yaxis()
ax_prox.set_xlabel("Positive Score")
ax_prox.set_title("Proximity Words (by pos score)")
ax_prox.set_xlim(0, 1)

plt.tight_layout()
st.pyplot(fig_lex)
plt.close(fig_lex)

# --- Custom word explorer ---
st.subheader("Custom Word Explorer")

lookup_word = st.text_input("Look up any word in SentiWordNet", value="sleep", key="lookup_word")

if lookup_word.strip():
    result = cached_sentiwordnet_lookup(lookup_word.strip().lower())
    if result:
        pos_s, neg_s, obj_s, syn_name = result
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Positive", f"{pos_s:.3f}")
        lc2.metric("Negative", f"{neg_s:.3f}")
        lc3.metric("Objective", f"{obj_s:.3f}")

        # Comparison bar
        death_avg_neg = death_df["Neg Score"].mean()
        death_avg_pos = death_df["Pos Score"].mean()
        prox_avg_pos = prox_df["Pos Score"].mean()
        prox_avg_neg = prox_df["Neg Score"].mean()

        fig_cmp_word, ax_cmp_word = plt.subplots(figsize=(8, 3))
        bar_labels = ["Your word", "Death avg", "Proximity avg"]
        pos_vals = [pos_s, death_avg_pos, prox_avg_pos]
        neg_vals = [neg_s, death_avg_neg, prox_avg_neg]
        obj_vals = [obj_s, 1 - death_avg_pos - death_avg_neg, 1 - prox_avg_pos - prox_avg_neg]

        y = np.arange(len(bar_labels))
        ax_cmp_word.barh(y, pos_vals, 0.6, label="Positive", color="#228B22")
        ax_cmp_word.barh(y, neg_vals, 0.6, left=pos_vals, label="Negative", color="#B22222")
        ax_cmp_word.barh(
            y, obj_vals, 0.6,
            left=[p + n for p, n in zip(pos_vals, neg_vals)],
            label="Objective", color="#D3D3D3",
        )
        ax_cmp_word.set_yticks(y)
        ax_cmp_word.set_yticklabels(bar_labels)
        ax_cmp_word.set_xlim(0, 1)
        ax_cmp_word.legend(fontsize=8, loc="upper right")
        ax_cmp_word.set_title("Sentiment Composition")
        plt.tight_layout()
        st.pyplot(fig_cmp_word)
        plt.close(fig_cmp_word)

        st.caption(f"Best matching synset: **{syn_name}**")
    else:
        st.warning(f"'{lookup_word}' not found in SentiWordNet.")

# --- Lexicon density through text ---
st.subheader("Lexicon Density Through Text")

normalize_density = st.toggle("Normalize by window size", value=True, key="normalize_density")

death_density = cached_lexicon_density(episode_text, tuple(DEATH_WORDS), window_size)
prox_density = cached_lexicon_density(episode_text, tuple(PROXIMITY_WORDS), window_size)

fig_density, ax_density = plt.subplots(figsize=(14, 5))
if death_density:
    dx, dy = zip(*death_density)
    if not normalize_density:
        dy = [d * window_size for d in dy]
    ax_density.plot(dx, dy, color="#B22222", linewidth=2, label="Death words")
if prox_density:
    px, py = zip(*prox_density)
    if not normalize_density:
        py = [p * window_size for p in py]
    ax_density.plot(px, py, color="#228B22", linewidth=2, label="Proximity words")

ax_density.set_xlabel("Word Position")
ax_density.set_ylabel("Density (rate)" if normalize_density else "Count per window")
ax_density.set_title(f"Lexicon Density: {episode_label} (window={window_size})")
ax_density.legend()
plt.tight_layout()
st.pyplot(fig_density)
plt.close(fig_density)

# --- Build your own lexicon ---
st.subheader("Build Your Own Lexicon")

custom_col1, custom_col2 = st.columns(2)
with custom_col1:
    custom_death = st.text_area(
        "Custom death words (one per line)",
        value="\n".join(DEATH_WORDS),
        height=200,
        key="custom_death",
    )
with custom_col2:
    custom_prox = st.text_area(
        "Custom proximity words (one per line)",
        value="\n".join(PROXIMITY_WORDS),
        height=200,
        key="custom_prox",
    )

if st.button("Recalculate", key="recalculate_lexicon"):
    custom_death_words = [w.strip() for w in custom_death.strip().split("\n") if w.strip()]
    custom_prox_words = [w.strip() for w in custom_prox.strip().split("\n") if w.strip()]

    if custom_death_words or custom_prox_words:
        cd_table = cached_lexicon_table(custom_death_words) if custom_death_words else []
        cp_table = cached_lexicon_table(custom_prox_words) if custom_prox_words else []

        cc1, cc2 = st.columns(2)
        with cc1:
            if cd_table:
                cd_df = pd.DataFrame(cd_table)
                st.metric("Custom Death Avg Negativity", f"{cd_df['Neg Score'].mean():.3f}")
                st.dataframe(
                    cd_df.style.background_gradient(subset=["Neg Score"], cmap="Reds"),
                    use_container_width=True, hide_index=True,
                )
        with cc2:
            if cp_table:
                cp_df = pd.DataFrame(cp_table)
                st.metric("Custom Proximity Avg Positivity", f"{cp_df['Pos Score'].mean():.3f}")
                st.dataframe(
                    cp_df.style.background_gradient(subset=["Pos Score"], cmap="Greens"),
                    use_container_width=True, hide_index=True,
                )

        # Custom bar charts
        fig_custom, (ax_cd, ax_cp) = plt.subplots(1, 2, figsize=(14, max(4, max(len(cd_table), len(cp_table)) * 0.3)))

        if cd_table:
            cd_sorted = sorted(cd_table, key=lambda r: r["Neg Score"], reverse=True)
            cd_w = [r["Word"] for r in cd_sorted]
            cd_n = [r["Neg Score"] for r in cd_sorted]
            cd_c = ["#B22222" if v > 0 else "#B0B0B0" for v in cd_n]
            ax_cd.barh(range(len(cd_w)), cd_n, color=cd_c)
            ax_cd.set_yticks(range(len(cd_w)))
            ax_cd.set_yticklabels(cd_w, fontsize=8)
            ax_cd.invert_yaxis()
            ax_cd.set_xlim(0, 1)
            ax_cd.set_title("Custom Death Words")

        if cp_table:
            cp_sorted = sorted(cp_table, key=lambda r: r["Pos Score"], reverse=True)
            cp_w = [r["Word"] for r in cp_sorted]
            cp_p = [r["Pos Score"] for r in cp_sorted]
            cp_c = ["#228B22" if v > 0 else "#B0B0B0" for v in cp_p]
            ax_cp.barh(range(len(cp_w)), cp_p, color=cp_c)
            ax_cp.set_yticks(range(len(cp_w)))
            ax_cp.set_yticklabels(cp_w, fontsize=8)
            ax_cp.invert_yaxis()
            ax_cp.set_xlim(0, 1)
            ax_cp.set_title("Custom Proximity Words")

        plt.tight_layout()
        st.pyplot(fig_custom)
        plt.close(fig_custom)
    else:
        st.warning("Enter at least one word in either list.")

st.markdown("""
---

**What this week reveals:** VADER sentiment analysis, trained on social media,
struggles with literary language -- scoring funeral euphemisms as positive and missing
the affective weight of words like "rest" and "sleep" in a death context. The register
classifier exposes how Joyce modulates narrative voice across episodes, and
SentiWordNet's context-free scores reveal the fundamental limitation of bag-of-words
sentiment: meaning depends on where words appear, not just what they denote.
""")
