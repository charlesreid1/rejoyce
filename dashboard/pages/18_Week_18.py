"""
Week 18 — Penelope
Text segmentation, topic modeling, and return to tokenization.
"""

import contextlib
import io
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
from nltk.tokenize import word_tokenize, sent_tokenize, TextTilingTokenizer
from nltk.probability import FreqDist

for resource in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(resource, quiet=True)

from week18.week18_penelope import (
    prepare_for_texttiling,
    vocabulary_shift_segmentation,
    STOP_WORDS,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(
    page_title="Week 18 — Penelope",
    page_icon="📖",
    layout="wide",
)
st.title("Week 18 — Penelope")
st.caption("Text Segmentation, Topic Modeling & Return to Tokenization")


# ============================================================================
# Constants
# ============================================================================

TOPIC_SEEDS = {
    "Gibraltar/girlhood": {
        "gibraltar", "mulvey", "girl", "garden", "flower", "mountain",
        "spanish", "moor", "sun", "rock",
    },
    "Bloom/marriage": {
        "bloom", "leopold", "poldy", "husband", "marry", "howth",
        "proposal", "eccles", "house", "home",
    },
    "Boylan/desire": {
        "boylan", "blazes", "afternoon", "bed", "kiss", "love",
        "want", "body", "man", "handsome",
    },
    "Body/physicality": {
        "body", "breast", "blood", "skin", "hair", "dress",
        "clothes", "bath", "perfume", "beauty",
    },
    "Other women/judgment": {
        "woman", "women", "mrs", "jealous", "pretty", "hat",
        "fashion", "better", "worse", "dress",
    },
    "Memory/reflection": {
        "remember", "time", "years", "ago", "first", "always",
        "never", "used", "once", "old",
    },
}

TOPIC_COLORS = {
    "Gibraltar/girlhood": "#E07A5F",
    "Bloom/marriage": "#4A90D9",
    "Boylan/desire": "#9B59B6",
    "Body/physicality": "#81B29A",
    "Other women/judgment": "#F2CC8F",
    "Memory/reflection": "#264653",
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
def cached_segment(text, chunk_size):
    """Segment text using TextTiling, falling back to vocabulary shift."""
    tokens = word_tokenize(text)
    paragraphs = []
    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i:i + chunk_size])
        paragraphs.append(chunk)
    prepared = "\n\n".join(paragraphs)

    try:
        tiler = TextTilingTokenizer(w=20, k=10)
        segments = tiler.tokenize(prepared)
        tt_boundaries = []
        pos = 0
        for seg in segments[:-1]:
            pos += len(word_tokenize(seg))
            tt_boundaries.append(pos)
    except Exception:
        segments = vocabulary_shift_segmentation(text)
        tt_boundaries = []
        pos = 0
        for seg in segments[:-1]:
            pos += len(word_tokenize(seg))
            tt_boundaries.append(pos)

    # Joyce's 8 sentences (approximate)
    total_words = len(tokens)
    joyce_boundaries = [i * total_words // 8 for i in range(1, 8)]

    return segments, tt_boundaries, joyce_boundaries, total_words


@st.cache_data
def cached_jaccard_curve(text, window_size=200):
    """Compute Jaccard distance between adjacent sliding windows."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    distances = []
    positions = []
    step = window_size // 2

    for i in range(0, len(tokens) - 2 * window_size, step):
        w1 = set(tokens[i:i + window_size]) - STOP_WORDS
        w2 = set(tokens[i + window_size:i + 2 * window_size]) - STOP_WORDS
        union = w1 | w2
        if union:
            jaccard = 1.0 - len(w1 & w2) / len(union)
        else:
            jaccard = 0.0
        distances.append(jaccard)
        positions.append(i + window_size)

    return positions, distances


@st.cache_data
def cached_topic_proportions(text, n_windows):
    """Compute topic proportions per window using keyword matching."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    total = len(tokens)
    window_size = total // n_windows

    # Overall proportions
    overall_counts = {topic: 0 for topic in TOPIC_SEEDS}
    for token in tokens:
        for topic, seeds in TOPIC_SEEDS.items():
            if token in seeds:
                overall_counts[topic] += 1

    overall_total = sum(overall_counts.values())
    overall_proportions = {}
    for topic, count in overall_counts.items():
        overall_proportions[topic] = count / overall_total if overall_total else 0

    # Per-window proportions
    window_proportions = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else total
        window_tokens = tokens[start:end]

        counts = {topic: 0 for topic in TOPIC_SEEDS}
        for token in window_tokens:
            for topic, seeds in TOPIC_SEEDS.items():
                if token in seeds:
                    counts[topic] += 1

        win_total = sum(counts.values())
        props = {}
        for topic, count in counts.items():
            props[topic] = count / win_total if win_total else 0
        window_proportions.append(props)

    return overall_proportions, window_proportions


@st.cache_data
def cached_episode_metrics(episode_file):
    """Compute basic metrics for an episode: tokens, types, ttr, hapax, avg_sent_len."""
    text = cached_load_episode(episode_file)
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    types = set(alpha_tokens)
    freq = Counter(alpha_tokens)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    sentences = sent_tokenize(text)
    avg_sent_len = len(alpha_tokens) / len(sentences) if sentences else 0

    return {
        "total_tokens": len(alpha_tokens),
        "total_types": len(types),
        "ttr": len(types) / len(alpha_tokens) if alpha_tokens else 0,
        "hapax_count": hapax,
        "hapax_ratio": hapax / len(types) if types else 0,
        "avg_sent_len": avg_sent_len,
        "num_sentences": len(sentences),
    }


@st.cache_data
def cached_particle_positions(text, particles):
    """Track positions of structural particles through the text."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    positions = {}
    for particle in particles:
        p_lower = particle.lower()
        positions[particle] = [i for i, t in enumerate(tokens) if t == p_lower]
    return positions, len(tokens)


@st.cache_data
def cached_rolling_frequency(text, particles, window_size):
    """Compute rolling frequency of particles across the text."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    total = len(tokens)
    results = {p: [] for p in particles}
    x_positions = []

    step = window_size // 4 if window_size > 4 else 1
    for i in range(0, total - window_size, step):
        window = tokens[i:i + window_size]
        x_positions.append(i + window_size // 2)
        for particle in particles:
            count = sum(1 for t in window if t == particle.lower())
            results[particle].append(count / window_size)

    return x_positions, results


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=17,  # Penelope
    caption="Week 18: Segmentation, Topic Modeling & Tokenization",
)

with st.sidebar:
    chunk_size = st.slider(
        "Chunk size for TextTiling",
        500, 2000, 1000, step=100,
        key="chunk_size",
        help="Number of words per artificial paragraph for TextTiling segmentation.",
    )
    st.divider()
    st.markdown(
        "**Penelope** is Molly Bloom's unpunctuated interior monologue — "
        "eight enormous sentences with almost no punctuation, streaming through "
        "memory, desire, and judgment. Without sentence boundaries, segmentation "
        "must rely on vocabulary shifts rather than syntax."
    )

# Load data
episode_text = cached_load_episode(episode_file)


# ============================================================================
# Section 1: Text Segmentation
# ============================================================================

st.header("1. Text Segmentation")

st.markdown(
    "Penelope's unpunctuated stream resists conventional sentence-based analysis. "
    "**TextTiling** detects topic boundaries by measuring vocabulary shifts between "
    "adjacent blocks — where the words change, a new topic begins. We compare these "
    "algorithmically detected boundaries against Joyce's own 8-sentence structure "
    "to see how well computational methods can recover the author's divisions."
)

segments, tt_boundaries, joyce_boundaries, total_words = cached_segment(
    episode_text, chunk_size
)

# --- Metrics row ---
# Compute overlap score: how many TextTiling boundaries fall within 5% of a Joyce boundary
tolerance = total_words * 0.05
overlap = 0
for tb in tt_boundaries:
    for jb in joyce_boundaries:
        if abs(tb - jb) < tolerance:
            overlap += 1
            break

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Words", f"{total_words:,}")
m2.metric("TextTiling Boundaries", len(tt_boundaries))
m3.metric("Joyce's 8 Sentences", f"7 boundaries")
m4.metric("Overlap Score", f"{overlap}/{len(tt_boundaries)}")

# --- Dual segmentation plot ---
st.subheader("Dual Segmentation: Joyce vs. TextTiling")

fig_seg, (ax_joyce, ax_tt) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

# Top panel: Joyce's boundaries
ax_joyce.axhline(y=0.5, color="#CCCCCC", linewidth=0.5)
for jb in joyce_boundaries:
    ax_joyce.axvline(x=jb, color="#4A90D9", linewidth=2, alpha=0.8)
ax_joyce.set_ylabel("Joyce's\n8 Sentences", fontsize=9)
ax_joyce.set_yticks([])
ax_joyce.set_title("Segmentation Boundaries Compared")
ax_joyce.set_xlim(0, total_words)

# Label Joyce segments
for i in range(8):
    start = joyce_boundaries[i - 1] if i > 0 else 0
    end = joyce_boundaries[i] if i < 7 else total_words
    mid = (start + end) / 2
    ax_joyce.text(mid, 0.5, f"S{i+1}", ha="center", va="center", fontsize=8, color="#4A90D9")

# Bottom panel: TextTiling boundaries
ax_tt.axhline(y=0.5, color="#CCCCCC", linewidth=0.5)
for tb in tt_boundaries:
    ax_tt.axvline(x=tb, color="#E07A5F", linewidth=2, alpha=0.8)
ax_tt.set_ylabel("TextTiling\nBoundaries", fontsize=9)
ax_tt.set_yticks([])
ax_tt.set_xlabel("Word Position")
ax_tt.set_xlim(0, total_words)

# Highlight overlapping boundaries
for tb in tt_boundaries:
    for jb in joyce_boundaries:
        if abs(tb - jb) < tolerance:
            ax_joyce.axvline(x=jb, color="#81B29A", linewidth=3, alpha=0.4)
            ax_tt.axvline(x=tb, color="#81B29A", linewidth=3, alpha=0.4)

plt.tight_layout()
st.pyplot(fig_seg)
plt.close(fig_seg)

st.caption(
    "Blue = Joyce's sentence boundaries, Red = TextTiling boundaries. "
    "Green highlights mark overlaps (within 5% tolerance)."
)

# --- Vocabulary shift curve ---
st.subheader("Vocabulary Shift Curve (Jaccard Distance)")

st.markdown(
    "The Jaccard distance between adjacent sliding windows measures how much "
    "vocabulary changes from one region to the next. Peaks indicate topic shifts — "
    "the raw signal that TextTiling uses to place boundaries."
)

positions, distances = cached_jaccard_curve(episode_text, window_size=200)

if positions:
    fig_jac, ax_jac = plt.subplots(figsize=(14, 4))
    ax_jac.plot(positions, distances, color="#4A90D9", alpha=0.7, linewidth=1)
    ax_jac.fill_between(positions, distances, alpha=0.15, color="#4A90D9")

    # Mark Joyce boundaries
    for jb in joyce_boundaries:
        ax_jac.axvline(x=jb, color="#E07A5F", linestyle="--", alpha=0.5, linewidth=1)

    # Mark TextTiling boundaries
    for tb in tt_boundaries:
        ax_jac.axvline(x=tb, color="#81B29A", linestyle=":", alpha=0.5, linewidth=1)

    ax_jac.set_xlabel("Word Position")
    ax_jac.set_ylabel("Jaccard Distance")
    ax_jac.set_title("Vocabulary Shift Between Adjacent Windows")
    ax_jac.legend(
        ["Jaccard distance", "Joyce boundary", "TextTiling boundary"],
        fontsize=8,
    )
    plt.tight_layout()
    st.pyplot(fig_jac)
    plt.close(fig_jac)

# --- Segment reader ---
st.subheader("Segment Reader")

if segments:
    seg_options = [f"Segment {i+1} ({len(word_tokenize(seg))} words)" for i, seg in enumerate(segments)]
    selected_seg = st.selectbox("Select a TextTiling segment", seg_options, key="seg_reader")
    seg_idx = seg_options.index(selected_seg)

    seg_text = segments[seg_idx]
    seg_tokens = word_tokenize(seg_text)
    seg_alpha = [t.lower() for t in seg_tokens if t.isalpha() and t.lower() not in STOP_WORDS]
    seg_freq = Counter(seg_alpha).most_common(10)

    st.markdown(f"**Top keywords:** {', '.join(f'{w} ({c})' for w, c in seg_freq)}")

    with st.expander("View segment text"):
        st.write(seg_text[:2000] + ("..." if len(seg_text) > 2000 else ""))
else:
    st.info("No segments produced. Try adjusting the chunk size.")


# ============================================================================
# Section 2: Topic Modeling
# ============================================================================

st.header("2. Topic Modeling")

is_penelope = episode_file == "18penelope.txt"

st.markdown(
    "Seed-based topic modeling tracks six thematic strands through Penelope using "
    "keyword matching. Each topic is defined by a set of seed words drawn from "
    "the episode's major preoccupations: Gibraltar and girlhood, Bloom and marriage, "
    "Boylan and desire, the body, other women, and memory."
)

if not is_penelope:
    st.info(
        "The Penelope topic seeds (Gibraltar/girlhood, Boylan/desire, etc.) are specific "
        "to Molly Bloom's monologue and will not produce meaningful results for other episodes. "
        "Select **18 — Penelope** to explore topic modeling."
    )

with st.sidebar:
    n_windows = st.slider(
        "Number of topic windows",
        4, 20, 6,
        key="n_topic_windows",
        help="How many windows to divide the text into for topic tracking.",
    )

overall_props, window_props = cached_topic_proportions(episode_text, n_windows)

# --- Topic proportion pie chart ---
st.subheader("Overall Topic Proportions")

fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
topics = list(TOPIC_SEEDS.keys())
sizes = [overall_props[t] for t in topics]
colors = [TOPIC_COLORS[t] for t in topics]

# Filter out zero-proportion topics
nonzero = [(t, s, c) for t, s, c in zip(topics, sizes, colors) if s > 0]
if nonzero:
    pie_topics, pie_sizes, pie_colors = zip(*nonzero)
    wedges, texts, autotexts = ax_pie.pie(
        pie_sizes,
        labels=pie_topics,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
    )
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(7)
    centre = plt.Circle((0, 0), 0.45, fc="white")
    ax_pie.add_artist(centre)
ax_pie.set_title("Topic Proportions (Keyword Matching)")
plt.tight_layout()
st.pyplot(fig_pie)
plt.close(fig_pie)

# --- Stacked area chart ---
st.subheader("Topic Flow Across Windows")

fig_area, ax_area = plt.subplots(figsize=(14, 5))
x = np.arange(1, n_windows + 1)

# Build arrays for stacked area
topic_arrays = {}
for topic in topics:
    topic_arrays[topic] = np.array([wp.get(topic, 0) for wp in window_props])

y_stack = np.row_stack([topic_arrays[t] for t in topics])
ax_area.stackplot(
    x, y_stack,
    labels=topics,
    colors=[TOPIC_COLORS[t] for t in topics],
    alpha=0.8,
)
ax_area.set_xlabel("Window")
ax_area.set_ylabel("Topic Proportion")
ax_area.set_title("Topic Flow Through Penelope")
ax_area.set_xticks(x)
ax_area.legend(loc="upper left", fontsize=7, ncol=2)
ax_area.set_xlim(1, n_windows)
ax_area.set_ylim(0, 1)
plt.tight_layout()
st.pyplot(fig_area)
plt.close(fig_area)

# --- Topic heatmap ---
st.subheader("Topic Heatmap")

heatmap_data = np.zeros((len(topics), n_windows))
for j, topic in enumerate(topics):
    for i in range(n_windows):
        heatmap_data[j][i] = window_props[i].get(topic, 0)

fig_heat, ax_heat = plt.subplots(figsize=(max(8, n_windows * 0.8), max(4, len(topics) * 0.6)))
im = ax_heat.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
ax_heat.set_xticks(range(n_windows))
ax_heat.set_xticklabels([f"W{i+1}" for i in range(n_windows)], fontsize=9)
ax_heat.set_yticks(range(len(topics)))
ax_heat.set_yticklabels(topics, fontsize=9)
ax_heat.set_xlabel("Window")
ax_heat.set_title("Topic Intensity Heatmap")

# Annotate cells
for j in range(len(topics)):
    for i in range(n_windows):
        val = heatmap_data[j][i]
        color = "white" if val > 0.3 else "black"
        ax_heat.text(i, j, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

fig_heat.colorbar(im, ax=ax_heat, label="Proportion")
plt.tight_layout()
st.pyplot(fig_heat)
plt.close(fig_heat)


# ============================================================================
# Section 3: Return to Tokenization
# ============================================================================

st.header("3. Return to Tokenization")

st.markdown(
    "We began in Week 1 with Telemachus and basic tokenization metrics. Now we return "
    "to the same measurements on the selected episode to see how its language compares "
    "to Penelope — Joyce's final episode. If Penelope is selected, we compare against "
    "Calypso — Bloom's opening episode."
)

# --- Selected episode vs comparison ---
selected_metrics = cached_episode_metrics(episode_file)

# Compare against Penelope, unless Penelope is selected — then compare against Telemachus
if episode_file == "18penelope.txt":
    compare_file = "04calypso.txt"
    compare_name = "Calypso"
else:
    compare_file = "18penelope.txt"
    compare_name = "Penelope"

compare_metrics = cached_episode_metrics(compare_file)
selected_name = episode_label.split(" — ")[1]

# Metrics row with deltas
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric(
    "Total Tokens",
    f"{selected_metrics['total_tokens']:,}",
    delta=f"{selected_metrics['total_tokens'] - compare_metrics['total_tokens']:+,}",
)
mc2.metric(
    "Total Types",
    f"{selected_metrics['total_types']:,}",
    delta=f"{selected_metrics['total_types'] - compare_metrics['total_types']:+,}",
)
mc3.metric(
    "TTR",
    f"{selected_metrics['ttr']:.4f}",
    delta=f"{selected_metrics['ttr'] - compare_metrics['ttr']:+.4f}",
)
mc4.metric(
    "Hapax Ratio",
    f"{selected_metrics['hapax_ratio']:.4f}",
    delta=f"{selected_metrics['hapax_ratio'] - compare_metrics['hapax_ratio']:+.4f}",
)
mc5.metric(
    "Avg Sentence Length",
    f"{selected_metrics['avg_sent_len']:.0f}",
    delta=f"{selected_metrics['avg_sent_len'] - compare_metrics['avg_sent_len']:+.0f}",
)

st.caption(f"Deltas shown relative to {compare_name}.")

# --- Full comparison table ---
st.subheader(f"{selected_name} vs. {compare_name}")

comparison_rows = []
metrics_labels = {
    "total_tokens": "Total Tokens",
    "total_types": "Total Types",
    "ttr": "Type-Token Ratio",
    "hapax_count": "Hapax Legomena",
    "hapax_ratio": "Hapax Ratio",
    "avg_sent_len": "Avg Sentence Length",
    "num_sentences": "Number of Sentences",
}

for key, label in metrics_labels.items():
    sel_val = selected_metrics[key]
    cmp_val = compare_metrics[key]
    if isinstance(sel_val, float):
        comparison_rows.append({
            "Metric": label,
            selected_name: f"{sel_val:.4f}",
            compare_name: f"{cmp_val:.4f}",
            "Delta": f"{sel_val - cmp_val:+.4f}",
        })
    else:
        comparison_rows.append({
            "Metric": label,
            selected_name: f"{sel_val:,}",
            compare_name: f"{cmp_val:,}",
            "Delta": f"{sel_val - cmp_val:+,}",
        })

st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

# --- Structural particles scatter plot ---
st.subheader("Structural Particles: Position Scatter")

st.markdown(
    "Without punctuation, Penelope relies on structural particles like *yes*, *no*, "
    "and *and* to create rhythm and emphasis. Tracking their positions through the "
    "text reveals Molly's rhetorical patterns — the famous 'yes' that opens and "
    "closes the episode, the 'and' that propels her forward."
)

default_particles = ["yes", "no", "and"]
particle_positions, total_token_count = cached_particle_positions(
    episode_text, default_particles
)

fig_scatter, ax_scatter = plt.subplots(figsize=(14, 3))
scatter_colors = ["#4A90D9", "#E07A5F", "#81B29A"]

for i, particle in enumerate(default_particles):
    positions = particle_positions.get(particle, [])
    if positions:
        ax_scatter.scatter(
            positions, [i] * len(positions),
            c=scatter_colors[i % len(scatter_colors)],
            s=8, alpha=0.5, label=f'"{particle}" ({len(positions)})',
        )

ax_scatter.set_yticks(range(len(default_particles)))
ax_scatter.set_yticklabels(default_particles, fontsize=10)
ax_scatter.set_xlabel("Token Position")
ax_scatter.set_title("Structural Particle Positions Through the Episode")
ax_scatter.set_xlim(0, total_token_count)
ax_scatter.legend(fontsize=8, loc="upper right")
plt.tight_layout()
st.pyplot(fig_scatter)
plt.close(fig_scatter)

# --- Rolling frequency chart ---
st.subheader("Rolling Frequency of Structural Particles")

with st.sidebar:
    rolling_window = st.slider(
        "Rolling frequency window size",
        100, 1000, 500, step=50,
        key="rolling_window",
        help="Window size (in tokens) for computing rolling frequency.",
    )

all_particles = list(default_particles)

# Custom particle tracker
custom_particle = st.text_input(
    "Add a custom word to track",
    value="",
    key="custom_particle",
)
if custom_particle.strip():
    all_particles.append(custom_particle.strip().lower())

x_positions, rolling_results = cached_rolling_frequency(
    episode_text, all_particles, rolling_window
)

if x_positions:
    fig_rolling, ax_rolling = plt.subplots(figsize=(14, 5))
    rolling_colors = ["#4A90D9", "#E07A5F", "#81B29A", "#9B59B6", "#F2CC8F"]

    for i, particle in enumerate(all_particles):
        freqs = rolling_results.get(particle, [])
        if freqs:
            ax_rolling.plot(
                x_positions, freqs,
                color=rolling_colors[i % len(rolling_colors)],
                alpha=0.8, linewidth=1.5,
                label=f'"{particle}"',
            )

    # Mark Joyce boundaries
    for jb in joyce_boundaries:
        ax_rolling.axvline(x=jb, color="#CCCCCC", linestyle="--", alpha=0.4, linewidth=0.8)

    ax_rolling.set_xlabel("Token Position")
    ax_rolling.set_ylabel("Frequency (per token)")
    ax_rolling.set_title(f"Rolling Frequency (window={rolling_window})")
    ax_rolling.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_rolling)
    plt.close(fig_rolling)

    st.caption(
        "Dashed grey lines mark Joyce's approximate sentence boundaries."
    )


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Penelope brings us full circle — from the basic tokenization
of Telemachus in Week 1 to the same measurements applied to Joyce's most radical prose.
TextTiling's vocabulary-shift boundaries partially recover Joyce's 8-sentence structure,
confirming that even without punctuation the thematic fabric shifts at recognizable points.
Topic modeling traces Molly's wandering mind through six recurring preoccupations, showing
how Gibraltar memories, desire for Boylan, and reflections on Bloom weave in and out like
motifs in a fugue. And the return to tokenization metrics reveals the paradox of Penelope:
dramatically different in structure (8 sentences vs. hundreds, no punctuation, enormous
sentence lengths) yet surprisingly consistent in vocabulary richness — as if Joyce's
linguistic fingerprint persists regardless of the formal constraints he imposes. The
"yes" that opens and closes the episode is not just Molly's affirmation but a return to
the beginning, completing the circle that Ulysses traces from Telemachus to Penelope.
""")
