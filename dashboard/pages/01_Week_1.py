"""
Week 01 — Telemachus
Tokenization, concordance, and frequency analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.text import Text
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from dashboard.shared import (
    cached_load_episode,
    cached_profile,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 01 — Telemachus", page_icon="📖", layout="wide")
st.title("Week 01 — Telemachus")
st.caption("Tokenization, Concordance & Frequency")

# --- Sidebar ---
episode_file, episode_label = episode_sidebar(
    default_index=0,
    caption="Week 1: Tokenization, Concordance & Frequency",
    description=(
        "**Week 1** explores tokenization, concordance (keyword-in-context), "
        "and word frequency distributions on Joyce's *Ulysses*."
    ),
)

# Load primary text
episode_text = cached_load_episode(episode_file)

# ============================================================================
# Section 1: Corpus Profile & Comparison
# ============================================================================

st.header("1. Corpus Profile & Comparison")

ep_stats = cached_profile(episode_text, label=episode_label)

# Metrics row for the selected episode
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tokens", f"{ep_stats['total_tokens']:,}")
c2.metric("Type-Token Ratio", f"{ep_stats['type_token_ratio']:.4f}")
c3.metric("Avg Sentence Length", f"{ep_stats['avg_sentence_length']:.1f}")
c4.metric("Hapax Ratio", f"{ep_stats['hapax_ratio']:.4f}")

# --- Multi-chapter radar comparison ---
st.subheader("Profile Radar — Compare Chapters")

# Chapter selector (exclude the currently selected episode — it's always shown)
other_labels = [l for l in EPISODE_LABELS if l != episode_label]
compare_chapters = st.multiselect(
    "Select chapters to compare",
    other_labels,
    default=[],
    help="The selected episode is always shown. Pick additional chapters to overlay.",
)

radar_keys = [
    "type_token_ratio",
    "hapax_ratio",
    "avg_sentence_length",
    "total_types",
    "total_sentences",
]
radar_axis_labels = ["TTR", "Hapax ratio", "Avg sent. len.", "Unique types", "Sentences"]

# Gather profiles for all selected chapters
all_selected = [episode_label] + compare_chapters
all_profiles = {}
for label in all_selected:
    fname = EPISODE_FILES[EPISODE_LABELS.index(label)]
    text = cached_load_episode(fname)
    all_profiles[label] = cached_profile(text, label=label)

# Compute global max per axis across ALL 18 episodes (not just selected)
# so the radar shape is meaningful even with a single chapter
@st.cache_data
def compute_global_max():
    all_vals = {k: [] for k in radar_keys}
    for fname, elabel in EPISODE_MAP.items():
        text = cached_load_episode(fname)
        profile = cached_profile(text, label=elabel)
        for k in radar_keys:
            all_vals[k].append(abs(profile[k]))
    # Use 95th percentile to avoid outliers (e.g. Penelope's 2 sentences)
    # crushing all other chapters to zero
    return {k: max(float(np.percentile(v, 95)), 1e-9) for k, v in all_vals.items()}

global_max = compute_global_max()

angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
angles += angles[:1]  # close polygon

# Reorder tab20: take the bold colors (even indices) first, then the pale ones
_all20 = plt.cm.tab20(range(20))
colors = np.vstack([_all20[0::2], _all20[1::2]])

fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for idx, label in enumerate(all_selected):
    profile = all_profiles[label]
    raw_vals = [profile[k] for k in radar_keys]
    plot_vals = [min(profile[k] / global_max[k], 1.1) for k in radar_keys]
    plot_vals += plot_vals[:1]
    linewidth = 2.5 if label == episode_label else 1.5
    alpha_fill = 0.15 if label == episode_label else 0.05
    ax_radar.plot(angles, plot_vals, "o-", linewidth=linewidth, label=label, color=colors[idx])
    ax_radar.fill(angles, plot_vals, alpha=alpha_fill, color=colors[idx])
    # Annotate each point with the actual value
    for angle, pv, rv in zip(angles[:-1], plot_vals[:-1], raw_vals):
        fmt = f"{rv:.3f}" if rv < 1 else (f"{rv:.1f}" if rv < 100 else f"{rv:,.0f}")
        ax_radar.annotate(
            fmt, (angle, pv),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color=colors[idx], fontweight="bold",
        )

ax_radar.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], radar_axis_labels)
ax_radar.set_ylim(0, 1.1)
ax_radar.set_yticklabels([])  # hide 0-1 scale — actual values are annotated
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=7)
ax_radar.set_title("Chapter Profile Comparison (scaled to fit, actual values labeled)", fontsize=10, pad=20)
st.pyplot(fig_radar)
plt.close(fig_radar)

st.markdown("""
All metrics are computed **per chapter** — each chapter is profiled independently.

- **TTR (Type-Token Ratio):** Unique word types in this chapter divided by total word tokens in this chapter. Higher TTR means more varied vocabulary relative to chapter length — Joyce's later episodes tend to score higher than conventional prose.
- **Hapax Ratio:** Words that appear exactly once in this chapter (hapax legomena) divided by the total unique word types in this chapter. A high hapax ratio means a large share of the chapter's vocabulary is used only once — signaling experimental or rare word use like neologisms, foreign words, and one-off coinages.
- **Avg Sentence Length:** Total tokens in this chapter divided by the number of sentences in this chapter. Longer averages can indicate stream-of-consciousness or complex syntax; shorter averages suggest dialogue-heavy or fragmented prose.
- **Unique Types:** The count of distinct word forms in this chapter. Larger chapters naturally have more types, but comparing chapters of similar length reveals differences in lexical range.
- **Sentences:** Total sentence count in this chapter. Combined with avg sentence length, this reveals whether a chapter is built from many short bursts or fewer, sprawling constructions.
""")

# Profile table for selected chapters
METRIC_KEYS = [
    "total_tokens",
    "total_alpha_tokens",
    "total_types",
    "type_token_ratio",
    "total_sentences",
    "avg_sentence_length",
    "hapax_legomena",
    "hapax_ratio",
]
METRIC_LABELS = [
    "Total tokens",
    "Alpha tokens",
    "Unique types",
    "Type-token ratio",
    "Sentences",
    "Avg sentence length",
    "Hapax legomena",
    "Hapax ratio",
]

table_rows = []
for label in all_selected:
    profile = all_profiles[label]
    row = {"Episode": label}
    for key, mlabel in zip(METRIC_KEYS, METRIC_LABELS):
        v = profile[key]
        row[mlabel] = f"{v:.4f}" if isinstance(v, float) else f"{v:,}"
    table_rows.append(row)

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ============================================================================
# Section 2: Concordance Explorer
# ============================================================================

st.header("2. Concordance Explorer")

# Expanded stopword list: NLTK English stopwords + minimal fiction filler
# (short function words, dialogue tags, and articles only — keep nouns/thematic words)
FICTION_STOPWORDS = {
    "said", "asked", "say", "says",
    "one", "two", "three",
    "mr", "mrs", "sir",
    "got", "get", "let", "go", "put", "see",
    "come", "came", "went", "made", "make",
    "upon", "also", "yet", "much", "well",
    "may", "shall", "could", "would",
    "us", "old", "new",
}
_all_stopwords = set(stopwords.words("english")) | FICTION_STOPWORDS


@st.cache_data
def top_content_words(text, n=10):
    """Return the top N content words after removing stopwords and fiction filler."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    fdist = FreqDist(t for t in tokens if t not in _all_stopwords)
    return [w for w, _ in fdist.most_common(n)]


suggested_keywords = top_content_words(episode_text, n=10)

keyword_col, width_col = st.columns([3, 1])
with keyword_col:
    keyword = st.text_input("Keyword", value=suggested_keywords[0] if suggested_keywords else "")
    btn_rows = [suggested_keywords[:5], suggested_keywords[5:]]
    for row_words in btn_rows:
        btn_cols = st.columns(len(row_words))
        for i, w in enumerate(row_words):
            if btn_cols[i].button(w, key=f"kw_{w}"):
                keyword = w

with width_col:
    context_width = st.slider("Context width (chars)", 40, 120, 80)


@st.cache_data
def get_concordance_lines(text, word, width):
    """Get KWIC concordance lines for a word."""
    tokens = word_tokenize(text)
    t = Text(tokens)
    return t.concordance_list(word, width=width, lines=100)


lines = get_concordance_lines(episode_text, keyword, context_width)
st.metric("Occurrences", len(lines))

if lines:
    kwic_rows = []
    for line in lines:
        left = " ".join(line.left)
        right = " ".join(line.right)
        kwic_rows.append(
            {"Left Context": left, "Keyword": " ".join(line.query), "Right Context": right}
        )
    kwic_df = pd.DataFrame(kwic_rows)
    st.dataframe(kwic_df, use_container_width=True, hide_index=True)
else:
    st.info(f"No occurrences of '{keyword}' found in this episode.")


# Co-occurrence heatmap
st.subheader("Keyword Co-occurrence")


@st.cache_data
def compute_cooccurrence(text, keywords, window=50):
    """Count how often pairs of keywords appear within `window` tokens of each other."""
    tokens = [t.lower() for t in word_tokenize(text)]
    n = len(keywords)
    matrix = np.zeros((n, n), dtype=int)

    positions = {}
    for kw in keywords:
        positions[kw] = [i for i, t in enumerate(tokens) if t == kw.lower()]

    for i in range(n):
        for j in range(i, n):
            count = 0
            for pos_i in positions[keywords[i]]:
                for pos_j in positions[keywords[j]]:
                    if i == j and pos_i == pos_j:
                        continue
                    if abs(pos_i - pos_j) <= window:
                        count += 1
            matrix[i][j] = count
            matrix[j][i] = count
    return matrix


cooc = compute_cooccurrence(episode_text, suggested_keywords)

fig_cooc, ax_cooc = plt.subplots(figsize=(8, 7))
im = ax_cooc.imshow(cooc, cmap="YlOrRd", aspect="auto")
ax_cooc.set_xticks(range(len(suggested_keywords)))
ax_cooc.set_xticklabels(suggested_keywords, rotation=45, ha="right")
ax_cooc.set_yticks(range(len(suggested_keywords)))
ax_cooc.set_yticklabels(suggested_keywords)
for i in range(len(suggested_keywords)):
    for j in range(len(suggested_keywords)):
        ax_cooc.text(j, i, str(cooc[i, j]), ha="center", va="center", fontsize=9)
fig_cooc.colorbar(im, ax=ax_cooc, label="Co-occurrences (within 50 tokens)")
ax_cooc.set_title("Keyword Co-occurrence Heatmap")
plt.tight_layout()
st.pyplot(fig_cooc)
plt.close(fig_cooc)

# Dispersion plot
st.subheader("Keyword Dispersion")

disp_keywords = st.multiselect(
    "Keywords to plot",
    suggested_keywords,
    default=["mother", "sea"],
)


@st.cache_data
def get_token_positions(text):
    """Return lowercased token list."""
    return [t.lower() for t in word_tokenize(text)]


if disp_keywords:
    all_tokens = get_token_positions(episode_text)
    total = len(all_tokens)

    fig_disp, ax_disp = plt.subplots(figsize=(10, max(1.5, 0.6 * len(disp_keywords))))
    colors_disp = plt.cm.Set1(np.linspace(0, 1, len(disp_keywords)))

    for idx, kw in enumerate(disp_keywords):
        positions = [i / total * 100 for i, t in enumerate(all_tokens) if t == kw.lower()]
        ax_disp.scatter(
            positions,
            [idx] * len(positions),
            c=[colors_disp[idx]],
            marker="|",
            s=100,
            linewidths=1.5,
            label=kw,
        )

    ax_disp.set_yticks(range(len(disp_keywords)))
    ax_disp.set_yticklabels(disp_keywords)
    ax_disp.set_xlabel("Position in episode (%)")
    ax_disp.set_title("Keyword Dispersion")
    ax_disp.set_xlim(0, 100)
    ax_disp.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_disp)
    plt.close(fig_disp)

# ============================================================================
# Section 3: Frequency & Zipf's Law
# ============================================================================

st.header("3. Frequency & Zipf's Law")

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
with ctrl1:
    remove_stopwords = st.toggle("Remove stopwords", value=False)
with ctrl2:
    top_n = st.slider("Top N words", 10, 100, 30)
with ctrl3:
    custom_stops = st.text_input("Custom stopwords (comma-separated)", value="")

stop_words = set(stopwords.words("english"))
extra_stops = {w.strip().lower() for w in custom_stops.split(",") if w.strip()}
stop_words = stop_words | extra_stops


@st.cache_data
def compute_freq(text):
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    return FreqDist(alpha_tokens), alpha_tokens


raw_fdist, alpha_tokens = compute_freq(episode_text)

if remove_stopwords:
    filtered = {w: c for w, c in raw_fdist.items() if w not in stop_words}
    display_fdist = FreqDist(filtered)
else:
    display_fdist = raw_fdist

top_words = display_fdist.most_common(top_n)

# Frequency bar chart
fig_freq, ax_freq = plt.subplots(figsize=(12, 5))
words = [w for w, _ in top_words]
counts = [c for _, c in top_words]
bar_colors = ["gray" if w in stop_words else "coral" for w in words]
ax_freq.bar(words, counts, color=bar_colors)
ax_freq.set_ylabel("Frequency")
stopword_status = "stopwords removed" if remove_stopwords else "with stopwords (gray = stopword)"
ax_freq.set_title(f"Top {top_n} Words ({stopword_status})")
ax_freq.tick_params(axis="x", rotation=60)
plt.tight_layout()
st.pyplot(fig_freq)
plt.close(fig_freq)

# Word frequency lookup
st.subheader("Word Frequency Lookup")
lookup_word = st.text_input("Look up a word", value="")

if lookup_word:
    w = lookup_word.strip().lower()
    if w in raw_fdist:
        rank = sorted(raw_fdist, key=raw_fdist.get, reverse=True).index(w) + 1
        count = raw_fdist[w]
        pct = count / len(alpha_tokens) * 100
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Rank", f"#{rank}")
        lc2.metric("Count", f"{count:,}")
        lc3.metric("% of tokens", f"{pct:.3f}%")
    else:
        st.warning(f"'{w}' not found in this episode.")

# Zipf's law plot
st.subheader("Zipf's Law")

other_labels_zipf = [l for l in EPISODE_LABELS if l != episode_label]
zipf_compare = st.multiselect(
    "Compare Zipf curves with other chapters",
    other_labels_zipf,
    default=[],
    key="zipf_compare",
)


@st.cache_data
def compute_zipf_data(text):
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    fdist = FreqDist(alpha_tokens)
    sorted_words = fdist.most_common()
    ranks = list(range(1, len(sorted_words) + 1))
    freqs = [c for _, c in sorted_words]
    word_rank_map = {w: r for r, (w, _) in enumerate(sorted_words, 1)}

    from scipy.stats import linregress

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
    r_squared = r_value**2

    return ranks, freqs, word_rank_map, r_squared, freqs[0]


ranks, freqs, word_rank_map, r_squared, C = compute_zipf_data(episode_text)

st.metric("R-squared", f"{r_squared:.4f}")

fig_zipf, ax_zipf = plt.subplots(figsize=(10, 6))

# Plot primary episode
ax_zipf.loglog(ranks, freqs, ".", markersize=4, alpha=0.7, label=episode_label, color=colors[0])

# Plot comparison chapters
zipf_all = [episode_label] + zipf_compare
for idx, label in enumerate(zipf_compare, start=1):
    fname = EPISODE_FILES[EPISODE_LABELS.index(label)]
    text = cached_load_episode(fname)
    comp_ranks, comp_freqs, _, comp_r2, _ = compute_zipf_data(text)
    ax_zipf.loglog(
        comp_ranks, comp_freqs, ".", markersize=3, alpha=0.5,
        label=f"{label} (R²={comp_r2:.3f})", color=colors[idx],
    )

# Ideal Zipf line — black dashed
ideal = [C / r for r in ranks]
ax_zipf.loglog(ranks, ideal, "k--", alpha=0.5, linewidth=1.5, label="Ideal Zipf (C/r)")

# Highlight lookup word
if lookup_word:
    w = lookup_word.strip().lower()
    if w in word_rank_map:
        wr = word_rank_map[w]
        wf = freqs[wr - 1]
        ax_zipf.plot(wr, wf, "o", color="orange", markersize=10, zorder=5)
        ax_zipf.annotate(
            w,
            (wr, wf),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
            color="orange",
            fontweight="bold",
        )

ax_zipf.set_xlabel("Rank (log)")
ax_zipf.set_ylabel("Frequency (log)")
ax_zipf.set_title(f"Zipf's Law — {episode_label} (R² = {r_squared:.4f})")
ax_zipf.legend(fontsize=7)
plt.tight_layout()
st.pyplot(fig_zipf)
plt.close(fig_zipf)

st.markdown("""
**What Zipf's Law tells you about a text:**

Zipf's Law predicts that in natural language, the frequency of a word is inversely proportional to its rank — the 2nd most common word appears half as often as the 1st, the 3rd a third as often, and so on. The ideal line on the plot represents this perfect `frequency = C / rank` relationship. When a text closely follows Zipf's Law (high R²), it suggests a "typical" balance between a small core of high-frequency function words and a long tail of rare content words — the kind of distribution that emerges from ordinary, communicative language use.

**What deviation means:**

When a chapter's curve bows away from the ideal line, it means the word frequency distribution is less predictable. Common causes include:

- **Excess repetition:** Stylistic or musical repetition (as in *Sirens* or *Circe*) inflates certain mid-rank words beyond what Zipf predicts, pulling the curve above the line.
- **Excess vocabulary:** Chapters with heavy neologism, technical language, or multilingual passages (like *Oxen of the Sun*, R²=0.880) pack more unique words into the tail, flattening the curve.
- **Structural experiments:** When Joyce abandons conventional prose — catechism format in *Ithaca*, unpunctuated monologue in *Penelope*, dramatic script in *Circe* — the resulting word distribution no longer resembles the statistical "norm" of natural language.

**Is it entropy?** Not directly, but they're related. A text that follows Zipf's Law has a predictable information structure — you can guess roughly how often any word will appear given its rank. Lower R² means higher surprise: the text's vocabulary is distributed in a less predictable way, which corresponds loosely to higher entropy. Shannon entropy and Zipf's exponent are mathematically linked — a steeper Zipf slope implies lower entropy (more predictable), while a flatter slope implies higher entropy (more uniform word usage, more surprise per word).
""")

