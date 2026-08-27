"""
Week 13 — Nausicaa
Stylometry and authorship attribution: Gerty vs. Bloom, Burrows' Delta, cliche detection.
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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg

for resource in ["punkt", "punkt_tab", "gutenberg"]:
    nltk.download(resource, quiet=True)

from week13.week13_nausicaa import (
    split_nausicaa,
    stylometric_profile,
    burrows_delta,
    extract_ngrams,
    FUNCTION_WORDS,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 13 — Nausicaa", page_icon="📖", layout="wide")
st.title("Week 13 — Nausicaa")
st.caption("Stylometry & Authorship Attribution")


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
def cached_split_nausicaa(text):
    """Split Nausicaa into Gerty and Bloom halves."""
    gerty_text, bloom_text, split_idx = split_nausicaa(text)
    return gerty_text, bloom_text, split_idx


@st.cache_data
def cached_stylometric_profile(text, label):
    """Compute stylometric profile with caching."""
    return suppress_stdout(stylometric_profile, text, label)


@st.cache_data
def cached_burrows_delta(test_profile_tuple, corpus_profiles_tuple):
    """Compute Burrows' Delta with caching.

    Profiles are passed as tuples-of-items for hashability, then
    reconstructed into dicts before calling the library function.
    """
    test_profile = {k: dict(v) if isinstance(v, tuple) else v for k, v in test_profile_tuple}
    corpus_profiles = [
        {k: dict(v) if isinstance(v, tuple) else v for k, v in items}
        for label, items in corpus_profiles_tuple
    ]
    return suppress_stdout(burrows_delta, test_profile, corpus_profiles)


@st.cache_data
def cached_extract_ngrams(text, n):
    """Extract character n-grams with caching."""
    return suppress_stdout(extract_ngrams, text, (n, n))


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=12,  # Nausicaa
    caption="Week 13: Stylometry & Authorship Attribution",
    description=(
        "*Nausicaa is Joyce's great pastiche — the first half channels Gerty "
        "MacDowell through the language of sentimental romance fiction, while the "
        "second half returns to Bloom's characteristic interior monologue. "
        "Stylometry lets us measure the distance between these voices and attribute "
        "them against a reference corpus.*"
    ),
)

is_nausicaa = episode_file == "13nausicaa.txt"

# Load data
episode_text = cached_load_episode(episode_file)


# ============================================================================
# Section 1: Gerty vs. Bloom Stylometric Split
# ============================================================================

st.header("1. Gerty vs. Bloom Stylometric Split")

st.markdown(
    "Joyce divides Nausicaa into two distinct halves: Gerty MacDowell's section, "
    "written in the style of sentimental romance fiction, and Bloom's interior "
    "monologue. We split the text and compute a stylometric profile for each half, "
    "measuring type-token ratio (vocabulary richness), Yule's K (vocabulary "
    "concentration), average sentence length, and function word frequencies."
)

if not is_nausicaa:
    st.info(
        "The Gerty/Bloom split analysis is specific to Nausicaa — the only episode "
        "with a clear stylistic bifurcation. Select **13 — Nausicaa** to explore "
        "this section in full. Showing whole-episode profile below."
    )
    full_profile = cached_stylometric_profile(episode_text, episode_label)

    profile_rows = []
    for key, val in full_profile.items():
        if key == "fw_freqs":
            continue
        profile_rows.append({"Metric": key, episode_label: f"{val:.4f}" if isinstance(val, float) else str(val)})
    st.dataframe(pd.DataFrame(profile_rows), width="stretch", hide_index=True)

else:
    gerty_text, bloom_text, split_idx = cached_split_nausicaa(episode_text)
    gerty_profile = cached_stylometric_profile(gerty_text, "Gerty")
    bloom_profile = cached_stylometric_profile(bloom_text, "Bloom")

    # --- Metrics row: deltas ---
    ttr_delta = gerty_profile.get("ttr", 0) - bloom_profile.get("ttr", 0)
    yule_delta = gerty_profile.get("yule_k", 0) - bloom_profile.get("yule_k", 0)
    sent_len_delta = gerty_profile.get("mean_sent_len", 0) - bloom_profile.get("mean_sent_len", 0)

    # Exclamation rate: count exclamation marks per sentence
    gerty_sents = sent_tokenize(gerty_text)
    bloom_sents = sent_tokenize(bloom_text)
    gerty_excl_rate = sum(1 for s in gerty_sents if "!" in s) / max(len(gerty_sents), 1)
    bloom_excl_rate = sum(1 for s in bloom_sents if "!" in s) / max(len(bloom_sents), 1)
    excl_delta = gerty_excl_rate - bloom_excl_rate

    metrics = [
        ("TTR", f"{gerty_profile.get('ttr', 0):.4f}", f"{bloom_profile.get('ttr', 0):.4f}", f"{ttr_delta:+.4f}"),
        ("Yule's K", f"{gerty_profile.get('yule_k', 0):.2f}", f"{bloom_profile.get('yule_k', 0):.2f}", f"{yule_delta:+.2f}"),
        ("Avg Sent Len", f"{gerty_profile.get('mean_sent_len', 0):.1f}", f"{bloom_profile.get('mean_sent_len', 0):.1f}", f"{sent_len_delta:+.1f}"),
        ("Excl Rate", f"{gerty_excl_rate:.2%}", f"{bloom_excl_rate:.2%}", f"{excl_delta:+.2%}"),
    ]
    for label, gerty_val, bloom_val, delta_val in metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{label} (Gerty)", gerty_val)
        c2.metric(f"{label} (Bloom)", bloom_val)
        c3.metric(f"{label} (Delta)", delta_val, delta=delta_val)

    # --- Profile comparison dataframe ---
    st.subheader("Profile Comparison")

    all_keys = [k for k in gerty_profile if k != "fw_freqs"]
    profile_rows = []
    for key in all_keys:
        g_val = gerty_profile.get(key, 0)
        b_val = bloom_profile.get(key, 0)
        g_str = f"{g_val:.4f}" if isinstance(g_val, float) else str(g_val)
        b_str = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
        if isinstance(g_val, (int, float)) and isinstance(b_val, (int, float)):
            diff = g_val - b_val
            d_str = f"{diff:+.4f}" if isinstance(diff, float) else f"{diff:+d}"
        else:
            d_str = "—"
        profile_rows.append({"Metric": key, "Gerty": g_str, "Bloom": b_str, "Delta": d_str})

    st.dataframe(pd.DataFrame(profile_rows), width="stretch", hide_index=True)

    # --- 4-panel matplotlib figure ---
    st.subheader("Stylometric Visualization")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Sentence length distributions
    ax1 = axes[0, 0]
    gerty_sent_lens = [len(word_tokenize(s)) for s in gerty_sents]
    bloom_sent_lens = [len(word_tokenize(s)) for s in bloom_sents]
    bins = np.linspace(0, max(max(gerty_sent_lens, default=1), max(bloom_sent_lens, default=1)), 30)
    ax1.hist(gerty_sent_lens, bins=bins, alpha=0.6, color="#E07A5F", label="Gerty", density=True)
    ax1.hist(bloom_sent_lens, bins=bins, alpha=0.6, color="#4A90D9", label="Bloom", density=True)
    ax1.set_xlabel("Sentence Length (tokens)")
    ax1.set_ylabel("Density")
    ax1.set_title("Sentence Length Distribution")
    ax1.legend()

    # Panel 2: Vocabulary richness comparison
    ax2 = axes[0, 1]
    richness_metrics = ["ttr", "hapax_ratio"]
    richness_labels = ["Type-Token Ratio", "Hapax Ratio"]
    gerty_vals = [gerty_profile.get(m, 0) for m in richness_metrics]
    bloom_vals = [bloom_profile.get(m, 0) for m in richness_metrics]
    x = np.arange(len(richness_labels))
    width = 0.35
    ax2.bar(x - width / 2, gerty_vals, width, label="Gerty", color="#E07A5F", alpha=0.8)
    ax2.bar(x + width / 2, bloom_vals, width, label="Bloom", color="#4A90D9", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(richness_labels)
    ax2.set_ylabel("Ratio")
    ax2.set_title("Vocabulary Richness")
    ax2.legend()

    # Panel 3: Punctuation comparison
    ax3 = axes[1, 0]
    punct_marks = ["!", "?", ";", "—", "..."]
    gerty_tokens = gerty_text
    bloom_tokens = bloom_text
    gerty_punct = [gerty_tokens.count(p) / max(len(gerty_sents), 1) for p in punct_marks]
    bloom_punct = [bloom_tokens.count(p) / max(len(bloom_sents), 1) for p in punct_marks]
    x3 = np.arange(len(punct_marks))
    ax3.bar(x3 - width / 2, gerty_punct, width, label="Gerty", color="#E07A5F", alpha=0.8)
    ax3.bar(x3 + width / 2, bloom_punct, width, label="Bloom", color="#4A90D9", alpha=0.8)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(punct_marks)
    ax3.set_ylabel("Per Sentence")
    ax3.set_title("Punctuation Frequency")
    ax3.legend()

    # Panel 4: Function word frequency differences
    ax4 = axes[1, 1]
    gerty_fw = gerty_profile.get("fw_freqs", {})
    bloom_fw = bloom_profile.get("fw_freqs", {})
    # Compute absolute differences, take top 15
    all_fw = set(list(gerty_fw.keys()) + list(bloom_fw.keys()))
    fw_diffs = []
    for fw in all_fw:
        g = gerty_fw.get(fw, 0)
        b = bloom_fw.get(fw, 0)
        fw_diffs.append((fw, g - b, g, b))
    fw_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_fw = fw_diffs[:15]
    if top_fw:
        fw_labels = [t[0] for t in top_fw]
        fw_vals = [t[1] for t in top_fw]
        colors = ["#E07A5F" if v > 0 else "#4A90D9" for v in fw_vals]
        ax4.barh(range(len(fw_labels)), fw_vals, color=colors)
        ax4.set_yticks(range(len(fw_labels)))
        ax4.set_yticklabels(fw_labels, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel("Frequency Difference (Gerty - Bloom)")
        ax4.set_title("Function Word Divergence")
        ax4.axvline(x=0, color="gray", linewidth=0.5)
        from matplotlib.patches import Patch
        ax4.legend(
            handles=[
                Patch(facecolor="#E07A5F", label="Higher in Gerty"),
                Patch(facecolor="#4A90D9", label="Higher in Bloom"),
            ],
            loc="lower right",
            fontsize=8,
        )

    plt.suptitle(f"Stylometric Profiles — {episode_label}", fontsize=14, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# Section 2: Burrows' Delta
# ============================================================================

st.header("2. Burrows' Delta")

st.markdown(
    "Burrows' Delta measures stylistic distance between a test text and a reference "
    "corpus using z-scored function word frequencies. Lower delta = closer stylistic "
    "match. We test each half of Nausicaa against Bloom episodes (Calypso, "
    "Lestrygonians), and against Jane Austen's *Emma* from the NLTK Gutenberg corpus "
    "as a sentimental fiction baseline."
)

if not is_nausicaa:
    st.info(
        "The Burrows' Delta analysis is specific to Nausicaa — the Gerty/Bloom split "
        "provides the natural test case for authorship attribution. Select "
        "**13 — Nausicaa** to explore this section."
    )
else:
    # Build reference corpus profiles
    @st.cache_data
    def build_corpus_profiles():
        """Build reference corpus profiles for Burrows' Delta."""
        profiles = []

        # Bloom episodes
        for ep_file, ep_label in [
            ("04calypso.txt", "Calypso (Bloom)"),
            ("08lestrygonians.txt", "Lestrygonians (Bloom)"),
        ]:
            text = cached_load_episode(ep_file)
            prof = cached_stylometric_profile(text, ep_label)
            profiles.append((ep_label, prof))

        # Gutenberg: Melville's Moby-Dick
        melville_text = gutenberg.raw("melville-moby_dick.txt")
        melville_prof = cached_stylometric_profile(melville_text, "Melville — Moby-Dick")
        profiles.append(("Melville — Moby-Dick", melville_prof))

        return profiles

    corpus_profiles = build_corpus_profiles()

    # Make profiles hashable for caching
    def profile_to_tuple(prof):
        items = []
        for k, v in sorted(prof.items()):
            if isinstance(v, dict):
                items.append((k, tuple(sorted(v.items()))))
            else:
                items.append((k, v))
        return tuple(items)

    gerty_profile_t = profile_to_tuple(gerty_profile)
    bloom_profile_t = profile_to_tuple(bloom_profile)
    corpus_profiles_t = tuple(
        (label, profile_to_tuple(prof)) for label, prof in corpus_profiles
    )

    # Compute delta for Gerty half
    gerty_deltas = cached_burrows_delta(gerty_profile_t, corpus_profiles_t)
    bloom_deltas = cached_burrows_delta(bloom_profile_t, corpus_profiles_t)

    # --- Side-by-side bar charts ---
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Gerty Half")
        if gerty_deltas:
            g_labels = [label for label, _ in gerty_deltas]
            g_vals = [delta for _, delta in gerty_deltas]
            fig_g, ax_g = plt.subplots(figsize=(6, max(3, len(g_labels) * 0.5)))
            bar_colors_g = []
            for label in g_labels:
                if "Bloom" in label:
                    bar_colors_g.append("#4A90D9")
                elif "Austen" in label:
                    bar_colors_g.append("#81B29A")
                else:
                    bar_colors_g.append("#A0A0A0")
            ax_g.barh(range(len(g_labels)), g_vals, color=bar_colors_g)
            ax_g.set_yticks(range(len(g_labels)))
            ax_g.set_yticklabels(g_labels, fontsize=9)
            ax_g.invert_yaxis()
            ax_g.set_xlabel("Burrows' Delta")
            ax_g.set_title("Gerty Half — Delta Distances")
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)

            closest = gerty_deltas[0]
            st.markdown(
                f"Closest match: **{closest[0]}** (delta = {closest[1]:.4f})"
            )
        else:
            st.warning("Could not compute Burrows' Delta for Gerty half.")

    with right_col:
        st.subheader("Bloom Half")
        if bloom_deltas:
            b_labels = [label for label, _ in bloom_deltas]
            b_vals = [delta for _, delta in bloom_deltas]
            fig_b, ax_b = plt.subplots(figsize=(6, max(3, len(b_labels) * 0.5)))
            bar_colors_b = []
            for label in b_labels:
                if "Bloom" in label:
                    bar_colors_b.append("#4A90D9")
                elif "Austen" in label:
                    bar_colors_b.append("#81B29A")
                else:
                    bar_colors_b.append("#A0A0A0")
            ax_b.barh(range(len(b_labels)), b_vals, color=bar_colors_b)
            ax_b.set_yticks(range(len(b_labels)))
            ax_b.set_yticklabels(b_labels, fontsize=9)
            ax_b.invert_yaxis()
            ax_b.set_xlabel("Burrows' Delta")
            ax_b.set_title("Bloom Half — Delta Distances")
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close(fig_b)

            closest = bloom_deltas[0]
            st.markdown(
                f"Closest match: **{closest[0]}** (delta = {closest[1]:.4f})"
            )
        else:
            st.warning("Could not compute Burrows' Delta for Bloom half.")

    # --- Function word contribution analysis ---
    st.subheader("Function Word Contribution Analysis")

    st.markdown(
        "Which function words drive the delta score? For each word, we compute the "
        "absolute z-score difference between the test text and each reference. "
        "Words with large contributions are the stylistic markers that distinguish "
        "Gerty's romance idiom from Bloom's interior monologue."
    )

    # Compute function word z-score contributions
    gerty_fw = gerty_profile.get("fw_freqs", {})
    all_corpus_fw = {}
    for label, prof in corpus_profiles:
        all_corpus_fw[label] = prof.get("fw_freqs", {})

    # Compute mean and std across corpus for each function word
    fw_means = {}
    fw_stds = {}
    for fw in FUNCTION_WORDS:
        vals = [gerty_fw.get(fw, 0)]
        vals += [all_corpus_fw[label].get(fw, 0) for label in all_corpus_fw]
        fw_means[fw] = np.mean(vals)
        fw_stds[fw] = np.std(vals) if np.std(vals) > 0 else 1e-10

    # Z-score for Gerty
    contrib_rows = []
    for fw in FUNCTION_WORDS:
        gerty_z = abs(gerty_fw.get(fw, 0) - fw_means[fw]) / fw_stds[fw]
        bloom_z = abs(bloom_profile.get("fw_freqs", {}).get(fw, 0) - fw_means[fw]) / fw_stds[fw]
        contrib_rows.append({
            "Function Word": fw,
            "Gerty Freq": f"{gerty_fw.get(fw, 0):.4f}",
            "Bloom Freq": f"{bloom_profile.get('fw_freqs', {}).get(fw, 0):.4f}",
            "Gerty |z|": f"{gerty_z:.3f}",
            "Bloom |z|": f"{bloom_z:.3f}",
            "Difference": f"{gerty_z - bloom_z:+.3f}",
        })

    contrib_rows.sort(key=lambda r: abs(float(r["Difference"])), reverse=True)
    st.dataframe(
        pd.DataFrame(contrib_rows[:20]),
        width="stretch",
        hide_index=True,
    )


# ============================================================================
# Section 3: Cliche Detection via N-gram Overlap
# ============================================================================

st.header("3. Cliche Detection via N-gram Overlap")

st.markdown(
    "We extract word n-grams from the Gerty half and from Gutenberg "
    "romantic/sentimental texts (Austen), then find shared n-grams. "
    "Note: most shared n-grams at lower *n* are common English phrases, not "
    "genre-specific cliches. Higher *n* values (5-grams) are more likely to "
    "surface genuinely formulaic phrases, but matches become very sparse."
)

# --- N-gram range selector ---
ngram_choice = st.radio(
    "N-gram size",
    ["3-grams", "4-grams", "5-grams"],
    horizontal=True,
    key="cliche_ngram",
)
ngram_n = {"3-grams": 3, "4-grams": 4, "5-grams": 5}[ngram_choice]


@st.cache_data
def cached_gutenberg_romantic_text():
    """Load and concatenate Gutenberg romantic/sentimental texts."""
    texts = []
    for fileid in ["austen-emma.txt", "austen-sense.txt", "austen-persuasion.txt"]:
        try:
            texts.append(gutenberg.raw(fileid))
        except Exception:
            pass
    return " ".join(texts)


@st.cache_data
def cached_cliche_analysis(gerty_text_hash, gutenberg_text_hash, n):
    """Find shared n-grams between Gerty text and Gutenberg romantic fiction.

    We pass text hashes but use the actual texts stored in session state.
    """
    gerty_t = st.session_state.get("_gerty_text", "")
    gut_t = st.session_state.get("_gutenberg_text", "")
    gerty_ngrams = suppress_stdout(extract_ngrams, gerty_t, (n, n))
    gutenberg_ngrams = suppress_stdout(extract_ngrams, gut_t, (n, n))
    shared_keys = set(gerty_ngrams.keys()) & set(gutenberg_ngrams.keys())
    shared = {k: gerty_ngrams[k] for k in shared_keys}
    return gerty_ngrams, gutenberg_ngrams, shared


if is_nausicaa:
    gerty_text_for_cliche = gerty_text
else:
    gerty_text_for_cliche = episode_text

gutenberg_romantic = cached_gutenberg_romantic_text()

# Store texts in session state for the cached function to access
st.session_state["_gerty_text"] = gerty_text_for_cliche
st.session_state["_gutenberg_text"] = gutenberg_romantic

gerty_ngrams, gutenberg_ngrams, shared_ngrams = cached_cliche_analysis(
    hash(gerty_text_for_cliche), hash(gutenberg_romantic), ngram_n
)

# --- Metrics row ---
total_gerty_ngrams = len(gerty_ngrams)
total_gutenberg_ngrams = len(gutenberg_ngrams)
shared_count = len(shared_ngrams)
density = shared_count / max(total_gerty_ngrams, 1)

cm1, cm2, cm3, cm4 = st.columns(4)
cm1.metric("Gerty Unique N-grams", f"{total_gerty_ngrams:,}")
cm2.metric("Gutenberg Unique N-grams", f"{total_gutenberg_ngrams:,}")
cm3.metric("Shared (Cliches)", f"{shared_count:,}")
cm4.metric("Cliche Density", f"{density:.2%}")

# --- Cliche table ---
st.subheader("Top Shared N-grams (Cliches)")

if shared_ngrams:
    sorted_shared = sorted(shared_ngrams.items(), key=lambda x: -x[1])[:50]
    cliche_rows = []
    for ngram_tuple, count in sorted_shared:
        ngram_str = " ".join(ngram_tuple) if isinstance(ngram_tuple, tuple) else str(ngram_tuple)
        gut_count = gutenberg_ngrams.get(ngram_tuple, 0)
        cliche_rows.append({
            "N-gram": ngram_str,
            "Gerty Count": count,
            "Gutenberg Count": gut_count,
        })
    st.dataframe(pd.DataFrame(cliche_rows), width="stretch", hide_index=True)
else:
    st.info("No shared n-grams found at this n-gram size.")

# --- Bloom half comparison toggle ---
if is_nausicaa:
    st.subheader("Bloom Half Comparison")

    show_bloom_cliche = st.toggle("Compare Bloom half cliche density", key="bloom_cliche_toggle")

    if show_bloom_cliche:
        st.session_state["_bloom_text_cliche"] = bloom_text
        bloom_ngrams_raw = suppress_stdout(extract_ngrams, bloom_text, (ngram_n, ngram_n))
        bloom_shared_keys = set(bloom_ngrams_raw.keys()) & set(gutenberg_ngrams.keys())
        bloom_shared_count = len(bloom_shared_keys)
        bloom_density = bloom_shared_count / max(len(bloom_ngrams_raw), 1)

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Bloom Unique N-grams", f"{len(bloom_ngrams_raw):,}")
        bc2.metric("Bloom Shared (Cliches)", f"{bloom_shared_count:,}")
        bc3.metric(
            "Bloom Cliche Density",
            f"{bloom_density:.2%}",
            delta=f"{bloom_density - density:+.2%}",
        )

        # Side-by-side density bar
        fig_cmp, ax_cmp = plt.subplots(figsize=(6, 3))
        labels_cmp = ["Gerty", "Bloom"]
        vals_cmp = [density * 100, bloom_density * 100]
        colors_cmp = ["#E07A5F", "#4A90D9"]
        ax_cmp.bar(labels_cmp, vals_cmp, color=colors_cmp)
        ax_cmp.set_ylabel("Cliche Density (%)")
        ax_cmp.set_title(f"Cliche Density Comparison ({ngram_choice})")
        for i, v in enumerate(vals_cmp):
            ax_cmp.text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown(
            "A higher overlap density in the Gerty half is suggestive, but most shared "
            "n-grams are common English phrases rather than genre-specific cliches. "
            "This method is better at detecting broad stylistic similarity than "
            "identifying specific borrowed phrases."
        )


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Nausicaa is Joyce's most sustained exercise in stylistic
ventriloquism. Stylometric profiling quantifies what any reader senses — Gerty's half
has longer, more ornate sentences, different function word patterns, and measurably lower
vocabulary richness than Bloom's clipped interior monologue. Burrows' Delta confirms the
split: Gerty's prose clusters closer to sentimental fiction, while Bloom's half
aligns with his other episodes. The n-gram overlap analysis is more limited — most
shared n-grams are common English phrases rather than genre-specific cliches, so the
"cliche density" metric should be taken as a rough proxy rather than definitive evidence
of pastiche.
""")
