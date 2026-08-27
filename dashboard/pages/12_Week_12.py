"""
Week 12 — Cyclops
Text classification, feature engineering, and genre detection.
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
from nltk import pos_tag
from nltk.classify import NaiveBayesClassifier, accuracy
import random

for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

from week12.week12_cyclops import (
    segment_cyclops,
    extract_features,
    classify_interpolation_genre,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
)

st.set_page_config(page_title="Week 12 — Cyclops", page_icon="📖", layout="wide")
st.title("Week 12 — Cyclops")
st.caption("Text Classification, Feature Engineering & Genre Detection")


# ============================================================================
# Helpers
# ============================================================================


def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def numeric_features(text):
    """Compute numeric (float) features for a text segment."""
    tokens = word_tokenize(text)
    alpha = [t.lower() for t in tokens if t.isalpha()]
    sents = sent_tokenize(text)
    return {
        "avg_sent_len": len(tokens) / len(sents) if sents else 0,
        "ttr": len(set(alpha)) / len(alpha) if alpha else 0,
        "avg_word_len": sum(len(w) for w in alpha) / len(alpha) if alpha else 0,
    }


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_segment_cyclops(text):
    """Segment Cyclops into barfly and interpolation passages."""
    return suppress_stdout(segment_cyclops, text)


@st.cache_data
def cached_extract_features(text):
    """Extract categorical features from a text segment."""
    return suppress_stdout(extract_features, text)


@st.cache_data
def cached_classify_genre(text):
    """Classify an interpolation segment by genre."""
    return suppress_stdout(classify_interpolation_genre, text)


@st.cache_data
def cached_train_classifier(barfly_tuple, interpolation_tuple):
    """Train a NaiveBayesClassifier on barfly vs interpolation segments."""
    barfly_segments = list(barfly_tuple)
    interpolation_segments = list(interpolation_tuple)

    labeled = []
    for seg in barfly_segments:
        feats = suppress_stdout(extract_features, seg)
        labeled.append((feats, "barfly"))
    for seg in interpolation_segments:
        feats = suppress_stdout(extract_features, seg)
        labeled.append((feats, "interpolation"))

    random.seed(42)
    random.shuffle(labeled)

    split_point = int(len(labeled) * 0.7)
    train_set = labeled[:split_point]
    test_set = labeled[split_point:]

    classifier = NaiveBayesClassifier.train(train_set)
    acc = accuracy(classifier, test_set) if test_set else 0.0

    return classifier, acc, train_set, test_set


@st.cache_data
def cached_numeric_features_by_label(barfly_tuple, interpolation_tuple):
    """Compute average numeric features for barfly and interpolation segments."""
    barfly_segments = list(barfly_tuple)
    interpolation_segments = list(interpolation_tuple)

    barfly_feats = [numeric_features(seg) for seg in barfly_segments if len(seg.strip()) > 20]
    interp_feats = [numeric_features(seg) for seg in interpolation_segments if len(seg.strip()) > 20]

    def avg_dict(feat_list):
        if not feat_list:
            return {"avg_sent_len": 0, "ttr": 0, "avg_word_len": 0}
        keys = feat_list[0].keys()
        return {k: sum(d[k] for d in feat_list) / len(feat_list) for k in keys}

    return avg_dict(barfly_feats), avg_dict(interp_feats)


@st.cache_data
def cached_classify_episode_passages(episode_file, _classifier_train):
    """Classify all paragraphs of an episode and return scored passages."""
    text = cached_load_episode(episode_file)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 50]

    # Rebuild classifier from cached data (can't pass classifier directly)
    train_set = list(_classifier_train)
    classifier = NaiveBayesClassifier.train(train_set)

    scored_passages = []
    for para in paragraphs:
        feats = suppress_stdout(extract_features, para)
        prob_dist = classifier.prob_classify(feats)
        p_barfly = prob_dist.prob("barfly")
        scored_passages.append((p_barfly, para))

    return scored_passages


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=11,  # Cyclops
    caption="Week 12: Text Classification & Genre Detection",
)

is_cyclops = episode_file == "12cyclops.txt"

with st.sidebar:
    st.divider()
    st.markdown(
        "**Cyclops** alternates between a barfly narrator's colloquial first-person "
        "account and gigantic interpolations that parody legal, epic, biblical, and "
        "journalistic styles. This week we build a classifier to distinguish the two "
        "voices and detect genre in the interpolations."
    )

# Load data
episode_text = cached_load_episode(episode_file)


# ============================================================================
# Section 1: Barfly vs. Interpolation Classifier
# ============================================================================

st.header("1. Barfly vs. Interpolation Classifier")

st.markdown(
    "The Cyclops episode alternates between two distinct narrative voices. The **barfly** "
    "is an unnamed first-person narrator — a Dublin pub regular telling the story in "
    "colloquial slang (\"says he\", \"begob\", \"bloody\"). The **interpolations** are "
    "Joyce's gigantism technique: enormous passages that interrupt the barfly's account to "
    "parody formal literary genres — legal briefs, epic poetry, biblical genealogies, and "
    "journalistic reports. This classifier learns to distinguish the two voices using "
    "textual features like sentence length, vocabulary richness, word length, and the "
    "presence of slang vs. formal markers."
)

if is_cyclops:
    barfly_segments, interpolation_segments = cached_segment_cyclops(episode_text)

    barfly_tuple = tuple(barfly_segments)
    interp_tuple = tuple(interpolation_segments)

    classifier, acc, train_set, test_set = cached_train_classifier(barfly_tuple, interp_tuple)

    # --- Metrics row ---
    total_segments = len(barfly_segments) + len(interpolation_segments)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Segments", total_segments)
    m2.metric("Barfly Segments", len(barfly_segments))
    m3.metric("Interpolation Segments", len(interpolation_segments))
    m4.metric("Classifier Accuracy", f"{acc:.1%}")

    # --- Feature comparison ---
    st.subheader("Feature Comparison: Barfly vs. Interpolation")

    avg_barfly, avg_interp = cached_numeric_features_by_label(barfly_tuple, interp_tuple)

    comparison_rows = []
    feature_names = {"avg_sent_len": "Avg Sentence Length", "ttr": "Type-Token Ratio", "avg_word_len": "Avg Word Length"}
    for key, display_name in feature_names.items():
        comparison_rows.append({
            "Feature": display_name,
            "Barfly": f"{avg_barfly[key]:.3f}",
            "Interpolation": f"{avg_interp[key]:.3f}",
            "Difference": f"{avg_interp[key] - avg_barfly[key]:+.3f}",
        })

    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    # --- Feature comparison bar chart ---
    fig_feat, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, display_name) in zip(axes, feature_names.items()):
        vals = [avg_barfly[key], avg_interp[key]]
        bars = ax.bar(["Barfly", "Interpolation"], vals, color=["#4A90D9", "#E07A5F"])
        ax.set_title(display_name)
        ax.set_ylabel(display_name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("Numeric Feature Comparison", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig_feat)
    plt.close(fig_feat)

    # --- Segment browser ---
    st.subheader("Segment Browser")

    cb1, cb2 = st.columns(2)
    show_barfly = cb1.checkbox("Show barfly segments", value=True, key="show_barfly")
    show_interp = cb2.checkbox("Show interpolation segments", value=True, key="show_interp")

    all_segments = []
    if show_barfly:
        all_segments += [(seg, "barfly") for seg in barfly_segments]
    if show_interp:
        all_segments += [(seg, "interpolation") for seg in interpolation_segments]

    segment_labels = [
        f"[{label}] {seg[:70]}..." for seg, label in all_segments
    ]

    if segment_labels:
        selected_seg_label = st.selectbox("Select a segment", segment_labels, key="seg_browser")
        sel_idx = segment_labels.index(selected_seg_label)
        sel_seg, sel_true_label = all_segments[sel_idx]

        feats = suppress_stdout(extract_features, sel_seg)
        predicted_label = classifier.classify(feats)
        prob_dist = classifier.prob_classify(feats)
        p_barfly = prob_dist.prob("barfly")
        p_interp = prob_dist.prob("interpolation")

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("True Label", sel_true_label)
        bc2.metric("Predicted Label", predicted_label)
        bc3.metric("P(barfly)", f"{p_barfly:.3f}")

        st.markdown("**Features:**")
        st.json(feats)

        with st.expander("View full segment text"):
            st.write(sel_seg)

else:
    st.info(
        "The barfly vs. interpolation classifier is specific to Cyclops — the episode "
        "built on alternating voices. Select **12 — Cyclops** to explore this section."
    )


# ============================================================================
# Section 2: Barfly Fingerprint Across Episodes
# ============================================================================

st.header("2. Classifier Across Episodes")

st.markdown(
    "The barfly/interpolation classifier was trained on the two voices of Cyclops. "
    "What happens when we apply it to other episodes? Select an episode below to see "
    "which passages the classifier considers most barfly-like (colloquial, short sentences, "
    "first-person) and most interpolation-like (formal, long sentences, elaborate syntax). "
    "This reveals what the classifier actually latches onto — which may not always match "
    "literary intuition, since the classifier only knows about sentence length, vocabulary "
    "richness, and pronoun usage, not content or meaning."
)

if is_cyclops:
    train_tuple = tuple(train_set)

    selected_episode_label = st.selectbox(
        "Select an episode to scan",
        EPISODE_LABELS,
        index=EPISODE_LABELS.index("12 — Cyclops") if "12 — Cyclops" in EPISODE_LABELS else 0,
        key="classifier_episode",
    )

    ep_file = EPISODE_FILES[EPISODE_LABELS.index(selected_episode_label)]
    scored_passages = cached_classify_episode_passages(ep_file, train_tuple)

    if scored_passages:
        probs = [p for p, _ in scored_passages]

        # --- Distribution histogram ---
        st.subheader("P(barfly) Distribution")
        st.markdown(
            "Distribution of P(barfly) scores across all paragraphs in the episode. "
            "Passages near 1.0 look barfly-like to the classifier; passages near 0.0 "
            "look interpolation-like. The shape of this histogram reveals how stylistically "
            "uniform or varied an episode is. For example, **Lestrygonians** piles almost "
            "every paragraph near 1.0 — Bloom's short, punchy interior monologue looks "
            "uniformly \"barfly\" to the classifier. **Oxen of the Sun**, by contrast, "
            "shows a dramatic bimodal split: its early paragraphs parody archaic prose "
            "styles (long sentences, elaborate syntax) that the classifier flags as "
            "interpolation-like, while its later slang-filled passages score as barfly-like."
        )

        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(probs, bins=20, color="#4A90D9", edgecolor="white", alpha=0.8)
        ax_hist.set_xlabel("P(barfly)")
        ax_hist.set_ylabel("Number of Paragraphs")
        ax_hist.set_title(f"P(barfly) Distribution — {selected_episode_label.split(' — ')[1]}")
        ax_hist.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # --- Summary metrics ---
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Total Paragraphs", len(scored_passages))
        sm2.metric("Median P(barfly)", f"{sorted(probs)[len(probs) // 2]:.3f}")
        sm3.metric("Std Dev", f"{np.std(probs):.3f}")

        # --- Top barfly-like and interpolation-like passages ---
        sorted_by_barfly = sorted(scored_passages, key=lambda x: -x[0])
        sorted_by_interp = sorted(scored_passages, key=lambda x: x[0])

        top_barfly = sorted_by_barfly[:5]
        top_interp = sorted_by_interp[:5]

        col_b, col_i = st.columns(2)

        with col_b:
            st.subheader("Top 5 Barfly-Like Passages")
            for rank, (prob, passage) in enumerate(top_barfly):
                with st.expander(f"#{rank + 1} — P(barfly) = {prob:.3f}"):
                    st.write(passage[:500])

        with col_i:
            st.subheader("Top 5 Interpolation-Like Passages")
            for rank, (prob, passage) in enumerate(top_interp):
                with st.expander(f"#{rank + 1} — P(barfly) = {1 - prob:.3f}"):
                    st.write(passage[:500])
    else:
        st.info("No paragraphs found in this episode.")

else:
    st.info(
        "This section requires a trained classifier from Cyclops. "
        "Select **12 — Cyclops** first to train the classifier, then explore "
        "other episodes."
    )


# ============================================================================
# Section 3: Gigantism Analysis
# ============================================================================

st.header("3. Gigantism Analysis")

if is_cyclops:
    # Classify each interpolation by genre
    genre_labels = []
    genre_segments = {}
    for seg in interpolation_segments:
        genre = suppress_stdout(classify_interpolation_genre, seg)
        genre_labels.append(genre)
        if genre not in genre_segments:
            genre_segments[genre] = []
        genre_segments[genre].append(seg)

    genre_counts = Counter(genre_labels)

    # --- Metrics row ---
    gm1, gm2 = st.columns(2)
    gm1.metric("Interpolation Segments", len(interpolation_segments))
    gm2.metric("Genres Detected", len(genre_counts))

    # --- Genre distribution donut chart ---
    st.subheader("Genre Distribution")

    fig_donut, ax_donut = plt.subplots(figsize=(7, 5))
    genre_names = list(genre_counts.keys())
    genre_sizes = [genre_counts[g] for g in genre_names]
    genre_colors = ["#4A90D9", "#E07A5F", "#81B29A", "#DAA520", "#C05555", "#9B59B6"]
    colors_used = genre_colors[:len(genre_names)]

    wedges, texts, autotexts = ax_donut.pie(
        genre_sizes,
        labels=[f"{g} ({c})" for g, c in zip(genre_names, genre_sizes)],
        colors=colors_used,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
    )
    centre = plt.Circle((0, 0), 0.55, fc="white")
    ax_donut.add_artist(centre)
    ax_donut.set_title("Interpolation Genre Distribution")
    plt.tight_layout()
    st.pyplot(fig_donut)
    plt.close(fig_donut)

    # --- Amplification vs barfly baseline ---
    st.subheader("Amplification vs. Barfly Baseline")

    st.markdown(
        "How do the numeric features of each interpolation genre compare to the barfly "
        "baseline? Values above 1.0 indicate amplification (the interpolation exceeds "
        "the barfly norm); below 1.0 indicates compression."
    )

    avg_barfly_feats, _ = cached_numeric_features_by_label(barfly_tuple, interp_tuple)

    amp_rows = []
    genre_numeric = {}
    for genre, segs in genre_segments.items():
        feats_list = [numeric_features(seg) for seg in segs if len(seg.strip()) > 20]
        if feats_list:
            avg = {k: sum(d[k] for d in feats_list) / len(feats_list) for k in feats_list[0].keys()}
            genre_numeric[genre] = avg
            amp_rows.append({
                "Genre": genre,
                "Avg Sent Len": f"{avg['avg_sent_len']:.2f}",
                "TTR": f"{avg['ttr']:.3f}",
                "Avg Word Len": f"{avg['avg_word_len']:.2f}",
                "Sent Len Ratio": f"{avg['avg_sent_len'] / avg_barfly_feats['avg_sent_len']:.2f}x" if avg_barfly_feats["avg_sent_len"] else "N/A",
                "TTR Ratio": f"{avg['ttr'] / avg_barfly_feats['ttr']:.2f}x" if avg_barfly_feats["ttr"] else "N/A",
                "Word Len Ratio": f"{avg['avg_word_len'] / avg_barfly_feats['avg_word_len']:.2f}x" if avg_barfly_feats["avg_word_len"] else "N/A",
            })

    if amp_rows:
        st.dataframe(pd.DataFrame(amp_rows), width="stretch", hide_index=True)

    # --- Amplification radar/bar chart ---
    if genre_numeric:
        feature_keys = ["avg_sent_len", "ttr", "avg_word_len"]
        feature_display = ["Avg Sent Len", "TTR", "Avg Word Len"]

        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        x = np.arange(len(feature_display))
        width = 0.8 / (len(genre_numeric) + 1)

        # Barfly baseline bar
        barfly_vals = [avg_barfly_feats[k] for k in feature_keys]
        # Normalize: barfly = 1.0
        ax_amp.bar(x - 0.4 + width * 0, [1.0] * len(feature_keys), width,
                   label="Barfly (baseline)", color="#888888", alpha=0.5)

        for i, (genre, avg) in enumerate(genre_numeric.items()):
            ratios = [avg[k] / avg_barfly_feats[k] if avg_barfly_feats[k] else 0 for k in feature_keys]
            ax_amp.bar(x - 0.4 + width * (i + 1), ratios, width,
                       label=genre, color=genre_colors[i % len(genre_colors)])

        ax_amp.set_xticks(x)
        ax_amp.set_xticklabels(feature_display)
        ax_amp.set_ylabel("Ratio to Barfly Baseline")
        ax_amp.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax_amp.set_title("Genre Amplification vs. Barfly Baseline")
        ax_amp.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_amp)
        plt.close(fig_amp)

        st.markdown(
            "Each bar shows a genre's average feature value **divided by the barfly baseline** "
            "for that same feature. The dashed line at **1.0** represents the barfly norm — "
            "the colloquial narrator's typical sentence length, vocabulary richness, and word "
            "length. Bars above 1.0 indicate **amplification**: the interpolation genre "
            "exceeds the barfly's norm for that feature (e.g., longer sentences, bigger words). "
            "Bars below 1.0 indicate **compression**: the genre is more constrained than the "
            "barfly on that dimension. The further a bar is from the dashed line, the more "
            "dramatically that genre departs from the barfly's voice — which is the quantitative "
            "signature of Joyce's gigantism technique."
        )

    # --- Interpolation reader ---
    st.subheader("Interpolation Reader")

    genre_filter = st.selectbox(
        "Filter by genre",
        ["All"] + list(genre_segments.keys()),
        key="genre_filter",
    )

    if genre_filter == "All":
        filtered_segs = list(zip(genre_labels, interpolation_segments))
    else:
        filtered_segs = [(g, seg) for g, seg in zip(genre_labels, interpolation_segments) if g == genre_filter]

    if filtered_segs:
        reader_labels = [
            f"[{g}] {seg[:70]}..." for g, seg in filtered_segs
        ]
        selected_reader = st.selectbox("Select an interpolation", reader_labels, key="interp_reader")
        sel_reader_idx = reader_labels.index(selected_reader)
        sel_genre, sel_text = filtered_segs[sel_reader_idx]

        st.markdown(f"**Genre:** {sel_genre}")

        nf = numeric_features(sel_text)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Avg Sentence Length", f"{nf['avg_sent_len']:.1f}")
        rc2.metric("Type-Token Ratio", f"{nf['ttr']:.3f}")
        rc3.metric("Avg Word Length", f"{nf['avg_word_len']:.2f}")

        with st.expander("View full interpolation text", expanded=True):
            st.write(sel_text)
    else:
        st.info("No interpolations match the selected genre filter.")

else:
    st.info(
        "The gigantism analysis is specific to Cyclops — the episode whose interpolations "
        "parody distinct literary genres. Select **12 — Cyclops** to explore this section."
    )


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Cyclops is Joyce's experiment in deliberate style-switching —
a colloquial barfly narrator interrupted by gigantic interpolations that parody legal briefs,
epic catalogues, biblical genealogies, and journalistic reports. A Naive Bayes classifier
trained on simple textual features (sentence length, type-token ratio, word length) can
distinguish these voices with surprising accuracy, confirming that Joyce's style shifts are
not just thematic but measurably structural. The barfly fingerprint analysis shows that
other episodes vary in how "barfly-like" their prose is, while genre classification of the
interpolations reveals the specific rhetorical traditions Joyce was satirizing. The gap
between the barfly's intimate vernacular and the interpolations' inflated registers is the
engine of Cyclops' comedy — and text classification makes that gap quantifiable.
""")
