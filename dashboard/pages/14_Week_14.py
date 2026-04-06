"""
Week 14 — Oxen of the Sun
Diachronic style analysis: period profiling, Naive Bayes style dating, feature trajectories.
"""

import contextlib
import io
import os
import random
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
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.corpus import gutenberg

for resource in ["punkt", "punkt_tab", "gutenberg", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

from week14.week14_oxenofthesun import (
    segment_oxen,
    period_features,
    discretize_features,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 14 — Oxen of the Sun", page_icon="\U0001F4D6", layout="wide")
st.title("Week 14 — Oxen of the Sun")
st.caption("Diachronic Style Analysis: Period Profiling, Style Dating & Feature Trajectories")


# ============================================================================
# Helpers
# ============================================================================


def suppress_stdout(func, *args, **kwargs):
    """Run func while suppressing stdout (e.g. NLTK download messages)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_segment_oxen(episode_file):
    """Segment Oxen of the Sun text into period sections."""
    text = cached_load_episode(episode_file)
    return segment_oxen(text)


@st.cache_data
def cached_period_features(text_chunk):
    """Compute period features for a text chunk."""
    return period_features(text_chunk)


@st.cache_data
def cached_all_section_features(episode_file):
    """Compute features for every section of the episode."""
    segments = cached_segment_oxen(episode_file)
    results = []
    for label, section_text in segments:
        feats = period_features(section_text)
        feats["section"] = label
        results.append(feats)
    return results


@st.cache_data
def cached_gutenberg_reference_features():
    """Compute features for Gutenberg reference texts representing literary periods."""
    references = {
        "Bible KJV": gutenberg.raw("bible-kjv.txt")[:20000],
        "Shakespeare (Hamlet)": gutenberg.raw("shakespeare-hamlet.txt")[:20000],
        "Austen (Emma)": gutenberg.raw("austen-emma.txt")[:20000],
        "Melville (Moby Dick)": gutenberg.raw("melville-moby_dick.txt")[:20000],
        "Whitman (Leaves)": gutenberg.raw("whitman-leaves.txt")[:20000],
    }
    results = []
    for name, text in references.items():
        feats = period_features(text)
        feats["reference"] = name
        results.append(feats)
    return results


@st.cache_data
def cached_general_features(episode_file):
    """Compute features for the full episode text (non-Oxen episodes)."""
    text = cached_load_episode(episode_file)
    return period_features(text)


@st.cache_data
def cached_train_classifier():
    """Train a NaiveBayesClassifier on Gutenberg chunks with period labels."""
    corpus_map = {
        "anglo_saxon": [gutenberg.raw("bible-kjv.txt")],
        "elizabethan": [gutenberg.raw("shakespeare-hamlet.txt")],
        "augustan": [gutenberg.raw("austen-emma.txt")],
        "victorian": [
            gutenberg.raw("melville-moby_dick.txt"),
            gutenberg.raw("whitman-leaves.txt"),
        ],
    }

    training_data = []
    for period_label, texts in corpus_map.items():
        for raw_text in texts:
            sentences = sent_tokenize(raw_text)
            # Create chunks of 30 sentences
            for i in range(0, len(sentences) - 29, 30):
                chunk = " ".join(sentences[i:i + 30])
                feats = period_features(chunk)
                disc = discretize_features(feats)
                training_data.append((disc, period_label))

    random.seed(42)
    random.shuffle(training_data)

    split = int(len(training_data) * 0.8)
    train_set = training_data[:split]
    test_set = training_data[split:]

    classifier = NaiveBayesClassifier.train(train_set)
    acc = accuracy(classifier, test_set)

    return classifier, acc, len(train_set), len(test_set)


@st.cache_data
def cached_classify_sections(episode_file):
    """Classify each Oxen section using the trained classifier."""
    classifier, acc, _, _ = cached_train_classifier()
    segments = cached_segment_oxen(episode_file)

    results = []
    for label, section_text in segments:
        feats = period_features(section_text)
        disc = discretize_features(feats)
        predicted = classifier.classify(disc)
        prob_dist = classifier.prob_classify(disc)
        confidence = prob_dist.prob(predicted)
        results.append({
            "Section": label,
            "Predicted Period": predicted,
            "Confidence": confidence,
        })
    return results, acc


@st.cache_data
def cached_confusion_matrix():
    """Compute confusion matrix for the classifier on the test set."""
    corpus_map = {
        "anglo_saxon": [gutenberg.raw("bible-kjv.txt")],
        "elizabethan": [gutenberg.raw("shakespeare-hamlet.txt")],
        "augustan": [gutenberg.raw("austen-emma.txt")],
        "victorian": [
            gutenberg.raw("melville-moby_dick.txt"),
            gutenberg.raw("whitman-leaves.txt"),
        ],
    }

    all_data = []
    for period_label, texts in corpus_map.items():
        for raw_text in texts:
            sentences = sent_tokenize(raw_text)
            for i in range(0, len(sentences) - 29, 30):
                chunk = " ".join(sentences[i:i + 30])
                feats = period_features(chunk)
                disc = discretize_features(feats)
                all_data.append((disc, period_label))

    random.seed(42)
    random.shuffle(all_data)

    split = int(len(all_data) * 0.8)
    train_set = all_data[:split]
    test_set = all_data[split:]

    classifier = NaiveBayesClassifier.train(train_set)

    labels = ["anglo_saxon", "elizabethan", "augustan", "victorian"]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for feats, true_label in test_set:
        predicted = classifier.classify(feats)
        if true_label in label_to_idx and predicted in label_to_idx:
            matrix[label_to_idx[true_label]][label_to_idx[predicted]] += 1

    return matrix, labels


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=13,  # Oxen of the Sun
    caption="Week 14: Diachronic Style Analysis",
    description=(
        "*Oxen of the Sun compresses the entire history of English prose into a "
        "single episode. Joyce imitates styles from Anglo-Saxon alliterative verse "
        "through Elizabethan, Augustan, Victorian, and finally modern slang — nine "
        "distinct period pastiches that map the evolution of the English sentence. "
        "This week we profile each section's stylistic features, train a classifier "
        "to date prose by period, and trace the arc of English across the episode.*"
    ),
)

is_oxen = episode_file == "14oxenofthesun.txt"


# ============================================================================
# Section 1: Period Profiling
# ============================================================================

st.header("1. Period Profiling")

st.markdown(
    "Each section of Oxen of the Sun imitates a different period of English prose. "
    "We compute stylistic features for each section — average sentence length, "
    "type-token ratio, average word length, proportion of long words, function word "
    "proportion, adjective density, noun-verb ratio, comma rate, and semicolon rate — "
    "then compare against real Gutenberg reference texts from each period."
)

if is_oxen:
    # --- Compute section features ---
    section_features = cached_all_section_features(episode_file)

    # --- Period profiles dataframe ---
    st.subheader("Section Feature Profiles")

    feature_cols = [
        "avg_sent_len", "ttr", "avg_word_len", "long_word_prop",
        "func_word_prop", "adj_density", "noun_verb_ratio",
        "comma_per_sent", "semicolon_per_sent",
    ]
    display_names = {
        "avg_sent_len": "Avg Sent Len",
        "ttr": "TTR",
        "avg_word_len": "Avg Word Len",
        "long_word_prop": "Long Word %",
        "func_word_prop": "Func Word %",
        "adj_density": "Adj Density",
        "noun_verb_ratio": "N/V Ratio",
        "comma_per_sent": "Commas/Sent",
        "semicolon_per_sent": "Semicolons/Sent",
    }

    rows = []
    for sf in section_features:
        row = {"Section": sf["section"]}
        for fc in feature_cols:
            val = sf.get(fc, 0)
            row[display_names[fc]] = round(val, 3) if isinstance(val, float) else val
        rows.append(row)

    df_sections = pd.DataFrame(rows)
    st.dataframe(df_sections, use_container_width=True, hide_index=True)

    # --- Reference profiles ---
    st.subheader("Gutenberg Reference Profiles")

    ref_features = cached_gutenberg_reference_features()
    ref_rows = []
    for rf in ref_features:
        row = {"Reference": rf["reference"]}
        for fc in feature_cols:
            val = rf.get(fc, 0)
            row[display_names[fc]] = round(val, 3) if isinstance(val, float) else val
        ref_rows.append(row)

    df_refs = pd.DataFrame(ref_rows)
    st.dataframe(df_refs, use_container_width=True, hide_index=True)

    # --- Metrics row ---
    segments = cached_segment_oxen(episode_file)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sections", len(segments))
    avg_sent_lens = [sf.get("avg_sent_len", 0) for sf in section_features]
    m2.metric("Shortest Avg Sentence", f"{min(avg_sent_lens):.1f} words")
    m3.metric("Longest Avg Sentence", f"{max(avg_sent_lens):.1f} words")
    ttrs = [sf.get("ttr", 0) for sf in section_features]
    m4.metric("TTR Range", f"{min(ttrs):.3f} - {max(ttrs):.3f}")

    # --- Period text reader ---
    st.subheader("Period Text Reader")

    section_labels = [sf["section"] for sf in section_features]
    selected_section = st.selectbox(
        "Select a period section",
        section_labels,
        key="period_reader",
    )
    selected_idx = section_labels.index(selected_section)
    selected_segment_text = segments[selected_idx][1]
    selected_feats = section_features[selected_idx]

    # Show features for selected section
    feat_cols = st.columns(5)
    feat_cols[0].metric("Avg Sent Len", f"{selected_feats.get('avg_sent_len', 0):.1f}")
    feat_cols[1].metric("TTR", f"{selected_feats.get('ttr', 0):.3f}")
    feat_cols[2].metric("Avg Word Len", f"{selected_feats.get('avg_word_len', 0):.2f}")
    feat_cols[3].metric("Long Word %", f"{selected_feats.get('long_word_prop', 0):.3f}")
    feat_cols[4].metric("Func Word %", f"{selected_feats.get('func_word_prop', 0):.3f}")

    # Show text preview
    preview_len = st.slider("Preview length (characters)", 200, 2000, 500, key="preview_len")
    preview_text = selected_segment_text[:preview_len]
    if len(selected_segment_text) > preview_len:
        preview_text += "..."
    st.markdown(f"> {preview_text}")

else:
    # --- Non-Oxen episode: show general feature analysis ---
    st.info(
        f"**{episode_label}** is not Oxen of the Sun. Showing general feature analysis "
        f"for this episode alongside Gutenberg reference profiles."
    )

    gen_feats = cached_general_features(episode_file)

    feature_cols = [
        "avg_sent_len", "ttr", "avg_word_len", "long_word_prop",
        "func_word_prop", "adj_density", "noun_verb_ratio",
        "comma_per_sent", "semicolon_per_sent",
    ]
    display_names = {
        "avg_sent_len": "Avg Sent Len",
        "ttr": "TTR",
        "avg_word_len": "Avg Word Len",
        "long_word_prop": "Long Word %",
        "func_word_prop": "Func Word %",
        "adj_density": "Adj Density",
        "noun_verb_ratio": "N/V Ratio",
        "comma_per_sent": "Commas/Sent",
        "semicolon_per_sent": "Semicolons/Sent",
    }

    ep_row = {"Text": episode_label}
    for fc in feature_cols:
        val = gen_feats.get(fc, 0)
        ep_row[display_names[fc]] = round(val, 3) if isinstance(val, float) else val

    ref_features = cached_gutenberg_reference_features()
    all_rows = [ep_row]
    for rf in ref_features:
        row = {"Text": rf["reference"]}
        for fc in feature_cols:
            val = rf.get(fc, 0)
            row[display_names[fc]] = round(val, 3) if isinstance(val, float) else val
        all_rows.append(row)

    df_combined = pd.DataFrame(all_rows)
    st.dataframe(df_combined, use_container_width=True, hide_index=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Sent Len", f"{gen_feats.get('avg_sent_len', 0):.1f}")
    m2.metric("TTR", f"{gen_feats.get('ttr', 0):.3f}")
    m3.metric("Avg Word Len", f"{gen_feats.get('avg_word_len', 0):.2f}")


# ============================================================================
# Section 2: The Style Dating Game
# ============================================================================

st.header("2. The Style Dating Game")

st.markdown(
    "Can a machine tell Anglo-Saxon from Augustan? We train a Naive Bayes classifier "
    "on 30-sentence chunks from Gutenberg reference texts, each labeled with a period: "
    "**anglo_saxon** (Bible KJV), **elizabethan** (Shakespeare), **augustan** (Austen), "
    "**victorian** (Melville, Whitman). The classifier uses discretized stylistic features "
    "to predict which period produced each chunk. Then we ask it to date each section of "
    "Oxen of the Sun."
)

if st.button("Train Classifier & Classify Sections", key="train_classify"):
    with st.spinner("Training Naive Bayes classifier on Gutenberg chunks..."):
        if is_oxen:
            classifications, acc = cached_classify_sections(episode_file)
        else:
            # Still train and show accuracy, but classify the full episode
            classifier, acc, n_train, n_test = cached_train_classifier()
            ep_text = cached_load_episode(episode_file)
            feats = period_features(ep_text)
            disc = discretize_features(feats)
            predicted = classifier.classify(disc)
            prob_dist = classifier.prob_classify(disc)
            confidence = prob_dist.prob(predicted)
            classifications = [{
                "Section": episode_label,
                "Predicted Period": predicted,
                "Confidence": confidence,
            }]

        st.session_state["classifications"] = classifications
        st.session_state["classifier_acc"] = acc

if "classifications" in st.session_state:
    classifications = st.session_state["classifications"]
    acc = st.session_state["classifier_acc"]

    # --- Accuracy metric ---
    ac1, ac2 = st.columns(2)
    ac1.metric("Classifier Accuracy (test set)", f"{acc:.1%}")
    ac2.metric("Sections Classified", len(classifications))

    # --- Classification table ---
    st.subheader("Classification Results")

    class_rows = []
    for c in classifications:
        class_rows.append({
            "Section": c["Section"],
            "Predicted Period": c["Predicted Period"],
            "Confidence": f"{c['Confidence']:.3f}",
        })
    df_class = pd.DataFrame(class_rows)
    st.dataframe(df_class, use_container_width=True, hide_index=True)

    # --- Classification bar chart ---
    if len(classifications) > 1:
        period_colors = {
            "anglo_saxon": "#8B4513",
            "elizabethan": "#4A90D9",
            "augustan": "#81B29A",
            "victorian": "#E07A5F",
        }

        fig_class, ax_class = plt.subplots(figsize=(12, 5))
        x_pos = range(len(classifications))
        bars = ax_class.bar(
            x_pos,
            [c["Confidence"] for c in classifications],
            color=[period_colors.get(c["Predicted Period"], "#999999") for c in classifications],
            edgecolor="#333333",
            linewidth=0.5,
        )

        ax_class.set_xticks(x_pos)
        ax_class.set_xticklabels(
            [c["Section"][:20] for c in classifications],
            rotation=45, ha="right", fontsize=8,
        )
        ax_class.set_ylabel("Confidence")
        ax_class.set_title("Predicted Period by Section")
        ax_class.set_ylim(0, 1)

        # Add period label on each bar
        for i, c in enumerate(classifications):
            ax_class.text(
                i, c["Confidence"] + 0.02,
                c["Predicted Period"],
                ha="center", va="bottom", fontsize=7, fontstyle="italic",
            )

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=color, label=label)
            for label, color in period_colors.items()
        ]
        ax_class.legend(handles=legend_handles, loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_class)
        plt.close(fig_class)

    # --- Confusion matrix heatmap ---
    st.subheader("Confusion Matrix")

    with st.spinner("Computing confusion matrix..."):
        conf_matrix, conf_labels = cached_confusion_matrix()

    fig_conf, ax_conf = plt.subplots(figsize=(8, 6))
    im = ax_conf.imshow(conf_matrix, cmap="YlOrRd", aspect="auto")
    ax_conf.set_xticks(range(len(conf_labels)))
    ax_conf.set_xticklabels(conf_labels, rotation=45, ha="right", fontsize=9)
    ax_conf.set_yticks(range(len(conf_labels)))
    ax_conf.set_yticklabels(conf_labels, fontsize=9)
    ax_conf.set_xlabel("Predicted")
    ax_conf.set_ylabel("True")
    ax_conf.set_title("Confusion Matrix — Naive Bayes Period Classifier")

    # Annotate cells
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            val = conf_matrix[i, j]
            color = "white" if val > conf_matrix.max() * 0.6 else "black"
            ax_conf.text(j, i, str(val), ha="center", va="center",
                         fontsize=11, color=color, fontweight="bold")

    fig_conf.colorbar(im, ax=ax_conf, label="Count")
    plt.tight_layout()
    st.pyplot(fig_conf)
    plt.close(fig_conf)


# ============================================================================
# Section 3: The Arc of English
# ============================================================================

st.header("3. The Arc of English")

st.markdown(
    "How do stylistic features change across the nine sections of Oxen of the Sun? "
    "Each line traces a feature's value from the earliest pastiche to the latest, "
    "revealing how Joyce's imitation of English prose evolves: sentences shorten, "
    "vocabulary shifts, punctuation patterns change."
)

if is_oxen:
    section_features_arc = cached_all_section_features(episode_file)

    all_feature_names = [
        "avg_sent_len", "avg_word_len", "long_word_prop",
        "adj_density", "comma_per_sent", "ttr",
    ]
    feature_display = {
        "avg_sent_len": "Avg Sentence Length",
        "avg_word_len": "Avg Word Length",
        "long_word_prop": "Long Word Proportion",
        "adj_density": "Adjective Density",
        "comma_per_sent": "Commas per Sentence",
        "ttr": "Type-Token Ratio",
    }

    selected_features = st.multiselect(
        "Features to plot",
        options=all_feature_names,
        default=all_feature_names,
        format_func=lambda x: feature_display.get(x, x),
        key="arc_features",
    )

    if selected_features:
        # --- Multi-panel line chart (3x2 grid) ---
        n_features = len(selected_features)
        n_cols = 2
        n_rows = (n_features + 1) // 2

        fig_arc, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        section_labels_arc = [sf["section"] for sf in section_features_arc]
        x_positions = range(len(section_labels_arc))

        colors = ["#E07A5F", "#4A90D9", "#81B29A", "#F2CC8F", "#9B59B6", "#264653"]

        for idx, feat_name in enumerate(selected_features):
            ax = axes[idx]
            values = [sf.get(feat_name, 0) for sf in section_features_arc]
            color = colors[idx % len(colors)]

            ax.plot(x_positions, values, marker="o", color=color, linewidth=2, markersize=6)
            ax.fill_between(x_positions, values, alpha=0.1, color=color)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [lbl[:15] for lbl in section_labels_arc],
                rotation=45, ha="right", fontsize=7,
            )
            ax.set_title(feature_display.get(feat_name, feat_name), fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Feature Trajectories Across Oxen of the Sun", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_arc)
        plt.close(fig_arc)

        # --- Feature trend summary table ---
        st.subheader("Feature Trend Summary")

        trend_rows = []
        for feat_name in selected_features:
            values = [sf.get(feat_name, 0) for sf in section_features_arc]
            start_val = values[0]
            end_val = values[-1]
            if end_val > start_val * 1.05:
                direction = "increasing"
            elif end_val < start_val * 0.95:
                direction = "decreasing"
            else:
                direction = "stable"

            trend_rows.append({
                "Feature": feature_display.get(feat_name, feat_name),
                "Start Value": round(start_val, 3),
                "End Value": round(end_val, 3),
                "Change": round(end_val - start_val, 3),
                "Direction": direction,
            })

        df_trends = pd.DataFrame(trend_rows)
        st.dataframe(df_trends, use_container_width=True, hide_index=True)

    else:
        st.info("Select at least one feature to plot.")

else:
    st.info(
        f"**{episode_label}** is not Oxen of the Sun. The feature trajectory analysis "
        f"requires the nine period sections unique to episode 14. Select "
        f"**14 — Oxen of the Sun** from the sidebar to see the arc of English."
    )


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Joyce compresses the entire history of English prose style
into a single episode. Oxen of the Sun is not mere pastiche — it is a controlled experiment
in diachronic stylistics. By imitating Anglo-Saxon alliterative verse, Elizabethan
amplification, Augustan balance, and Victorian elaboration, Joyce demonstrates that
prose style is a measurable, classifiable signal: sentence length, vocabulary richness,
punctuation density, and function word proportion all shift systematically across the
nine sections. A simple Naive Bayes classifier trained on Gutenberg reference texts can
identify which period each section imitates, confirming that Joyce's pastiches capture
the statistical fingerprint of each era. The feature trajectories trace the arc of
English itself — from long, paratactic, heavily punctuated sentences to shorter, more
varied modern prose — compressed into a single chapter of *Ulysses*.
""")
