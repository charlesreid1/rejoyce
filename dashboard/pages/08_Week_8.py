"""
Week 08 — Lestrygonians
N-gram language models, perplexity, and associative chains (PMI).
"""

import math
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
from nltk.util import bigrams as nltk_bigrams, trigrams as nltk_trigrams

for resource in ["punkt", "punkt_tab", "gutenberg"]:
    nltk.download(resource, quiet=True)

from week08.week08_lestrygonians import (
    tokenize_sentences,
    train_ngram_model,
    generate_sentences,
    compute_perplexity,
    is_contraction_fragment,
    is_proper_name_pair,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 08 — Lestrygonians", page_icon="📖", layout="wide")
st.title("Week 08 — Lestrygonians")
st.caption("N-gram Models, Perplexity & Associative Chains")

# Character classification for perplexity color coding
BLOOM_EPISODES = {
    "04calypso.txt", "05lotuseaters.txt", "06hades.txt",
    "08lestrygonians.txt", "11sirens.txt", "12cyclops.txt",
    "13nausicaa.txt",
}
STEPHEN_EPISODES = {
    "01telemachus.txt", "02nestor.txt", "03proteus.txt",
    "09scyllacharybdis.txt",
}


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_train_and_generate(episode_file, n, num_sentences, _seed):
    """Train n-gram model and generate sentences. _seed forces fresh output."""
    text = cached_load_episode(episode_file)
    model = train_ngram_model(text, n=n)
    sentences = generate_sentences(model, num_sentences=num_sentences)
    return sentences


@st.cache_data
def cached_top_ngrams(episode_file, n, top_k=20):
    """Get top-k most frequent n-grams from an episode."""
    text = cached_load_episode(episode_file)
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if n == 2:
        ngram_list = list(nltk_bigrams(tokens))
    else:
        ngram_list = list(nltk_trigrams(tokens))
    freq = Counter(ngram_list)
    return freq.most_common(top_k)


@st.cache_data
def cached_perplexity(train_file, test_file_or_label, n):
    """Compute perplexity. test_file_or_label is either a filename or 'Bible (KJV)'."""
    train_text = cached_load_episode(train_file)
    if test_file_or_label == "Bible (KJV)":
        from nltk.corpus import gutenberg
        test_text = gutenberg.raw("bible-kjv.txt")[:len(train_text)]
    else:
        test_text = cached_load_episode(test_file_or_label)
    return compute_perplexity(train_text, test_text, n=n)


@st.cache_data
def cached_pmi_associations(episode_file, min_count=2):
    """Compute PMI for all bigrams in an episode, returning list of dicts."""
    text = cached_load_episode(episode_file)
    raw_tokens = word_tokenize(text)
    tokens = [t.lower() for t in raw_tokens if t.isalpha() or t in ["'", "-"]]

    # Filter out contraction fragments
    filtered_tokens = []
    i = 0
    while i < len(tokens):
        if i > 0 and tokens[i - 1] == "'" and is_contraction_fragment(tokens[i]):
            i += 1
        else:
            filtered_tokens.append(tokens[i])
            i += 1

    bigram_freq = Counter(nltk_bigrams(filtered_tokens))
    unigram_freq = Counter(filtered_tokens)
    total_words = len(filtered_tokens)

    results = []
    for (w1, w2), count in bigram_freq.items():
        if count < min_count:
            continue
        if is_contraction_fragment(w1) or is_contraction_fragment(w2):
            continue
        if w1 == "'" or w2 == "'":
            continue

        p_w1_w2 = count / total_words
        p_w1 = unigram_freq[w1] / total_words
        p_w2 = unigram_freq[w2] / total_words

        if p_w1 * p_w2 == 0:
            continue

        pmi = math.log(p_w1_w2 / (p_w1 * p_w2))
        if pmi <= 0:
            continue

        if is_proper_name_pair(w1, w2):
            category = "Name"
        else:
            category = "Content"

        results.append({
            "word1": w1,
            "word2": w2,
            "pmi": pmi,
            "count": count,
            "category": category,
        })

    results.sort(key=lambda x: -x["pmi"])

    # Also return summary stats
    total_bigrams = len([c for c in bigram_freq.values() if c >= min_count])
    positive_pmi = len(results)
    name_count = sum(1 for r in results if r["category"] == "Name")
    content_count = sum(1 for r in results if r["category"] == "Content")

    return results, {
        "total_bigrams": total_bigrams,
        "positive_pmi": positive_pmi,
        "name_pairs": name_count,
        "content_pairs": content_count,
    }


@st.cache_data
def cached_cross_sentence_transitions(episode_file):
    """Extract cross-sentence boundary bigrams with context."""
    text = cached_load_episode(episode_file)
    sentences = sent_tokenize(text)
    transitions = []
    for i in range(len(sentences) - 1):
        tokens_curr = [t.lower() for t in word_tokenize(sentences[i]) if t.isalpha()]
        tokens_next = [t.lower() for t in word_tokenize(sentences[i + 1]) if t.isalpha()]
        if tokens_curr and tokens_next:
            transitions.append({
                "last_word": tokens_curr[-1],
                "first_word": tokens_next[0],
                "sent_idx": i,
                "end_sentence": sentences[i],
                "start_sentence": sentences[i + 1],
            })
    return transitions


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=7,  # Lestrygonians
    caption="Week 8: N-gram Models, Perplexity & Associative Chains",
    description=(
        "*Lestrygonians is Bloom's lunchtime episode. The technique is peristaltic "
        "— each thought triggered by the preceding one, like muscular contractions "
        "moving food through the digestive tract. N-gram language models capture "
        "exactly this: the probability of the next word given only the preceding "
        "n-1 words. The result is text that sounds like Bloom locally but has no "
        "plan, no theme, no memory beyond its window.*"
    ),
)


# ============================================================================
# Section 1: N-gram Text Generation
# ============================================================================

st.header("1. N-gram Text Generation")

st.markdown(
    "Train bigram and trigram language models on episode text and generate sentences. "
    "The **Markov property** means the model has no memory beyond its window — bigrams "
    "(1-word context) produce wild associative sprawl, trigrams (2-word context) produce "
    "shorter, more grammatical output. The trade-off between local coherence and "
    "generative freedom mirrors stream-of-consciousness writing itself."
)

# --- Model controls ---
gen_c1, gen_c2, gen_c3 = st.columns(3)
with gen_c1:
    ngram_order = st.select_slider(
        "N-gram order",
        options=[2, 3],
        value=2,
        format_func=lambda x: "Bigram" if x == 2 else "Trigram",
        key="gen_ngram_order",
    )
with gen_c2:
    num_sentences = st.slider("Sentences to generate", 1, 10, 5, key="gen_num_sentences")
with gen_c3:
    compare_label = st.selectbox(
        "Comparison episode",
        EPISODE_LABELS,
        index=EPISODE_LABELS.index("03 — Proteus"),
        key="gen_compare_episode",
    )
    compare_file = EPISODE_FILES[EPISODE_LABELS.index(compare_label)]

# --- Generate button ---
if st.button("Generate Sentences", key="gen_button"):
    # Use a random seed to get fresh output each time
    seed = random.randint(0, 1_000_000)
    with st.spinner("Training n-gram models..."):
        primary_sentences = cached_train_and_generate(
            episode_file, ngram_order, num_sentences, seed
        )
        compare_sentences = cached_train_and_generate(
            compare_file, ngram_order, num_sentences, seed + 1
        )
    st.session_state["gen_primary"] = primary_sentences
    st.session_state["gen_compare"] = compare_sentences
    st.session_state["gen_primary_label"] = episode_label
    st.session_state["gen_compare_label"] = compare_label

if "gen_primary" in st.session_state:
    primary_sentences = st.session_state["gen_primary"]
    compare_sentences = st.session_state["gen_compare"]
    primary_label = st.session_state["gen_primary_label"]
    compare_label_display = st.session_state["gen_compare_label"]

    # Side-by-side generated output
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader(primary_label)
        for i, sent in enumerate(primary_sentences, 1):
            st.markdown(f"**{i}.** {sent}")
    with right_col:
        st.subheader(compare_label_display)
        for i, sent in enumerate(compare_sentences, 1):
            st.markdown(f"**{i}.** {sent}")

    # Metrics row
    primary_lengths = [len(s.split()) for s in primary_sentences]
    compare_lengths = [len(s.split()) for s in compare_sentences]
    avg_primary = sum(primary_lengths) / len(primary_lengths) if primary_lengths else 0
    avg_compare = sum(compare_lengths) / len(compare_lengths) if compare_lengths else 0

    primary_words = set()
    for s in primary_sentences:
        primary_words.update(w.lower() for w in s.split() if w.isalpha())
    compare_words = set()
    for s in compare_sentences:
        compare_words.update(w.lower() for w in s.split() if w.isalpha())

    overlap = primary_words & compare_words
    overlap_frac = len(overlap) / len(primary_words | compare_words) if (primary_words | compare_words) else 0
    unique_to_primary = primary_words - compare_words

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Avg sentence length (primary)", f"{avg_primary:.1f} words")
    mc2.metric(
        "Avg sentence length (comparison)",
        f"{avg_compare:.1f} words",
        delta=f"{avg_compare - avg_primary:+.1f}",
    )
    mc3.metric("Vocabulary overlap", f"{overlap_frac:.0%}")
    mc4.metric("Unique to primary", f"{len(unique_to_primary)} words")

# --- Top n-gram frequency chart ---
st.subheader(f"Top {'Bigram' if ngram_order == 2 else 'Trigram'} Frequencies")

primary_ngrams = cached_top_ngrams(episode_file, ngram_order, 20)
compare_top50 = cached_top_ngrams(compare_file, ngram_order, 50)
compare_ngram_set = set(ng for ng, _ in compare_top50)

if primary_ngrams:
    labels = [" ".join(ng) for ng, _ in primary_ngrams]
    counts = [c for _, c in primary_ngrams]
    colors = ["#808080" if ng in compare_ngram_set else "#E07A5F" for ng, _ in primary_ngrams]

    fig_ng, ax_ng = plt.subplots(figsize=(10, max(5, len(labels) * 0.35)))
    ax_ng.barh(range(len(labels)), counts, color=colors)
    ax_ng.set_yticks(range(len(labels)))
    ax_ng.set_yticklabels(labels, fontsize=8)
    ax_ng.invert_yaxis()
    ax_ng.set_xlabel("Frequency")
    ax_ng.set_title(f"Top 20 {'Bigrams' if ngram_order == 2 else 'Trigrams'} — {episode_label}")

    from matplotlib.patches import Patch
    ax_ng.legend(
        handles=[
            Patch(facecolor="#808080", label=f"Also in top 50 of {compare_label}"),
            Patch(facecolor="#E07A5F", label=f"Not in top 50 of {compare_label}"),
        ],
        loc="lower right",
        fontsize=8,
    )
    plt.tight_layout()
    st.pyplot(fig_ng)
    plt.close(fig_ng)

# --- "Why do these sound different?" expander ---
with st.expander("Why do these sound different?"):
    compare_ngrams_20 = cached_top_ngrams(compare_file, ngram_order, 20)
    max_len = max(len(primary_ngrams), len(compare_ngrams_20))

    rows = []
    for i in range(max_len):
        row = {"Rank": i + 1}
        if i < len(primary_ngrams):
            ng, c = primary_ngrams[i]
            row[f"{episode_label} N-gram"] = " ".join(ng)
            row[f"{episode_label} Count"] = c
        else:
            row[f"{episode_label} N-gram"] = ""
            row[f"{episode_label} Count"] = ""
        if i < len(compare_ngrams_20):
            ng, c = compare_ngrams_20[i]
            row[f"{compare_label} N-gram"] = " ".join(ng)
            row[f"{compare_label} Count"] = c
        else:
            row[f"{compare_label} N-gram"] = ""
            row[f"{compare_label} Count"] = ""
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ============================================================================
# Section 2: Perplexity as Style Measure
# ============================================================================

st.header("2. Perplexity as Style Measure")

st.markdown(
    "Train a language model on one episode, then measure how *surprised* it is by other texts. "
    "Lower perplexity = more stylistically similar. This quantifies what you intuit about "
    "character voice: a Bloom-trained model should find other Bloom episodes less surprising "
    "than Stephen episodes or the King James Bible."
)

# --- Training config ---
ppl_c1, ppl_c2 = st.columns(2)
with ppl_c1:
    ppl_ngram_order = st.select_slider(
        "N-gram order",
        options=[2, 3],
        value=2,
        format_func=lambda x: "Bigram" if x == 2 else "Trigram",
        key="ppl_ngram_order",
    )
with ppl_c2:
    st.caption(
        "Laplace smoothing is always used to avoid infinite perplexity on unseen "
        "n-grams in literary text with large vocabulary."
    )

# --- Test episode multiselect ---
test_options = EPISODE_LABELS + ["Bible (KJV)"]

# Build default selections
default_tests = [episode_label]
if "04 — Calypso" in EPISODE_LABELS and "04 — Calypso" != episode_label:
    default_tests.append("04 — Calypso")
if "03 — Proteus" in EPISODE_LABELS and "03 — Proteus" != episode_label:
    default_tests.append("03 — Proteus")
default_tests.append("Bible (KJV)")

selected_tests = st.multiselect(
    "Test texts",
    test_options,
    default=default_tests,
    key="ppl_test_texts",
)

# --- Compute Perplexity button ---
if st.button("Compute Perplexity", key="ppl_button"):
    if not selected_tests:
        st.warning("Select at least one test text.")
    else:
        progress = st.progress(0, text="Computing perplexity...")
        results = []
        for i, test_label in enumerate(selected_tests):
            # Map label to file
            if test_label == "Bible (KJV)":
                test_key = "Bible (KJV)"
            else:
                test_idx = EPISODE_LABELS.index(test_label)
                test_key = EPISODE_FILES[test_idx]

            ppl = cached_perplexity(episode_file, test_key, ppl_ngram_order)
            results.append((test_label, ppl))
            progress.progress((i + 1) / len(selected_tests), text=f"Computed {test_label}...")

        progress.empty()
        st.session_state["ppl_results"] = results
        st.session_state["ppl_train_label"] = episode_label
        st.session_state["ppl_train_file"] = episode_file

if "ppl_results" in st.session_state:
    results = st.session_state["ppl_results"]
    train_label = st.session_state["ppl_train_label"]
    train_file = st.session_state["ppl_train_file"]

    # Sort by perplexity (lowest first)
    results_sorted = sorted(results, key=lambda x: x[1])

    # --- Bar chart ---
    labels = [lbl for lbl, _ in results_sorted]
    values = [ppl for _, ppl in results_sorted]

    # Color coding
    bar_colors = []
    for lbl, _ in results_sorted:
        if lbl == train_label:
            bar_colors.append("#2E8B57")  # green - self
        elif lbl == "Bible (KJV)":
            bar_colors.append("#CD5C5C")  # red - external
        elif lbl in EPISODE_LABELS:
            file = EPISODE_FILES[EPISODE_LABELS.index(lbl)]
            if file in BLOOM_EPISODES:
                bar_colors.append("#4A90D9")  # blue - Bloom
            elif file in STEPHEN_EPISODES:
                bar_colors.append("#E89B3F")  # orange - Stephen
            else:
                bar_colors.append("#A0A0A0")  # gray - other
        else:
            bar_colors.append("#A0A0A0")

    fig_ppl, ax_ppl = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    ax_ppl.barh(range(len(labels)), values, color=bar_colors)
    ax_ppl.set_yticks(range(len(labels)))
    ax_ppl.set_yticklabels(labels, fontsize=9)
    ax_ppl.invert_yaxis()
    ax_ppl.set_xlabel("Perplexity")
    ax_ppl.set_title(f"Perplexity — Model Trained on {train_label}")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2E8B57", label="Training episode (self)"),
        Patch(facecolor="#4A90D9", label="Bloom episodes"),
        Patch(facecolor="#E89B3F", label="Stephen episodes"),
        Patch(facecolor="#CD5C5C", label="External text"),
        Patch(facecolor="#A0A0A0", label="Other episodes"),
    ]
    ax_ppl.legend(handles=legend_handles, loc="upper right", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_ppl)
    plt.close(fig_ppl)

    # --- Metrics row ---
    # Find self-perplexity as baseline
    self_ppl = None
    for lbl, ppl in results:
        if lbl == train_label:
            self_ppl = ppl
            break

    # Show up to 6 metrics
    display_results = results_sorted[:6]
    if display_results:
        cols = st.columns(len(display_results))
        for col, (lbl, ppl) in zip(cols, display_results):
            short_label = lbl.split(" — ")[1] if " — " in lbl else lbl
            if self_ppl is not None and lbl != train_label:
                col.metric(short_label, f"{ppl:.2f}", delta=f"{ppl - self_ppl:+.2f}")
            else:
                col.metric(short_label, f"{ppl:.2f}")

# --- "What does perplexity mean?" expander ---
with st.expander("What does perplexity mean?"):
    st.markdown(
        "A perplexity of 500 means the model is as confused as if it had to choose "
        "among 500 equally likely next words. Formally, perplexity = 2^(cross-entropy). "
        "Higher perplexity = more information content = the text is more *surprising* "
        "relative to what the model learned.\n\n"
        "**Self-perplexity** (training on and testing the same text) should be lowest — "
        "the model is least surprised by its own training data. Other Bloom episodes "
        "should cluster below Stephen episodes because they share vocabulary and "
        "syntactic patterns. The Bible, with completely different vocabulary and style, "
        "should produce the highest perplexity."
    )

# --- Cross-episode perplexity matrix ---
st.markdown(
    "Compute perplexity for every pair of episodes — training on each row, "
    "testing on each column. **Warning:** 18×18 = 324 computations, this takes "
    "several minutes."
)
if st.button("Compute full matrix", key="ppl_matrix_button"):
    n_eps = len(EPISODE_FILES)
    matrix = np.zeros((n_eps, n_eps))
    progress = st.progress(0, text="Computing perplexity matrix...")
    total = n_eps * n_eps
    count = 0
    for i, train_f in enumerate(EPISODE_FILES):
        for j, test_f in enumerate(EPISODE_FILES):
            ppl = cached_perplexity(train_f, test_f, ppl_ngram_order)
            matrix[i, j] = ppl
            count += 1
            progress.progress(count / total, text=f"Row {i+1}/{n_eps}, Col {j+1}/{n_eps}")
    progress.empty()
    st.session_state["ppl_matrix"] = matrix

if "ppl_matrix" in st.session_state:
    matrix = st.session_state["ppl_matrix"]
    short_labels = [f"{i+1} - {lbl.split(' — ')[1][:10]}" for i, lbl in enumerate(EPISODE_LABELS)]

    fig_mat, ax_mat = plt.subplots(figsize=(14, 12))
    im = ax_mat.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax_mat.set_xticks(range(len(short_labels)))
    ax_mat.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax_mat.set_yticks(range(len(short_labels)))
    ax_mat.set_yticklabels(short_labels, fontsize=7)
    ax_mat.set_xlabel("Test Episode")
    ax_mat.set_ylabel("Training Episode")
    ax_mat.set_title("Cross-Episode Perplexity Matrix")
    fig_mat.colorbar(im, ax=ax_mat, label="Perplexity")

    # Annotate cells with values (small font)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > matrix.mean() else "black"
            ax_mat.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=5, color=color)

    plt.tight_layout()
    st.pyplot(fig_mat)
    plt.close(fig_mat)

    st.markdown(
        "**How to read this matrix:** Each cell shows how *surprised* a language model "
        "trained on the **row** episode is when it reads the **column** episode. "
        "Lower perplexity (lighter color) means the test text feels familiar to the "
        "training model — they share vocabulary, syntax, and rhythms.\n\n"
        "**Why isn't it symmetric?** Training on Cyclops and testing on Penelope is a "
        "different question from the reverse. Cyclops is a long, stylistically varied "
        "episode — its model learns a broad vocabulary, so it handles Penelope reasonably "
        "well. But Penelope is Molly Bloom's unpunctuated monologue with its own narrow "
        "idiom — a model trained *only* on Penelope has never seen most of Cyclops's "
        "words, so it's far more surprised. In general, models trained on longer or more "
        "varied episodes are less surprised by other texts, while models trained on "
        "short or stylistically narrow episodes are easily confused by anything different. "
        "The diagonal (self-perplexity) is always lowest in each row — every episode is "
        "least surprising to its own model."
    )


# ============================================================================
# Section 3: Associative Chains (PMI)
# ============================================================================

st.header("3. Associative Chains (PMI)")

st.markdown(
    "Pointwise Mutual Information measures how much *more* often two words appear together "
    "than you'd expect by chance. PMI = log(P(w1,w2) / (P(w1) × P(w2))). A PMI of 5 means "
    "the pair appears ~150× more often than chance. High-PMI pairs reveal the episode's hidden "
    "associative logic: character names, fixed collocations, and thematic pairings that capture "
    "Bloom's preoccupations."
)

# --- Filter controls ---
pmi_c1, pmi_c2, pmi_c3 = st.columns(3)
with pmi_c1:
    min_bigram_count = st.slider("Minimum bigram frequency", 2, 20, 2, key="pmi_min_count")
with pmi_c2:
    category_filter = st.radio(
        "Show",
        ["All associations", "Content words only", "Proper names only"],
        key="pmi_category",
    )
with pmi_c3:
    top_n = st.slider("Number of associations", 10, 50, 20, key="pmi_top_n")

# Compute PMI
all_associations, pmi_stats = cached_pmi_associations(episode_file, min_bigram_count)

# Apply category filter
if category_filter == "Content words only":
    filtered = [a for a in all_associations if a["category"] == "Content"]
elif category_filter == "Proper names only":
    filtered = [a for a in all_associations if a["category"] == "Name"]
else:
    filtered = all_associations

display_assoc = filtered[:top_n]

# --- Metrics row ---
pm1, pm2, pm3, pm4 = st.columns(4)
pm1.metric("Total unique bigrams", f"{pmi_stats['total_bigrams']:,}")
pm2.metric("Positive PMI pairs", f"{pmi_stats['positive_pmi']:,}")
pm3.metric("Proper name pairs", f"{pmi_stats['name_pairs']:,}")
pm4.metric("Content word pairs", f"{pmi_stats['content_pairs']:,}")

# --- PMI association table ---
if display_assoc:
    table_rows = []
    for i, a in enumerate(display_assoc, 1):
        table_rows.append({
            "Rank": i,
            "Word 1": a["word1"],
            "Word 2": a["word2"],
            "PMI": f"{a['pmi']:.4f}",
            "Count": a["count"],
            "Category": a["category"],
        })

    df_pmi = pd.DataFrame(table_rows)

    st.dataframe(df_pmi, width="stretch", hide_index=True)
else:
    st.info("No associations found with current filter settings.")

# --- Cross-Sentence Transitions ---
st.subheader("Cross-Sentence Associative Links")

st.markdown(
    "Cross-sentence transitions reveal the *inter-sentence* logic — the channels "
    "(sensory, thematic, phonetic, idiosyncratic) through which one thought triggers the next. "
    "This is the peristaltic machinery of consciousness."
)

transitions = cached_cross_sentence_transitions(episode_file)

# Compute transition frequencies
trans_freq = Counter((t["last_word"], t["first_word"]) for t in transitions)
top_transitions = trans_freq.most_common(15)

# Metrics
tc1, tc2, tc3 = st.columns(3)
tc1.metric("Sentence boundaries analyzed", len(transitions))
tc2.metric("Unique transition types", len(trans_freq))
if top_transitions:
    most_common = top_transitions[0]
    tc3.metric(
        "Most common transition",
        f"{most_common[0][0]} → {most_common[0][1]}",
        delta=f"{most_common[1]} occurrences",
        delta_color="off",
    )

# Transition frequency table
if top_transitions:
    trans_rows = []
    for (w1, w2), count in top_transitions:
        trans_rows.append({"Last Word": w1, "First Word": w2, "Count": count})
    st.dataframe(pd.DataFrame(trans_rows), width="stretch", hide_index=True)

# Transition context explorer
if top_transitions:
    transition_options = [f"{w1} → {w2} ({c}×)" for (w1, w2), c in top_transitions]
    selected_trans = st.selectbox(
        "Select a transition to explore",
        transition_options,
        key="trans_explorer",
    )
    selected_idx = transition_options.index(selected_trans)
    sel_w1, sel_w2 = top_transitions[selected_idx][0]

    # Find matching transitions
    matching = [t for t in transitions if t["last_word"] == sel_w1 and t["first_word"] == sel_w2]

    for t in matching[:5]:
        end_sent = t["end_sentence"]
        start_sent = t["start_sentence"]
        # Bold the boundary words
        end_display = end_sent
        start_display = start_sent
        st.markdown(f"...{end_display[-120:]}")
        st.caption("→")
        st.markdown(f"{start_display[:120]}...")
        st.divider()

    # --- Transition dispersion strip ---
    st.subheader("Transition Dispersion")
    total_sents = len(transitions) + 1
    positions = [t["sent_idx"] / total_sents * 100 for t in matching]

    fig_disp, ax_disp = plt.subplots(figsize=(10, 1.5))
    ax_disp.scatter(positions, [0] * len(positions), marker="|", s=200, c="#E07A5F", linewidths=2)
    ax_disp.set_xlim(0, 100)
    ax_disp.set_xlabel("Position in episode (%)")
    ax_disp.set_yticks([])
    ax_disp.set_title(f'"{sel_w1} → {sel_w2}" — Where in the episode?')
    plt.tight_layout()
    st.pyplot(fig_disp)
    plt.close(fig_disp)

# --- Cross-episode comparison ---
with st.expander("Compare associations across episodes"):
    compare_pmi_label = st.selectbox(
        "Compare with",
        [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
        key="pmi_compare_episode",
    )
    compare_pmi_file = EPISODE_FILES[EPISODE_LABELS.index(compare_pmi_label)]

    compare_assoc, compare_stats = cached_pmi_associations(compare_pmi_file, min_bigram_count)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader(episode_label)
        top15_primary = all_associations[:15]
        if top15_primary:
            rows = [{"Word 1": a["word1"], "Word 2": a["word2"], "PMI": f"{a['pmi']:.4f}"}
                    for a in top15_primary]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with right_col:
        st.subheader(compare_pmi_label)
        top15_compare = compare_assoc[:15]
        if top15_compare:
            rows = [{"Word 1": a["word1"], "Word 2": a["word2"], "PMI": f"{a['pmi']:.4f}"}
                    for a in top15_compare]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Overlap metric
    primary_top20 = set((a["word1"], a["word2"]) for a in all_associations[:20])
    compare_top20 = set((a["word1"], a["word2"]) for a in compare_assoc[:20])
    shared = primary_top20 & compare_top20
    st.metric("Shared top-20 associations", f"{len(shared)}/20")


# --- Footer ---
st.markdown("""
---

**What this week reveals:** N-gram models capture the local texture of character voice — Bloom's
clipped, food-oriented fragments versus Stephen's philosophical, multilingual ones — but have no
memory beyond their window. Perplexity quantifies stylistic distance between episodes, confirming
that Bloom episodes share vocabulary patterns distinct from Stephen's. PMI reveals the hidden
associative logic of stream-of-consciousness: the specific word-pairs and cross-sentence
transitions that constitute the peristaltic machinery of Bloom's mind.
""")
