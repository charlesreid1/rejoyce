"""
Week 07 — Aeolus
TF-IDF, keyword extraction, rhetoric detection, and headline generation.
"""

import contextlib
import io
import math
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import nltk

for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

from week07.week07_aeolus import (
    split_aeolus_sections,
    compute_tfidf,
    detect_anaphora,
    detect_tricolon,
    STOP_WORDS,
    JOYCE_ARTIFACTS,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 07 — Aeolus", page_icon="📖", layout="wide")
st.title("Week 07 — Aeolus")
st.caption("TF-IDF, Keyword Extraction, Rhetoric Detection & Headline Generation")


# ============================================================================
# Helpers
# ============================================================================


def split_into_chunks(text, n_chunks=20):
    """Split non-Aeolus episodes into paragraph-based chunks.

    Returns list of (label, chunk_text) tuples matching the format of
    split_aeolus_sections.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < n_chunks:
        # Not enough paragraphs — use what we have
        return [(f"Chunk {i+1}", p) for i, p in enumerate(paragraphs)]

    # Group consecutive paragraphs into n_chunks chunks
    chunk_size = len(paragraphs) / n_chunks
    chunks = []
    for i in range(n_chunks):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        chunk_text = "\n\n".join(paragraphs[start:end])
        chunks.append((f"Chunk {i+1}", chunk_text))
    return chunks


def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_split_sections(text, is_aeolus):
    if is_aeolus:
        return split_aeolus_sections(text)
    return split_into_chunks(text)


@st.cache_data
def cached_compute_tfidf(sections_tuple):
    """Compute TF-IDF. sections_tuple is a tuple of (headline, text) pairs."""
    sections = list(sections_tuple)
    return compute_tfidf(sections)


@st.cache_data
def cached_detect_anaphora(text, min_repeat, include_stopwords):
    """Detect anaphora, optionally including stopword openers."""
    sentences = sent_tokenize(text)
    prefix_groups = defaultdict(list)

    for sent in sentences:
        tokens = word_tokenize(sent)
        if len(tokens) >= 2:
            prefix = tokens[0].lower()
            if prefix.isalpha() and len(prefix) > 1:
                prefix_groups[prefix].append(sent)

    if include_stopwords:
        anaphora = [
            (prefix, sents)
            for prefix, sents in prefix_groups.items()
            if len(sents) >= min_repeat
        ]
    else:
        anaphora = [
            (prefix, sents)
            for prefix, sents in prefix_groups.items()
            if len(sents) >= min_repeat and prefix not in STOP_WORDS
        ]

    anaphora.sort(key=lambda x: -len(x[1]))
    return anaphora


@st.cache_data
def cached_detect_tricolon(text):
    """Detect tricolons, suppressing print output."""
    return suppress_stdout(detect_tricolon, text)


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=6,  # Aeolus
    caption="Week 7: TF-IDF, Rhetoric & Headlines",
)

is_aeolus = episode_file == "07aeolus.txt"

with st.sidebar:
    top_k = st.slider("Top-k keywords per section", 1, 10, 5, key="top_k")
    st.divider()
    st.markdown(
        "**Aeolus** is Joyce's newspaper office, where headlines interrupt the "
        "narrative — making it a perfect test case for computational keyword "
        "extraction and the limits of bag-of-words summarization."
    )

# Load data
episode_text = cached_load_episode(episode_file)
sections = cached_split_sections(episode_text, is_aeolus)
sections_tuple = tuple((h, t) for h, t in sections)
tfidf_results = cached_compute_tfidf(sections_tuple)


# ============================================================================
# Section 1: TF-IDF Keyword Explorer
# ============================================================================

st.header("1. TF-IDF Keyword Explorer")

st.markdown(
    "TF-IDF (Term Frequency × Inverse Document Frequency) measures how distinctive "
    "a word is within a section relative to the entire episode. High TF-IDF = common "
    "in this section but rare elsewhere. The formula: **TF**(t,d) = count(t in d) / |d|, "
    "**IDF**(t) = log(N / df(t)), **TF-IDF** = TF × IDF."
)

# --- Metrics row ---
total_tokens = sum(
    len(word_tokenize(text)) for _, text in sections
)
avg_tokens = total_tokens // len(sections) if sections else 0

m1, m2, m3 = st.columns(3)
m1.metric("Sections Parsed", len(sections))
m2.metric("Total Tokens", f"{total_tokens:,}")
m3.metric("Avg Section Length", f"{avg_tokens:,} tokens")

# --- TF-IDF Heatmap ---
st.subheader("TF-IDF Heatmap")

# Collect union of top-k terms across all sections
all_top_terms = []
for idx in sorted(tfidf_results.keys()):
    for term, score in tfidf_results[idx][:top_k]:
        if term not in all_top_terms:
            all_top_terms.append(term)

# Limit columns for readability
heatmap_terms = all_top_terms[:30]
heatmap_sections = list(range(len(sections)))

# For large section counts, show top 15 initially
show_all_sections = True
if len(sections) > 15:
    show_all_sections = st.checkbox("Show all sections in heatmap", value=False)
    if not show_all_sections:
        heatmap_sections = heatmap_sections[:15]

if heatmap_terms:
    matrix = np.zeros((len(heatmap_sections), len(heatmap_terms)))
    for row, sec_idx in enumerate(heatmap_sections):
        scores_dict = dict(tfidf_results.get(sec_idx, []))
        for col, term in enumerate(heatmap_terms):
            matrix[row, col] = scores_dict.get(term, 0.0)

    section_labels = []
    for sec_idx in heatmap_sections:
        headline = sections[sec_idx][0]
        label = headline[:35] + "..." if len(headline) > 35 else headline
        section_labels.append(f"{sec_idx+1}. {label}")

    fig_heat, ax_heat = plt.subplots(
        figsize=(max(10, len(heatmap_terms) * 0.5), max(5, len(heatmap_sections) * 0.35))
    )
    im = ax_heat.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(len(heatmap_terms)))
    ax_heat.set_xticklabels(heatmap_terms, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(len(heatmap_sections)))
    ax_heat.set_yticklabels(section_labels, fontsize=7)
    fig_heat.colorbar(im, ax=ax_heat, label="TF-IDF Score")
    ax_heat.set_title("TF-IDF Keyword Heatmap Across Sections")
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)
else:
    st.info("No TF-IDF terms found.")

# --- Section Deep-Dive ---
st.subheader("Section Deep-Dive")

section_options = [
    f"{i+1}. {sections[i][0][:50]}" for i in range(len(sections))
]
selected_section_label = st.selectbox("Select a section", section_options, key="section_dive")
selected_idx = section_options.index(selected_section_label)

headline, section_text = sections[selected_idx]

if is_aeolus:
    st.markdown(f"**Joyce's headline:** {headline}")

# Bar chart of top-k keywords
section_scores = tfidf_results.get(selected_idx, [])[:top_k]
if section_scores:
    terms = [t for t, _ in section_scores]
    scores = [s for _, s in section_scores]

    fig_bar, ax_bar = plt.subplots(figsize=(8, max(3, len(terms) * 0.4)))
    ax_bar.barh(range(len(terms)), scores, color="#4A9D8E")
    ax_bar.set_yticks(range(len(terms)))
    ax_bar.set_yticklabels(terms)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("TF-IDF Score")
    ax_bar.set_title(f"Top-{top_k} Keywords — Section {selected_idx+1}")
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Score breakdown table
    all_scores = dict(tfidf_results.get(selected_idx, []))
    tokens = [
        t.lower()
        for t in word_tokenize(section_text)
        if t.isalpha() and t.lower() not in STOP_WORDS and len(t) > 2
        and t.lower() not in JOYCE_ARTIFACTS
    ]
    tf_counter = Counter(tokens)
    total = len(tokens) if tokens else 1
    N = len(sections)

    # Document frequency
    df_counts = Counter()
    for _, sec_t in sections:
        sec_tokens = set(
            t.lower() for t in word_tokenize(sec_t)
            if t.isalpha() and t.lower() not in STOP_WORDS and len(t) > 2
            and t.lower() not in JOYCE_ARTIFACTS
        )
        for term in sec_tokens:
            df_counts[term] += 1

    breakdown_rows = []
    for term, _ in section_scores:
        tf_val = tf_counter.get(term, 0) / total
        idf_val = math.log(N / df_counts[term]) if df_counts.get(term, 0) > 0 else 0
        breakdown_rows.append({
            "Term": term,
            "TF": f"{tf_val:.4f}",
            "IDF": f"{idf_val:.3f}",
            "TF-IDF": f"{tf_val * idf_val:.4f}",
        })
    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

with st.expander("View section text"):
    st.write(section_text)

# --- Headline vs. Keywords Comparison Table (Aeolus only) ---
if is_aeolus:
    st.subheader("Headline vs. Keywords Comparison")

    overlap_rows = []
    overlap_count = 0
    for i, (hl, _) in enumerate(sections):
        keywords = [term for term, _ in tfidf_results.get(i, [])[:top_k]]
        headline_words = set(
            w.lower().strip(".,!?;:") for w in hl.split() if w.isalpha()
        )
        keyword_set = set(keywords)
        overlap = headline_words & keyword_set
        overlap_pct = len(overlap) / len(headline_words) * 100 if headline_words else 0
        if overlap:
            overlap_count += 1
        overlap_rows.append({
            "Section": i + 1,
            "Joyce's Headline": hl,
            "Top Keywords": ", ".join(keywords),
            "Overlap Words": ", ".join(overlap) if overlap else "—",
            "Overlap %": f"{overlap_pct:.0f}%",
        })

    overall_pct = overlap_count / len(sections) * 100 if sections else 0
    st.metric(
        "Sections with any keyword-headline overlap",
        f"{overlap_count}/{len(sections)} ({overall_pct:.0f}%)",
    )
    st.dataframe(pd.DataFrame(overlap_rows), use_container_width=True, hide_index=True)

    st.markdown(
        "**Why overlap is low:** TF-IDF captures *statistical distinctiveness* — words "
        "that are common here but rare elsewhere. Joyce's headlines are *editorial* — "
        "ironic, thematic, allusive. They compress narrative events and cultural references, "
        "not word frequencies. The mismatch is the lesson."
    )
else:
    st.info(
        "Headline comparison is only available for Aeolus — the only episode "
        "with interpolated ALL-CAPS headlines."
    )

# --- Cross-Episode Keyword Comparison ---
st.subheader("Cross-Episode Keyword Comparison")

compare_episodes = st.multiselect(
    "Select additional episodes to compare",
    [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
    default=[],
    key="cross_ep_compare",
)

if compare_episodes:
    # Build a corpus: each selected episode is one document
    corpus_labels = [episode_label] + compare_episodes
    corpus_files = [episode_file] + [
        EPISODE_FILES[EPISODE_LABELS.index(lbl)] for lbl in compare_episodes
    ]
    corpus_texts = [cached_load_episode(f) for f in corpus_files]

    # Tokenize each episode
    corpus_tokens = []
    for text in corpus_texts:
        tokens = [
            t.lower()
            for t in word_tokenize(text)
            if t.isalpha() and t.lower() not in STOP_WORDS and len(t) > 2
            and t.lower() not in JOYCE_ARTIFACTS
        ]
        corpus_tokens.append(tokens)

    # Document frequency across episodes
    ep_df = Counter()
    for tokens in corpus_tokens:
        for term in set(tokens):
            ep_df[term] += 1

    N_eps = len(corpus_labels)
    comparison_rows = {}
    for i, (label, tokens) in enumerate(zip(corpus_labels, corpus_tokens)):
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        scores = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log(N_eps / ep_df[term]) if ep_df[term] > 0 else 0
            scores[term] = tf_val * idf_val
        top_terms = sorted(scores.items(), key=lambda x: -x[1])[:10]
        comparison_rows[label] = ", ".join(t for t, _ in top_terms)

    cmp_df = pd.DataFrame([
        {"Episode": lbl, "Top-10 Distinctive Keywords": kws}
        for lbl, kws in comparison_rows.items()
    ])
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)


# ============================================================================
# Section 2: Rhetoric Detection
# ============================================================================

st.header("2. Rhetoric Detection")

st.markdown(
    "Anaphora (repeated sentence openers for emphasis) and tricolon (three parallel "
    "phrases of similar structure) are classical rhetorical figures. Aeolus, set in a "
    "newspaper office full of speechmakers, is the natural place to look for them. "
    "Simple heuristics can detect candidates, but false positive rates of 40-60% "
    "reveal the limits of rule-based NLP."
)

# --- Anaphora Explorer ---
st.subheader("Anaphora Explorer")

ana_col1, ana_col2 = st.columns(2)
with ana_col1:
    min_repeat = st.slider("Minimum repetitions", 2, 10, 2, key="min_repeat")
with ana_col2:
    include_stopwords = st.toggle(
        "Include stopword openers",
        value=False,
        key="include_stopwords",
        help="Including stopwords reveals a flood of false positives (e.g., 'the' opening 150+ sentences).",
    )

anaphora = cached_detect_anaphora(episode_text, min_repeat, include_stopwords)

st.metric("Anaphora groups detected", len(anaphora))

if anaphora:
    # Bar chart of top-15 anaphora prefixes
    top_anaphora = anaphora[:15]
    total_sentences = len(sent_tokenize(episode_text))

    fig_ana, ax_ana = plt.subplots(figsize=(10, max(4, len(top_anaphora) * 0.4)))
    prefixes = [prefix for prefix, _ in top_anaphora]
    counts = [len(sents) for _, sents in top_anaphora]
    colors = [
        "#808080" if c / total_sentences > 0.20 else "#2E8B8B"
        for c in counts
    ]
    ax_ana.barh(range(len(prefixes)), counts, color=colors)
    ax_ana.set_yticks(range(len(prefixes)))
    ax_ana.set_yticklabels([f'"{p}"' for p in prefixes])
    ax_ana.invert_yaxis()
    ax_ana.set_xlabel("Sentence Count")
    ax_ana.set_title(f"Top Anaphora Prefixes — {episode_label}")

    from matplotlib.patches import Patch
    ax_ana.legend(
        handles=[
            Patch(facecolor="#2E8B8B", label="Rhetorical candidate"),
            Patch(facecolor="#808080", label="Likely false positive (>20% of sentences)"),
        ],
        loc="lower right",
        fontsize=8,
    )
    plt.tight_layout()
    st.pyplot(fig_ana)
    plt.close(fig_ana)

    # Expandable examples per prefix
    for prefix, sents in top_anaphora[:10]:
        with st.expander(f'"{prefix}" — {len(sents)} sentences'):
            for s in sents[:5]:
                st.markdown(f"- {s[:200]}")
else:
    st.info("No anaphora detected with current settings.")

# --- Tricolon Explorer ---
st.subheader("Tricolon Explorer")

tricolons = cached_detect_tricolon(episode_text)

st.metric("Tricolons detected", len(tricolons))

if tricolons:
    for i, (triple, lengths) in enumerate(tricolons[:8]):
        st.markdown(f"**Tricolon {i+1}** — phrase lengths: {lengths}")
        cols = st.columns(3)
        for j, (phrase, length) in enumerate(zip(triple, lengths)):
            with cols[j]:
                st.markdown(f"*{phrase.strip()}*")
                st.caption(f"{length} tokens")
        st.divider()

    # Grouped bar chart of phrase lengths
    if len(tricolons) >= 2:
        fig_tri, ax_tri = plt.subplots(figsize=(10, 4))
        n_show = min(8, len(tricolons))
        x = np.arange(n_show)
        width = 0.25

        for phrase_idx, color in zip(range(3), ["#E07A5F", "#4A90D9", "#81B29A"]):
            vals = [tricolons[i][1][phrase_idx] for i in range(n_show)]
            ax_tri.bar(x + phrase_idx * width, vals, width, label=f"Phrase {phrase_idx+1}", color=color)

        ax_tri.set_xticks(x + width)
        ax_tri.set_xticklabels([f"Tricolon {i+1}" for i in range(n_show)], fontsize=8)
        ax_tri.set_ylabel("Token Length")
        ax_tri.set_title("Tricolon Phrase Lengths — Parallelism Check")
        ax_tri.legend()
        plt.tight_layout()
        st.pyplot(fig_tri)
        plt.close(fig_tri)

    st.markdown(
        "Tricolons are detected by finding three consecutive comma/semicolon-separated "
        "phrases with similar token counts and matching POS patterns. False positives arise "
        "from coincidental comma-separated structures and speech attributions."
    )

# --- Cross-Episode Rhetoric Comparison ---
if not is_aeolus:
    st.subheader("Rhetoric Comparison with Aeolus")
    aeolus_text = cached_load_episode("07aeolus.txt")
    aeolus_anaphora = cached_detect_anaphora(aeolus_text, min_repeat, False)
    aeolus_tricolons = cached_detect_tricolon(aeolus_text)

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric(f"Anaphora ({episode_label.split(' — ')[1]})", len(anaphora))
    rc2.metric("Anaphora (Aeolus)", len(aeolus_anaphora))
    rc3.metric(f"Tricolons ({episode_label.split(' — ')[1]})", len(tricolons))
    rc4.metric("Tricolons (Aeolus)", len(aeolus_tricolons))

    st.markdown(
        "*Does the newspaper episode really have more rhetorical figures than others?* "
        "Compare the counts above. Aeolus's setting among journalists and speechmakers "
        "should produce denser rhetoric — but heuristic detection may not capture the difference."
    )
elif is_aeolus:
    # Show baseline comparison option
    with st.expander("Compare rhetoric density with another episode"):
        compare_rhet_label = st.selectbox(
            "Compare with",
            [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
            key="rhet_compare",
        )
        compare_rhet_file = EPISODE_FILES[EPISODE_LABELS.index(compare_rhet_label)]
        compare_rhet_text = cached_load_episode(compare_rhet_file)

        cmp_anaphora = cached_detect_anaphora(compare_rhet_text, min_repeat, False)
        cmp_tricolons = cached_detect_tricolon(compare_rhet_text)

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Anaphora (Aeolus)", len(anaphora))
        rc2.metric(f"Anaphora ({compare_rhet_label.split(' — ')[1]})", len(cmp_anaphora))
        rc3.metric("Tricolons (Aeolus)", len(tricolons))
        rc4.metric(f"Tricolons ({compare_rhet_label.split(' — ')[1]})", len(cmp_tricolons))


# ============================================================================
# Section 3: Headline Generation & the Gap
# ============================================================================

st.header("3. Headline Generation & the Gap")

st.markdown(
    "Bag-of-words headline generation joins top TF-IDF keywords in ALL CAPS. "
    "The result is word salad — it has no access to narrative events, character "
    "actions, irony, or cultural allusions. Joyce's headline 'EXIT BLOOM' compresses "
    "a narrative event; TF-IDF might produce 'CROSSBLIND WHITE SEEMS' because those "
    "words are statistically distinctive. The gap motivates everything from sequence "
    "models to attention mechanisms."
)

if is_aeolus:
    # --- Generated vs. Joyce Headlines Table ---
    st.subheader("Generated vs. Joyce Headlines")

    gen_rows = []
    overlap_sections = 0
    total_overlap_pct = 0

    for i, (hl, text) in enumerate(sections):
        keywords = [term for term, _ in tfidf_results.get(i, [])[:top_k]]
        generated = " ".join(kw.upper() for kw in keywords)

        headline_words = set(
            w.lower().strip(".,!?;:") for w in hl.split() if w.isalpha()
        )
        keyword_set = set(keywords)
        overlap = headline_words & keyword_set
        overlap_pct = len(overlap) / len(headline_words) * 100 if headline_words else 0
        if overlap:
            overlap_sections += 1
        total_overlap_pct += overlap_pct

        gen_rows.append({
            "Section": i + 1,
            "Joyce's Headline": hl,
            "Generated Headline": generated,
            "Overlap": ", ".join(overlap) if overlap else "—",
        })

    st.dataframe(pd.DataFrame(gen_rows), use_container_width=True, hide_index=True)

    # --- "Build Your Own Headline" Interactive ---
    st.subheader("Build Your Own Headline")

    build_section_label = st.selectbox(
        "Select a section",
        section_options,
        key="build_headline_section",
    )
    build_idx = section_options.index(build_section_label)
    build_headline, build_text = sections[build_idx]

    # Show section text
    with st.expander("View section text"):
        st.write(build_text)

    # Top-10 keywords as selectable options
    build_scores = tfidf_results.get(build_idx, [])[:10]
    build_terms = [t for t, _ in build_scores]
    default_selection = build_terms[:3]

    selected_words = st.multiselect(
        "Select keywords for your headline",
        build_terms,
        default=default_selection,
        key="headline_words",
    )

    custom_words = st.text_input(
        "Add your own words (comma-separated)",
        value="",
        key="custom_headline_words",
    )

    all_headline_words = list(selected_words)
    if custom_words.strip():
        all_headline_words.extend(w.strip() for w in custom_words.split(",") if w.strip())

    if all_headline_words:
        your_headline = " ".join(w.upper() for w in all_headline_words)
        st.markdown(f"**Your headline:** {your_headline}")
        st.markdown(f"**Joyce's headline:** {build_headline}")
    else:
        st.info("Select or type words to build a headline.")

    # --- Keyword Score Explorer ---
    st.subheader("Keyword Score Explorer")

    build_all_scores = tfidf_results.get(build_idx, [])[:20]
    if build_all_scores:
        joyce_words = set(
            w.lower().strip(".,!?;:") for w in build_headline.split() if w.isalpha()
        )

        score_terms = [t for t, _ in build_all_scores]
        score_vals = [s for _, s in build_all_scores]
        score_colors = [
            "#DAA520" if t in joyce_words else "#A0A0A0"
            for t in score_terms
        ]

        fig_kw, ax_kw = plt.subplots(figsize=(8, max(3, len(score_terms) * 0.35)))
        ax_kw.barh(range(len(score_terms)), score_vals, color=score_colors)
        ax_kw.set_yticks(range(len(score_terms)))
        ax_kw.set_yticklabels(score_terms, fontsize=9)
        ax_kw.invert_yaxis()
        ax_kw.set_xlabel("TF-IDF Score")
        ax_kw.set_title(f"Keyword Scores — Section {build_idx+1}")

        from matplotlib.patches import Patch
        ax_kw.legend(
            handles=[
                Patch(facecolor="#DAA520", label="In Joyce's headline"),
                Patch(facecolor="#A0A0A0", label="Not in headline"),
            ],
            loc="lower right",
            fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig_kw)
        plt.close(fig_kw)

        st.markdown(
            "Gold bars show words Joyce actually used in the headline. Notice how "
            "Joyce's choices often have *low* TF-IDF scores — they're chosen for "
            "meaning, irony, or narrative function, not statistical distinctiveness."
        )

    # --- Whole-Episode Summary ---
    st.subheader("Whole-Episode Summary")

    # Compute per-section overlap stats
    overlaps_pct = []
    best_match_idx = 0
    best_match_pct = 0
    worst_match_idx = 0
    worst_match_pct = 100

    for i, (hl, _) in enumerate(sections):
        keywords = set(term for term, _ in tfidf_results.get(i, [])[:top_k])
        headline_words = set(
            w.lower().strip(".,!?;:") for w in hl.split() if w.isalpha()
        )
        overlap = keywords & headline_words
        pct = len(overlap) / len(headline_words) * 100 if headline_words else 0
        overlaps_pct.append(pct)
        if pct > best_match_pct:
            best_match_pct = pct
            best_match_idx = i
        if pct < worst_match_pct:
            worst_match_pct = pct
            worst_match_idx = i

    avg_overlap = sum(overlaps_pct) / len(overlaps_pct) if overlaps_pct else 0
    sections_with_overlap = sum(1 for p in overlaps_pct if p > 0)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Sections with overlap",
        f"{sections_with_overlap}/{len(sections)} ({sections_with_overlap/len(sections)*100:.0f}%)",
    )
    s2.metric("Average overlap", f"{avg_overlap:.1f}%")
    s3.metric(
        "Best match",
        f"Section {best_match_idx+1} ({best_match_pct:.0f}%)",
        help=f"Headline: {sections[best_match_idx][0]}",
    )
    s4.metric(
        "Worst match",
        f"Section {worst_match_idx+1} ({worst_match_pct:.0f}%)",
        help=f"Headline: {sections[worst_match_idx][0]}",
    )

else:
    # Non-Aeolus: still show generated headlines from chunks
    st.subheader("Generated Headlines from Chunks")

    st.info(
        "Headline comparison with Joyce is only available for Aeolus. "
        "Below are TF-IDF-generated headlines for each chunk of this episode."
    )

    chunk_rows = []
    for i, (label, text) in enumerate(sections):
        keywords = [term for term, _ in tfidf_results.get(i, [])[:top_k]]
        generated = " ".join(kw.upper() for kw in keywords)
        chunk_rows.append({
            "Section": label,
            "Generated Headline": generated,
        })
    st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)

    # Still show Build Your Own Headline
    st.subheader("Build Your Own Headline")

    build_section_label = st.selectbox(
        "Select a section",
        section_options,
        key="build_headline_section_non_aeolus",
    )
    build_idx = section_options.index(build_section_label)
    build_text = sections[build_idx][1]

    with st.expander("View section text"):
        st.write(build_text)

    build_scores = tfidf_results.get(build_idx, [])[:10]
    build_terms = [t for t, _ in build_scores]

    selected_words = st.multiselect(
        "Select keywords for your headline",
        build_terms,
        default=build_terms[:3],
        key="headline_words_non_aeolus",
    )

    custom_words = st.text_input(
        "Add your own words (comma-separated)",
        value="",
        key="custom_headline_words_non_aeolus",
    )

    all_headline_words = list(selected_words)
    if custom_words.strip():
        all_headline_words.extend(w.strip() for w in custom_words.split(",") if w.strip())

    if all_headline_words:
        st.markdown(f"**Your headline:** {' '.join(w.upper() for w in all_headline_words)}")


st.markdown("""
---

**What this week reveals:** TF-IDF quantifies what's statistically distinctive, but Joyce's
headlines reveal what's *narratively meaningful*. The gap between computational keywords and
editorial headlines is not a failure of the algorithm — it's a precise measurement of what
bag-of-words methods cannot capture: irony, event compression, thematic weight, and cultural
allusion. Rhetoric detection with simple heuristics similarly exposes the distance between
pattern-matching and genuine linguistic understanding. These gaps motivate the move from
frequency-based to sequence-based NLP.
""")
