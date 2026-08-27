"""
Week 10 — Wandering Rocks
Text similarity, interpolation detection, and entity tracking across the labyrinth.
"""

import contextlib
import io
import math
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import networkx as nx

for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]:
    nltk.download(resource, quiet=True)

from week10.week10_wanderingrocks import (
    split_wandering_rocks,
    tfidf_vectors,
    cosine_similarity,
    sentence_tfidf_vector,
    extract_entities_from_section,
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
    page_title="Week 10 — Wandering Rocks",
    page_icon="📖",
    layout="wide",
)
st.title("Week 10 — Wandering Rocks")
st.caption(
    "Text Similarity, Interpolation Detection & Entity Tracking Across the Labyrinth"
)

# Section labels from scholarly consensus
SECTION_LABELS = [
    "Father Conmee's walk",
    "Denis Maginni",
    "Corny Kelleher (funeral director)",
    "Corny Kelleher continued",
    "One-legged sailor",
    "Father Conmee continued",
    "The throwaway skiff on the Liffey",
    "Blazes Boylan shopping",
    "Lenehan looking for Boylan",
    "Lenehan at the pub",
    "Dilly Dedalus",
    "Mr Kernan",
    "The viceregal cavalcade",
    "Mr Kernan continued",
    "Stephen Dedalus",
    "Father Conmee's return",
    "Ben Dollard mentioned",
    "Ben Dollard appears",
    "Blazes Boylan's return",
]


# ============================================================================
# Helpers
# ============================================================================


def split_into_chunks(text, n_chunks=19):
    """Split non-Wandering-Rocks episodes into paragraph-based chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < n_chunks:
        return paragraphs
    chunk_size = len(paragraphs) / n_chunks
    chunks = []
    for i in range(n_chunks):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        chunks.append("\n\n".join(paragraphs[start:end]))
    return chunks


def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_split_sections(text, n_sections, is_wandering_rocks):
    if is_wandering_rocks:
        return split_wandering_rocks(text)
    return split_into_chunks(text, n_sections)


@st.cache_data
def cached_tfidf_vectors(sections_tuple):
    sections = list(sections_tuple)
    return tfidf_vectors(sections)


@st.cache_data
def cached_similarity_matrix(sections_tuple):
    """Compute pairwise cosine similarity matrix."""
    sections = list(sections_tuple)
    vectors, vocab, df = tfidf_vectors(sections)
    n = len(sections)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(vectors[i], vectors[j])
    return matrix, vectors, vocab, df


@st.cache_data
def cached_detect_interpolations(sections_tuple, threshold, min_tokens):
    """Detect interpolations with configurable threshold and minimum token count."""
    sections = list(sections_tuple)
    vectors, vocab, df = tfidf_vectors(sections)
    N = len(sections)
    all_anomalies = []

    for sec_idx, section in enumerate(sections):
        sec_vec = vectors[sec_idx]
        if not sec_vec:
            continue
        sentences = sent_tokenize(section)
        if len(sentences) < 3:
            continue
        scored = []
        for sent in sentences:
            sent_vec = sentence_tfidf_vector(sent, df, N)
            if not sent_vec:
                continue
            sim = cosine_similarity(sent_vec, sec_vec)
            scored.append((sent, sim))
        scored.sort(key=lambda x: x[1])
        for sent, sim in scored[:2]:
            if sim < threshold and len(word_tokenize(sent)) > min_tokens:
                all_anomalies.append((sec_idx + 1, sim, sent))

    all_anomalies.sort(key=lambda x: x[1])
    return all_anomalies


@st.cache_data
def cached_sentence_scores(sections_tuple):
    """Compute similarity scores for all sentences in all sections."""
    sections = list(sections_tuple)
    vectors, vocab, df = tfidf_vectors(sections)
    N = len(sections)
    all_scores = []  # list of (section_idx_1based, similarity, sentence)

    for sec_idx, section in enumerate(sections):
        sec_vec = vectors[sec_idx]
        if not sec_vec:
            continue
        sentences = sent_tokenize(section)
        for sent in sentences:
            sent_vec = sentence_tfidf_vector(sent, df, N)
            if not sent_vec:
                continue
            sim = cosine_similarity(sent_vec, sec_vec)
            all_scores.append((sec_idx + 1, sim, sent))

    return all_scores


@st.cache_data
def cached_entity_tracking(sections_tuple):
    """Track entities across sections."""
    sections = list(sections_tuple)
    section_entities = []
    entity_sections = defaultdict(list)

    for i, section in enumerate(sections):
        entities = extract_entities_from_section(section)
        section_entities.append(entities)
        for entity in entities:
            entity_sections[entity].append(i + 1)

    # Normalize entity names
    normalized = defaultdict(list)
    for entity, secs in entity_sections.items():
        if entity in ("Father", "Conmee"):
            normalized["Father Conmee"].extend(secs)
        else:
            normalized[entity].extend(secs)

    for entity in normalized:
        normalized[entity] = sorted(set(normalized[entity]))

    # Convert to regular dict for caching
    return dict(normalized), [list(s) for s in section_entities]


@st.cache_data
def cached_sentence_cross_section_similarity(sentence, sections_tuple):
    """Compute similarity of a sentence to every section centroid."""
    sections = list(sections_tuple)
    vectors, vocab, df = tfidf_vectors(sections)
    N = len(sections)
    sent_vec = sentence_tfidf_vector(sentence, df, N)
    sims = []
    for i, sec_vec in enumerate(vectors):
        sims.append(cosine_similarity(sent_vec, sec_vec))
    return sims


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=9,  # Wandering Rocks
    caption="Week 10: Text Similarity & Entity Tracking",
)

is_wandering_rocks = episode_file == "10wanderingrocks.txt"

with st.sidebar:
    n_sections = st.slider(
        "Number of sections",
        5, 30, 19,
        key="n_sections",
        help="For Wandering Rocks, always uses 19 (scholarly consensus). "
             "For other episodes, controls paragraph-based chunking.",
    )
    st.divider()
    st.markdown(
        "**Wandering Rocks** is Joyce's labyrinth — 19 interlocking sections "
        "following different Dubliners simultaneously through Dublin. "
        "**Interpolations** (sentences from one section intruding into another) "
        "mark simultaneous events, threading the city together."
    )

# Load data
episode_text = cached_load_episode(episode_file)
sections = cached_split_sections(episode_text, n_sections, is_wandering_rocks)
sections_tuple = tuple(sections)


def section_label(idx):
    """Return a display label for a section (0-indexed)."""
    if is_wandering_rocks and idx < len(SECTION_LABELS):
        return f"{idx + 1}. {SECTION_LABELS[idx]}"
    return f"Section {idx + 1}"


# ============================================================================
# Section 1: Section Similarity Matrix
# ============================================================================

st.header("1. Section Similarity Matrix")

st.markdown(
    "TF-IDF vectors for each section are compared pairwise using cosine similarity. "
    "High similarity between non-adjacent sections may indicate shared characters, "
    "themes, or interpolated content linking them across the labyrinth."
)

matrix, vectors, vocab, df = cached_similarity_matrix(sections_tuple)
n = len(sections)

# --- Metrics row ---
# Average pairwise similarity (excluding diagonal)
upper_tri = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
avg_sim = sum(upper_tri) / len(upper_tri) if upper_tri else 0

m1, m2, m3 = st.columns(3)
m1.metric("Sections Parsed", n)
m2.metric("Unique Vocabulary", f"{len(vocab):,}")
m3.metric("Avg Pairwise Similarity", f"{avg_sim:.4f}")

# --- Similarity heatmap ---
st.subheader("Similarity Heatmap")

section_labels_list = [section_label(i) for i in range(n)]

fig_heat, ax_heat = plt.subplots(
    figsize=(max(8, n * 0.55), max(6, n * 0.45))
)
# Scale color to off-diagonal range so subtle variation isn't crushed by the diagonal
off_diag = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
vmin = min(off_diag) if off_diag else 0
vmax = max(off_diag) if off_diag else 1
im = ax_heat.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
ax_heat.set_xticks(range(n))
ax_heat.set_xticklabels(
    [f"{i+1}" for i in range(n)], fontsize=8
)
ax_heat.set_yticks(range(n))
ax_heat.set_yticklabels(section_labels_list, fontsize=7)
ax_heat.set_xlabel("Section")
ax_heat.set_ylabel("Section")
ax_heat.set_title(f"Section Similarity Matrix — {episode_label}")
fig_heat.colorbar(im, ax=ax_heat, label="Cosine Similarity")
plt.tight_layout()
st.pyplot(fig_heat)
plt.close(fig_heat)

# --- Top similar pairs table ---
st.subheader("Top Similar Pairs")

pairs = []
for i in range(n):
    for j in range(i + 1, n):
        # Find top shared keywords
        shared = set(vectors[i].keys()) & set(vectors[j].keys())
        top_shared = sorted(
            shared,
            key=lambda t: -(vectors[i].get(t, 0) + vectors[j].get(t, 0)),
        )[:5]
        pairs.append((i, j, matrix[i][j], ", ".join(top_shared)))
pairs.sort(key=lambda x: -x[2])

pair_rows = []
for i, j, sim, keywords in pairs[:10]:
    pair_rows.append({
        "Section A": section_label(i),
        "Section B": section_label(j),
        "Cosine Similarity": f"{sim:.4f}",
        "Top Shared Keywords": keywords,
    })
st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)

# --- Section deep-dive ---
st.subheader("Section Deep-Dive")

section_options = [section_label(i) for i in range(n)]
selected_section = st.selectbox(
    "Select a section", section_options, key="sim_section_dive"
)
selected_idx = section_options.index(selected_section)

# Bar chart of similarity to all other sections
other_sims = [(i, matrix[selected_idx][i]) for i in range(n) if i != selected_idx]
other_sims.sort(key=lambda x: -x[1])

fig_bar, ax_bar = plt.subplots(figsize=(8, max(3, (n - 1) * 0.3)))
bar_labels = [section_label(i) for i, _ in other_sims]
bar_vals = [s for _, s in other_sims]
ax_bar.barh(range(len(bar_labels)), bar_vals, color="#4A9D8E")
ax_bar.set_yticks(range(len(bar_labels)))
ax_bar.set_yticklabels(bar_labels, fontsize=7)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Cosine Similarity")
ax_bar.set_title(f"Similarity to {section_label(selected_idx)}")
plt.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

# Top TF-IDF keywords for this section
top_terms = sorted(vectors[selected_idx].items(), key=lambda x: -x[1])[:10]
if top_terms:
    st.markdown(
        f"**Top TF-IDF keywords:** {', '.join(f'{t} ({s:.3f})' for t, s in top_terms)}"
    )

with st.expander("View section text"):
    st.write(sections[selected_idx])

# --- Cross-episode comparison ---
st.subheader("Cross-Episode Comparison")

compare_episodes = st.multiselect(
    "Compare intra-episode similarity with other episodes",
    [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
    default=[],
    key="sim_cross_ep",
)

if compare_episodes:
    comparison_data = [{"Episode": episode_label, "Avg Pairwise Similarity": f"{avg_sim:.4f}"}]

    for cmp_label in compare_episodes:
        cmp_file = EPISODE_FILES[EPISODE_LABELS.index(cmp_label)]
        cmp_text = cached_load_episode(cmp_file)
        cmp_is_wr = cmp_file == "10wanderingrocks.txt"
        cmp_sections = cached_split_sections(cmp_text, n_sections, cmp_is_wr)
        cmp_tuple = tuple(cmp_sections)
        cmp_matrix, _, _, _ = cached_similarity_matrix(cmp_tuple)
        cmp_n = len(cmp_sections)
        cmp_upper = [cmp_matrix[i][j] for i in range(cmp_n) for j in range(i + 1, cmp_n)]
        cmp_avg = sum(cmp_upper) / len(cmp_upper) if cmp_upper else 0
        comparison_data.append({
            "Episode": cmp_label,
            "Avg Pairwise Similarity": f"{cmp_avg:.4f}",
        })

    st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)
    st.markdown(
        "*Lower average similarity suggests more fragmented, diverse content across sections. "
        "Wandering Rocks' labyrinthine structure typically produces lower intra-episode "
        "similarity than more continuous episodes.*"
    )


# ============================================================================
# Section 2: Interpolation Detection
# ============================================================================

st.header("2. Interpolation Detection")

st.markdown(
    "Joyce threads Wandering Rocks together with **interpolations** — sentences from "
    "one section intruding into another to mark simultaneous events. We flag sentences "
    "with abnormally low cosine similarity to their section's TF-IDF centroid as "
    "potential interpolations. If a flagged sentence matches a *different* section "
    "better than its own, it may truly be an interpolation."
)

# --- Threshold controls ---
thresh_col1, thresh_col2 = st.columns(2)
with thresh_col1:
    sim_threshold = st.slider(
        "Similarity threshold (anomaly cutoff)",
        0.0, 0.3, 0.1, step=0.01,
        key="interp_threshold",
        help="Sentences with similarity below this value are flagged as anomalies.",
    )
with thresh_col2:
    min_token_count = st.slider(
        "Minimum token count",
        3, 15, 5,
        key="interp_min_tokens",
        help="Ignore very short sentences that produce unreliable TF-IDF vectors.",
    )

anomalies = cached_detect_interpolations(sections_tuple, sim_threshold, min_token_count)

# --- Metrics row ---
sections_with_anomalies = len(set(sec for sec, _, _ in anomalies))
avg_anomaly_score = (
    sum(sim for _, sim, _ in anomalies) / len(anomalies)
    if anomalies else 0
)

am1, am2, am3 = st.columns(3)
am1.metric("Anomalies Detected", len(anomalies))
am2.metric("Sections with Anomalies", sections_with_anomalies)
am3.metric("Avg Anomaly Score", f"{avg_anomaly_score:.4f}")

# --- Anomaly table ---
if anomalies:
    st.subheader("Anomalous Sentences")
    anomaly_rows = []
    for sec, sim, sent in anomalies:
        anomaly_rows.append({
            "Section": section_label(sec - 1),
            "Similarity": f"{sim:.4f}",
            "Sentence": sent[:150] + ("..." if len(sent) > 150 else ""),
        })
    st.dataframe(
        pd.DataFrame(anomaly_rows), width="stretch", hide_index=True
    )

# --- Strip chart: sentence similarity by section ---
st.subheader("Sentence Similarity by Section")

all_sentence_scores = cached_sentence_scores(sections_tuple)

if all_sentence_scores:
    fig_strip, ax_strip = plt.subplots(figsize=(10, max(4, n * 0.3)))

    anomaly_set = set((sec, sent) for sec, _, sent in anomalies)

    normal_x, normal_y = [], []
    anom_x, anom_y = [], []
    for sec, sim, sent in all_sentence_scores:
        if (sec, sent) in anomaly_set:
            anom_x.append(sim)
            anom_y.append(sec)
        else:
            normal_x.append(sim)
            normal_y.append(sec)

    ax_strip.scatter(normal_x, normal_y, c="#4A90D9", alpha=0.4, s=12, label="Normal")
    ax_strip.scatter(anom_x, anom_y, c="#E07A5F", alpha=0.8, s=20, label="Anomaly", zorder=5)
    ax_strip.axvline(x=sim_threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({sim_threshold})")
    ax_strip.set_xlabel("Cosine Similarity to Section Centroid")
    ax_strip.set_ylabel("Section")
    ax_strip.set_yticks(range(1, n + 1))
    ax_strip.set_yticklabels(
        [section_label(i) for i in range(n)], fontsize=7
    )
    ax_strip.invert_yaxis()
    ax_strip.set_title("Sentence Similarity — Anomalies in Red")
    ax_strip.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_strip)
    plt.close(fig_strip)

# --- Sentence inspector ---
if anomalies:
    st.subheader("Sentence Inspector")

    inspector_options = [
        f"[Sec {sec}, sim={sim:.4f}] {sent[:80]}..."
        for sec, sim, sent in anomalies[:20]
    ]
    selected_anomaly = st.selectbox(
        "Select an anomalous sentence",
        inspector_options,
        key="sentence_inspector",
    )
    anom_idx = inspector_options.index(selected_anomaly)
    anom_sec, anom_sim, anom_sent = anomalies[anom_idx]

    st.markdown(f"**Full sentence:** {anom_sent}")
    st.markdown(f"**Found in:** {section_label(anom_sec - 1)}")
    st.markdown(f"**Similarity to own section:** {anom_sim:.4f}")

    # Cross-section similarity for this sentence
    cross_sims = cached_sentence_cross_section_similarity(anom_sent, sections_tuple)
    best_match = max(range(len(cross_sims)), key=lambda i: cross_sims[i])

    fig_cross, ax_cross = plt.subplots(figsize=(8, max(3, n * 0.3)))
    colors = []
    for i in range(len(cross_sims)):
        if i == anom_sec - 1:
            colors.append("#E07A5F")  # own section
        elif i == best_match:
            colors.append("#81B29A")  # best match
        else:
            colors.append("#4A90D9")

    ax_cross.barh(range(len(cross_sims)), cross_sims, color=colors)
    ax_cross.set_yticks(range(len(cross_sims)))
    ax_cross.set_yticklabels([section_label(i) for i in range(len(cross_sims))], fontsize=7)
    ax_cross.invert_yaxis()
    ax_cross.set_xlabel("Cosine Similarity")
    ax_cross.set_title("Sentence Similarity to Each Section Centroid")

    from matplotlib.patches import Patch
    ax_cross.legend(
        handles=[
            Patch(facecolor="#E07A5F", label="Own section"),
            Patch(facecolor="#81B29A", label="Best matching section"),
            Patch(facecolor="#4A90D9", label="Other sections"),
        ],
        loc="lower right",
        fontsize=8,
    )
    plt.tight_layout()
    st.pyplot(fig_cross)
    plt.close(fig_cross)

    if best_match != anom_sec - 1:
        st.success(
            f"This sentence matches **{section_label(best_match)}** "
            f"(sim={cross_sims[best_match]:.4f}) better than its own section "
            f"(sim={cross_sims[anom_sec - 1]:.4f}) — likely a genuine interpolation!"
        )
    else:
        st.info(
            "This sentence matches its own section best — it may be anomalous "
            "in topic but not necessarily an interpolation from another section."
        )


# ============================================================================
# Section 3: Entity Tracking Across the Labyrinth
# ============================================================================

st.header("3. Entity Tracking Across the Labyrinth")

st.markdown(
    "Named Entity Recognition (NER) via NLTK extracts character and place names from "
    "each section. Tracking which entities appear across multiple sections reveals how "
    "Joyce weaves characters through the labyrinth — and which section pairs are most "
    "connected by shared characters."
)

entity_sections_dict, section_entities_list = cached_entity_tracking(sections_tuple)

# Filter to multi-section entities
multi_section = {e: secs for e, secs in entity_sections_dict.items() if len(secs) > 1}
sorted_entities = sorted(multi_section.items(), key=lambda x: -len(x[1]))
all_entities_list = [e for e, _ in sorted_entities[:20]]

# Shared entity pairs between sections
shared_pairs = Counter()
for entity, secs in entity_sections_dict.items():
    for j in range(len(secs)):
        for k in range(j + 1, len(secs)):
            shared_pairs[(secs[j], secs[k])] += 1

most_connected = shared_pairs.most_common(1)
most_connected_str = (
    f"Sec {most_connected[0][0][0]} & {most_connected[0][0][1]} "
    f"({most_connected[0][1]} shared)"
    if most_connected else "—"
)

# --- Metrics row ---
em1, em2, em3 = st.columns(3)
em1.metric("Unique Entities", len(entity_sections_dict))
em2.metric("Entities Spanning 2+ Sections", len(multi_section))
em3.metric("Most Connected Pair", most_connected_str)

# --- Entity-section heatmap ---
st.subheader("Entity-Section Heatmap")

if all_entities_list:
    ent_matrix = np.zeros((n, len(all_entities_list)))
    for col, entity in enumerate(all_entities_list):
        for sec in entity_sections_dict.get(entity, []):
            if sec - 1 < n:
                ent_matrix[sec - 1][col] = 1

    fig_ent, ax_ent = plt.subplots(
        figsize=(max(8, len(all_entities_list) * 0.6), max(5, n * 0.35))
    )
    im_ent = ax_ent.imshow(ent_matrix, cmap="YlOrRd", aspect="auto")
    ax_ent.set_xticks(range(len(all_entities_list)))
    ax_ent.set_xticklabels(all_entities_list, rotation=45, ha="right", fontsize=8)
    ax_ent.set_yticks(range(n))
    ax_ent.set_yticklabels([section_label(i) for i in range(n)], fontsize=7)
    ax_ent.set_title("Entity Presence Across Sections (Top 20 Multi-Section Entities)")
    fig_ent.colorbar(im_ent, ax=ax_ent, label="Present", ticks=[0, 1])
    plt.tight_layout()
    st.pyplot(fig_ent)
    plt.close(fig_ent)
else:
    st.info("No multi-section entities detected.")

# --- Entity explorer ---
st.subheader("Entity Explorer")

if all_entities_list:
    selected_entities = st.multiselect(
        "Select entities to explore",
        all_entities_list,
        default=all_entities_list[:3] if len(all_entities_list) >= 3 else all_entities_list,
        key="entity_explorer",
    )

    if selected_entities:
        for entity in selected_entities:
            secs = entity_sections_dict.get(entity, [])
            presence = ["present" if (i + 1) in secs else "" for i in range(n)]
            st.markdown(
                f"**{entity}** — appears in sections: "
                f"{', '.join(str(s) for s in secs)}"
            )

        # Co-appearing entities in the same sections
        selected_secs = set()
        for entity in selected_entities:
            selected_secs.update(entity_sections_dict.get(entity, []))
        co_entities = set()
        for entity, secs in entity_sections_dict.items():
            if entity not in selected_entities and set(secs) & selected_secs:
                co_entities.add(entity)
        if co_entities:
            st.markdown(
                f"**Co-appearing entities:** {', '.join(sorted(co_entities)[:15])}"
            )

# --- Section connectivity chart ---
st.subheader("Section Connectivity by Shared Entities")

top_pairs = shared_pairs.most_common(10)
if top_pairs:
    fig_conn, ax_conn = plt.subplots(figsize=(8, max(3, len(top_pairs) * 0.4)))
    pair_labels = [f"Sec {s1} & Sec {s2}" for (s1, s2), _ in top_pairs]
    pair_counts = [c for _, c in top_pairs]
    ax_conn.barh(range(len(pair_labels)), pair_counts, color="#4A9D8E")
    ax_conn.set_yticks(range(len(pair_labels)))
    ax_conn.set_yticklabels(pair_labels, fontsize=8)
    ax_conn.invert_yaxis()
    ax_conn.set_xlabel("Shared Entity Count")
    ax_conn.set_title("Top Connected Section Pairs")
    plt.tight_layout()
    st.pyplot(fig_conn)
    plt.close(fig_conn)

# --- Bipartite network graph ---
st.subheader("Bipartite Network Graph")

min_appearances = st.slider(
    "Minimum section appearances (filter low-frequency entities)",
    1, max(5, max((len(s) for s in entity_sections_dict.values()), default=1)),
    3,
    key="min_appearances",
)

filtered_entities = {
    e: secs for e, secs in sorted_entities
    if len(secs) >= min_appearances
}

if filtered_entities:
    B = nx.Graph()

    section_nodes = [f"Sec {i+1}" for i in range(n)]
    entity_nodes = list(filtered_entities.keys())

    B.add_nodes_from(section_nodes, bipartite=0)
    B.add_nodes_from(entity_nodes, bipartite=1)

    for entity, secs in filtered_entities.items():
        for sec in secs:
            if sec <= n:
                B.add_edge(f"Sec {sec}", entity)

    # Compute degree for node coloring
    section_degrees = [B.degree(f"Sec {i+1}") for i in range(n)]
    entity_degrees = [B.degree(e) for e in entity_nodes]

    # Layout
    pos = {}
    for i, node in enumerate(section_nodes):
        pos[node] = (0, -i)
    for i, node in enumerate(entity_nodes):
        pos[node] = (2, -i * (n / max(len(entity_nodes), 1)))

    fig_net, ax_net = plt.subplots(
        figsize=(12, max(6, max(n, len(entity_nodes)) * 0.35))
    )

    # Draw edges
    nx.draw_networkx_edges(B, pos, ax=ax_net, alpha=0.3, edge_color="#888888")

    # Draw section nodes
    nx.draw_networkx_nodes(
        B, pos,
        nodelist=section_nodes,
        node_color=section_degrees,
        cmap=plt.cm.Blues,
        node_size=300,
        ax=ax_net,
    )

    # Draw entity nodes
    nx.draw_networkx_nodes(
        B, pos,
        nodelist=entity_nodes,
        node_color=entity_degrees,
        cmap=plt.cm.Oranges,
        node_size=200,
        node_shape="s",
        ax=ax_net,
    )

    # Labels
    nx.draw_networkx_labels(
        B, pos,
        labels={node: node for node in section_nodes},
        font_size=7,
        ax=ax_net,
    )
    nx.draw_networkx_labels(
        B, pos,
        labels={node: node for node in entity_nodes},
        font_size=7,
        ax=ax_net,
    )

    ax_net.set_title(
        f"Section-Entity Bipartite Graph ({len(filtered_entities)} entities, "
        f"min {min_appearances} appearances)"
    )
    ax_net.axis("off")
    plt.tight_layout()
    st.pyplot(fig_net)
    plt.close(fig_net)

    st.caption(
        f"{len(B.nodes())} nodes, {len(B.edges())} edges. "
        f"Circles = sections (colored by degree), squares = entities."
    )
else:
    st.info("No entities meet the minimum appearances filter.")


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Wandering Rocks is Joyce's most structurally fragmented episode —
19 scenes threaded together by interpolations that mark simultaneous events across Dublin.
TF-IDF similarity quantifies how distinct each section is; interpolation detection uses
that distinctiveness to flag intruding sentences; entity tracking maps the character network
that binds the labyrinth together. The combination reveals not just *what* Joyce wrote, but
*how* he wove a city's simultaneous life into a single narrative thread.
""")
