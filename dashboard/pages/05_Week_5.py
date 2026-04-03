"""
Week 05 — Lotus Eaters
WordNet semantic similarity, malapropisms, and substitution chains.
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
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize

for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "cmudict"]:
    nltk.download(resource, quiet=True)

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

from week05.week05_lotuseaters import THEMATIC_WORDS as LOTUS_EATERS_WORDS

# --- Constants ---

DEFAULT_WORD_PAIRS = [
    ("world", "word"),
    ("flower", "flour"),
    ("altar", "alter"),
    ("body", "bawdy"),
    ("sole", "soul"),
    ("sun", "son"),
    ("holy", "wholly"),
    ("rite", "right"),
    ("bread", "bred"),
    ("wine", "whine"),
]

LOTUS_EATERS_SUGGESTIONS = ["opium", "priest", "mass", "sleep", "perfume"]

LOTUS_EATERS_CHAIN_DEFAULTS = ["body", "bread", "flower", "drug", "water"]


# --- Stopwords for content word extraction ---
# Short function words, articles, dialogue tags — not thematic
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to",
    "from", "by", "with", "in", "of", "is", "it", "be", "as", "do", "so",
    "was", "were", "been", "are", "am", "has", "had", "have", "will", "would",
    "could", "should", "shall", "may", "might", "must", "did", "does", "done",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "not", "no", "yes", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "only", "own", "same", "into", "over", "after", "before",
    "about", "between", "through", "during", "under", "again", "then", "once",
    "here", "there", "also", "back", "much", "many", "well", "still", "even",
    "said", "says", "like", "know", "think", "come", "came", "went", "going",
    "make", "made", "take", "took", "give", "gave", "tell", "told", "look",
    "looked", "them", "they", "their", "your", "you", "him", "his", "her",
    "she", "he", "we", "our", "its", "my", "me", "out", "up", "now",
}

# Color palette
CLUSTER_COLORS = {
    "physical": "#4A90D9",
    "substance": "#81B29A",
    "activity": "#E07A5F",
    "other": "#999999",
}

RELATION_COLORS = {
    "synonym": "#4A90D9",
    "hypernym": "#50C878",
    "hyponym": "#E07A5F",
    "part_meronym": "#9B59B6",
    "substance_meronym": "#9B59B6",
    "part_holonym": "#F2CC8F",
    "substance_holonym": "#F2CC8F",
}

# --- Page config ---

st.set_page_config(page_title="Week 05 — Lotus Eaters", page_icon="📖", layout="wide")
st.title("Week 05 — Lotus Eaters")
st.caption(
    "WordNet Semantic Similarity, Malapropisms & Substitution Chains"
)

# --- Sidebar ---
episode_file, episode_label = episode_sidebar(
    default_index=4,  # Lotus Eaters
    caption="Week 5: WordNet & Semantic Similarity",
)

with st.sidebar:
    st.divider()
    st.markdown(
        "**Week 5** explores how Joyce's Lotus Eaters vocabulary forms a web of "
        "narcotic dissolution where religious, bodily, and chemical terms converge "
        "through shared WordNet hypernyms — and Martha Clifford's malapropisms "
        "('other world' / 'other word') exploit the gap between sound and meaning "
        "that taxonomies cannot bridge."
    )

episode_text = cached_load_episode(episode_file)


# ============================================================================
# Cached computation functions
# ============================================================================


@st.cache_data
def extract_thematic_words(text, n=15):
    """Extract top N content words with WordNet synsets from episode text.

    Filters out stopwords and short words, then ranks by frequency,
    keeping only words that have at least one WordNet synset.
    """
    tokens = word_tokenize(text)
    content_words = [
        t.lower() for t in tokens
        if t.isalpha() and len(t) > 3 and t.lower() not in _STOPWORDS
    ]
    freq = Counter(content_words)
    # Keep only words with synsets (meaningful in WordNet)
    result = []
    for word, _ in freq.most_common(n * 3):  # oversample then filter
        if wn.synsets(word):
            result.append(word)
            if len(result) >= n:
                break
    return result


@st.cache_data
def extract_chain_seeds(text, n=5):
    """Extract N good chain-starting words from episode text.

    Picks frequent content nouns with high synset counts (polysemous words
    make more interesting chains).
    """
    tokens = word_tokenize(text)
    content_words = [
        t.lower() for t in tokens
        if t.isalpha() and len(t) > 3 and t.lower() not in _STOPWORDS
    ]
    freq = Counter(content_words)
    # Score by frequency * synset count (prefer polysemous frequent words)
    scored = []
    for word, count in freq.most_common(50):
        synsets = wn.synsets(word)
        # Prefer nouns for chain exploration
        noun_synsets = [s for s in synsets if s.pos() == "n"]
        if noun_synsets:
            scored.append((word, count * len(noun_synsets)))
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:n]]


@st.cache_data
def compute_word_synset_data(words):
    """Compute synset data for each word: synsets, depth, path, definition, cluster."""
    results = {}
    for word in words:
        synsets = wn.synsets(word)
        if not synsets:
            continue

        # Prefer noun/verb over satellite adjective
        ss = synsets[0]
        if ss.pos() == "s" and len(ss.hypernym_paths()) > 0:
            shortest = min(ss.hypernym_paths(), key=len)
            if len(shortest) <= 1:
                for s in synsets:
                    if s.pos() in ["n", "v"] and len(s.hypernym_paths()) > 0:
                        ss = s
                        break

        paths = ss.hypernym_paths()
        shortest_path = min(paths, key=len) if paths else []
        path_names = [s.name() for s in shortest_path]

        # Determine cluster by top-level hypernym
        cluster = "other"
        for ancestor in shortest_path:
            name = ancestor.name()
            if "physical" in name or "whole.n" in name or "object.n" in name:
                cluster = "physical"
                break
            elif "substance" in name:
                cluster = "substance"
                break
            elif "activity" in name or "act.n" in name or "event.n" in name:
                cluster = "activity"
                break

        results[word] = {
            "synset_name": ss.name(),
            "definition": ss.definition(),
            "num_synsets": len(synsets),
            "path": path_names,
            "depth": len(shortest_path),
            "cluster": cluster,
            "all_synsets": [(s.name(), s.definition()) for s in synsets],
        }
    return results


@st.cache_data
def compute_similarity_matrix(words):
    """Compute NxN Wu-Palmer similarity matrix."""
    n = len(words)
    matrix = np.zeros((n, n))
    lcs_matrix = [["" for _ in range(n)] for _ in range(n)]

    synset_map = {}
    for word in words:
        synsets = wn.synsets(word)
        if synsets:
            ss = synsets[0]
            if ss.pos() == "s":
                for s in synsets:
                    if s.pos() in ["n", "v"]:
                        ss = s
                        break
            synset_map[word] = ss

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
                continue
            w1, w2 = words[i], words[j]
            if w1 in synset_map and w2 in synset_map:
                ss1, ss2 = synset_map[w1], synset_map[w2]
                sim = ss1.wup_similarity(ss2) or 0
                matrix[i][j] = sim
                matrix[j][i] = sim
                lcs_list = ss1.lowest_common_hypernyms(ss2)
                if lcs_list:
                    lcs_matrix[i][j] = lcs_list[0].name()
                    lcs_matrix[j][i] = lcs_list[0].name()
    return matrix, lcs_matrix


@st.cache_data
def compute_malapropism_data(pairs):
    """Compute semantic and phonological similarity for word pairs."""
    try:
        pronouncing = cmudict.dict()
    except Exception:
        pronouncing = {}

    results = []
    for w1, w2 in pairs:
        ss1 = wn.synsets(w1)
        ss2 = wn.synsets(w2)

        max_path_sim = 0
        max_wup_sim = 0
        if ss1 and ss2:
            for s1 in ss1:
                for s2 in ss2:
                    path_sim = s1.path_similarity(s2) or 0
                    wup_sim = s1.wup_similarity(s2) or 0
                    max_path_sim = max(max_path_sim, path_sim)
                    max_wup_sim = max(max_wup_sim, wup_sim)

        phon_dist = None
        p1_phonemes = None
        p2_phonemes = None
        if w1.lower() in pronouncing and w2.lower() in pronouncing:
            p1 = pronouncing[w1.lower()][0]
            p2 = pronouncing[w2.lower()][0]
            p1_phonemes = p1
            p2_phonemes = p2
            phon_dist = nltk.edit_distance(p1, p2)

        results.append(
            {
                "w1": w1,
                "w2": w2,
                "path_sim": max_path_sim,
                "wup_sim": max_wup_sim,
                "phon_dist": phon_dist,
                "p1_phonemes": p1_phonemes,
                "p2_phonemes": p2_phonemes,
            }
        )
    return results


@st.cache_data
def compute_chain(word, steps, relations):
    """Build a substitution chain through WordNet relations.

    Returns list of dicts with word, definition, relation_used.
    """
    chain = [{"word": word, "definition": "", "relation": "start"}]
    synsets = wn.synsets(word)
    if synsets:
        chain[0]["definition"] = synsets[0].definition()

    current = word
    visited = {word.lower()}

    relation_map = {
        "Synonyms": "synonym",
        "Hypernyms": "hypernym",
        "Hyponyms": "hyponym",
        "Part meronyms": "part_meronym",
        "Substance meronyms": "substance_meronym",
        "Part holonyms": "part_holonym",
        "Substance holonyms": "substance_holonym",
    }

    for _ in range(steps):
        synsets = wn.synsets(current)
        found = False

        for ss in synsets:
            if found:
                break

            # Try each relation type in order
            candidates = []

            if "Synonyms" in relations:
                for lemma in ss.lemmas():
                    c = lemma.name().replace("_", " ")
                    if c.lower() != current.lower() and c.lower() not in visited:
                        candidates.append((c, ss.definition(), "synonym"))

            if "Hypernyms" in relations:
                for h in ss.hypernyms():
                    for lemma in h.lemmas():
                        c = lemma.name().replace("_", " ")
                        if c.lower() not in visited:
                            candidates.append((c, h.definition(), "hypernym"))

            if "Hyponyms" in relations:
                for h in ss.hyponyms():
                    for lemma in h.lemmas():
                        c = lemma.name().replace("_", " ")
                        if c.lower() not in visited:
                            candidates.append((c, h.definition(), "hyponym"))

            if "Part meronyms" in relations:
                for m in ss.part_meronyms():
                    for lemma in m.lemmas():
                        c = lemma.name().replace("_", " ")
                        if c.lower() not in visited:
                            candidates.append((c, m.definition(), "part_meronym"))

            if "Substance meronyms" in relations:
                for m in ss.substance_meronyms():
                    for lemma in m.lemmas():
                        c = lemma.name().replace("_", " ")
                        if c.lower() not in visited:
                            candidates.append((c, m.definition(), "substance_meronym"))

            if "Part holonyms" in relations:
                for h in ss.part_holonyms():
                    for lemma in h.lemmas():
                        c = lemma.name().replace("_", " ")
                        if c.lower() not in visited:
                            candidates.append((c, h.definition(), "part_holonym"))

            if "Substance holonyms" in relations:
                for h in ss.substance_holonyms():
                    for lemma in h.lemmas():
                        c = lemma.name().replace("_", " ")
                        if c.lower() not in visited:
                            candidates.append((c, h.definition(), "substance_holonym"))

            if candidates:
                best = candidates[0]
                chain.append(
                    {"word": best[0], "definition": best[1], "relation": best[2]}
                )
                visited.add(best[0].lower())
                current = best[0]
                found = True

        if not found:
            break

    return chain


@st.cache_data
def compute_polysemy(episode_file):
    """Compute average synset count per content word for an episode."""
    text = cached_load_episode(episode_file)
    tokens = word_tokenize(text)
    content_words = [t.lower() for t in tokens if t.isalpha() and len(t) > 3]

    total_synsets = 0
    words_with_synsets = 0
    for w in content_words:
        ss = wn.synsets(w)
        if ss:
            total_synsets += len(ss)
            words_with_synsets += 1

    avg = total_synsets / words_with_synsets if words_with_synsets else 0
    coverage = words_with_synsets / len(content_words) * 100 if content_words else 0
    return {
        "avg_synsets": avg,
        "content_words": len(content_words),
        "words_with_synsets": words_with_synsets,
        "coverage": coverage,
    }


@st.cache_data
def compute_chapter_coherence(episode_file):
    """Compute average pairwise WuP similarity of top 15 content words."""
    text = cached_load_episode(episode_file)
    tokens = word_tokenize(text)
    content_words = [t.lower() for t in tokens if t.isalpha() and len(t) > 3]
    freq = Counter(content_words)
    top_words = [w for w, _ in freq.most_common(15)]

    synset_map = {}
    for w in top_words:
        synsets = wn.synsets(w)
        if synsets:
            synset_map[w] = synsets[0]

    words_with_ss = [w for w in top_words if w in synset_map]
    if len(words_with_ss) < 2:
        return 0.0

    total_sim = 0
    count = 0
    for i in range(len(words_with_ss)):
        for j in range(i + 1, len(words_with_ss)):
            sim = synset_map[words_with_ss[i]].wup_similarity(
                synset_map[words_with_ss[j]]
            )
            if sim is not None:
                total_sim += sim
                count += 1

    return total_sim / count if count else 0.0


# ============================================================================
# Section 1: Semantic Fields of Narcosis
# ============================================================================

st.header("1. Semantic Fields of Narcosis")

st.markdown(
    "How do Joyce's thematic words relate in WordNet's taxonomy? "
    "This section reveals hidden conceptual clusters beneath the surface vocabulary — "
    "body/blood/bread converging through shared hypernyms, water/bath/floating "
    "clustering in substance, altar/sacrament/communion in activity."
)

# --- Word selection (dynamic based on selected episode) ---
is_lotus_eaters = episode_file == "05lotuseaters.txt"

if is_lotus_eaters:
    default_thematic = LOTUS_EATERS_WORDS
    suggestions = LOTUS_EATERS_SUGGESTIONS
else:
    default_thematic = extract_thematic_words(episode_text, n=15)
    # Suggest Lotus Eaters words that aren't already in the dynamic list
    suggestions = [w for w in LOTUS_EATERS_WORDS if w not in default_thematic][:5]

all_options = list(dict.fromkeys(default_thematic + suggestions))

selected_words = st.multiselect(
    "Thematic words",
    options=all_options,
    default=default_thematic,
    key="thematic_words",
)

# Quick-add buttons
if suggestions:
    st.caption("Quick-add suggestions:")
    add_cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with add_cols[i]:
            if st.button(suggestion, key=f"add_{suggestion}"):
                if suggestion not in selected_words:
                    selected_words.append(suggestion)

threshold = st.slider(
    "WuP similarity threshold", 0.1, 0.8, 0.3, 0.05, key="wup_threshold"
)

if selected_words:
    words_tuple = tuple(selected_words)
    word_data = compute_word_synset_data(words_tuple)
    valid_words = [w for w in selected_words if w in word_data]

    if valid_words:
        sim_matrix, lcs_matrix = compute_similarity_matrix(tuple(valid_words))

        # --- Metrics row ---
        m1, m2, m3, m4 = st.columns(4)
        total_synsets = sum(word_data[w]["num_synsets"] for w in valid_words)
        avg_depth = np.mean([word_data[w]["depth"] for w in valid_words])
        n = len(valid_words)
        pairs_above = sum(
            1
            for i in range(n)
            for j in range(i + 1, n)
            if sim_matrix[i][j] > threshold
        )

        m1.metric("Words Analyzed", len(valid_words))
        m2.metric("Total Synsets", total_synsets)
        m3.metric("Avg Hypernym Depth", f"{avg_depth:.1f}")
        m4.metric("Pairs Above Threshold", pairs_above)

        # --- Semantic Similarity Heatmap ---
        st.subheader("Semantic Similarity Heatmap")

        # Hierarchical clustering for word ordering
        display_order = list(range(n))
        if SCIPY_AVAILABLE and n > 2:
            # Convert similarity to distance
            dist_matrix = 1 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)
            # Ensure symmetry and no negative values
            dist_matrix = np.maximum(dist_matrix, 0)
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method="average")
            display_order = list(leaves_list(Z))

        ordered_words = [valid_words[i] for i in display_order]
        ordered_matrix = sim_matrix[np.ix_(display_order, display_order)]

        fig_heat, ax_heat = plt.subplots(
            figsize=(max(8, n * 0.7), max(7, n * 0.6))
        )
        im = ax_heat.imshow(ordered_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax_heat.set_xticks(range(n))
        ax_heat.set_xticklabels(ordered_words, rotation=45, ha="right", fontsize=9)
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(ordered_words, fontsize=9)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = ordered_matrix[i][j]
                    color = "white" if val > 0.6 else "black"
                    ax_heat.text(
                        j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color,
                    )

        fig_heat.colorbar(im, ax=ax_heat, label="Wu-Palmer Similarity")
        ax_heat.set_title("Semantic Similarity (hierarchically clustered)")
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        # --- Hypernym Depth Bar Chart ---
        st.subheader("Hypernym Depth by Word")

        depths = [(w, word_data[w]["depth"], word_data[w]["cluster"]) for w in valid_words]
        depths.sort(key=lambda x: -x[1])

        fig_depth, ax_depth = plt.subplots(figsize=(10, max(4, len(depths) * 0.35)))
        labels = [d[0] for d in depths]
        values = [d[1] for d in depths]
        colors = [CLUSTER_COLORS.get(d[2], "#999999") for d in depths]

        ax_depth.barh(range(len(labels)), values, color=colors)
        ax_depth.set_yticks(range(len(labels)))
        ax_depth.set_yticklabels(labels, fontsize=9)
        ax_depth.set_xlabel("Hypernym Depth (distance from root entity.n.01)")
        ax_depth.set_title("Hypernym Depth — Color by Top-Level Cluster")
        ax_depth.invert_yaxis()

        from matplotlib.patches import Patch

        ax_depth.legend(
            handles=[
                Patch(facecolor="#4A90D9", label="Physical objects"),
                Patch(facecolor="#81B29A", label="Substances"),
                Patch(facecolor="#E07A5F", label="Activities"),
                Patch(facecolor="#999999", label="Other"),
            ],
            loc="lower right",
            fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig_depth)
        plt.close(fig_depth)

        # --- Semantic Network Graph ---
        st.subheader("Semantic Network Graph")

        if not NETWORKX_AVAILABLE:
            st.warning("Install `networkx` for the network graph.")
        else:
            G = nx.Graph()
            for i in range(n):
                G.add_node(
                    valid_words[i],
                    synset_count=word_data[valid_words[i]]["num_synsets"],
                    cluster=word_data[valid_words[i]]["cluster"],
                )

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i][j] > threshold:
                        lcs_name = lcs_matrix[i][j]
                        lcs_short = lcs_name.split(".")[0] if lcs_name else ""
                        G.add_edge(
                            valid_words[i],
                            valid_words[j],
                            weight=sim_matrix[i][j],
                            lcs=lcs_short,
                        )

            if len(G.edges()) > 0:
                fig_net, ax_net = plt.subplots(figsize=(12, 9))
                pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(len(G.nodes())))

                node_sizes = [
                    word_data[node]["num_synsets"] * 80 + 200 for node in G.nodes()
                ]
                node_colors = [
                    CLUSTER_COLORS.get(word_data[node]["cluster"], "#999999")
                    for node in G.nodes()
                ]

                edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
                max_w = max(edge_weights) if edge_weights else 1

                nx.draw_networkx_edges(
                    G, pos,
                    width=[w / max_w * 4 + 0.5 for w in edge_weights],
                    alpha=0.3, edge_color="#999999", ax=ax_net,
                )
                nx.draw_networkx_nodes(
                    G, pos, node_size=node_sizes, node_color=node_colors,
                    alpha=0.9, edgecolors="#333333", linewidths=1, ax=ax_net,
                )
                nx.draw_networkx_labels(
                    G, pos, font_size=10, font_family="sans-serif",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
                    ax=ax_net,
                )

                # Edge labels with LCS
                edge_labels = {
                    (u, v): G[u][v]["lcs"]
                    for u, v in G.edges()
                    if G[u][v]["lcs"]
                }
                if edge_labels:
                    nx.draw_networkx_edge_labels(
                        G, pos, edge_labels=edge_labels, font_size=7, ax=ax_net,
                    )

                ax_net.set_title(
                    f"Semantic Network (WuP > {threshold}) — node size = synset count"
                )
                ax_net.axis("off")
                plt.tight_layout()
                st.pyplot(fig_net)
                plt.close(fig_net)
            else:
                st.info(
                    "No pairs above threshold. Try lowering the WuP similarity threshold."
                )

        # --- Hypernym Paths Explorer ---
        with st.expander("Hypernym Paths Explorer"):
            explorer_word = st.selectbox(
                "Select a word", valid_words, key="hypernym_explorer"
            )
            if explorer_word in word_data:
                data = word_data[explorer_word]
                path_display = " → ".join(
                    s.split(".")[0] for s in data["path"]
                )
                st.markdown(f"**Hypernym path:** {path_display}")
                st.markdown(f"**Primary synset:** `{data['synset_name']}` — {data['definition']}")
                st.markdown(f"**Depth:** {data['depth']} | **Total synsets:** {data['num_synsets']}")
                st.markdown("**All synsets:**")
                for sname, sdef in data["all_synsets"]:
                    st.markdown(f"- `{sname}`: {sdef}")

        # --- Semantic Coherence across chapters ---
        with st.expander("Semantic coherence across all 18 chapters"):
            if st.button("Compute coherence for all episodes", key="compute_coherence"):
                coherence_data = []
                progress = st.progress(0)
                for i, ef in enumerate(EPISODE_FILES):
                    coh = compute_chapter_coherence(ef)
                    coherence_data.append(
                        {"episode": EPISODE_MAP[ef], "coherence": coh}
                    )
                    progress.progress((i + 1) / len(EPISODE_FILES))
                progress.empty()

                fig_coh, ax_coh = plt.subplots(figsize=(14, 5))
                labels = [d["episode"] for d in coherence_data]
                values = [d["coherence"] for d in coherence_data]
                colors = [
                    "#E07A5F" if ef == episode_file else "#B0B0B0"
                    for ef in EPISODE_FILES
                ]
                ax_coh.bar(range(len(labels)), values, color=colors)
                ax_coh.set_xticks(range(len(labels)))
                ax_coh.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax_coh.set_ylabel("Avg Pairwise WuP Similarity")
                ax_coh.set_title("Semantic Coherence Across Episodes")
                plt.tight_layout()
                st.pyplot(fig_coh)
                plt.close(fig_coh)
else:
    st.info("Select at least one thematic word to begin.")


# ============================================================================
# Section 2: Martha's Malapropism — Sound vs. Meaning
# ============================================================================

st.header("2. Martha's Malapropism — Sound vs. Meaning")

st.markdown(
    "Martha Clifford's letter confuses 'I do not like that other **world**' "
    "with 'other **word**' — a pun that exposes the gap between phonological "
    "and semantic similarity. Near-homophones sound alike but mean nothing alike; "
    "the best puns live in this mismatch."
)

# --- Pair selection ---
pair_labels = [f"{w1} / {w2}" for w1, w2 in DEFAULT_WORD_PAIRS]
selected_pair_labels = st.multiselect(
    "Word pairs", pair_labels, default=pair_labels, key="pair_select"
)

selected_pairs = [
    DEFAULT_WORD_PAIRS[pair_labels.index(lbl)] for lbl in selected_pair_labels
]

# Custom pair input
cp1, cp2, cp3 = st.columns([2, 2, 1])
with cp1:
    custom_w1 = st.text_input("Word 1", value="", key="custom_w1")
with cp2:
    custom_w2 = st.text_input("Word 2", value="", key="custom_w2")
with cp3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Add pair", key="add_pair"):
        if custom_w1.strip() and custom_w2.strip():
            selected_pairs.append((custom_w1.strip().lower(), custom_w2.strip().lower()))

metric_choice = st.radio(
    "Similarity metric", ["Wu-Palmer", "Path Similarity", "Both"], horizontal=True,
    key="sim_metric",
)

if selected_pairs:
    mal_data = compute_malapropism_data(tuple(tuple(p) for p in selected_pairs))

    # --- Metrics row ---
    mm1, mm2, mm3, mm4 = st.columns(4)
    avg_sem = np.mean([d["wup_sim"] for d in mal_data])
    phon_dists = [d["phon_dist"] for d in mal_data if d["phon_dist"] is not None]
    avg_phon = np.mean(phon_dists) if phon_dists else 0

    # Best pun: largest gap between phonological closeness and semantic distance
    best_pun = "—"
    best_gap = -1
    for d in mal_data:
        if d["phon_dist"] is not None and d["phon_dist"] > 0:
            # Normalize: low phon_dist = sounds alike, low wup_sim = means different
            gap = (1.0 / d["phon_dist"]) * (1.0 - d["wup_sim"])
            if gap > best_gap:
                best_gap = gap
                best_pun = f"{d['w1']}/{d['w2']}"

    mm1.metric("Pairs Analyzed", len(mal_data))
    mm2.metric("Avg Semantic Similarity", f"{avg_sem:.3f}")
    mm3.metric("Avg Phonological Distance", f"{avg_phon:.1f}")
    mm4.metric("Best Pun", best_pun)

    # --- Pun Gap Scatter Plot ---
    st.subheader("The Pun Gap: Sound vs. Meaning")

    scatter_data = [d for d in mal_data if d["phon_dist"] is not None]

    if scatter_data:
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))

        x_vals = [d["phon_dist"] for d in scatter_data]
        y_vals = [d["wup_sim"] for d in scatter_data]
        labels_sc = [f"{d['w1']}/{d['w2']}" for d in scatter_data]

        # Color: default pairs coral, custom blue
        n_default = len([p for p in selected_pair_labels if p in pair_labels])
        colors = ["#E07A5F"] * min(n_default, len(scatter_data))
        colors += ["#4A90D9"] * (len(scatter_data) - len(colors))

        ax_scatter.scatter(x_vals, y_vals, c=colors[:len(scatter_data)], s=120,
                          edgecolors="#333333", linewidths=1, zorder=3)

        for i, label in enumerate(labels_sc):
            ax_scatter.annotate(
                label, (x_vals[i], y_vals[i]),
                textcoords="offset points", xytext=(8, 8), fontsize=9,
            )

        # Reference diagonal
        max_x = max(x_vals) + 1
        ax_scatter.plot([0, max_x], [0, 1], "--", color="#CCCCCC", alpha=0.5,
                       label="No pun zone")

        ax_scatter.set_xlabel("Phonological Distance (CMU edit distance) — lower = sounds more alike")
        ax_scatter.set_ylabel("Semantic Similarity (max WuP) — higher = means more alike")
        ax_scatter.set_title("The Pun Gap: Best puns live in the bottom-left")
        ax_scatter.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

    # --- Paired Horizontal Bar Chart ---
    st.subheader("Phonological vs. Semantic Similarity")

    # Sort by pun gap descending
    sorted_mal = sorted(
        mal_data,
        key=lambda d: (
            (1.0 / max(d["phon_dist"], 0.1)) * (1.0 - d["wup_sim"])
            if d["phon_dist"] is not None
            else 0
        ),
        reverse=True,
    )

    fig_bars, ax_bars = plt.subplots(figsize=(10, max(4, len(sorted_mal) * 0.5)))
    y_pos = np.arange(len(sorted_mal))
    bar_h = 0.35

    # Normalize phonological distance to 0-1 scale
    max_phon = max((d["phon_dist"] for d in sorted_mal if d["phon_dist"] is not None), default=1)
    if max_phon == 0:
        max_phon = 1

    phon_normalized = [
        (d["phon_dist"] / max_phon if d["phon_dist"] is not None else 0)
        for d in sorted_mal
    ]
    sem_vals = [d["wup_sim"] for d in sorted_mal]
    bar_labels = [f"{d['w1']} / {d['w2']}" for d in sorted_mal]

    ax_bars.barh(y_pos - bar_h / 2, phon_normalized, bar_h,
                label="Phonological distance (normalized)", color="#4A90D9")
    ax_bars.barh(y_pos + bar_h / 2, sem_vals, bar_h,
                label="Semantic similarity (WuP)", color="#E07A5F")

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(bar_labels, fontsize=9)
    ax_bars.set_xlabel("Score (0–1)")
    ax_bars.set_title("Phonological Distance vs. Semantic Similarity — mismatch = pun potential")
    ax_bars.legend(fontsize=8)
    ax_bars.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_bars)
    plt.close(fig_bars)

    # --- Phoneme details expander ---
    with st.expander("Phoneme details"):
        phoneme_rows = []
        for d in mal_data:
            phoneme_rows.append(
                {
                    "Word 1": d["w1"],
                    "CMU Phonemes (1)": " ".join(d["p1_phonemes"]) if d["p1_phonemes"] else "N/A",
                    "Word 2": d["w2"],
                    "CMU Phonemes (2)": " ".join(d["p2_phonemes"]) if d["p2_phonemes"] else "N/A",
                    "Edit Distance": d["phon_dist"] if d["phon_dist"] is not None else "N/A",
                    "Shared Phonemes": (
                        len(set(d["p1_phonemes"]) & set(d["p2_phonemes"]))
                        if d["p1_phonemes"] and d["p2_phonemes"]
                        else "N/A"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(phoneme_rows), use_container_width=True, hide_index=True)

    # --- Test your own pun ---
    with st.expander("Test your own pun"):
        tp1, tp2 = st.columns(2)
        with tp1:
            test_w1 = st.text_input("Test word 1", value="night", key="test_w1")
        with tp2:
            test_w2 = st.text_input("Test word 2", value="knight", key="test_w2")

        if test_w1.strip() and test_w2.strip():
            test_data = compute_malapropism_data(
                ((test_w1.strip().lower(), test_w2.strip().lower()),)
            )[0]

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("WuP Similarity", f"{test_data['wup_sim']:.3f}")
            tc2.metric("Path Similarity", f"{test_data['path_sim']:.3f}")
            tc3.metric(
                "Phonological Distance",
                str(test_data["phon_dist"]) if test_data["phon_dist"] is not None else "N/A",
            )

            # Verdict
            if test_data["phon_dist"] is not None:
                if test_data["phon_dist"] <= 2 and test_data["wup_sim"] < 0.3:
                    st.success("**Strong pun** — sounds very alike, means very different!")
                elif test_data["phon_dist"] <= 3 and test_data["wup_sim"] < 0.5:
                    st.warning("**Weak pun** — some sound similarity, moderate meaning overlap.")
                else:
                    st.info("**Not a pun** — too different phonologically or too similar semantically.")
            else:
                st.info("One or both words not found in CMU Pronouncing Dictionary.")

            # Show synsets
            for w in [test_w1.strip().lower(), test_w2.strip().lower()]:
                ss = wn.synsets(w)
                if ss:
                    st.markdown(f"**Synsets for '{w}':**")
                    for s in ss[:5]:
                        st.markdown(f"- `{s.name()}`: {s.definition()}")
else:
    st.info("Select at least one word pair.")


# ============================================================================
# Section 3: Substitution Chains
# ============================================================================

st.header("3. Substitution Chains")

st.markdown(
    "Words transform through WordNet relations — synonyms, hypernyms, hyponyms, "
    "meronyms, holonyms — drifting from one meaning to the next. "
    "A linguistic analogy for the narcotic drift of the Lotus Eaters episode."
)

# --- Controls (dynamic based on selected episode) ---
if is_lotus_eaters:
    chain_defaults = LOTUS_EATERS_CHAIN_DEFAULTS
else:
    chain_defaults = extract_chain_seeds(episode_text, n=5)

chain_word = st.text_input(
    "Starting word", value=chain_defaults[0] if chain_defaults else "body",
    key="chain_start",
)

if chain_defaults:
    st.caption("Quick-start words:")
    qcols = st.columns(len(chain_defaults))
    for i, default_word in enumerate(chain_defaults):
        with qcols[i]:
            if st.button(default_word, key=f"chain_{default_word}"):
                chain_word = default_word

chain_steps = st.slider("Maximum chain steps", 3, 20, 10, key="chain_steps")

ALL_RELATIONS = [
    "Synonyms", "Hypernyms", "Hyponyms",
    "Part meronyms", "Substance meronyms",
    "Part holonyms", "Substance holonyms",
]
chain_relations = st.multiselect(
    "Relation types to follow", ALL_RELATIONS, default=ALL_RELATIONS,
    key="chain_relations",
)

if chain_word.strip() and chain_relations:
    chain = compute_chain(chain_word.strip().lower(), chain_steps, tuple(chain_relations))

    # --- Metrics row ---
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Chain Length", len(chain) - 1)

    start_synsets = wn.synsets(chain_word.strip().lower())
    cm2.metric("Starting Word Synsets", len(start_synsets))

    # Unique POS types in the chain
    pos_types = set()
    for step in chain:
        ss = wn.synsets(step["word"])
        for s in ss:
            pos_types.add(s.pos())
    cm3.metric("Unique POS Types", len(pos_types))

    dead_end = len(chain) - 1 < chain_steps
    cm4.metric("Dead End?", "Yes" if dead_end else "No")

    # --- Chain Step Diagram ---
    st.subheader("Chain Step Diagram")

    if len(chain) > 1:
        fig_chain, ax_chain = plt.subplots(
            figsize=(max(14, len(chain) * 2.5), 3)
        )
        ax_chain.set_xlim(-0.5, len(chain) * 2.5)
        ax_chain.set_ylim(-1, 2)
        ax_chain.axis("off")

        for i, step in enumerate(chain):
            x = i * 2.5
            color = RELATION_COLORS.get(step["relation"], "#999999")
            if step["relation"] == "start":
                color = "#333333"

            # Draw box
            bbox = dict(
                boxstyle="round,pad=0.4", facecolor=color, alpha=0.3,
                edgecolor=color,
            )
            word_display = step["word"]
            defn = step["definition"][:40] + "..." if len(step["definition"]) > 40 else step["definition"]
            ax_chain.text(
                x, 1, word_display, ha="center", va="center",
                fontsize=11, fontweight="bold", bbox=bbox,
            )
            ax_chain.text(
                x, 0.3, defn, ha="center", va="center", fontsize=7,
                color="#666666", style="italic",
            )

            # Draw arrow
            if i > 0:
                ax_chain.annotate(
                    "", xy=(x - 0.6, 1), xytext=(x - 1.9, 1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                )
                ax_chain.text(
                    x - 1.25, 1.5, step["relation"].replace("_", " "),
                    ha="center", va="center", fontsize=7, color=color,
                )

        ax_chain.set_title("Substitution Chain", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_chain)
        plt.close(fig_chain)

    # --- Multi-Chain Convergence Network ---
    st.subheader("Multi-Chain Convergence Network")

    if not NETWORKX_AVAILABLE:
        st.warning("Install `networkx` for the convergence network.")
    else:
        chain_colors_list = ["#E07A5F", "#4A90D9", "#81B29A", "#F2CC8F", "#9B59B6"]
        all_chains = {}
        for cw in chain_defaults:
            all_chains[cw] = compute_chain(cw, chain_steps, tuple(ALL_RELATIONS))

        G_conv = nx.Graph()

        for ci, cw in enumerate(chain_defaults):
            ch = all_chains[cw]
            color = chain_colors_list[ci % len(chain_colors_list)]

            for j, step in enumerate(ch):
                w = step["word"]
                if not G_conv.has_node(w):
                    G_conv.add_node(w, size=1, colors=set())
                G_conv.nodes[w]["colors"].add(color)
                G_conv.nodes[w]["size"] += 1

                if j > 0:
                    prev_w = ch[j - 1]["word"]
                    G_conv.add_edge(prev_w, w, color=color)

        if len(G_conv.nodes()) > 1:
            fig_conv, ax_conv = plt.subplots(figsize=(14, 10))
            pos = nx.spring_layout(G_conv, seed=42, k=1.5 / np.sqrt(len(G_conv.nodes())))

            # Draw edges colored by chain
            for u, v, data in G_conv.edges(data=True):
                nx.draw_networkx_edges(
                    G_conv, pos, edgelist=[(u, v)],
                    width=1.5, alpha=0.4, edge_color=data["color"], ax=ax_conv,
                )

            # Node sizing: larger for start words and convergence points
            node_sizes = []
            node_colors = []
            for node in G_conv.nodes():
                n_colors = len(G_conv.nodes[node]["colors"])
                if node in chain_defaults:
                    node_sizes.append(800)
                elif n_colors > 1:
                    node_sizes.append(500)  # Convergence point
                else:
                    node_sizes.append(200)
                # Mix colors or use first
                node_colors.append(list(G_conv.nodes[node]["colors"])[0])

            nx.draw_networkx_nodes(
                G_conv, pos, node_size=node_sizes, node_color=node_colors,
                alpha=0.9, edgecolors="#333333", linewidths=1, ax=ax_conv,
            )
            nx.draw_networkx_labels(
                G_conv, pos, font_size=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                ax=ax_conv,
            )

            ax_conv.set_title("Multi-Chain Convergence — shared nodes show where meanings meet")
            ax_conv.axis("off")

            # Legend
            from matplotlib.patches import Patch

            legend_handles = [
                Patch(facecolor=chain_colors_list[i], label=chain_defaults[i])
                for i in range(len(chain_defaults))
            ]
            ax_conv.legend(handles=legend_handles, loc="upper left", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_conv)
            plt.close(fig_conv)

        # --- Chain comparison table ---
        with st.expander("Chain comparison table"):
            max_len = max(len(all_chains[cw]) for cw in chain_defaults)
            table_data = {}
            for cw in chain_defaults:
                ch = all_chains[cw]
                table_data[cw] = [
                    step["word"] if i < len(ch) else ""
                    for i in range(max_len)
                ]
            df_chains = pd.DataFrame(table_data, index=[f"Step {i}" for i in range(max_len)])
            st.dataframe(df_chains, use_container_width=True)

            # Show shared words
            all_words_per_chain = {
                cw: set(step["word"].lower() for step in all_chains[cw])
                for cw in chain_defaults
            }
            st.markdown("**Convergence points:**")
            for i, w1 in enumerate(chain_defaults):
                for w2 in chain_defaults[i + 1 :]:
                    overlap = all_words_per_chain[w1] & all_words_per_chain[w2]
                    if overlap:
                        st.markdown(f"- **{w1}** and **{w2}** share: {', '.join(sorted(overlap))}")
else:
    st.info("Enter a starting word and select at least one relation type.")


# ============================================================================
# Bonus: Polysemy Across Ulysses
# ============================================================================

st.header("Bonus: Polysemy Across Ulysses")

st.markdown(
    "How polysemous is Joyce's vocabulary? Average synset count per content word "
    "measures lexical richness — words with more WordNet senses carry more "
    "potential meanings, fueling the ambiguity Joyce exploits."
)

poly_data = compute_polysemy(episode_file)

pm1, pm2, pm3 = st.columns(3)
pm1.metric("Avg Synsets/Word", f"{poly_data['avg_synsets']:.2f}")
pm2.metric("Content Words", poly_data["content_words"])
pm3.metric("Coverage %", f"{poly_data['coverage']:.1f}%")

with st.expander("Compute polysemy for all 18 episodes"):
    if st.button("Compute All", key="compute_polysemy_all"):
        poly_results = []
        progress = st.progress(0)
        for i, ef in enumerate(EPISODE_FILES):
            pd_ep = compute_polysemy(ef)
            poly_results.append(
                {
                    "Episode": EPISODE_MAP[ef],
                    "file": ef,
                    "Content Words": pd_ep["content_words"],
                    "Words with Synsets": pd_ep["words_with_synsets"],
                    "Avg Synsets/Word": pd_ep["avg_synsets"],
                    "Coverage %": pd_ep["coverage"],
                }
            )
            progress.progress((i + 1) / len(EPISODE_FILES))
        progress.empty()

        # Bar chart
        fig_poly, ax_poly = plt.subplots(figsize=(14, 5))
        ep_labels = [r["Episode"] for r in poly_results]
        ep_vals = [r["Avg Synsets/Word"] for r in poly_results]
        ep_colors = [
            "#E07A5F" if r["file"] == episode_file else "#B0B0B0"
            for r in poly_results
        ]

        ax_poly.bar(range(len(ep_labels)), ep_vals, color=ep_colors)
        ax_poly.set_xticks(range(len(ep_labels)))
        ax_poly.set_xticklabels(ep_labels, rotation=45, ha="right", fontsize=8)
        ax_poly.set_ylabel("Avg Synsets per Content Word")
        ax_poly.set_title("Polysemy Richness Across Episodes")
        plt.tight_layout()
        st.pyplot(fig_poly)
        plt.close(fig_poly)

        # Data table
        display_df = pd.DataFrame(poly_results).drop(columns=["file"])
        display_df["Avg Synsets/Word"] = display_df["Avg Synsets/Word"].round(2)
        display_df["Coverage %"] = display_df["Coverage %"].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("""
---

**What this week reveals:** Joyce's Lotus Eaters vocabulary is a web of narcotic
dissolution where religious, bodily, and chemical terms converge through shared
WordNet hypernyms. The semantic heatmap shows body/blood/bread clustering together,
water/bath/floating forming a substance group, and altar/sacrament/communion sharing
activity roots. Martha Clifford's malapropisms ("other world" / "other word") exploit
the gap between sound and meaning that taxonomies cannot bridge — near-homophones
that WordNet sees as unrelated but the ear treats as interchangeable. The substitution
chains show how quickly any word drifts into unexpected territory, a lexical analogy
for the narcotic dissolution that pervades the episode.
""")
