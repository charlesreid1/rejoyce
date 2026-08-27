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
        "**Week 5** uses WordNet to measure the semantic distance between Joyce's "
        "thematic words in Lotus Eaters — how long is the bridge from *blood* to "
        "*wine*, or from *altar* to *bath*? Martha Clifford's malapropisms "
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
def discover_near_homophones(text, max_pairs=10):
    """Find near-homophone pairs among words that appear in the episode text.

    Extracts frequent content words that exist in both CMU dict and WordNet,
    then finds pairs with low phonological distance and low semantic similarity.
    """
    try:
        pronouncing = cmudict.dict()
    except Exception:
        return []

    tokens = word_tokenize(text)
    content_words = [
        t.lower() for t in tokens
        if t.isalpha() and len(t) > 3 and t.lower() not in _STOPWORDS
    ]
    freq = Counter(content_words)

    # Keep top 150 frequent words that are in both CMU and WordNet
    candidates = []
    for word, _ in freq.most_common(300):
        if word in pronouncing and wn.synsets(word):
            candidates.append(word)
            if len(candidates) >= 150:
                break

    # Find pairs with phonological distance ≤ 3
    pairs = []
    for i in range(len(candidates)):
        p1 = pronouncing[candidates[i]][0]
        for j in range(i + 1, len(candidates)):
            p2 = pronouncing[candidates[j]][0]
            phon_dist = nltk.edit_distance(p1, p2)
            if phon_dist <= 3:
                ss1 = wn.synsets(candidates[i])
                ss2 = wn.synsets(candidates[j])
                wup = ss1[0].wup_similarity(ss2[0]) or 0
                phon_closeness = 1.0 - phon_dist / 4.0
                pun_gap = phon_closeness * (1.0 - wup)
                pairs.append((candidates[i], candidates[j], phon_dist, wup, pun_gap))

    # Sort by pun gap descending, return top pairs
    pairs.sort(key=lambda x: -x[4])
    return [(a, b) for a, b, _, _, _ in pairs[:max_pairs]]


@st.cache_data
def discover_near_homophones_corpus(max_candidates=1000, max_pairs=15):
    """Find near-homophone pairs across the entire Ulysses corpus.

    Concatenates all 18 episodes and searches the top N most frequent
    content words for near-homophones. Slower but finds more pairs.
    """
    all_text = "\n".join(cached_load_episode(ef) for ef in EPISODE_FILES)
    try:
        pronouncing = cmudict.dict()
    except Exception:
        return []

    tokens = word_tokenize(all_text)
    content_words = [
        t.lower() for t in tokens
        if t.isalpha() and len(t) > 3 and t.lower() not in _STOPWORDS
    ]
    freq = Counter(content_words)

    candidates = []
    for word, _ in freq.most_common(max_candidates * 2):
        if word in pronouncing and wn.synsets(word):
            candidates.append(word)
            if len(candidates) >= max_candidates:
                break

    pairs = []
    for i in range(len(candidates)):
        p1 = pronouncing[candidates[i]][0]
        for j in range(i + 1, len(candidates)):
            p2 = pronouncing[candidates[j]][0]
            if abs(len(p1) - len(p2)) > 3:
                continue
            phon_dist = nltk.edit_distance(p1, p2)
            if phon_dist <= 3:
                ss1 = wn.synsets(candidates[i])
                ss2 = wn.synsets(candidates[j])
                wup = ss1[0].wup_similarity(ss2[0]) or 0
                phon_closeness = 1.0 - phon_dist / 4.0
                pun_gap = phon_closeness * (1.0 - wup)
                pairs.append((candidates[i], candidates[j], phon_dist, wup, pun_gap))

    pairs.sort(key=lambda x: -x[4])
    return [(a, b) for a, b, _, _, _ in pairs[:max_pairs]]


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
# Section 1: Semantic Fields
# ============================================================================

st.header("1. Semantic Fields")

st.markdown(
    "How do Joyce's thematic words relate in WordNet's taxonomy? "
    "This section maps the semantic distances between the episode's key vocabulary — "
    "how close or far apart are words like *blood* and *bread*, *altar* and *bath*? "
    "The heatmap and network graph reveal which words are nearby neighbors in "
    "meaning and which are connected only by long bridges through abstract concepts."
)

# --- Word selection (dynamic based on selected episode) ---
is_lotus_eaters = episode_file == "05lotuseaters.txt"

if is_lotus_eaters:
    default_thematic = LOTUS_EATERS_WORDS
else:
    default_thematic = extract_thematic_words(episode_text, n=15)

all_options = list(dict.fromkeys(default_thematic))

# Reset word selection and pair selection when the episode changes
if st.session_state.get("_prev_episode") != episode_file:
    st.session_state["_prev_episode"] = episode_file
    st.session_state["thematic_words"] = default_thematic
    st.session_state.pop("pair_select", None)
    st.rerun()

selected_words = st.multiselect(
    "Thematic words",
    options=all_options,
    default=default_thematic,
    key="thematic_words",
)


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

        st.markdown(
            "WordNet organizes words into a tree of increasingly general categories "
            "called **hypernyms** — for example, *lotus* is a kind of *plant*, which "
            "is a kind of *organism*, which is a kind of *entity*. **Hypernym depth** "
            "counts how many steps a word sits below the root (*entity*). "
            "Deeper words are more specific and concrete (e.g., *sacrament* at depth 10), "
            "while shallower words are broader and more abstract (e.g., *body* at depth 4). "
            "Colors group words by their top-level ancestor: "
            "blue = physical objects, green = substances, coral = activities, grey = other."
        )

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

        st.markdown(
            f"Each word is a node in the graph, connected to other words whose "
            f"**Wu-Palmer similarity** exceeds the threshold you set above "
            f"(currently **{threshold}**). Wu-Palmer (WuP) measures how closely "
            f"two words relate in WordNet's noun hierarchy — 1.0 means identical "
            f"meaning, while values below ~0.2 indicate very distant concepts. "
            f"**Larger nodes** have more synsets (distinct dictionary senses), "
            f"meaning Joyce can exploit more shades of meaning. "
            f"**Edge labels** show the lowest common hypernym — the most specific "
            f"ancestor two words share in WordNet's taxonomy — revealing the "
            f"hidden conceptual bridge between them. "
            f"Node colors indicate top-level cluster: "
            f"blue = physical objects, green = substances, coral = activities, grey = other."
        )

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
            st.markdown(
                "**Semantic coherence** measures how tightly a chapter's vocabulary "
                "hangs together conceptually. For each episode, we take the 15 most "
                "frequent content words and compute the average Wu-Palmer similarity "
                "across every pair. Higher scores mean the chapter's key words are "
                "close semantic neighbors; lower scores mean the vocabulary is more "
                "spread out across unrelated domains. Lotus Eaters scores relatively "
                "low — its key words (*body*, *flower*, *altar*, *drug*) span several "
                "distant branches of WordNet's hierarchy, which is precisely what makes "
                "the semantic bridges in Section 3 interesting: Joyce is yoking together "
                "concepts that the taxonomy considers far apart."
            )
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
    "the best puns live in this mismatch.\n\n"
    "**Algorithmic note:** These pairs are discovered automatically from each "
    "episode's text. First, we extract the most frequent content words that appear "
    "in both the CMU Pronouncing Dictionary (for phoneme sequences) and WordNet "
    "(for semantic similarity). Next, we compare every pair's phonological edit "
    "distance — the number of phoneme insertions, deletions, or substitutions needed "
    "to transform one pronunciation into the other — and keep only pairs within 3 edits. "
    "Finally, we rank by a **pun gap** score: pairs that sound very alike (low "
    "phonological distance) but mean very different things (low Wu-Palmer similarity) "
    "score highest. The result is a ranked list of the episode's best latent puns."
)

# --- Pair selection (dynamic based on selected episode) ---
if is_lotus_eaters:
    episode_word_pairs = DEFAULT_WORD_PAIRS
else:
    episode_word_pairs = discover_near_homophones(episode_text, max_pairs=10)
    if not episode_word_pairs:
        episode_word_pairs = DEFAULT_WORD_PAIRS  # fallback

# Corpus-wide search (triggered by button, results stored in session state)
with st.expander("Search all 18 episodes for pun pairs"):
    st.markdown(
        "The pairs above are discovered from the selected episode's vocabulary. "
        "For a broader search, scan the top 1,000 most frequent words across all "
        "of *Ulysses* — this takes longer but surfaces puns that span the full novel."
    )
    if st.button("Search full Ulysses corpus", key="corpus_pun_search"):
        with st.spinner("Scanning corpus..."):
            corpus_pairs = discover_near_homophones_corpus(max_candidates=1000, max_pairs=15)
            st.session_state["_corpus_pun_pairs"] = corpus_pairs
            # Reset pair selector so it picks up the new merged list
            st.session_state.pop("pair_select", None)
            st.rerun()
    if "_corpus_pun_pairs" in st.session_state:
        corpus_pairs = st.session_state["_corpus_pun_pairs"]
        st.success(f"Found {len(corpus_pairs)} pun pairs across all episodes.")
        # Merge corpus pairs with episode pairs, deduplicating
        existing = set((a, b) for a, b in episode_word_pairs)
        for pair in corpus_pairs:
            if pair not in existing and (pair[1], pair[0]) not in existing:
                episode_word_pairs.append(pair)
                existing.add(pair)

pair_labels = [f"{w1} / {w2}" for w1, w2 in episode_word_pairs]

# Ensure all pairs are selected by default (including after episode switch)
if "pair_select" not in st.session_state:
    st.session_state["pair_select"] = pair_labels

selected_pair_labels = st.multiselect(
    "Word pairs", pair_labels, default=pair_labels, key="pair_select"
)

selected_pairs = [
    episode_word_pairs[pair_labels.index(lbl)] for lbl in selected_pair_labels
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

    # --- Pun Gap Chart ---
    st.subheader("The Pun Gap: Sound vs. Meaning")

    scatter_data = [d for d in mal_data if d["phon_dist"] is not None]

    if scatter_data:
        # Compute pun-gap score for later use
        max_phon_sc = max(d["phon_dist"] for d in scatter_data) or 1
        for d in scatter_data:
            phon_closeness = 1.0 - d["phon_dist"] / max_phon_sc
            sem_distance = 1.0 - d["wup_sim"]
            d["pun_gap"] = phon_closeness * sem_distance

        x_raw = [d["phon_dist"] for d in scatter_data]
        y_vals = [d["wup_sim"] for d in scatter_data]
        labels_sc = [f"{d['w1']} / {d['w2']}" for d in scatter_data]

        # Jitter overlapping points: shift x slightly when two points
        # share the same integer x and have y values within 0.05
        rng = np.random.RandomState(42)
        x_vals = list(x_raw)
        for i in range(len(x_vals)):
            for j in range(i):
                if x_raw[i] == x_raw[j] and abs(y_vals[i] - y_vals[j]) < 0.05:
                    x_vals[i] += rng.uniform(-0.15, 0.15)
                    break

        # Distinct markers for each pair
        marker_list = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h",
                       "<", ">", "8", "H", "d"]
        # Color palette
        color_list = [
            "#E07A5F", "#4A90D9", "#81B29A", "#F2CC8F", "#9B59B6",
            "#E76F51", "#264653", "#2A9D8F", "#E9C46A", "#F4A261",
            "#606C38", "#BC6C25", "#023047", "#FB8500", "#8338EC",
        ]

        fig_gap, ax_gap = plt.subplots(figsize=(10, 5))

        for i, d in enumerate(scatter_data):
            marker = marker_list[i % len(marker_list)]
            color = color_list[i % len(color_list)]
            ax_gap.scatter(
                x_vals[i], y_vals[i],
                marker=marker, c=color, s=140,
                edgecolors="#333333", linewidths=0.8, zorder=3,
                label=labels_sc[i],
            )

        # "Best pun" zone shading (bottom-left)
        ax_gap.axhspan(0, 0.3, xmin=0, xmax=0.4, alpha=0.06, color="#2A9D8F")
        ax_gap.text(
            0.3, 0.02, "strong pun zone", fontsize=8, color="#2A9D8F",
            fontstyle="italic", alpha=0.7,
        )

        # Reference diagonal
        max_x = max(x_raw) + 1
        ax_gap.plot([0, max_x], [0, 1], "--", color="#CCCCCC", alpha=0.5)

        ax_gap.set_xticks(range(0, int(max(x_raw)) + 2))
        ax_gap.set_xlabel("Phonological Distance (CMU edit distance)\nlower = sounds more alike", fontsize=9)
        ax_gap.set_ylabel("Semantic Similarity (max WuP)\nlower = means more different", fontsize=9)
        ax_gap.set_title("The Pun Gap\n(best puns live in the bottom-left)", fontsize=11)

        # Legend outside the plot area
        ax_gap.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9,
            frameon=True, framealpha=0.9, borderaxespad=0,
        )
        fig_gap.subplots_adjust(right=0.72)
        plt.tight_layout(rect=[0, 0, 0.72, 1])
        st.pyplot(fig_gap)
        plt.close(fig_gap)

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
        st.dataframe(pd.DataFrame(phoneme_rows), width="stretch", hide_index=True)

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

            # Compute pun gap score on the same scale as the chart above
            test_phon = test_data["phon_dist"]
            test_sem = test_data["wup_sim"]
            if test_phon is not None:
                test_phon_closeness = 1.0 - test_phon / max_phon if max_phon else 0
                test_pun_gap = max(0, test_phon_closeness) * (1.0 - test_sem)
            else:
                test_pun_gap = None

            # Show comparison bar chart: your pair vs the existing pairs
            fig_test, ax_test = plt.subplots(figsize=(10, max(3, (len(sorted_mal) + 1) * 0.45)))

            # Existing pairs sorted by pun gap (reuse sorted_mal)
            existing_gaps = []
            for d in sorted_mal:
                if d["phon_dist"] is not None:
                    pc = 1.0 - d["phon_dist"] / max_phon if max_phon else 0
                    existing_gaps.append((f"{d['w1']} / {d['w2']}", max(0, pc) * (1.0 - d["wup_sim"]),
                                          d["phon_dist"], d["wup_sim"]))
            existing_gaps.sort(key=lambda x: -x[1])

            # Insert test pair
            test_label = f"{test_w1.strip().lower()} / {test_w2.strip().lower()}"
            all_bars = [(test_label, test_pun_gap, test_phon, test_sem)] + existing_gaps

            bar_labels_test = [b[0] for b in all_bars]
            bar_vals_test = [b[1] if b[1] is not None else 0 for b in all_bars]
            bar_colors = ["#4A90D9"] + ["#E07A5F"] * len(existing_gaps)

            y_pos_test = np.arange(len(all_bars))
            ax_test.barh(y_pos_test, bar_vals_test, color=bar_colors, edgecolor="#333333", linewidth=0.5)
            ax_test.set_yticks(y_pos_test)
            ax_test.set_yticklabels(bar_labels_test, fontsize=10)
            ax_test.invert_yaxis()
            ax_test.set_xlabel("Pun Gap Score (higher = better pun)")
            ax_test.set_title("Your pair (blue) vs. Joyce's malapropisms (coral)")

            # Annotate each bar
            for i, (_, gap, ph, sm) in enumerate(all_bars):
                if gap is not None and ph is not None:
                    ax_test.text(
                        gap + 0.01, i,
                        f"phon dist {ph}, sem sim {sm:.2f}",
                        va="center", fontsize=8, color="#666666",
                    )

            max_bar = max(bar_vals_test) if bar_vals_test else 1
            ax_test.set_xlim(0, max_bar * 1.45)
            plt.tight_layout()
            st.pyplot(fig_test)
            plt.close(fig_test)

            # Verdict
            if test_phon is not None:
                if test_phon <= 2 and test_sem < 0.3:
                    st.success("**Strong pun** — sounds very alike, means very different!")
                elif test_phon <= 3 and test_sem < 0.5:
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
                    st.markdown(f"**Synsets for '{w}':** none found in WordNet")
else:
    st.info("Select at least one word pair.")


# ============================================================================
# Section 3: Semantic Bridges
# ============================================================================

st.header("3. Semantic Bridges")

st.markdown(
    "In Lotus Eaters, Joyce yokes together words that seem unrelated on the surface — "
    "*body* and *bread*, *flower* and *water*, *drug* and *communion* — but which "
    "share a hidden ancestor in the tree of meaning. WordNet organizes every noun into "
    "a hierarchy from specific to general: *bread* is a kind of *food*, which is a kind "
    "of *substance*, which is a kind of *matter*. Two words connect where their paths "
    "upward meet at a **lowest common ancestor** (LCA) — the most specific concept that "
    "encompasses both.\n\n"
    "The deeper that meeting point sits in the tree, the more specific the shared "
    "concept and the tighter the connection. A shallow meeting point (like *entity*) "
    "means the words share almost nothing; a deep one (like *substance*) means Joyce "
    "is exploiting a real conceptual bridge."
)


@st.cache_data
def compute_bridge(word_a, word_b):
    """Compute the hypernym bridge between two words.

    Returns the path from each word up to their lowest common hypernym,
    plus the LCA synset info.
    """
    ss_a_list = wn.synsets(word_a)
    ss_b_list = wn.synsets(word_b)
    if not ss_a_list or not ss_b_list:
        return None

    # Use the first (most common) noun synset for each word, falling back
    # to the first synset of any POS if no noun sense exists.
    def first_noun_synset(synsets):
        for s in synsets:
            if s.pos() == "n":
                return s
        return synsets[0]

    sa = first_noun_synset(ss_a_list)
    sb = first_noun_synset(ss_b_list)
    best_sim = sa.wup_similarity(sb) or 0
    lca_list = sa.lowest_common_hypernyms(sb)
    if not lca_list:
        return None
    lca = lca_list[0]

    # Get path from each synset up to the LCA
    def path_to_ancestor(synset, ancestor):
        """BFS to find path from synset up to ancestor."""
        from collections import deque
        queue = deque([(synset, [synset])])
        visited = {synset}
        while queue:
            current, path = queue.popleft()
            if current == ancestor:
                return path
            for hyp in current.hypernyms():
                if hyp not in visited:
                    visited.add(hyp)
                    queue.append((hyp, path + [hyp]))
        return [synset, ancestor]  # fallback

    path_a = path_to_ancestor(sa, lca)
    path_b = path_to_ancestor(sb, lca)

    return {
        "word_a": word_a,
        "word_b": word_b,
        "synset_a": sa.name(),
        "synset_b": sb.name(),
        "def_a": sa.definition(),
        "def_b": sb.definition(),
        "lca": lca.name(),
        "lca_def": lca.definition(),
        "lca_depth": lca.min_depth(),
        "path_a": [(word_a, sa.definition())] + [(s.name().split(".")[0], s.definition()) for s in path_a[1:]],
        "path_b": [(word_b, sb.definition())] + [(s.name().split(".")[0], s.definition()) for s in path_b[1:]],
        "wup_sim": best_sim,
    }


# --- Default interesting pairs for Lotus Eaters ---
if is_lotus_eaters:
    BRIDGE_PAIRS = [
        ("blood", "wine"),
        ("altar", "bath"),
        ("lotus", "flower"),
        ("body", "bread"),
        ("flower", "water"),
    ]
else:
    # Pick pairs from the thematic words with highest similarity
    _bridge_words = selected_words[:8] if selected_words else []
    BRIDGE_PAIRS = []
    if len(_bridge_words) >= 2:
        _scored_pairs = []
        for _i in range(len(_bridge_words)):
            for _j in range(_i + 1, len(_bridge_words)):
                _ss1 = wn.synsets(_bridge_words[_i])
                _ss2 = wn.synsets(_bridge_words[_j])
                if _ss1 and _ss2:
                    _sim = _ss1[0].wup_similarity(_ss2[0]) or 0
                    if 0.15 < _sim < 0.85:  # interesting range
                        _scored_pairs.append((_bridge_words[_i], _bridge_words[_j], _sim))
        _scored_pairs.sort(key=lambda x: -x[2])
        BRIDGE_PAIRS = [(a, b) for a, b, _ in _scored_pairs[:5]]

# --- Pair selector ---
pair_options = [f"{a} ↔ {b}" for a, b in BRIDGE_PAIRS]
bc1, bc2 = st.columns([3, 1])
with bc1:
    if pair_options:
        bridge_choice = st.selectbox(
            "Choose a word pair to bridge",
            pair_options,
            key="bridge_pair",
        )
        bridge_idx = pair_options.index(bridge_choice)
        bridge_word_a, bridge_word_b = BRIDGE_PAIRS[bridge_idx]
    else:
        bridge_word_a, bridge_word_b = "", ""
with bc2:
    custom_a = st.text_input("Or enter word A", key="bridge_a")
    custom_b = st.text_input("And word B", key="bridge_b")
    if custom_a.strip() and custom_b.strip():
        bridge_word_a = custom_a.strip().lower()
        bridge_word_b = custom_b.strip().lower()

if bridge_word_a and bridge_word_b:
    bridge = compute_bridge(bridge_word_a, bridge_word_b)

    if bridge:
        # --- Metrics ---
        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("WuP Similarity", f"{bridge['wup_sim']:.3f}")
        bm2.metric("Meeting Point Depth", bridge["lca_depth"])
        bm3.metric("Meeting Concept", bridge["lca"].split(".")[0])

        # --- Bridge Diagram ---
        st.subheader("Hypernym Bridge")

        st.markdown(
            f"Reading from left to right, **{bridge_word_a}** climbs up through "
            f"increasingly general categories until it meets **{bridge_word_b}**'s "
            f"path at **{bridge['lca'].split('.')[0]}** "
            f"(*{bridge['lca_def']}*). "
            f"That shared ancestor is the conceptual bridge Joyce exploits."
        )

        path_a = bridge["path_a"]
        path_b = bridge["path_b"]

        # Build the full bridge: word_a path → LCA ← word_b path (reversed)
        # path_a goes [word_a, ..., LCA], path_b goes [word_b, ..., LCA]
        # Display: path_a then path_b reversed (excluding LCA duplicate)
        full_path = path_a + list(reversed(path_b[:-1]))
        lca_index = len(path_a) - 1  # index of the LCA in full_path

        n_nodes = len(full_path)
        fig_bridge, ax_bridge = plt.subplots(
            figsize=(max(14, n_nodes * 2.2), 3.5)
        )
        ax_bridge.set_xlim(-0.5, n_nodes * 2.2)
        ax_bridge.set_ylim(-0.5, 2.5)
        ax_bridge.axis("off")

        for i, (name, defn) in enumerate(full_path):
            x = i * 2.2

            # Color: word_a's side blue, LCA gold, word_b's side coral
            if i < lca_index:
                color = "#4A90D9"
            elif i == lca_index:
                color = "#F2CC8F"
            else:
                color = "#E07A5F"

            # Make start/end/LCA words bold and larger
            is_key = (i == 0 or i == n_nodes - 1 or i == lca_index)
            fontsize = 12 if is_key else 9
            fontweight = "bold" if is_key else "normal"

            bbox = dict(
                boxstyle="round,pad=0.4", facecolor=color,
                alpha=0.4 if is_key else 0.2,
                edgecolor=color,
            )
            ax_bridge.text(
                x, 1.5, name, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, bbox=bbox,
            )

            # Definition below
            defn_short = defn[:35] + "..." if len(defn) > 35 else defn
            ax_bridge.text(
                x, 0.7, defn_short, ha="center", va="center", fontsize=6,
                color="#666666", style="italic",
            )

            # Arrow
            if i > 0:
                arrow_color = "#4A90D9" if i <= lca_index else "#E07A5F"
                # Arrows point toward the LCA from both sides
                if i <= lca_index:
                    ax_bridge.annotate(
                        "", xy=(x - 0.4, 1.5), xytext=(x - 1.8, 1.5),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                    )
                else:
                    ax_bridge.annotate(
                        "", xy=(x - 1.8, 1.5), xytext=(x - 0.4, 1.5),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                    )

            # Label the LCA
            if i == lca_index:
                ax_bridge.text(
                    x, 2.2, "▼ lowest common ancestor",
                    ha="center", va="center", fontsize=8, color="#996633",
                    fontweight="bold",
                )

        ax_bridge.set_title(
            f"Semantic Bridge: {bridge_word_a} → {bridge['lca'].split('.')[0]} ← {bridge_word_b}",
            fontsize=13,
        )
        plt.tight_layout()
        st.pyplot(fig_bridge)
        plt.close(fig_bridge)

        # --- All-pairs bridge summary ---
        st.subheader("Bridge Summary for All Pairs")

        st.markdown(
            "How do all the suggested word pairs connect? The table below shows "
            "each pair's meeting point, its depth in the hierarchy, and their "
            "Wu-Palmer similarity. Deeper meeting points = tighter conceptual links."
        )

        bridge_rows = []
        for wa, wb in BRIDGE_PAIRS:
            b = compute_bridge(wa, wb)
            if b:
                bridge_rows.append({
                    "Word A": wa,
                    "Word B": wb,
                    "Meeting Point": b["lca"].split(".")[0],
                    "Meeting Depth": b["lca_depth"],
                    "Path A Length": len(b["path_a"]),
                    "Path B Length": len(b["path_b"]),
                    "WuP Similarity": round(b["wup_sim"], 3),
                })
        if bridge_rows:
            df_bridges = pd.DataFrame(bridge_rows)
            df_bridges = df_bridges.sort_values("WuP Similarity", ascending=False)
            st.dataframe(df_bridges, width="stretch", hide_index=True)
    else:
        st.warning(
            f"Could not find a WordNet connection between "
            f"**{bridge_word_a}** and **{bridge_word_b}**. "
            f"Try words that are common English nouns."
        )
else:
    st.info("Select or enter a word pair to explore their semantic bridge.")


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
        st.dataframe(display_df, width="stretch", hide_index=True)

st.markdown("""
---

**What this week reveals:** Joyce's Lotus Eaters vocabulary spans distant branches
of WordNet's taxonomy — *body*, *flower*, *altar*, and *drug* sit far apart in the
hierarchy, giving the episode low semantic coherence compared to other chapters.
That distance is the point: the semantic bridges show exactly how far meaning must
travel to connect these words. Blood and wine meet through *substance*; altar and
bath through *instrumentality*; lotus and flower deep in the taxonomy at *vascular
plant*. Some bridges are short (confirming intuitive links), others are surprisingly
long (revealing connections Joyce builds through literary context rather than
dictionary meaning). Martha Clifford's malapropisms ("other world" / "other word")
exploit a different kind of gap — phonological rather than semantic — where
near-homophones that WordNet sees as unrelated sound interchangeable to the ear.
""")
