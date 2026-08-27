"""
Week 15 — Circe
NER, speaker extraction, and network analysis.
"""

import contextlib
import io
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "words"]:
    nltk.download(resource, quiet=True)

from week15.week15_circe import (
    extract_speakers,
    classify_entity,
    build_interaction_graph,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 15 — Circe", page_icon="📖", layout="wide")
st.title("Week 15 — Circe")
st.caption("Named Entity Recognition, Speaker Extraction & Network Analysis")


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
def cached_extract_speakers(episode_file):
    """Extract speakers from an episode with caching."""
    text = cached_load_episode(episode_file)
    speakers, stage_directions, scenes = suppress_stdout(extract_speakers, text)
    return speakers, stage_directions, scenes


@st.cache_data
def cached_classify_all(speaker_names):
    """Classify all speakers and return list of (name, category) tuples."""
    results = []
    for name in speaker_names:
        cat = classify_entity(name)
        results.append((name, cat))
    return results


@st.cache_data
def cached_build_interaction_graph(scenes_tuple, min_degree):
    """Build interaction graph with caching. scenes_tuple must be hashable."""
    # Reconstruct scenes list from the hashable tuple representation
    scenes = [(fs, lines) for fs, lines in scenes_tuple]
    filtered_nodes, filtered_edges, node_degree = suppress_stdout(
        build_interaction_graph, scenes, min_degree
    )
    return filtered_nodes, filtered_edges, node_degree


@st.cache_data
def cached_cumulative_entities():
    """Scan all 18 episodes and extract proper-noun entities via regex.

    Returns:
        episode_entities: dict of ep_num -> set of entity strings
        all_entities: Counter of entity -> episode_count
    """
    episode_files = [
        ("01", "01telemachus.txt"),
        ("02", "02nestor.txt"),
        ("03", "03proteus.txt"),
        ("04", "04calypso.txt"),
        ("05", "05lotuseaters.txt"),
        ("06", "06hades.txt"),
        ("07", "07aeolus.txt"),
        ("08", "08lestrygonians.txt"),
        ("09", "09scyllacharybdis.txt"),
        ("10", "10wanderingrocks.txt"),
        ("11", "11sirens.txt"),
        ("12", "12cyclops.txt"),
        ("13", "13nausicaa.txt"),
        ("14", "14oxenofthesun.txt"),
        ("15", "15circe.txt"),
        ("16", "16eumaeus.txt"),
        ("17", "17ithaca.txt"),
        ("18", "18penelope.txt"),
    ]

    stop_names = {
        "The", "A", "An", "And", "But", "For", "His", "Her", "Its",
        "He", "She", "It", "They", "We", "You", "My", "Who", "What",
        "That", "This", "There", "Then", "Than", "These", "Those",
        "With", "From", "Into", "Upon", "Your", "Their", "Our",
        "Not", "Nor", "Now", "How", "Where", "When", "Why", "Which",
        "Have", "Has", "Had", "Was", "Were", "Are", "Been", "Being",
        "Will", "Would", "Could", "Should", "Shall", "May", "Might",
        "Do", "Does", "Did", "Can", "Must", "Need", "Dare",
        "So", "No", "Yes", "Oh", "Ah", "If", "Or", "As", "At",
        "By", "In", "On", "To", "Of", "Up", "Out", "Off",
        "All", "Each", "Every", "Some", "Any", "Such", "Only",
        "About", "After", "Before", "Over", "Under", "Between",
        "Mr", "Mrs", "Miss", "Sir", "Lord", "Saint", "Father", "Mother",
        "Points", "Laughs", "Cries", "Calls", "Turns", "Takes",
        "Looks", "Stands", "Walks", "Sits", "Gets", "Puts", "Runs",
        "Comes", "Goes", "Says", "Speaks", "Sings", "Whispers",
    }
    proper_noun_pat = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

    def extract_entities(text):
        entities = set()
        for sent in sent_tokenize(text):
            trimmed = sent.split(None, 1)
            if len(trimmed) < 2:
                continue
            remainder = trimmed[1]
            for match in proper_noun_pat.finditer(remainder):
                name = match.group(1)
                words = name.split()
                if any(w not in stop_names for w in words):
                    entities.add(name)
        return entities

    episode_entities = {}
    all_entities = Counter()

    for ep_num, filename in episode_files:
        text = cached_load_episode(filename)
        entities = extract_entities(text)
        episode_entities[ep_num] = entities
        for e in entities:
            all_entities[e] += 1

    return episode_entities, all_entities


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=14,  # Circe
    caption="Week 15: NER, Speaker Extraction & Network Analysis",
    description=(
        "*Circe is Joyce's hallucinatory play-within-a-novel — the longest episode, "
        "written entirely as dramatic script. Every character from the book returns, "
        "the dead speak, objects come alive, and identities dissolve. Computational "
        "entity extraction reveals the dramatis personae that Joyce assembles for "
        "this theatrical climax.*"
    ),
)

# Load data
episode_text = cached_load_episode(episode_file)
is_circe = episode_file == "15circe.txt"


# ============================================================================
# Section 1: Dramatis Personae
# ============================================================================

st.header("1. Dramatis Personae")

if not is_circe:
    st.info(
        "Speaker extraction is designed for Circe's dramatic format (ALL-CAPS speaker tags). "
        "Results on other episodes will reflect any lines that happen to match this pattern. "
        "Select **15 — Circe** for the full analysis."
    )

speakers, stage_directions, scenes = cached_extract_speakers(episode_file)

# Classify each speaker
speaker_names = tuple(speakers.keys())
classifications = cached_classify_all(speaker_names)
name_to_cat = {name: cat for name, cat in classifications}

# Compute category counts
category_counts = Counter()
for name, cat in classifications:
    category_counts[cat] += 1

living_count = category_counts.get("person (living)", 0)
dead_count = category_counts.get("dead/hallucinated", 0)
nonhuman_count = (
    category_counts.get("object", 0)
    + category_counts.get("animal", 0)
    + category_counts.get("abstraction", 0)
)

# --- Metrics row ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Unique Speakers", len(speakers))
m2.metric("Living Persons", living_count)
m3.metric("Dead / Hallucinated", dead_count)
m4.metric("Non-Human", nonhuman_count)

# --- Category bar chart (horizontal) ---
st.subheader("Entity Categories")

cat_colors = {
    "person (living)": "#4A90D9",
    "dead/hallucinated": "#9B59B6",
    "object": "#E07A5F",
    "animal": "#81B29A",
    "abstraction": "#F2CC8F",
}

if category_counts:
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    cat_names = [c for c, _ in sorted_cats]
    cat_vals = [v for _, v in sorted_cats]
    bar_colors = [cat_colors.get(c, "#A0A0A0") for c in cat_names]

    fig_cat, ax_cat = plt.subplots(figsize=(8, max(3, len(cat_names) * 0.6)))
    ax_cat.barh(range(len(cat_names)), cat_vals, color=bar_colors)
    ax_cat.set_yticks(range(len(cat_names)))
    ax_cat.set_yticklabels(cat_names, fontsize=9)
    ax_cat.invert_yaxis()
    ax_cat.set_xlabel("Number of Entities")
    ax_cat.set_title(f"Entity Categories — {episode_label}")
    plt.tight_layout()
    st.pyplot(fig_cat)
    plt.close(fig_cat)

# --- Full cast table ---
st.subheader("Full Cast Table")

all_categories = sorted(set(name_to_cat.values()))
selected_cats = st.multiselect(
    "Filter by category",
    all_categories,
    default=all_categories,
    key="cast_filter",
)

cast_rows = []
for name in speakers:
    cat = name_to_cat.get(name, "unknown")
    if cat in selected_cats:
        cast_rows.append({
            "Entity": name,
            "Category": cat,
            "Line Count": speakers[name],
        })

if cast_rows:
    df_cast = pd.DataFrame(cast_rows)
    df_cast = df_cast.sort_values("Line Count", ascending=False).reset_index(drop=True)
    st.dataframe(df_cast, width="stretch", hide_index=True)
else:
    st.info("No entities match the selected categories.")

# --- Speaker frequency bar chart: top 20 ---
st.subheader("Top 20 Speakers by Line Count")

top_speakers = speakers.most_common(20)
if top_speakers:
    sp_names = [s for s, _ in top_speakers]
    sp_counts = [c for _, c in top_speakers]
    sp_colors = [cat_colors.get(name_to_cat.get(s, ""), "#A0A0A0") for s in sp_names]

    fig_sp, ax_sp = plt.subplots(figsize=(10, max(5, len(sp_names) * 0.35)))
    ax_sp.barh(range(len(sp_names)), sp_counts, color=sp_colors)
    ax_sp.set_yticks(range(len(sp_names)))
    ax_sp.set_yticklabels(sp_names, fontsize=8)
    ax_sp.invert_yaxis()
    ax_sp.set_xlabel("Line Count")
    ax_sp.set_title(f"Top 20 Speakers — {episode_label}")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=cat) for cat, c in cat_colors.items()]
    ax_sp.legend(handles=legend_handles, loc="lower right", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_sp)
    plt.close(fig_sp)

# --- Entity browser ---
st.subheader("Entity Browser")

if speakers:
    entity_options = [s for s, _ in speakers.most_common()]
    selected_entity = st.selectbox("Select a speaker", entity_options, key="entity_browser")
    sel_cat = name_to_cat.get(selected_entity, "unknown")
    sel_count = speakers[selected_entity]
    bc1, bc2 = st.columns(2)
    bc1.metric("Line Count", sel_count)
    bc2.metric("Category", sel_cat)


# ============================================================================
# Section 2: Interaction Network
# ============================================================================

st.header("2. Interaction Network")

st.markdown(
    "Build a co-appearance graph from speaker sequences. Two speakers are connected "
    "when they appear in the same sliding window of consecutive turns. Edge weight "
    "reflects how often they co-appear."
)

min_degree = st.slider("Minimum degree (filter weak nodes)", 1, 10, 2, key="min_degree")

# Make scenes hashable for caching
scenes_tuple = tuple((fs, tuple(lines)) for fs, lines in scenes)
filtered_nodes, filtered_edges, node_degree = cached_build_interaction_graph(
    scenes_tuple, min_degree
)

# Edge weight slider
max_edge_weight = max(filtered_edges.values()) if filtered_edges else 1
min_edge_weight = st.slider(
    "Minimum edge weight",
    1,
    max(1, max_edge_weight),
    1,
    key="min_edge_weight",
)
display_edges = {e: w for e, w in filtered_edges.items() if w >= min_edge_weight}
display_nodes = set()
for (a, b) in display_edges:
    display_nodes.add(a)
    display_nodes.add(b)

# Metrics
total_nodes = len(display_nodes)
total_edges = len(display_edges)
max_possible = total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 1
density = total_edges / max_possible if max_possible > 0 else 0
most_central = node_degree.most_common(1)[0][0] if node_degree else "N/A"

nm1, nm2, nm3, nm4 = st.columns(4)
nm1.metric("Nodes", total_nodes)
nm2.metric("Edges", total_edges)
nm3.metric("Density", f"{density:.4f}")
nm4.metric("Most Central", most_central)

# --- Network visualization ---
st.subheader("Network Visualization")

if display_nodes and display_edges:
    G = nx.Graph()
    for node in display_nodes:
        G.add_node(node)
    for (a, b), w in display_edges.items():
        G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Node colors by category, sizes by line count
    node_colors_list = []
    node_sizes_list = []
    for node in G.nodes():
        cat = name_to_cat.get(node, "person (living)")
        node_colors_list.append(cat_colors.get(cat, "#A0A0A0"))
        lc = speakers.get(node, 1)
        node_sizes_list.append(max(50, min(lc * 8, 800)))

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 3.0 * (w / max_w) for w in edge_weights]
    edge_alphas = [0.2 + 0.6 * (w / max_w) for w in edge_weights]

    fig_net, ax_net = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(
        G, pos, ax=ax_net, width=edge_widths,
        alpha=0.3, edge_color="#999999",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax_net, node_color=node_colors_list,
        node_size=node_sizes_list, alpha=0.8, edgecolors="white", linewidths=0.5,
    )
    # Label only top nodes
    top_node_set = {n for n, _ in node_degree.most_common(20)}
    labels = {n: n for n in G.nodes() if n in top_node_set}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax_net, font_size=7, font_weight="bold",
    )
    ax_net.set_title(f"Speaker Interaction Network — {episode_label}", fontsize=14)
    ax_net.axis("off")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=cat) for cat, c in cat_colors.items()]
    ax_net.legend(handles=legend_handles, loc="lower left", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_net)
    plt.close(fig_net)
else:
    st.info("No nodes/edges to display with current filter settings. Try lowering the minimum degree or edge weight.")

# --- Degree centrality table ---
st.subheader("Degree Centrality")

if display_nodes:
    centrality_rows = []
    for node, deg in node_degree.most_common():
        if node in display_nodes:
            centrality_rows.append({
                "Speaker": node,
                "Category": name_to_cat.get(node, "unknown"),
                "Degree": deg,
                "Line Count": speakers.get(node, 0),
            })
    if centrality_rows:
        df_cent = pd.DataFrame(centrality_rows)
        st.dataframe(df_cent, width="stretch", hide_index=True)


# ============================================================================
# Section 3: Cumulative Entity Tracking
# ============================================================================

st.header("3. Cumulative Entity Tracking")

st.markdown(
    "Scan all 18 episodes for proper-noun entities using regex-based extraction. "
    "This reveals how Circe reactivates characters from earlier episodes — the "
    "hallucinatory return of the repressed. This computation is expensive because "
    "it processes every episode."
)

if st.button("Run Cumulative Entity Scan", key="cumulative_button"):
    with st.spinner("Scanning all 18 episodes for proper-noun entities..."):
        episode_entities, all_entities = cached_cumulative_entities()
    st.session_state["cumulative_entities"] = episode_entities
    st.session_state["cumulative_all"] = all_entities

if "cumulative_entities" in st.session_state:
    episode_entities = st.session_state["cumulative_entities"]
    all_entities = st.session_state["cumulative_all"]

    # Multi-episode entities
    multi_ep = {e: c for e, c in all_entities.items() if c > 1}

    # Circe reactivation
    circe_ents = episode_entities.get("15", set())
    prior_ents = set()
    for ep in ["01", "02", "03", "04", "05", "06", "07", "08", "09",
               "10", "11", "12", "13", "14"]:
        prior_ents |= episode_entities.get(ep, set())

    reactivated = circe_ents & prior_ents
    new_in_circe = circe_ents - prior_ents
    reactivation_rate = (
        len(reactivated) / (len(reactivated) + len(new_in_circe)) * 100
        if (reactivated or new_in_circe) else 0
    )

    total_all = sum(len(ents) for ents in episode_entities.values())

    # --- Metrics ---
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Total Entities (all episodes)", f"{total_all:,}")
    cm2.metric("Entities in Circe from Prior Episodes", len(reactivated))
    cm3.metric("New Entities in Circe", len(new_in_circe))
    cm4.metric("Reactivation Rate", f"{reactivation_rate:.1f}%")

    # --- Episode-entity heatmap: top 20 multi-episode entities ---
    st.subheader("Episode-Entity Heatmap")

    top_multi = sorted(multi_ep.items(), key=lambda x: -x[1])[:20]
    if top_multi:
        ep_nums = [f"{int(ep):02d}" for ep in sorted(episode_entities.keys(), key=int)]
        entity_names = [e for e, _ in top_multi]

        heatmap_data = np.zeros((len(entity_names), len(ep_nums)))
        for i, entity in enumerate(entity_names):
            for j, ep in enumerate(ep_nums):
                if entity in episode_entities.get(ep, set()):
                    heatmap_data[i, j] = 1.0

        fig_heat, ax_heat = plt.subplots(figsize=(14, max(5, len(entity_names) * 0.4)))
        im = ax_heat.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", interpolation="nearest")
        ax_heat.set_xticks(range(len(ep_nums)))
        ax_heat.set_xticklabels(ep_nums, fontsize=8)
        ax_heat.set_yticks(range(len(entity_names)))
        ax_heat.set_yticklabels(entity_names, fontsize=8)
        ax_heat.set_xlabel("Episode")
        ax_heat.set_title("Top 20 Multi-Episode Entities")

        # Highlight Circe column
        circe_idx = ep_nums.index("15") if "15" in ep_nums else None
        if circe_idx is not None:
            ax_heat.axvline(x=circe_idx - 0.5, color="blue", linewidth=1.5, linestyle="--", alpha=0.7)
            ax_heat.axvline(x=circe_idx + 0.5, color="blue", linewidth=1.5, linestyle="--", alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    # --- Reactivation table ---
    st.subheader("Reactivation Table")

    if reactivated:
        react_rows = []
        for entity in sorted(reactivated):
            prior_eps = []
            for ep in sorted(episode_entities.keys(), key=int):
                if ep == "15":
                    continue
                if entity in episode_entities.get(ep, set()):
                    prior_eps.append(ep)
            react_rows.append({
                "Entity": entity,
                "Prior Episodes": ", ".join(prior_eps),
                "Episode Count": len(prior_eps) + 1,  # +1 for Circe
            })
        df_react = pd.DataFrame(react_rows)
        df_react = df_react.sort_values("Episode Count", ascending=False).reset_index(drop=True)
        st.dataframe(df_react, width="stretch", hide_index=True)
    else:
        st.info("No reactivated entities found.")


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Circe is where Joyce's novel becomes a total recall machine.
Computational entity extraction shows that the episode's dramatic format produces a
recoverable dramatis personae — speakers are tagged in ALL-CAPS, making NER almost
trivially accurate compared to the stream-of-consciousness prose of other episodes.
The interaction network reveals how densely interconnected the episode's speakers are:
living characters, the dead, objects, and abstractions all share the same conversational
space. The cumulative entity scan across all 18 episodes shows Circe's role as the novel's
recapitulation — reactivating names and presences from throughout the book in a
hallucinatory theatrical climax. The reactivation rate quantifies what readers feel
intuitively: that Circe is where everything comes back.
""")
