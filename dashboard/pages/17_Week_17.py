"""
Week 17 — Ithaca
Information extraction, knowledge graphs, and topic distribution.
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

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize

for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

from week17.week17_ithaca import (
    parse_catechism,
    classify_question,
    extract_triples,
    STOPWORDS,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 17 — Ithaca", page_icon="📖", layout="wide")
st.title("Week 17 — Ithaca")
st.caption("Information Extraction, Knowledge Graphs & Topic Distribution")

# ============================================================================
# Topic keyword dictionary (replicated from week17_ithaca.topic_distribution)
# ============================================================================

TOPIC_KEYWORDS = {
    "physical_objects": {"drawer", "table", "chair", "cup", "key", "door", "shelf", "bed", "lamp", "range", "kitchen", "book", "furniture", "contents", "articles", "clothes", "hat", "coat", "shoes", "umbrella", "bag", "box", "bottle", "glass", "plate", "fork", "knife", "spoon", "window", "wall", "floor", "ceiling", "roof", "garden", "tree", "flower", "letter", "paper", "pen", "ink"},
    "human_relations": {"bloom", "stephen", "molly", "father", "mother", "son", "wife", "husband", "friend", "companion", "family", "relative", "neighbor", "stranger", "man", "woman", "person", "people", "child", "children", "baby", "girl", "boy", "sister", "brother", "uncle", "aunt", "cousin", "grandfather", "grandmother"},
    "abstract_concepts": {"reason", "cause", "purpose", "meaning", "thought", "memory", "belief", "feeling", "emotion", "desire", "idea", "concept", "truth", "knowledge", "wisdom", "understanding", "consciousness", "spirit", "mind", "soul", "will", "intention", "decision", "choice", "value", "principle", "ethics"},
    "science_math": {"water", "temperature", "light", "weight", "distance", "calculation", "measurement", "star", "astronomical", "mathematical", "scientific", "physics", "chemistry", "biology", "mathematics", "geometry", "algebra", "equation", "formula", "experiment", "hypothesis", "theory", "law", "energy", "matter", "atom", "molecule", "cell", "organ", "body"},
    "economics": {"money", "cost", "budget", "expenditure", "income", "savings", "financial", "price", "pound", "dollar", "profit", "loss", "investment", "bank", "account", "tax", "debt", "credit", "loan", "business", "commerce", "trade", "market", "shop", "buy", "sell", "purchase"},
    "geography": {"street", "road", "route", "dublin", "eccles", "city", "house", "garden", "direction", "church", "place", "square", "avenue", "lane", "park", "river", "bridge", "hill", "mountain", "valley", "country", "nation", "island", "sea", "ocean", "lake", "bay", "harbor", "port"},
    "daily_routine": {"breakfast", "lunch", "dinner", "meal", "eat", "drink", "sleep", "wake", "walk", "go", "come", "return", "arrive", "leave", "start", "finish", "begin", "end", "morning", "evening", "afternoon", "night", "day", "today", "yesterday", "tomorrow", "time", "clock", "hour", "minute", "second", "schedule", "routine", "habit", "wash", "bathe", "dress"},
    "food_drink": {"bread", "butter", "cheese", "milk", "tea", "coffee", "water", "wine", "beer", "whiskey", "food", "drink", "soup", "meat", "fish", "egg", "fruit", "vegetable", "apple", "orange", "banana", "potato", "onion", "salt", "pepper", "sugar", "flour", "oil", "vinegar", "juice", "cake", "cookie", "sandwich", "breakfast", "lunch", "dinner", "supper"},
    "body_health": {"head", "eye", "ear", "nose", "mouth", "teeth", "tongue", "lip", "face", "hair", "neck", "shoulder", "arm", "hand", "finger", "thumb", "leg", "foot", "toe", "heart", "lung", "stomach", "liver", "brain", "blood", "skin", "bone", "muscle", "pain", "ache", "ill", "sick", "healthy", "doctor", "medicine", "pill", "drug", "hospital", "health", "body", "physical", "strength", "weakness", "tired", "rest", "exercise"},
    "history_memory": {"past", "present", "future", "history", "memory", "remember", "forget", "recall", "recollection", "experience", "event", "incident", "story", "narrative", "chronicle", "record", "date", "year", "century", "era", "period", "age", "time", "moment", "occasion", "yesterday", "ancient", "modern", "old", "young", "tradition", "heritage", "legacy"},
    "emotion_inner_life": {"happy", "sad", "angry", "furious", "calm", "peaceful", "excited", "bored", "tired", "energetic", "hopeful", "desperate", "afraid", "brave", "lonely", "loved", "hated", "jealous", "grateful", "guilty", "innocent", "proud", "ashamed", "confident", "shy", "curious", "surprised", "disappointed", "satisfied", "frustrated", "anxious", "relaxed", "stressed", "emotion", "feeling", "mood", "attitude", "perspective", "outlook", "dream", "fantasy", "imagination"},
}


# ============================================================================
# Helpers
# ============================================================================


def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def assign_topic(question, answer, topic_keywords):
    """Assign a single Q&A pair to a topic based on keyword overlap.

    Tokenizes the question plus the first 200 characters of the answer,
    then returns the topic with the most keyword matches (or 'other').
    """
    combined = question + " " + answer[:200]
    tokens = set(
        t.lower() for t in word_tokenize(combined) if t.isalpha()
    )
    best_topic = "other"
    best_count = 0
    for topic, keywords in topic_keywords.items():
        overlap = len(tokens & keywords)
        if overlap > best_count:
            best_count = overlap
            best_topic = topic
    return best_topic


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_parse_catechism(episode_file):
    """Parse catechism Q&A pairs and classify each question."""
    text = cached_load_episode(episode_file)
    qa_pairs = suppress_stdout(parse_catechism, text)
    classified = []
    for question, answer in qa_pairs:
        q_type = classify_question(question)
        classified.append({
            "question": question,
            "answer": answer,
            "type": q_type,
            "answer_length": len(answer),
        })
    return classified


@st.cache_data
def cached_extract_triples(episode_file):
    """Extract knowledge triples from the episode's Q&A pairs."""
    text = cached_load_episode(episode_file)
    qa_pairs = suppress_stdout(parse_catechism, text)
    triples = suppress_stdout(extract_triples, qa_pairs)
    return triples


@st.cache_data
def cached_topic_distribution(episode_file):
    """Compute topic distribution for each Q&A pair."""
    classified = cached_parse_catechism(episode_file)
    topic_counts = Counter()
    qa_topics = []
    for item in classified:
        topic = assign_topic(item["question"], item["answer"], TOPIC_KEYWORDS)
        topic_counts[topic] += 1
        qa_topics.append(topic)
    return topic_counts, qa_topics


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=16,  # Ithaca
    caption="Week 17: Information Extraction & Knowledge Graphs",
    description=(
        "*Ithaca is Joyce's catechism episode — every fact about Bloom's night rendered "
        "in impersonal question-and-answer form, as if the universe itself were being "
        "cross-examined. This makes it uniquely amenable to information extraction: "
        "the Q&A structure is already halfway to a knowledge base.*"
    ),
)

is_ithaca = episode_file == "17ithaca.txt"

# Load data
episode_text = cached_load_episode(episode_file)


# ============================================================================
# Section 1: Parsing the Catechism
# ============================================================================

st.header("1. Parsing the Catechism")

if not is_ithaca:
    st.info(
        "The catechism parsing analysis is designed for Ithaca's Q&A structure. "
        "Other episodes will still be parsed, but results may be less meaningful — "
        "the parser looks for question-answer patterns that are characteristic of Ithaca."
    )

classified = cached_parse_catechism(episode_file)

if not classified:
    st.warning("No Q&A pairs found in this episode.")
else:
    # --- Metrics row ---
    total_qa = len(classified)
    avg_answer_len = sum(c["answer_length"] for c in classified) / total_qa
    type_counts = Counter(c["type"] for c in classified)
    most_common_type = type_counts.most_common(1)[0][0]
    longest_answer = max(classified, key=lambda c: c["answer_length"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Q&A Pairs", total_qa)
    m2.metric("Avg Answer Length", f"{avg_answer_len:.0f} chars")
    m3.metric("Most Common Type", most_common_type)
    m4.metric("Longest Answer", f"{longest_answer['answer_length']:,} chars")

    # --- Question type bar chart (horizontal) ---
    st.subheader("Question Type Distribution")

    type_sorted = type_counts.most_common()
    type_labels = [t for t, _ in type_sorted]
    type_values = [c for _, c in type_sorted]

    fig_types, ax_types = plt.subplots(figsize=(10, max(4, len(type_labels) * 0.4)))
    ax_types.barh(range(len(type_labels)), type_values, color="#4A90D9")
    ax_types.set_yticks(range(len(type_labels)))
    ax_types.set_yticklabels(type_labels, fontsize=9)
    ax_types.invert_yaxis()
    ax_types.set_xlabel("Count")
    ax_types.set_title(f"Question Types — {episode_label}")
    plt.tight_layout()
    st.pyplot(fig_types)
    plt.close(fig_types)

    # --- Q&A browser ---
    st.subheader("Q&A Browser")

    filter_type = st.selectbox(
        "Filter by question type",
        ["All"] + [t for t, _ in type_sorted],
        key="qa_type_filter",
    )

    if filter_type == "All":
        filtered_qa = classified
    else:
        filtered_qa = [c for c in classified if c["type"] == filter_type]

    st.caption(f"Showing {len(filtered_qa)} of {total_qa} Q&A pairs")

    if filtered_qa:
        qa_options = [
            f"Q{i+1}: {c['question'][:80]}..."
            for i, c in enumerate(filtered_qa)
        ]
        selected_qa = st.selectbox("Select a Q&A pair", qa_options, key="qa_browser")
        sel_idx = qa_options.index(selected_qa)
        sel_item = filtered_qa[sel_idx]

        st.markdown(f"**Question:** {sel_item['question']}")
        st.markdown(f"**Type:** {sel_item['type']}")
        st.markdown(f"**Answer length:** {sel_item['answer_length']:,} characters")
        with st.expander("View full answer"):
            st.write(sel_item["answer"])

    # --- Answer length histogram ---
    st.subheader("Answer Length Distribution")

    answer_lengths = [c["answer_length"] for c in classified]

    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.hist(answer_lengths, bins=30, color="#81B29A", edgecolor="white", alpha=0.8)
    ax_hist.axvline(avg_answer_len, color="#E07A5F", linestyle="--", label=f"Mean ({avg_answer_len:.0f})")
    ax_hist.set_xlabel("Answer Length (characters)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Answer Length Distribution — {episode_label}")
    ax_hist.legend()
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)


# ============================================================================
# Section 2: Knowledge Triple Extraction
# ============================================================================

st.header("2. Knowledge Triple Extraction")

if not is_ithaca:
    st.info(
        "Knowledge triple extraction works best on Ithaca's structured Q&A format. "
        "Results for other episodes may contain noise from narrative prose."
    )

triples = cached_extract_triples(episode_file)

if not triples:
    st.warning("No triples extracted from this episode.")
else:
    # --- Metrics row ---
    total_triples = len(triples)
    unique_subjects = len(set(t[0] for t in triples))
    unique_predicates = len(set(t[1] for t in triples))
    unique_objects = len(set(t[2] for t in triples))

    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("Total Triples", total_triples)
    tm2.metric("Unique Subjects", unique_subjects)
    tm3.metric("Unique Predicates", unique_predicates)
    tm4.metric("Unique Objects", unique_objects)

    # --- Top subjects, predicates, objects bar charts ---
    st.subheader("Top Subjects, Predicates & Objects")

    subject_counts = Counter(t[0] for t in triples).most_common(15)
    predicate_counts = Counter(t[1] for t in triples).most_common(15)
    object_counts = Counter(t[2] for t in triples).most_common(15)

    fig_spo, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, data, title, color in zip(
        axes,
        [subject_counts, predicate_counts, object_counts],
        ["Top Subjects", "Top Predicates", "Top Objects"],
        ["#4A90D9", "#E07A5F", "#81B29A"],
    ):
        if data:
            labels = [item[0][:25] for item in data]
            values = [item[1] for item in data]
            ax.barh(range(len(labels)), values, color=color)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            ax.set_title(title)

    plt.tight_layout()
    st.pyplot(fig_spo)
    plt.close(fig_spo)

    # --- Knowledge graph visualization ---
    st.subheader("Knowledge Graph")

    # Limit to top 60 triples by subject frequency
    subject_freq = Counter(t[0] for t in triples)
    top_subjects = set(s for s, _ in subject_freq.most_common(20))
    graph_triples = [t for t in triples if t[0] in top_subjects][:60]

    if graph_triples:
        G = nx.DiGraph()
        for subj, pred, obj in graph_triples:
            G.add_edge(subj, obj, label=pred)

        fig_graph, ax_graph = plt.subplots(figsize=(14, 10))

        try:
            pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        except Exception:
            pos = nx.shell_layout(G)

        # Node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [max(100, degrees.get(n, 1) * 80) for n in G.nodes()]

        # Color subjects vs objects
        subject_set = set(t[0] for t in graph_triples)
        object_set = set(t[2] for t in graph_triples)
        node_colors = []
        for n in G.nodes():
            if n in subject_set and n in object_set:
                node_colors.append("#DAA520")  # both
            elif n in subject_set:
                node_colors.append("#4A90D9")  # subject
            else:
                node_colors.append("#81B29A")  # object

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax_graph)
        nx.draw_networkx_edges(G, pos, edge_color="#888888", alpha=0.4, arrows=True, arrowsize=12, ax=ax_graph)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax_graph)

        # Edge labels (predicates) — only show for a manageable number
        if len(graph_triples) <= 30:
            edge_labels = {(s, o): p for s, p, o in graph_triples}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax_graph)

        ax_graph.set_title(f"Knowledge Graph — {episode_label} (top {len(graph_triples)} triples)")
        ax_graph.axis("off")

        from matplotlib.patches import Patch
        ax_graph.legend(
            handles=[
                Patch(facecolor="#4A90D9", label="Subject only"),
                Patch(facecolor="#81B29A", label="Object only"),
                Patch(facecolor="#DAA520", label="Both subject & object"),
            ],
            loc="upper left",
            fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig_graph)
        plt.close(fig_graph)

        st.caption(
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges. "
            f"Arrow direction: subject -> object."
        )

    # --- Triple browser ---
    st.subheader("Triple Browser")

    subject_filter = st.text_input(
        "Filter by subject (case-insensitive)",
        value="",
        key="triple_subject_filter",
    )

    if subject_filter:
        display_triples = [
            t for t in triples
            if subject_filter.lower() in t[0].lower()
        ]
    else:
        display_triples = triples

    st.caption(f"Showing {len(display_triples)} of {total_triples} triples")

    triple_rows = [
        {"Subject": t[0], "Predicate": t[1], "Object": t[2]}
        for t in display_triples[:200]
    ]
    if triple_rows:
        st.dataframe(
            pd.DataFrame(triple_rows),
            use_container_width=True,
            hide_index=True,
        )

    # --- Download button for triples as TSV ---
    tsv_lines = ["Subject\tPredicate\tObject"]
    for t in triples:
        tsv_lines.append(f"{t[0]}\t{t[1]}\t{t[2]}")
    tsv_content = "\n".join(tsv_lines)

    st.download_button(
        label="Download triples as TSV",
        data=tsv_content,
        file_name=f"triples_{episode_file.replace('.txt', '')}.tsv",
        mime="text/tab-separated-values",
        key="download_triples",
    )


# ============================================================================
# Section 3: Topic Distribution
# ============================================================================

st.header("3. Topic Distribution")

st.markdown(
    "Each Q&A pair is assigned to a topic based on keyword overlap between "
    "the question text (plus the first 200 characters of the answer) and "
    "curated keyword sets for 11 thematic categories. This reveals Ithaca's "
    "encyclopedic ambition — its restless cataloguing of physical objects, "
    "human relations, science, economics, and everything else in Bloom's world."
)

topic_counts, qa_topics = cached_topic_distribution(episode_file)

if not topic_counts:
    st.warning("No topic assignments found.")
else:
    # --- Topic proportion pie chart ---
    st.subheader("Topic Proportions")

    topic_sorted = topic_counts.most_common()
    pie_labels = [t.replace("_", " ").title() for t, _ in topic_sorted]
    pie_values = [c for _, c in topic_sorted]

    cmap = plt.cm.Set3
    pie_colors = [cmap(i / len(pie_labels)) for i in range(len(pie_labels))]

    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax_pie.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        startangle=90,
        pctdistance=0.8,
    )
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(7)
    ax_pie.set_title(f"Topic Distribution — {episode_label}")
    plt.tight_layout()
    st.pyplot(fig_pie)
    plt.close(fig_pie)

    # --- Cross-episode comparison bar chart ---
    st.subheader("Cross-Episode Topic Comparison")

    compare_label = st.selectbox(
        "Compare with episode",
        [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
        index=max(0, [i for i, lbl in enumerate(EPISODE_LABELS) if lbl != episode_label and "Calypso" in lbl][:1] or [0]),
        key="topic_compare_episode",
    )
    compare_file = EPISODE_FILES[EPISODE_LABELS.index(compare_label)]

    compare_topic_counts, _ = cached_topic_distribution(compare_file)

    # Build comparison dataframe
    all_topics = sorted(set(list(topic_counts.keys()) + list(compare_topic_counts.keys())))
    all_topics = [t for t in all_topics if t != "other"] + (["other"] if "other" in all_topics else [])

    primary_vals = [topic_counts.get(t, 0) for t in all_topics]
    compare_vals = [compare_topic_counts.get(t, 0) for t in all_topics]

    # Normalize to proportions
    primary_total = sum(primary_vals) or 1
    compare_total = sum(compare_vals) or 1
    primary_pcts = [v / primary_total * 100 for v in primary_vals]
    compare_pcts = [v / compare_total * 100 for v in compare_vals]

    topic_display = [t.replace("_", " ").title() for t in all_topics]

    x = np.arange(len(all_topics))
    width = 0.35

    fig_comp, ax_comp = plt.subplots(figsize=(max(10, len(all_topics) * 1.2), 6))
    ax_comp.bar(x - width / 2, primary_pcts, width, label=episode_label.split(" — ")[1], color="#4A90D9", alpha=0.8)
    ax_comp.bar(x + width / 2, compare_pcts, width, label=compare_label.split(" — ")[1], color="#E07A5F", alpha=0.8)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(topic_display, rotation=45, ha="right", fontsize=8)
    ax_comp.set_ylabel("Proportion (%)")
    ax_comp.set_title("Topic Distribution Comparison")
    ax_comp.legend()
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # --- Topic arc across Q&A windows ---
    st.subheader("Topic Arc Across the Episode")

    st.markdown(
        "How do topics shift as the episode progresses? Each row is a topic; "
        "the x-axis divides Q&A pairs into windows. Color intensity shows the "
        "proportion of that topic within each window."
    )

    n_windows = st.slider("Number of windows", 5, 20, 10, key="topic_windows")

    if len(qa_topics) >= n_windows:
        window_size = len(qa_topics) // n_windows
        topic_list = sorted(set(qa_topics))
        topic_list = [t for t in topic_list if t != "other"] + (["other"] if "other" in topic_list else [])

        arc_matrix = np.zeros((len(topic_list), n_windows))
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size if w < n_windows - 1 else len(qa_topics)
            window_topics = qa_topics[start:end]
            window_counts = Counter(window_topics)
            window_total = len(window_topics) or 1
            for ti, topic in enumerate(topic_list):
                arc_matrix[ti, w] = window_counts.get(topic, 0) / window_total

        fig_arc, ax_arc = plt.subplots(figsize=(12, max(4, len(topic_list) * 0.4)))
        im_arc = ax_arc.imshow(arc_matrix, cmap="YlOrRd", aspect="auto")
        ax_arc.set_yticks(range(len(topic_list)))
        ax_arc.set_yticklabels(
            [t.replace("_", " ").title() for t in topic_list],
            fontsize=8,
        )
        ax_arc.set_xticks(range(n_windows))
        ax_arc.set_xticklabels(
            [f"Q&A {w * window_size + 1}-{min((w + 1) * window_size, len(qa_topics))}" for w in range(n_windows)],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax_arc.set_title(f"Topic Heatmap Across Episode — {episode_label}")
        fig_arc.colorbar(im_arc, ax=ax_arc, label="Proportion")
        plt.tight_layout()
        st.pyplot(fig_arc)
        plt.close(fig_arc)
    else:
        st.info("Not enough Q&A pairs to compute topic arc with the selected window count.")


# ============================================================================
# Footer
# ============================================================================

st.markdown("""
---

**What this week reveals:** Ithaca's catechism form — relentless, impersonal,
encyclopedic — is uniquely suited to computational analysis. The question-answer
structure parses cleanly into typed queries, the answers yield extractable knowledge
triples, and the topic distribution reveals the episode's restless ambition to
catalogue *everything*: the weight of water, the cost of a funeral, the distance to
the stars, the contents of a drawer. Where other episodes resist extraction (Penelope's
unpunctuated flow, Circe's hallucinatory drama), Ithaca *invites* it — and in doing
so reveals that Joyce's encyclopedism is not random but systematically structured,
moving through physical objects, human relations, science, economics, and geography
with the methodical thoroughness of a catechism examining the whole of Bloom's world.
""")
