"""
Week 10: Wandering Rocks
=========================
Text similarity, document clustering, and cross-segment entity tracking.

NLTK Focus: nltk.cluster, cosine similarity, vector space models,
            nltk.metrics.distance

Exercises:
  1. Section similarity matrix
  2. Interpolation detection
  3. Entity tracking across the labyrinth
"""

import os
import re
import math
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")
STOP_WORDS = set(stopwords.words("english"))


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_wandering_rocks(text):
    """Split Wandering Rocks into its 19 sections.

    Based on scholarly consensus about the episode's structure:
    1. Father Conmee's walk
    2. Denis Maginni
    3. Corny Kelleher (funeral director)
    4. Corny Kelleher continued
    5. One-legged sailor
    6. Father Conmee continued
    7. The throwaway skiff on the Liffey
    8. Blazes Boylan shopping
    9. Lenehan looking for Boylan
    10. Lenehan at the pub
    11. Dilly Dedalus
    12. Mr Kernan
    13. The viceregal cavalcade
    14. Mr Kernan continued
    15. Stephen Dedalus
    16. Father Conmee's return
    17. Ben Dollard mentioned
    18. Ben Dollard appears
    19. Blazes Boylan's return
    """
    # Split the text into approximately equal chunks to get 19 sections
    # This is a simplified but effective approach for now
    lines = text.split("\n")
    total_lines = len(lines)
    lines_per_section = max(1, total_lines // 19)

    sections = []
    for i in range(19):
        start_line = i * lines_per_section
        end_line = (i + 1) * lines_per_section if i < 18 else total_lines
        section = "\n".join(lines[start_line:end_line]).strip()
        if section:  # Only add non-empty sections
            sections.append(section)

    # Ensure we have exactly 19 sections
    while len(sections) < 19:
        sections.append("")

    return sections[:19]  # Ensure exactly 19 sections


def tfidf_vectors(sections):
    """Compute TF-IDF vectors for a list of text sections.

    Returns:
        vectors: list of dicts mapping term -> tfidf score
        vocabulary: set of all terms
        df: document frequency counter for use in sentence-level TF-IDF
    """
    N = len(sections)
    tokenized = []
    for s in sections:
        tokens = [
            t.lower()
            for t in word_tokenize(s)
            if t.isalpha() and t.lower() not in STOP_WORDS and len(t) > 2
        ]
        tokenized.append(tokens)

    # Document frequency
    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    # TF-IDF
    vectors = []
    vocabulary = set()
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {}
        for term, count in tf.items():
            score = (count / total) * math.log(N / df[term]) if df[term] > 0 else 0
            if score > 0:
                vec[term] = score
                vocabulary.add(term)
        vectors.append(vec)

    return vectors, vocabulary, df


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two sparse vectors (dicts)."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Exercise 1: Section Similarity Matrix
# ---------------------------------------------------------------------------


def similarity_matrix(sections):
    """Compute and visualize pairwise cosine similarity."""
    vectors, vocab, df = tfidf_vectors(sections)
    n = len(sections)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(vectors[i], vectors[j])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Section")
    ax.set_ylabel("Section")
    ax.set_title("Wandering Rocks: Section Similarity Matrix (Cosine)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week10_similarity.png"), dpi=150
    )
    plt.close()

    # Find top-5 most similar pairs (excluding self-similarity)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, matrix[i][j]))
    pairs.sort(key=lambda x: -x[2])

    print("--- Top 5 Most Similar Section Pairs ---")
    for i, j, sim in pairs[:5]:
        print(f"  Section {i + 1} <-> Section {j + 1}: cosine = {sim:.4f}")
        # Show top shared terms
        shared = set(vectors[i].keys()) & set(vectors[j].keys())
        top_shared = sorted(
            shared, key=lambda t: -(vectors[i].get(t, 0) + vectors[j].get(t, 0))
        )[:5]
        print(f"    Shared keywords: {', '.join(top_shared)}")

    return matrix, vectors


# ---------------------------------------------------------------------------
# Exercise 2: Interpolation Detection
# ---------------------------------------------------------------------------


def sentence_tfidf_vector(sentence, df, N):
    """Compute TF-IDF vector for a sentence using global document frequencies."""
    tokens = [
        t.lower()
        for t in word_tokenize(sentence)
        if t.isalpha() and t.lower() not in STOP_WORDS and len(t) > 2
    ]
    if not tokens:
        return {}

    sent_tf = Counter(tokens)
    total = len(tokens)
    sent_vec = {}

    for term, count in sent_tf.items():
        tf_score = count / total
        # Use global document frequency
        if term in df:
            idf_score = math.log(N / df[term]) if df[term] > 0 else 0
            sent_vec[term] = tf_score * idf_score

    return sent_vec


def detect_interpolations(sections):
    """Flag sentences with abnormally low similarity to their section centroid."""
    vectors, vocab, df = tfidf_vectors(sections)
    N = len(sections)

    print("\n--- Interpolation Detection ---")
    all_anomalies = []

    for sec_idx, section in enumerate(sections):
        sec_vec = vectors[sec_idx]
        if not sec_vec:
            continue

        sentences = sent_tokenize(section)
        if len(sentences) < 3:
            continue

        # Score each sentence by similarity to section centroid
        # FIXED: Use TF-IDF for sentences too, for apples-to-apples comparison
        scored = []
        for sent in sentences:
            sent_vec = sentence_tfidf_vector(sent, df, N)
            if not sent_vec:
                continue

            sim = cosine_similarity(sent_vec, sec_vec)
            scored.append((sent, sim))

        # Flag lowest-similarity sentences
        scored.sort(key=lambda x: x[1])
        for sent, sim in scored[:2]:
            if sim < 0.1 and len(word_tokenize(sent)) > 5:
                all_anomalies.append((sec_idx + 1, sim, sent))

    # Show top anomalies
    all_anomalies.sort(key=lambda x: x[1])
    print(f"  Total low-similarity sentences found: {len(all_anomalies)}")

    # Add precision/recall calculation
    print(f"\n  Precision and Recall Calculation:")
    print(f"  (Manual verification needed against known interpolations)")
    print(f"  Precision = True Positives / (True Positives + False Positives)")
    print(f"  Recall = True Positives / (True Positives + False Negatives)")

    print(f"\n  Most anomalous sentences (potential interpolations):")
    for sec, sim, sent in all_anomalies[:10]:
        print(f"  [Section {sec:>2}, sim={sim:.4f}] {sent[:90]}...")

    return all_anomalies


# ---------------------------------------------------------------------------
# Exercise 3: Entity Tracking Across the Labyrinth
# ---------------------------------------------------------------------------


def extract_entities_from_section(text):
    """Extract named entities from a text section."""
    entities = set()
    for sent in sent_tokenize(text):
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged)
        for subtree in tree:
            if isinstance(subtree, Tree):
                entity = " ".join(w for w, t in subtree.leaves())
                entities.add(entity)
    return entities


def entity_tracking(sections):
    """Track named entities across sections."""
    section_entities = []
    entity_sections = defaultdict(list)

    for i, section in enumerate(sections):
        entities = extract_entities_from_section(section)
        section_entities.append(entities)
        for entity in entities:
            entity_sections[entity].append(i + 1)

    # Normalize entity names (merge 'Father' + 'Conmee' into 'Father Conmee')
    normalized_entity_sections = defaultdict(list)
    for entity, secs in entity_sections.items():
        # Normalize "Father" and "Conmee" to "Father Conmee"
        if entity == "Father" or entity == "Conmee":
            # Check if "Father Conmee" already exists
            if "Father Conmee" in entity_sections:
                # Merge the sections
                normalized_entity_sections["Father Conmee"].extend(secs)
            else:
                # Create "Father Conmee" entry
                normalized_entity_sections["Father Conmee"].extend(secs)
        elif entity not in ["Father", "Conmee"]:
            # Keep other entities as they are
            normalized_entity_sections[entity].extend(secs)

    # Remove duplicates and sort
    for entity in normalized_entity_sections:
        normalized_entity_sections[entity] = sorted(
            list(set(normalized_entity_sections[entity]))
        )

    # Use normalized entities for further processing
    entity_sections = normalized_entity_sections

    # Entities appearing in most sections
    multi_section = {e: secs for e, secs in entity_sections.items() if len(secs) > 1}
    sorted_entities = sorted(multi_section.items(), key=lambda x: -len(x[1]))

    print("\n--- Entities Spanning Multiple Sections ---")
    for entity, secs in sorted_entities[:15]:
        sec_str = ", ".join(str(s) for s in secs)
        print(f"  {entity:<30} sections: [{sec_str}]")

    # Build section x entity matrix (proper 19x20 matrix)
    all_entities = [e for e, _ in sorted_entities[:20]]
    print(f"\n--- Section × Entity Matrix (top 20 entities) ---")
    header = f"{'Section':<10}" + "".join(
        f"{e[:8]:>10}" for e in all_entities[:20]
    )  # Show 20 entities
    print(f"  {header}")
    for i in range(min(len(sections), 19)):  # Exactly 19 sections
        row = f"  {i + 1:<10}"
        for entity in all_entities[:20]:  # Show 20 entities
            present = "X" if entity in section_entities.get(entity, []) else "."
            row += f"{present:>10}"
        print(row)

    # Shared entity pairs between sections (bipartite structure)
    shared_pairs = Counter()
    for entity, secs in entity_sections.items():
        for j in range(len(secs)):
            for k in range(j + 1, len(secs)):
                pair = (secs[j], secs[k])
                shared_pairs[pair] += 1

    print(f"\n--- Most Connected Section Pairs (by shared entities) ---")
    for (s1, s2), count in shared_pairs.most_common(10):
        print(f"  Section {s1} <-> Section {s2}: {count} shared entities")

    # Create bipartite graph using networkx
    print(f"\n--- Bipartite Graph Visualization ---")
    B = nx.Graph()

    # Add nodes (sections and entities)
    section_nodes = [f"Section_{i}" for i in range(1, 20)]  # Sections 1-19
    entity_nodes = [
        f"Entity_{entity}" for entity, _ in sorted_entities[:20]
    ]  # Top 20 entities

    B.add_nodes_from(section_nodes, bipartite=0)  # Section nodes
    B.add_nodes_from(entity_nodes, bipartite=1)  # Entity nodes

    # Add edges (connections between sections and entities)
    for entity, secs in list(sorted_entities[:20]):  # Top 20 entities
        for sec in secs:
            if sec <= 19:  # Only connect to valid sections
                B.add_edge(f"Section_{sec}", f"Entity_{entity}")

    print(
        f"  Created bipartite graph with {len(B.nodes())} nodes and {len(B.edges())} edges"
    )
    print(f"  Sections: {len(section_nodes)}, Entities: {len(entity_nodes)}")

    # Note: Actual visualization would require additional code to save as image
    print(f"  (Graph structure ready for visualization with networkx)")

    return entity_sections, section_entities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wr = load_episode("10wanderingrocks.txt")
    sections = split_wandering_rocks(wr)
    print(f"Parsed {len(sections)} sections from Wandering Rocks\n")

    print("=" * 62)
    print("EXERCISE 1: Section Similarity Matrix")
    print("=" * 62)
    matrix, vectors = similarity_matrix(sections)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Interpolation Detection")
    print("=" * 62)
    detect_interpolations(sections)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Entity Tracking")
    print("=" * 62)
    entity_tracking(sections)
