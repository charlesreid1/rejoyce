"""
Week 04: Calypso
=================
Chunking and Named Entity Recognition.

NLTK Focus: nltk.chunk, ne_chunk, RegexpParser, noun phrase extraction

Exercises:
  1. NER as characterization
  2. Noun phrase chunking
  3. Entity co-occurrence and narrative structure
"""

import os
import sys
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
from nltk.tree import Tree
import matplotlib.pyplot as plt

try:
    import networkx as nx
    import matplotlib.pyplot as plt_networkx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("networkx not available. Install with: pip install networkx")

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print(
        "spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm"
    )

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]:
    nltk.download(resource, quiet=True)

SHORT = True  # True = limit to 100 sentences per text (fast); False = process all
SHORT_LIMIT = 100

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_named_entities_spacy(text, nlp):
    """Extract named entities from text using spaCy.

    Returns:
        entities: list of (entity_text, entity_type) tuples
        type_counts: Counter of entity types
    """
    sentences = sent_tokenize(text)
    if SHORT:
        sentences = sentences[:SHORT_LIMIT]
    entities = []
    total = len(sentences)

    for i, sent in enumerate(sentences):
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(
                f"\r  spaCy NER processing: {i + 1}/{total} sentences",
                end="",
                flush=True,
            )
        doc = nlp(sent)

        for ent in doc.ents:
            entities.append((ent.text, ent.label_))

    print()  # newline after progress
    type_counts = Counter(etype for _, etype in entities)
    return entities, type_counts


def extract_named_entities(text):
    """Extract named entities from text using NLTK's ne_chunk.

    Returns:
        entities: list of (entity_text, entity_type) tuples
        type_counts: Counter of entity types
    """
    sentences = sent_tokenize(text)
    if SHORT:
        sentences = sentences[:SHORT_LIMIT]
    entities = []
    total = len(sentences)

    for i, sent in enumerate(sentences):
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"\r  NER processing: {i + 1}/{total} sentences", end="", flush=True)
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged)

        for subtree in tree:
            if isinstance(subtree, Tree):
                entity_text = " ".join(word for word, tag in subtree.leaves())
                entity_type = subtree.label()
                entities.append((entity_text, entity_type))

    print()  # newline after progress
    type_counts = Counter(etype for _, etype in entities)
    return entities, type_counts


# ---------------------------------------------------------------------------
# Exercise 1: NER as Characterization
# ---------------------------------------------------------------------------


def ner_as_characterization():
    """Compare NER density between Calypso (Bloom) and Proteus (Stephen)."""
    calypso = load_episode("04calypso.txt")
    proteus = load_episode("03proteus.txt")

    cal_entities, cal_types = extract_named_entities(calypso)
    pro_entities, pro_types = extract_named_entities(proteus)

    # Tokenize only the sentences that were actually processed
    cal_sentences = sent_tokenize(calypso)
    pro_sentences = sent_tokenize(proteus)

    if SHORT:
        cal_sentences = cal_sentences[:SHORT_LIMIT]
        pro_sentences = pro_sentences[:SHORT_LIMIT]

    cal_tokens = sum(len(word_tokenize(sent)) for sent in cal_sentences)
    pro_tokens = sum(len(word_tokenize(sent)) for sent in pro_sentences)

    cal_density = len(cal_entities) / cal_tokens * 1000
    pro_density = len(pro_entities) / pro_tokens * 1000

    print("--- NER Comparison: Calypso vs. Proteus ---")
    print(f"{'Metric':<35} {'Calypso':>12} {'Proteus':>12}")
    print("-" * 60)
    print(f"{'Total tokens':<35} {cal_tokens:>12} {pro_tokens:>12}")
    print(
        f"{'Named entities found':<35} {len(cal_entities):>12} {len(pro_entities):>12}"
    )
    print(
        f"{'Entities per 1,000 tokens':<35} {cal_density:>12.2f} {pro_density:>12.2f}"
    )

    print("\n--- Entity Type Distribution ---")
    all_types = sorted(set(list(cal_types.keys()) + list(pro_types.keys())))
    print(f"{'Type':<15} {'Calypso':>10} {'Proteus':>10}")
    print("-" * 37)
    for etype in all_types:
        print(
            f"{etype:<15} {cal_types.get(etype, 0):>10} {pro_types.get(etype, 0):>10}"
        )

    # Show sample entities
    print("\n--- Sample Calypso Entities ---")
    for ent, etype in cal_entities[:20]:
        print(f"  [{etype:<12}] {ent}")

    print("\n--- Sample Proteus Entities ---")
    for ent, etype in pro_entities[:20]:
        print(f"  [{etype:<12}] {ent}")

    # Add spaCy comparison if available
    if SPACY_AVAILABLE:
        print("\n--- spaCy NER Comparison ---")
        try:
            nlp = spacy.load("en_core_web_sm")
            cal_entities_spacy, cal_types_spacy = extract_named_entities_spacy(
                calypso, nlp
            )
            pro_entities_spacy, pro_types_spacy = extract_named_entities_spacy(
                proteus, nlp
            )

            cal_density_spacy = len(cal_entities_spacy) / cal_tokens * 1000
            pro_density_spacy = len(pro_entities_spacy) / pro_tokens * 1000

            print(f"{'Metric':<35} {'Calypso':>12} {'Proteus':>12}")
            print("-" * 60)
            print(
                f"{'Named entities found (spaCy)':<35} {len(cal_entities_spacy):>12} {len(pro_entities_spacy):>12}"
            )
            print(
                f"{'Entities per 1,000 tokens (spaCy)':<35} {cal_density_spacy:>12.2f} {pro_density_spacy:>12.2f}"
            )

            print("\n--- Sample Calypso Entities (spaCy) ---")
            for ent, etype in cal_entities_spacy[:20]:
                print(f"  [{etype:<12}] {ent}")

            print("\n--- Sample Proteus Entities (spaCy) ---")
            for ent, etype in pro_entities_spacy[:20]:
                print(f"  [{etype:<12}] {ent}")
        except OSError:
            print(
                "spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm"
            )

    return (cal_entities, cal_types, cal_density), (
        pro_entities,
        pro_types,
        pro_density,
    )


# ---------------------------------------------------------------------------
# Exercise 2: Noun Phrase Chunking
# ---------------------------------------------------------------------------


def noun_phrase_chunking(text, label="Calypso"):
    """Extract noun phrases and prepositional phrases using RegexpParser.

    Returns:
        np_freq: Counter of noun phrase strings
        pp_freq: Counter of prepositional phrase strings
    """
    # Use combined grammar to ensure PP rule can match already-chunked NPs
    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}
        PP: {<IN><NP>}
    """

    parser = RegexpParser(grammar)

    sentences = sent_tokenize(text)
    if SHORT:
        sentences = sentences[:SHORT_LIMIT]
    np_freq = Counter()
    pp_freq = Counter()
    total = len(sentences)

    for i, sent in enumerate(sentences):
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"\r  Chunking: {i + 1}/{total} sentences", end="", flush=True)
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)

        # Parse with combined grammar
        tree = parser.parse(tagged)

        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join(word.lower() for word, tag in subtree.leaves())
                # Enhanced filtering to remove noise artifacts
                if (
                    len(phrase) >= 2  # At least 2 characters
                    and not all(
                        c in ".,;:!?-\"'—–()[]{}" for c in phrase
                    )  # Not all punctuation
                    and not phrase.startswith("'")  # Doesn't start with apostrophe
                    and not phrase.endswith("'")  # Doesn't end with apostrophe
                    and phrase
                    not in [
                        "' t",
                        "prr",
                        "t",
                        "s",
                        "d",
                        "ll",
                        "ve",
                        "re",
                        "stiffly round",
                        "—",
                    ]  # Common contractions and noise
                    and any(c.isalpha() for c in phrase)  # Contains at least one letter
                    and not (  # Additional filtering for common noise patterns
                        len(phrase.split()) == 2
                        and all(
                            word
                            in [
                                "stiffly",
                                "round",
                                "cleanly",
                                "evenly",
                                "widely",
                                "highly",
                                "deeply",
                                "fully",
                            ]
                            for word in phrase.split()
                        )
                    )
                ):
                    np_freq[phrase] += 1
            elif subtree.label() == "PP":
                phrase = " ".join(word.lower() for word, tag in subtree.leaves())
                # Enhanced filtering for PPs
                if (
                    len(phrase) >= 3  # At least 3 characters for PPs
                    and not all(
                        c in ".,;:!?-\"'—–()[]{}" for c in phrase
                    )  # Not all punctuation
                    and not phrase.startswith("'")  # Doesn't start with apostrophe
                    and not phrase.endswith("'")  # Doesn't end with apostrophe
                    and phrase != "stiffly round"  # Remove noise artifacts
                ):
                    pp_freq[phrase] += 1

    print()  # newline after progress
    print(f"\n--- Top 25 Noun Phrases: {label} ---")
    for phrase, count in np_freq.most_common(25):
        print(f"  {count:>4}  {phrase}")

    print(f"\n--- Top 25 Prepositional Phrases: {label} ---")
    for phrase, count in pp_freq.most_common(25):
        print(f"  {count:>4}  {phrase}")

    print(f"\n  Total unique NPs: {len(np_freq)}")
    print(f"  Total unique PPs: {len(pp_freq)}")

    return np_freq, pp_freq


# ---------------------------------------------------------------------------
# Exercise 3: Entity Co-occurrence and Narrative Structure
# ---------------------------------------------------------------------------


def entity_cooccurrence(text, label="Calypso"):
    """Build entity co-occurrence matrix by paragraph.

    Returns:
        cooccurrence: dict of (entity_a, entity_b) -> count
        entity_paragraphs: dict of entity -> list of paragraph indices
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 5:
        # Fall back to splitting on single newlines for denser text
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if SHORT:
        paragraphs = paragraphs[:SHORT_LIMIT]

    entity_paragraphs = defaultdict(list)
    cooccurrence = Counter()

    total_paras = len(paragraphs)
    for i, para in enumerate(paragraphs):
        if (i + 1) % 10 == 0 or i + 1 == total_paras:
            print(
                f"\r  Co-occurrence: {i + 1}/{total_paras} paragraphs",
                end="",
                flush=True,
            )
        entities_in_para = set()
        sentences = sent_tokenize(para)
        for sent in sentences:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            tree = ne_chunk(tagged)
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity_text = " ".join(w for w, t in subtree.leaves())
                    entities_in_para.add(entity_text)

        for entity in entities_in_para:
            entity_paragraphs[entity].append(i)

        # Co-occurrence: all pairs in same paragraph
        entity_list = sorted(entities_in_para)
        for j in range(len(entity_list)):
            for k in range(j + 1, len(entity_list)):
                pair = (entity_list[j], entity_list[k])
                cooccurrence[pair] += 1

    print()  # newline after progress
    print(f"\n--- Entity Co-occurrence: {label} ---")
    print(f"  Total paragraphs/segments: {len(paragraphs)}")
    print(f"  Unique entities found: {len(entity_paragraphs)}")

    print(f"\n--- Entities Appearing in Most Paragraphs ---")
    sorted_entities = sorted(entity_paragraphs.items(), key=lambda x: -len(x[1]))
    for entity, paras in sorted_entities[:15]:
        print(f"  {entity:<30} appears in {len(paras):>3} paragraphs")

    print(f"\n--- Top 15 Co-occurring Entity Pairs ---")
    for (e1, e2), count in cooccurrence.most_common(15):
        print(f"  {count:>3}  {e1} <-> {e2}")

    return cooccurrence, entity_paragraphs


def plot_entity_network(cooccurrence, label="Calypso"):
    """Create and save a networkx graph visualization of entity co-occurrence."""
    if not NETWORKX_AVAILABLE:
        print("NetworkX not available, skipping network visualization.")
        return

    if len(cooccurrence) == 0:
        print("No co-occurrence data available for network visualization.")
        return

    # Create graph
    G = nx.Graph()

    # Add edges with weights based on co-occurrence counts
    for (e1, e2), count in cooccurrence.most_common(50):  # Top 50 co-occurrences
        if count > 0:  # Only add edges with actual co-occurrence
            G.add_edge(e1, e2, weight=count)

    if len(G.nodes()) == 0:
        print("No strong enough connections for network visualization.")
        return

    # Create visualization
    plt_networkx.figure(figsize=(12, 10))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", alpha=0.7)

    # Draw edges with width proportional to weight
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        G, pos, width=[w * 0.5 for w in weights], alpha=0.5, edge_color="gray"
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

    plt_networkx.title(f"Entity Co-occurrence Network: {label}")
    plt_networkx.axis("off")
    plt_networkx.tight_layout()

    # Save plot
    network_filename = os.path.join(
        os.path.dirname(__file__), f"week04_entity_network_{label.lower()}.png"
    )
    plt_networkx.savefig(network_filename, dpi=150, bbox_inches="tight")
    plt_networkx.close()

    print(f"  Entity network graph saved to {network_filename}")


def plot_entity_trajectory(entity_paragraphs, entities_to_track=None, label="Calypso"):
    """Plot when entities appear across the episode's paragraphs."""
    if entities_to_track is None:
        # Track the top 8 most frequent entities
        sorted_ents = sorted(entity_paragraphs.items(), key=lambda x: -len(x[1]))
        entities_to_track = [e for e, _ in sorted_ents[:8]]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, entity in enumerate(entities_to_track):
        paras = entity_paragraphs.get(entity, [])
        ax.scatter(paras, [i] * len(paras), s=30, alpha=0.7, label=entity)

    ax.set_yticks(range(len(entities_to_track)))
    ax.set_yticklabels(entities_to_track, fontsize=9)
    ax.set_xlabel("Paragraph Index")
    ax.set_title(f"Entity Trajectory: {label}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week04_entity_trajectory.png"), dpi=150
    )
    plt.close()
    print(f"\n  Entity trajectory plot saved to week04/week04_entity_trajectory.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    calypso = load_episode("04calypso.txt")

    print("=" * 62)
    print("EXERCISE 1: NER as Characterization")
    print("=" * 62)
    ner_as_characterization()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Noun Phrase Chunking")
    print("=" * 62)
    noun_phrase_chunking(calypso)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Entity Co-occurrence")
    print("=" * 62)
    cooccurrence, entity_paras = entity_cooccurrence(calypso)
    plot_entity_trajectory(entity_paras)
    plot_entity_network(cooccurrence)
