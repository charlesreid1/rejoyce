"""
Week 15: Circe
===============
Named Entity Recognition, speaker extraction, and network/graph analysis.

Focus: NER feeding into network visualization (networkx, matplotlib)

Exercises:
  1. Extract the dramatis personae — all of them
  2. Build the interaction graph
  3. The novel in one graph (cumulative entity network)
"""

import os
import re
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                 'averaged_perceptron_tagger_eng',
                 'words']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Circe Speaker Extraction
# ---------------------------------------------------------------------------

def extract_speakers(text):
    """Extract speaker tags from Circe's dramatic format.

    Circe uses a play-like format where speakers are indicated by
    ALL-CAPS names or parenthetical stage directions.

    Returns:
        speakers: Counter of speaker name -> line count
        stage_directions: list of stage direction strings
        scenes: list of (speakers_in_scene, dialogue_lines) tuples
    """
    lines = text.split('\n')
    speakers = Counter()
    stage_directions = []
    speaker_sequence = []

    # Pattern for speaker tags: ALL CAPS followed by text
    speaker_pattern = re.compile(r'^([A-Z][A-Z\s]{2,})\s*[:\.]?\s*(.*)$')
    # Pattern for stage directions: text in parentheses
    stage_pattern = re.compile(r'^\((.+)\)$')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for stage direction
        stage_match = stage_pattern.match(stripped)
        if stage_match:
            stage_directions.append(stage_match.group(1))
            continue

        # Check for speaker tag
        speaker_match = speaker_pattern.match(stripped)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            # Filter out very short or unlikely speakers
            if len(speaker) > 1 and speaker not in {'THE', 'AND', 'BUT', 'ALL', 'HIS', 'HER'}:
                speakers[speaker] += 1
                speaker_sequence.append(speaker)

    # Build scenes using a sliding window over the speaker sequence.
    # Each window of consecutive speaker turns forms a "scene" capturing
    # who is talking near whom.
    window_size = 5
    scenes = []
    for i in range(len(speaker_sequence)):
        window = speaker_sequence[i:i + window_size]
        scene_speakers = frozenset(window)
        if len(scene_speakers) > 1:
            scenes.append((scene_speakers, []))

    return speakers, stage_directions, scenes


def classify_entity(name):
    """Classify a Circe entity: person (living), person (dead/hallucinated),
    object, animal, abstraction, collective."""
    # Known dead/hallucinated characters
    dead = {'RUDOLPH', 'RUDY', 'PADDY DIGNAM', 'VIRAG', 'THE MOTHER',
            'ELLEN BLOOM', 'QUEEN VICTORIA'}
    objects = {'THE CAP', 'THE GASJET', 'THE WATERFALL', 'A BUTTON',
               'THE SOAP', 'THE FAN', 'THE BELL', 'THE PIANOLA',
               'THE NYMPH', 'THE YEWS'}
    abstractions = {'THE END OF THE WORLD', 'THE VOICE', 'ALL',
                    'THE ECHO', 'THE BRACELETS'}
    animals = {'THE CAT', 'THE DOG', 'THE MOTH'}

    name_upper = name.upper()
    if name_upper in dead:
        return 'dead/hallucinated'
    elif name_upper in objects:
        return 'object'
    elif name_upper in abstractions:
        return 'abstraction'
    elif name_upper in animals:
        return 'animal'
    else:
        return 'person (living)'


# ---------------------------------------------------------------------------
# Exercise 1: Dramatis Personae
# ---------------------------------------------------------------------------

def dramatis_personae():
    """Extract and classify all speaking entities in Circe."""
    circe = load_episode('15circe.txt')
    speakers, stage_dirs, scenes = extract_speakers(circe)

    print(f"--- Dramatis Personae ---")
    print(f"  Total unique speakers: {len(speakers)}")
    print(f"  Total stage directions: {len(stage_dirs)}")
    print(f"  Total scenes: {len(scenes)}")

    # Classify entities
    categories = Counter()
    classified = []
    for speaker, count in speakers.most_common():
        cat = classify_entity(speaker)
        categories[cat] += 1
        classified.append((speaker, count, cat))

    print(f"\n--- Entity Categories ---")
    for cat, count in categories.most_common():
        print(f"  {cat:<25} {count:>5}")

    print(f"\n--- Top 30 Speakers by Line Count ---")
    for speaker, count, cat in classified[:30]:
        print(f"  {speaker:<30} {count:>5} lines  [{cat}]")

    # Plot category distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = [c for c, _ in categories.most_common()]
    counts = [n for _, n in categories.most_common()]
    ax.barh(cats, counts, color='steelblue')
    ax.set_xlabel('Number of Entities')
    ax.set_title('Circe: Entity Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week15_categories.png'), dpi=150)
    plt.close()

    return speakers, classified, scenes


# ---------------------------------------------------------------------------
# Exercise 2: Build the Interaction Graph
# ---------------------------------------------------------------------------

def build_interaction_graph(scenes, min_degree=2):
    """Build a co-appearance graph from scenes.

    Nodes = speakers, edges = co-appearance in same scene, weighted by count.
    Returns adjacency dict and edge weights.
    """
    edges = Counter()
    node_scenes = defaultdict(int)

    for scene_speakers, _ in scenes:
        speakers_list = sorted(scene_speakers)
        for s in speakers_list:
            node_scenes[s] += 1
        for i in range(len(speakers_list)):
            for j in range(i + 1, len(speakers_list)):
                edge = (speakers_list[i], speakers_list[j])
                edges[edge] += 1

    # Filter to nodes with minimum degree
    node_degree = Counter()
    for (a, b), w in edges.items():
        node_degree[a] += w
        node_degree[b] += w

    filtered_nodes = {n for n, d in node_degree.items() if d >= min_degree}
    filtered_edges = {e: w for e, w in edges.items()
                      if e[0] in filtered_nodes and e[1] in filtered_nodes}

    # Graph metrics
    total_nodes = len(filtered_nodes)
    total_edges = len(filtered_edges)
    max_possible = total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 1
    density = total_edges / max_possible if max_possible > 0 else 0

    # Betweenness centrality approximation (degree centrality)
    degree_central = sorted(node_degree.items(), key=lambda x: -x[1])

    print(f"\n--- Interaction Graph ---")
    print(f"  Nodes (speakers):  {total_nodes}")
    print(f"  Edges:             {total_edges}")
    print(f"  Graph density:     {density:.4f}")

    print(f"\n--- Most Central Nodes (by degree) ---")
    for node, deg in degree_central[:15]:
        print(f"  {node:<30} degree: {deg:>5}")

    print(f"\n--- Strongest Edges (most co-appearances) ---")
    for (a, b), w in edges.most_common(15):
        print(f"  {a:<25} <-> {b:<25} weight: {w}")

    return filtered_nodes, filtered_edges, node_degree


# ---------------------------------------------------------------------------
# Exercise 3: Cumulative Entity Network
# ---------------------------------------------------------------------------

def cumulative_entity_network():
    """Build entity networks for each episode and show how Circe reconnects them."""
    episode_files = [
        ('01', '01telemachus.txt'),
        ('02', '02nestor.txt'),
        ('03', '03proteus.txt'),
        ('04', '04calypso.txt'),
        ('05', '05lotuseaters.txt'),
        ('06', '06hades.txt'),
        ('07', '07aeolus.txt'),
        ('08', '08lestrygonians.txt'),
        ('09', '09scyllacharybdis.txt'),
        ('10', '10wanderingrocks.txt'),
        ('11', '11sirens.txt'),
        ('12', '12cyclops.txt'),
        ('13', '13nausicaa.txt'),
        ('14', '14oxenofthesun.txt'),
        ('15', '15circe.txt'),
        ('16', '16eumaeus.txt'),
        ('17', '17ithaca.txt'),
        ('18', '18penelope.txt'),
    ]

    # Regex-based proper noun extraction: find sequences of capitalized words
    # that aren't at sentence starts. Filter common false positives.
    stop_names = {
        # Articles, conjunctions, prepositions, pronouns
        'The', 'A', 'An', 'And', 'But', 'For', 'His', 'Her', 'Its',
        'He', 'She', 'It', 'They', 'We', 'You', 'My', 'Who', 'What',
        'That', 'This', 'There', 'Then', 'Than', 'These', 'Those',
        'With', 'From', 'Into', 'Upon', 'Your', 'Their', 'Our',
        'Not', 'Nor', 'Now', 'How', 'Where', 'When', 'Why', 'Which',
        'Have', 'Has', 'Had', 'Was', 'Were', 'Are', 'Been', 'Being',
        'Will', 'Would', 'Could', 'Should', 'Shall', 'May', 'Might',
        'Do', 'Does', 'Did', 'Can', 'Must', 'Need', 'Dare',
        'So', 'No', 'Yes', 'Oh', 'Ah', 'If', 'Or', 'As', 'At',
        'By', 'In', 'On', 'To', 'Of', 'Up', 'Out', 'Off',
        'All', 'Each', 'Every', 'Some', 'Any', 'Such', 'Only',
        'About', 'After', 'Before', 'Over', 'Under', 'Between',
        # Titles
        'Mr', 'Mrs', 'Miss', 'Sir', 'Lord', 'Saint', 'Father', 'Mother',
        # Circe stage direction verbs
        'Points', 'Laughs', 'Cries', 'Calls', 'Turns', 'Takes',
        'Looks', 'Stands', 'Walks', 'Sits', 'Gets', 'Puts', 'Runs',
        'Comes', 'Goes', 'Says', 'Speaks', 'Sings', 'Whispers',
    }
    proper_noun_pat = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

    def extract_entities(text):
        """Extract likely proper nouns via regex on capitalized word sequences."""
        entities = set()
        for sent in sent_tokenize(text):
            # Skip the first word of each sentence (always capitalized)
            trimmed = sent.split(None, 1)
            if len(trimmed) < 2:
                continue
            remainder = trimmed[1]
            for match in proper_noun_pat.finditer(remainder):
                name = match.group(1)
                words = name.split()
                # Filter: at least one word must not be a stop word
                if any(w not in stop_names for w in words):
                    entities.add(name)
        return entities

    episode_entities = {}
    all_entities = Counter()

    for ep_num, filename in episode_files:
        print(f"  Processing episode {ep_num} ({filename})...", end='', flush=True)
        text = load_episode(filename)
        entities = extract_entities(text)
        episode_entities[ep_num] = entities
        for e in entities:
            all_entities[e] += 1
        print(f" done ({len(entities)} entities)")

    # Entities shared across episodes
    multi_ep = {e: c for e, c in all_entities.items() if c > 1}

    print(f"\n--- Cumulative Entity Network ---")
    for ep_num, entities in episode_entities.items():
        print(f"  Episode {ep_num}: {len(entities)} entities")

    print(f"\n  Entities appearing in multiple episodes: {len(multi_ep)}")
    print(f"\n--- Most Cross-Referenced Entities ---")
    for entity, count in sorted(multi_ep.items(), key=lambda x: -x[1])[:15]:
        eps = [ep for ep, ents in episode_entities.items() if entity in ents]
        print(f"  {entity:<30} in episodes: {', '.join(eps)}")

    # Circe reactivation analysis
    circe_ents = episode_entities.get('15', set())
    prior_ents = set()
    for ep in ['01', '04', '06', '10', '12']:
        prior_ents |= episode_entities.get(ep, set())

    reactivated = circe_ents & prior_ents
    new_in_circe = circe_ents - prior_ents

    print(f"\n--- Circe Reactivation ---")
    print(f"  Entities reactivated from prior episodes: {len(reactivated)}")
    print(f"  New entities in Circe:                    {len(new_in_circe)}")
    print(f"  Reactivation ratio:                       "
          f"{len(reactivated)/(len(reactivated)+len(new_in_circe))*100:.1f}%"
          if (reactivated or new_in_circe) else "")

    return episode_entities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 62)
    print("EXERCISE 1: Dramatis Personae")
    print("=" * 62)
    speakers, classified, scenes = dramatis_personae()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Interaction Graph")
    print("=" * 62)
    build_interaction_graph(scenes)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Cumulative Entity Network")
    print("=" * 62)
    cumulative_entity_network()
