"""
Week 05: Lotus Eaters
======================
WordNet and lexical semantics.

NLTK Focus: nltk.corpus.wordnet, synsets, hypernymy, hyponymy, meronymy,
            semantic similarity (wup_similarity, path_similarity)

Exercises:
  1. Semantic fields of narcosis
  2. Martha's malapropism and semantic distance
  3. Substitution chains
"""

import os
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'cmudict']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Semantic Fields of Narcosis
# ---------------------------------------------------------------------------

THEMATIC_WORDS = [
    'flower', 'languid', 'body', 'pin', 'altar', 'bath',
    'drug', 'floating', 'dissolve', 'communion', 'bread',
    'blood', 'lotus', 'water', 'sacrament',
]


def semantic_fields(words=None):
    """Map hypernym trees for thematic words and find convergence points.

    Returns dict mapping each word to its hypernym paths.
    """
    if words is None:
        words = THEMATIC_WORDS

    hypernym_data = {}
    common_ancestors = defaultdict(list)

    print("--- Semantic Fields: Hypernym Analysis ---")
    for word in words:
        synsets = wn.synsets(word)
        if not synsets:
            print(f"  {word:<15} — not in WordNet")
            continue

        # Use first synset (most common sense)
        ss = synsets[0]
        paths = ss.hypernym_paths()
        shortest_path = min(paths, key=len) if paths else []

        hypernym_data[word] = {
            'synset': ss,
            'definition': ss.definition(),
            'num_synsets': len(synsets),
            'path': [s.name() for s in shortest_path],
            'depth': len(shortest_path),
        }

        path_str = ' → '.join(s.name().split('.')[0] for s in shortest_path[-4:])
        print(f"  {word:<15} [{ss.name():<25}] depth={len(shortest_path):>2}  ...{path_str}")

    # Find convergence: common ancestors between thematic word pairs
    print("\n--- Hypernym Convergence (Lowest Common Subsumer) ---")
    pairs_checked = set()
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            if w1 not in hypernym_data or w2 not in hypernym_data:
                continue
            pair = tuple(sorted([w1, w2]))
            if pair in pairs_checked:
                continue
            pairs_checked.add(pair)

            ss1 = hypernym_data[w1]['synset']
            ss2 = hypernym_data[w2]['synset']
            lcs_list = ss1.lowest_common_hypernyms(ss2)
            if lcs_list:
                lcs = lcs_list[0]
                sim = ss1.wup_similarity(ss2) or 0
                if sim > 0.3:  # Only show meaningfully related pairs
                    print(f"  {w1:<12} + {w2:<12} → LCS: {lcs.name():<30} "
                          f"WuP sim: {sim:.3f}")

    return hypernym_data


# ---------------------------------------------------------------------------
# Exercise 2: Martha's Malapropism and Semantic Distance
# ---------------------------------------------------------------------------

def marthas_malapropism():
    """Compute semantic distance between 'world' and 'word', and other near-homophones.

    Also computes phonological distance using CMU Pronouncing Dictionary.
    """
    print("\n--- Martha's Malapropism: world/word ---")

    # Semantic distances
    pairs = [
        ('world', 'word'),
        ('flower', 'flour'),
        ('altar', 'alter'),
        ('body', 'bawdy'),
        ('sole', 'soul'),
        ('sun', 'son'),
        ('holy', 'wholly'),
        ('rite', 'right'),
        ('bread', 'bred'),
        ('wine', 'whine'),
    ]

    # Load CMU dict for phonological comparison
    try:
        pronouncing = cmudict.dict()
    except Exception:
        pronouncing = {}

    print(f"{'Pair':<25} {'Path Sim':>10} {'WuP Sim':>10} {'Phon Dist':>10}")
    print("-" * 57)

    for w1, w2 in pairs:
        ss1 = wn.synsets(w1)
        ss2 = wn.synsets(w2)

        if ss1 and ss2:
            path_sim = ss1[0].path_similarity(ss2[0]) or 0
            wup_sim = ss1[0].wup_similarity(ss2[0]) or 0
        else:
            path_sim = 0
            wup_sim = 0

        # Phonological distance (edit distance on phoneme sequences)
        phon_dist = '-'
        if w1 in pronouncing and w2 in pronouncing:
            p1 = pronouncing[w1][0]
            p2 = pronouncing[w2][0]
            phon_dist = nltk.edit_distance(p1, p2)

        print(f"  {w1}/{w2:<20} {path_sim:>10.3f} {wup_sim:>10.3f} {str(phon_dist):>10}")

    print("\n  Key insight: phonological closeness (low phon_dist) does NOT predict")
    print("  semantic closeness (high similarity). The pun lives in this gap.")


# ---------------------------------------------------------------------------
# Exercise 3: Substitution Chains
# ---------------------------------------------------------------------------

def substitution_chain(start_word, steps=10):
    """Build a chain of synonym substitutions through WordNet.

    At each step, replace the current word with its closest synonym
    (the first lemma of the first synset that isn't the word itself).

    Returns the chain as a list of words.
    """
    chain = [start_word]
    current = start_word
    visited = {start_word}

    for _ in range(steps):
        synsets = wn.synsets(current)
        found_next = False

        for ss in synsets:
            for lemma in ss.lemmas():
                candidate = lemma.name().replace('_', ' ')
                if candidate.lower() != current.lower() and candidate.lower() not in visited:
                    chain.append(candidate)
                    visited.add(candidate.lower())
                    current = candidate
                    found_next = True
                    break
            if found_next:
                break

        if not found_next:
            chain.append('[dead end]')
            break

    return chain


def run_substitution_chains():
    """Run substitution chains for key thematic words."""
    start_words = ['body', 'bread', 'flower', 'drug', 'water']

    print("\n--- Substitution Chains (10 steps) ---")
    all_chains = {}
    for word in start_words:
        chain = substitution_chain(word, steps=10)
        all_chains[word] = chain
        print(f"\n  {word}:")
        print(f"    → {' → '.join(chain)}")

    # Check for convergence
    print("\n--- Chain Convergence Check ---")
    for i, w1 in enumerate(start_words):
        for w2 in start_words[i+1:]:
            c1 = set(w.lower() for w in all_chains[w1])
            c2 = set(w.lower() for w in all_chains[w2])
            overlap = c1 & c2
            if overlap:
                print(f"  {w1} and {w2} converge at: {overlap}")

    return all_chains


# ---------------------------------------------------------------------------
# Exercise Bonus: Synset richness per content word
# ---------------------------------------------------------------------------

def avg_synset_count(text):
    """Compute average number of WordNet synsets per content word."""
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
    print(f"\n  Average synsets per content word: {avg:.2f}")
    print(f"  ({words_with_synsets} words with synsets out of {len(content_words)} content words)")
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    lotus = load_episode('05lotuseaters.txt')

    print("=" * 62)
    print("EXERCISE 1: Semantic Fields of Narcosis")
    print("=" * 62)
    semantic_fields()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Martha's Malapropism")
    print("=" * 62)
    marthas_malapropism()

    print("\n" + "=" * 62)
    print("EXERCISE 3: Substitution Chains")
    print("=" * 62)
    run_substitution_chains()

    print("\n" + "=" * 62)
    print("BONUS: Average Synset Count")
    print("=" * 62)
    avg_synset_count(lotus)
