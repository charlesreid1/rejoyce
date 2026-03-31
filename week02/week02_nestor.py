"""
Week 02: Nestor
================
Part-of-speech tagging and basic morphological analysis.

NLTK Focus: nltk.pos_tag, nltk.corpus.wordnet (lemmatization), tagged corpus exploration

Exercises:
  1. Tag and tabulate
  2. Deasy vs. Stephen
  3. Lemmatization and the weight of history
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import brown, wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                 'averaged_perceptron_tagger_eng',
                 'brown', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Tag and Tabulate
# ---------------------------------------------------------------------------

def tag_and_tabulate(text, label="Nestor"):
    """POS-tag a text and compute tag frequency statistics.

    Returns:
        tagged: list of (word, tag) tuples
        tag_freq: Counter of tag frequencies
        ratios: dict of key ratios (noun/verb, adj/adv)
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tag_freq = Counter(tag for _, tag in tagged)

    # Noun tags: NN, NNS, NNP, NNPS
    nouns = sum(c for t, c in tag_freq.items() if t.startswith('NN'))
    # Verb tags: VB, VBD, VBG, VBN, VBP, VBZ
    verbs = sum(c for t, c in tag_freq.items() if t.startswith('VB'))
    # Adjectives: JJ, JJR, JJS
    adjs = sum(c for t, c in tag_freq.items() if t.startswith('JJ'))
    # Adverbs: RB, RBR, RBS
    advs = sum(c for t, c in tag_freq.items() if t.startswith('RB'))

    ratios = {
        'noun_verb_ratio': nouns / verbs if verbs else float('inf'),
        'adj_adv_ratio': adjs / advs if advs else float('inf'),
        'noun_count': nouns,
        'verb_count': verbs,
        'adj_count': adjs,
        'adv_count': advs,
    }

    print(f"\n--- POS Tag Distribution: {label} ---")
    for tag, count in tag_freq.most_common(15):
        print(f"  {tag:<6} {count:>6}")
    print(f"\n  Noun/Verb ratio:  {ratios['noun_verb_ratio']:.3f}")
    print(f"  Adj/Adv ratio:    {ratios['adj_adv_ratio']:.3f}")

    return tagged, tag_freq, ratios


def compare_to_brown(tag_freq):
    """Compare episode POS distribution to the Brown Corpus."""
    brown_tags = Counter(tag for _, tag in brown.tagged_words(tagset='universal'))
    # Map Penn Treebank to simplified for comparison
    episode_universal = Counter()
    mapping = {
        'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
        'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB',
        'VBP': 'VERB', 'VBZ': 'VERB',
        'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
        'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
        'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
        'DT': 'DET', 'IN': 'ADP', 'CC': 'CONJ',
    }
    for tag, count in tag_freq.items():
        univ = mapping.get(tag, 'X')
        episode_universal[univ] += count

    total_ep = sum(episode_universal.values())
    total_brown = sum(brown_tags.values())

    print("\n--- Nestor vs. Brown Corpus (Universal Tags) ---")
    print(f"{'Tag':<8} {'Nestor %':>10} {'Brown %':>10} {'Diff':>10}")
    print("-" * 40)
    for tag in sorted(set(list(episode_universal.keys()) + list(brown_tags.keys()))):
        ep_pct = 100.0 * episode_universal.get(tag, 0) / total_ep
        br_pct = 100.0 * brown_tags.get(tag, 0) / total_brown
        diff = ep_pct - br_pct
        if ep_pct > 0.5 or br_pct > 0.5:
            print(f"{tag:<8} {ep_pct:>9.2f}% {br_pct:>9.2f}% {diff:>+9.2f}%")


# ---------------------------------------------------------------------------
# Exercise 2: Deasy vs. Stephen
# ---------------------------------------------------------------------------

def split_deasy_stephen(text):
    """Heuristic split of Deasy's dialogue from Stephen's interior monologue.

    Deasy's dialogue: lines that follow attribution patterns like
    '—' dash-prefixed dialogue in the second half of the episode.
    Stephen's interior: italicized or unattributed passages of reflection.

    This is approximate — the exercise acknowledges the difficulty.
    """
    lines = text.split('\n')

    # Heuristic: dialogue lines start with em-dash (—)
    # Deasy speaks in the second half of the episode (after the classroom scene)
    # We'll collect all dash-dialogue as "dialogue" and non-dash as "narration/interior"
    dialogue_lines = []
    interior_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('—') or stripped.startswith('--'):
            dialogue_lines.append(stripped)
        elif stripped:
            interior_lines.append(stripped)

    dialogue_text = ' '.join(dialogue_lines)
    interior_text = ' '.join(interior_lines)

    return dialogue_text, interior_text


def compare_voices(text):
    """Compare POS distributions of dialogue vs. interior monologue."""
    dialogue, interior = split_deasy_stephen(text)

    print("\n--- Dialogue vs. Interior Monologue ---")
    print(f"  Dialogue tokens:  {len(word_tokenize(dialogue))}")
    print(f"  Interior tokens:  {len(word_tokenize(interior))}")

    _, dtags, dratios = tag_and_tabulate(dialogue, "Dialogue")
    _, itags, iratios = tag_and_tabulate(interior, "Interior/Narration")

    print("\n--- Voice Comparison ---")
    print(f"{'Metric':<25} {'Dialogue':>12} {'Interior':>12}")
    print("-" * 50)
    for key in ['noun_verb_ratio', 'adj_adv_ratio', 'noun_count', 'verb_count']:
        print(f"{key:<25} {dratios[key]:>12.3f} {iratios[key]:>12.3f}")

    return dratios, iratios


# ---------------------------------------------------------------------------
# Exercise 3: Lemmatization and the Weight of History
# ---------------------------------------------------------------------------

def get_wordnet_pos(treebank_tag):
    """Map Penn Treebank POS tag to WordNet POS."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN  # default


def lemmatize_and_compare(text_a, text_b, label_a="Nestor", label_b="Telemachus"):
    """Lemmatize two episode texts and find distinctive lemmas.

    Returns the top lemmas more frequent in text_a than text_b (normalized).
    """
    lemmatizer = WordNetLemmatizer()

    def get_lemma_freq(text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        lemmas = []
        for word, tag in tagged:
            if word.isalpha():
                wn_pos = get_wordnet_pos(tag)
                lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
                lemmas.append(lemma)
        return FreqDist(lemmas), len(lemmas)

    freq_a, total_a = get_lemma_freq(text_a)
    freq_b, total_b = get_lemma_freq(text_b)

    # Normalized difference: (freq_a/total_a) - (freq_b/total_b)
    distinctive = {}
    for lemma in freq_a:
        norm_a = freq_a[lemma] / total_a
        norm_b = freq_b.get(lemma, 0) / total_b
        if freq_a[lemma] >= 3:  # minimum frequency threshold
            distinctive[lemma] = norm_a - norm_b

    top_distinctive = sorted(distinctive.items(), key=lambda x: -x[1])[:20]

    print(f"\n--- Top 20 Lemmas More Frequent in {label_a} than {label_b} ---")
    print(f"{'Lemma':<20} {label_a + ' (norm)':>15} {label_b + ' (norm)':>15} {'Diff':>10}")
    print("-" * 62)
    for lemma, diff in top_distinctive:
        na = freq_a[lemma] / total_a
        nb = freq_b.get(lemma, 0) / total_b
        print(f"{lemma:<20} {na:>15.5f} {nb:>15.5f} {diff:>+10.5f}")

    return top_distinctive


def lemmatization_loss_examples(text):
    """Show cases where lemmatization collapses meaningful distinctions."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Find words that lemmatize to the same form but may carry different meanings
    lemma_groups = {}
    for word, tag in tagged:
        if word.isalpha() and len(word) > 3:
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            if lemma != word.lower():
                if lemma not in lemma_groups:
                    lemma_groups[lemma] = set()
                lemma_groups[lemma].add(word.lower())

    print("\n--- Lemmatization Collapses (surface forms → single lemma) ---")
    for lemma, forms in sorted(lemma_groups.items(), key=lambda x: -len(x[1])):
        if len(forms) > 1:
            print(f"  {lemma:<20} ← {', '.join(sorted(forms))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    nestor = load_episode('02nestor.txt')
    telemachus = load_episode('01telemachus.txt')

    print("=" * 62)
    print("EXERCISE 1: Tag and Tabulate")
    print("=" * 62)
    tagged, tag_freq, ratios = tag_and_tabulate(nestor)
    compare_to_brown(tag_freq)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Deasy vs. Stephen")
    print("=" * 62)
    compare_voices(nestor)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Lemmatization and the Weight of History")
    print("=" * 62)
    lemmatize_and_compare(nestor, telemachus)
    lemmatization_loss_examples(nestor)
