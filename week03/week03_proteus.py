"""
Week 03: Proteus
=================
Stemming, morphological analysis, and language identification.

NLTK Focus: nltk.stem (Porter, Lancaster, Snowball), edit_distance, stopword-based
            language detection heuristics, WordNet derivational morphology

Exercises:
  1. The stemmer's struggle
  2. Multilingual detection
  3. Derivational morphology and neologism
"""

import os
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.metrics.distance import edit_distance
from nltk.corpus import stopwords, wordnet
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: The Stemmer's Struggle
# ---------------------------------------------------------------------------

def stemmers_struggle(text, top_n=10):
    """Apply three stemmers and find the most aggressive reductions.

    For each stemmer, identifies the top-N cases where the stemmed form
    is most distant from the original (by edit distance).

    Returns dict mapping stemmer name -> list of (word, stem, distance).
    """
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')

    tokens = word_tokenize(text)
    alpha_tokens = list(set(t.lower() for t in tokens if t.isalpha() and len(t) > 2))

    stemmers = {
        'Porter': porter,
        'Lancaster': lancaster,
        'Snowball': snowball,
    }

    results = {}
    for name, stemmer in stemmers.items():
        reductions = []
        for word in alpha_tokens:
            stem = stemmer.stem(word)
            dist = edit_distance(word, stem)
            reductions.append((word, stem, dist))

        reductions.sort(key=lambda x: -x[2])
        results[name] = reductions[:top_n]

        print(f"\n--- {name} Stemmer: Top {top_n} Most Aggressive Reductions ---")
        print(f"  {'Original':<35} {'Stem':<25} {'Edit Dist':>10}")
        print("  " + "-" * 72)
        for word, stem, dist in reductions[:top_n]:
            print(f"  {word:<35} {stem:<25} {dist:>10}")

    # Disagreement rate between stemmers
    disagree_count = 0
    for word in alpha_tokens:
        stems = set(s.stem(word) for s in stemmers.values())
        if len(stems) > 1:
            disagree_count += 1

    disagree_rate = disagree_count / len(alpha_tokens) if alpha_tokens else 0
    print(f"\n  Stemmer disagreement rate: {disagree_rate:.3f} "
          f"({disagree_count}/{len(alpha_tokens)} unique words)")

    return results, disagree_rate


# ---------------------------------------------------------------------------
# Exercise 2: Multilingual Detection
# ---------------------------------------------------------------------------

# Simple Latin stopword list (NLTK doesn't include Latin)
LATIN_STOPWORDS = {
    'et', 'in', 'est', 'non', 'ad', 'cum', 'sed', 'ut', 'de', 'ex',
    'per', 'ab', 'si', 'qui', 'quod', 'aut', 'nec', 'quam', 'iam',
    'tam', 'hoc', 'ille', 'ipse', 'ego', 'tu', 'nos', 'vos', 'suo',
    'sua', 'suis', 'eius', 'ante', 'post', 'inter', 'sub', 'pro',
    'contra', 'super', 'omnis', 'deus', 'dei', 'corpus', 'anima',
}


def detect_languages(text, window_size=1):
    """Sliding-window language detector using stopword overlap.

    For each sentence, computes the proportion of tokens in each language's
    stopword list and assigns the most likely language.

    Returns list of (sentence, detected_language, scores_dict).
    """
    lang_stops = {
        'english': set(stopwords.words('english')),
        'french': set(stopwords.words('french')),
        'german': set(stopwords.words('german')),
        'italian': set(stopwords.words('italian')),
        'latin': LATIN_STOPWORDS,
    }

    sentences = sent_tokenize(text)
    results = []
    lang_counts = Counter()

    for sent in sentences:
        tokens = [t.lower() for t in word_tokenize(sent) if t.isalpha()]
        if not tokens:
            continue

        scores = {}
        for lang, stops in lang_stops.items():
            overlap = sum(1 for t in tokens if t in stops)
            scores[lang] = overlap / len(tokens)

        # Default to English if no clear winner
        best_lang = max(scores, key=scores.get)
        # Only flag non-English if the non-English score is at least as high
        # and there's meaningful non-English stopword presence
        if best_lang != 'english' and scores[best_lang] < 0.1:
            best_lang = 'english'

        lang_counts[best_lang] += 1
        results.append((sent[:80] + ('...' if len(sent) > 80 else ''), best_lang, scores))

    print("\n--- Language Detection Summary ---")
    print(f"  Total sentences analyzed: {len(results)}")
    for lang, count in lang_counts.most_common():
        print(f"  {lang:<12} {count:>5} sentences ({100*count/len(results):.1f}%)")

    # Show non-English detections
    print("\n--- Non-English Detections ---")
    non_english = [(s, l, sc) for s, l, sc in results if l != 'english']
    for sent, lang, scores in non_english[:15]:
        print(f"  [{lang:>8}] {sent}")

    return results, lang_counts


def non_english_token_proportion(text):
    """Estimate the proportion of non-English tokens in the episode."""
    english_stops = set(stopwords.words('english'))
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]

    # Heuristic: words not in an English word list
    # Use WordNet as a rough English dictionary
    english_words = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            english_words.add(lemma.name().lower().replace('_', ' '))

    non_english = [t for t in tokens if t not in english_words and len(t) > 2]
    proportion = len(non_english) / len(tokens) if tokens else 0

    print(f"\n  Non-English token proportion (heuristic): {proportion:.3f}")
    print(f"  ({len(non_english)} of {len(tokens)} alpha tokens)")
    print(f"  Sample non-English tokens: {non_english[:20]}")

    return proportion


# ---------------------------------------------------------------------------
# Exercise 3: Derivational Morphology and Neologism
# ---------------------------------------------------------------------------

INTERESTING_WORDS = [
    'ineluctable', 'nacheinander', 'nebeneinander', 'diaphane',
    'adiaphane', 'maestro', 'dogsbody', 'contransmagnificandjewbangtantiality',
    'snotgreen', 'scrotumtightening', 'seaspawn', 'wavespeech',
    'bridebed', 'childbed', 'deathbed', 'omphalos', 'thalatta',
    'augur', 'protean', 'metempsychosis',
]


def morphological_analysis(words=None):
    """Trace derivational history of unusual words via WordNet.

    For words not in WordNet, hypothesize a morphological parse.
    """
    if words is None:
        words = INTERESTING_WORDS

    lemmatizer = nltk.WordNetLemmatizer()

    print("\n--- Derivational Morphology Analysis ---")
    print(f"{'Word':<40} {'In WordNet?':<12} {'Synsets':<8} {'Analysis'}")
    print("-" * 100)

    in_wordnet = 0
    not_in_wordnet = 0

    for word in words:
        synsets = wordnet.synsets(word)
        if synsets:
            in_wordnet += 1
            # Get hypernym chain
            top_synset = synsets[0]
            hypernyms = top_synset.hypernym_paths()
            depth = len(hypernyms[0]) if hypernyms else 0
            definition = top_synset.definition()[:50]
            print(f"  {word:<38} {'Yes':<12} {len(synsets):<8} {definition}")
        else:
            not_in_wordnet += 1
            # Attempt morphological decomposition
            analysis = hypothesize_parse(word)
            print(f"  {word:<38} {'No':<12} {0:<8} {analysis}")

    print(f"\n  In WordNet: {in_wordnet}/{len(words)}")
    print(f"  Not in WordNet: {not_in_wordnet}/{len(words)}")
    return in_wordnet, not_in_wordnet


def hypothesize_parse(word):
    """Attempt to decompose a compound or neologism into recognizable parts."""
    word_lower = word.lower()

    # Check for known compound patterns
    # Try splitting at every position and checking both halves
    best_split = None
    best_score = 0

    for i in range(3, len(word_lower) - 2):
        left = word_lower[:i]
        right = word_lower[i:]
        left_in = bool(wordnet.synsets(left))
        right_in = bool(wordnet.synsets(right))
        score = int(left_in) + int(right_in)
        if score > best_score:
            best_score = score
            best_split = (left, right, left_in, right_in)

    if best_split and best_score >= 1:
        left, right, li, ri = best_split
        parts = []
        if li:
            parts.append(f"'{left}' (in WN)")
        else:
            parts.append(f"'{left}' (not in WN)")
        if ri:
            parts.append(f"'{right}' (in WN)")
        else:
            parts.append(f"'{right}' (not in WN)")
        return f"Compound: {' + '.join(parts)}"

    # Check for known prefixes/suffixes
    prefixes = ['un', 'in', 'dis', 'non', 'pre', 'post', 'anti', 'contra', 'trans']
    for prefix in prefixes:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            remainder = word_lower[len(prefix):]
            if wordnet.synsets(remainder):
                return f"Prefix '{prefix}-' + '{remainder}' (in WN)"

    return "Sui generis / foreign borrowing"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    proteus = load_episode('03proteus.txt')

    print("=" * 62)
    print("EXERCISE 1: The Stemmer's Struggle")
    print("=" * 62)
    stemmers_struggle(proteus)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Multilingual Detection")
    print("=" * 62)
    detect_languages(proteus)
    non_english_token_proportion(proteus)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Derivational Morphology and Neologism")
    print("=" * 62)
    morphological_analysis()
