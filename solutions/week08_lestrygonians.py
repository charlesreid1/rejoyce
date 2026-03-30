"""
Week 08: Lestrygonians
=======================
N-gram language models, Markov chains, and probabilistic text generation.

NLTK Focus: nltk.lm, nltk.util.bigrams/trigrams, MLE, Laplace smoothing

Exercises:
  1. Train and generate
  2. Perplexity as style measure
  3. Associative chains
"""

import os
import random
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import bigrams, trigrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def tokenize_sentences(text):
    """Tokenize text into list of sentence token-lists (lowercased alpha)."""
    sentences = sent_tokenize(text)
    return [word_tokenize(s.lower()) for s in sentences]


# ---------------------------------------------------------------------------
# Exercise 1: Train and Generate
# ---------------------------------------------------------------------------

def train_ngram_model(text, n=3):
    """Train an n-gram language model using NLTK's MLE.

    Returns the trained model and the vocabulary.
    """
    tokenized = tokenize_sentences(text)
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized)
    model = MLE(n)
    model.fit(train_data, padded_vocab)
    return model


def generate_sentences(model, num_sentences=10, max_words=30):
    """Generate sentences from a trained language model."""
    generated = []
    for _ in range(num_sentences):
        try:
            words = list(model.generate(max_words, random_seed=random.randint(0, 10000)))
            # Clean up: remove padding tokens
            words = [w for w in words if w not in ('<s>', '</s>')]
            sentence = ' '.join(words)
            generated.append(sentence)
        except Exception:
            generated.append("[generation failed]")
    return generated


def train_and_compare():
    """Train models on Lestrygonians and Proteus, generate from each."""
    lestry = load_episode('08lestrygonians.txt')
    proteus = load_episode('03proteus.txt')

    for label, text, n in [("Lestrygonians (bigram)", lestry, 2),
                            ("Lestrygonians (trigram)", lestry, 3),
                            ("Proteus (bigram)", proteus, 2),
                            ("Proteus (trigram)", proteus, 3)]:
        print(f"\n--- Generated from {label} ---")
        model = train_ngram_model(text, n=n)
        sentences = generate_sentences(model, num_sentences=5)
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent[:100]}")

    return


# ---------------------------------------------------------------------------
# Exercise 2: Perplexity as Style Measure
# ---------------------------------------------------------------------------

def compute_perplexity(train_text, test_text, n=2):
    """Compute perplexity of a model trained on train_text evaluated on test_text.

    Uses Laplace smoothing to avoid zero probabilities.
    """
    train_sents = tokenize_sentences(train_text)
    test_sents = tokenize_sentences(test_text)

    train_data, vocab = padded_everygram_pipeline(n, train_sents)
    model = Laplace(n)
    model.fit(train_data, vocab)

    # Compute perplexity on test sentences
    test_data, _ = padded_everygram_pipeline(n, test_sents)
    # Flatten test ngrams
    test_ngrams = []
    for sent_ngrams in test_data:
        test_ngrams.extend(list(sent_ngrams))

    try:
        ppl = model.perplexity(test_ngrams)
    except Exception:
        ppl = float('inf')

    return ppl


def perplexity_comparison():
    """Compare perplexity of Bloom-trained model on various test texts."""
    lestry = load_episode('08lestrygonians.txt')
    calypso = load_episode('04calypso.txt')
    proteus = load_episode('03proteus.txt')

    # Use a short reference text from Gutenberg
    try:
        from nltk.corpus import gutenberg
        nltk.download('gutenberg', quiet=True)
        reference = gutenberg.raw('austen-emma.txt')[:len(lestry)]
        ref_label = "Emma (Austen)"
    except Exception:
        reference = "The quick brown fox jumps over the lazy dog. " * 100
        ref_label = "Reference prose"

    test_texts = [
        ("Lestrygonians (self)", lestry),
        ("Calypso (Bloom)", calypso),
        ("Proteus (Stephen)", proteus),
        (ref_label, reference),
    ]

    print("--- Perplexity: Lestrygonians-trained bigram model ---")
    print(f"{'Test Text':<35} {'Perplexity':>12}")
    print("-" * 49)

    perplexities = []
    for label, test in test_texts:
        ppl = compute_perplexity(lestry, test, n=2)
        perplexities.append((label, ppl))
        print(f"  {label:<33} {ppl:>12.2f}")

    return perplexities


# ---------------------------------------------------------------------------
# Exercise 3: Associative Chains
# ---------------------------------------------------------------------------

def associative_chains(text, top_n=20):
    """Extract bigrams ranked by conditional probability P(w2|w1).

    Also identifies cross-sentence boundary bigrams.
    """
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    bigram_freq = Counter(bigrams(tokens))
    unigram_freq = Counter(tokens)

    # Conditional probability: P(w2|w1) = count(w1,w2) / count(w1)
    conditional = {}
    for (w1, w2), count in bigram_freq.items():
        if unigram_freq[w1] >= 3:  # minimum frequency
            prob = count / unigram_freq[w1]
            conditional[(w1, w2)] = prob

    top_assoc = sorted(conditional.items(), key=lambda x: -x[1])[:top_n]

    print(f"\n--- Top {top_n} Strongest Bigram Associations ---")
    print(f"  {'Bigram':<35} {'P(w2|w1)':>10} {'Count':>8}")
    print("  " + "-" * 55)
    for (w1, w2), prob in top_assoc:
        count = bigram_freq[(w1, w2)]
        print(f"  {w1 + ' → ' + w2:<35} {prob:>10.4f} {count:>8}")

    # Cross-sentence bigrams
    sentences = sent_tokenize(text)
    cross_sentence = []
    for i in range(len(sentences) - 1):
        tokens_curr = [t.lower() for t in word_tokenize(sentences[i]) if t.isalpha()]
        tokens_next = [t.lower() for t in word_tokenize(sentences[i+1]) if t.isalpha()]
        if tokens_curr and tokens_next:
            cross_sentence.append((tokens_curr[-1], tokens_next[0], i))

    print(f"\n--- Cross-Sentence Associative Links (sample) ---")
    cross_freq = Counter((w1, w2) for w1, w2, _ in cross_sentence)
    for (w1, w2), count in cross_freq.most_common(15):
        print(f"  {w1} → {w2}  ({count} times)")

    # Show specific cross-sentence transitions
    print(f"\n--- Sample Cross-Sentence Transitions ---")
    sample_indices = random.sample(range(len(cross_sentence)),
                                    min(10, len(cross_sentence)))
    for idx in sorted(sample_indices):
        w1, w2, sent_i = cross_sentence[idx]
        end_sent = sentences[sent_i][-60:]
        start_sent = sentences[sent_i + 1][:60]
        print(f"  ...{end_sent}")
        print(f"     [{w1}] → [{w2}]")
        print(f"  {start_sent}...")
        print()

    return top_assoc, cross_sentence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    lestry = load_episode('08lestrygonians.txt')

    print("=" * 62)
    print("EXERCISE 1: Train and Generate")
    print("=" * 62)
    train_and_compare()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Perplexity as Style Measure")
    print("=" * 62)
    perplexity_comparison()

    print("\n" + "=" * 62)
    print("EXERCISE 3: Associative Chains")
    print("=" * 62)
    associative_chains(lestry)
