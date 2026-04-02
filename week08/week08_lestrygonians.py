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

for resource in ["punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_sentences(text):
    """Tokenize text into list of sentence token-lists (lowercased alpha)."""
    sentences = sent_tokenize(text)
    return [word_tokenize(s.lower()) for s in sentences]


# ---------------------------------------------------------------------------
# Exercise 1: Train and Generate
# ---------------------------------------------------------------------------


def train_ngram_model(text, n=3):
    """Train an n-gram language model using NLTK's MLE or Laplace smoothing.

    Uses Laplace smoothing for trigram models to avoid degenerate generation.
    Returns the trained model and the vocabulary.
    """
    tokenized = tokenize_sentences(text)
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized)

    # Use Laplace smoothing for trigram models to avoid degenerate generation
    if n >= 3:
        model = Laplace(n)
    else:
        model = MLE(n)

    model.fit(train_data, padded_vocab)
    return model


def generate_sentences(model, num_sentences=10, max_words=30):
    """Generate sentences from a trained language model.

    Implements custom generation to avoid degenerate short outputs.
    """
    generated = []

    for _ in range(num_sentences):
        try:
            # Try multiple generation attempts for each sentence
            best_sentence = ""
            best_length = 0

            for attempt in range(5):  # Try 5 times to get a good sentence
                words = list(
                    model.generate(max_words, random_seed=random.randint(0, 10000))
                )
                # Clean up: remove padding tokens
                words = [w for w in words if w not in ("<s>", "</s>")]

                # Create sentence
                sentence = " ".join(words)

                # Keep the longest valid sentence
                if len(words) > best_length and len(sentence.strip()) > 5:
                    best_sentence = sentence
                    best_length = len(words)

            # If we got a reasonable sentence, use it
            if best_length >= 3:
                generated.append(best_sentence)
            else:
                generated.append("[generation failed - insufficient content]")
        except Exception:
            generated.append("[generation failed - exception]")

    return generated


def train_and_compare():
    """Train models on Lestrygonians and Proteus, generate from each."""
    lestry = load_episode("08lestrygonians.txt")
    proteus = load_episode("03proteus.txt")

    for label, text, n in [
        ("Lestrygonians (bigram)", lestry, 2),
        ("Lestrygonians (trigram)", lestry, 3),
        ("Proteus (bigram)", proteus, 2),
        ("Proteus (trigram)", proteus, 3),
    ]:
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
    # Flatten test ngrams and filter to only include ngrams of order n
    test_ngrams = []
    for sent_ngrams in test_data:
        # Filter ngrams to only include those of length n (order n)
        filtered_ngrams = [ngram for ngram in sent_ngrams if len(ngram) == n]
        test_ngrams.extend(filtered_ngrams)

    try:
        ppl = model.perplexity(test_ngrams)
    except Exception:
        ppl = float("inf")

    return ppl


def perplexity_comparison():
    """Compare perplexity of Bloom-trained model on various test texts."""
    lestry = load_episode("08lestrygonians.txt")
    calypso = load_episode("04calypso.txt")
    proteus = load_episode("03proteus.txt")

    # Use a short reference text from Gutenberg (non-fiction/journalistic alternative)
    try:
        from nltk.corpus import gutenberg

        nltk.download("gutenberg", quiet=True)
        # Using Bible as a non-fiction reference text instead of Austen's novel
        reference = gutenberg.raw("bible-kjv.txt")[: len(lestry)]
        ref_label = "Bible (KJV)"
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


def is_contraction_fragment(token):
    """Check if token is a contraction fragment like 't' or 's'."""
    return token in ["t", "s", "d", "ll", "ve", "re", "m"]


def is_proper_name_pair(w1, w2):
    """Check if the bigram consists of two proper names."""
    return w1.istitle() and w2.istitle()


import math


def associative_chains(text, top_n=20):
    """Extract bigrams ranked by PMI (Pointwise Mutual Information).

    Also identifies cross-sentence boundary bigrams.
    Filters out contraction fragments and separates proper name pairs.
    """
    # Tokenize and filter out punctuation
    raw_tokens = word_tokenize(text)
    tokens = [t.lower() for t in raw_tokens if t.isalpha() or t in ["'", "-"]]

    # Filter out contraction fragments
    filtered_tokens = []
    i = 0
    while i < len(tokens):
        if i > 0 and tokens[i - 1] == "'" and is_contraction_fragment(tokens[i]):
            # Skip contraction fragment
            i += 1
        else:
            filtered_tokens.append(tokens[i])
            i += 1

    bigram_freq = Counter(bigrams(filtered_tokens))
    unigram_freq = Counter(filtered_tokens)

    # Calculate total number of words for PMI
    total_words = len(filtered_tokens)

    # PMI: log(P(w1,w2) / (P(w1) * P(w2))) = log(count(w1,w2) * N / (count(w1) * count(w2)))
    pmi_scores = {}
    content_word_bigrams = {}  # Separate container for content-word bigrams
    name_bigrams = {}  # Separate container for proper name bigrams

    for (w1, w2), count in bigram_freq.items():
        # Skip bigrams with very low frequency (increased threshold)
        if unigram_freq[w1] < 10 or unigram_freq[w2] < 10:
            continue

        # Skip contraction fragments
        if is_contraction_fragment(w1) or is_contraction_fragment(w2):
            continue

        # Calculate PMI
        p_w1_w2 = count / total_words
        p_w1 = unigram_freq[w1] / total_words
        p_w2 = unigram_freq[w2] / total_words

        # Avoid division by zero
        if p_w1 * p_w2 == 0:
            continue

        pmi = math.log(p_w1_w2 / (p_w1 * p_w2))
        pmi_scores[(w1, w2)] = pmi

        # Categorize bigrams
        if is_proper_name_pair(w1, w2):
            name_bigrams[(w1, w2)] = pmi
        else:
            content_word_bigrams[(w1, w2)] = pmi

    # Sort and display top associations
    top_assoc = sorted(pmi_scores.items(), key=lambda x: -x[1])[:top_n]
    top_content = sorted(content_word_bigrams.items(), key=lambda x: -x[1])[:top_n]
    top_names = sorted(name_bigrams.items(), key=lambda x: -x[1])[:top_n]

    print(f"\n--- Top {top_n} Bigram Associations by PMI (All) ---")
    print(f"  {'Bigram':<35} {'PMI':>10} {'Count':>8}")
    print("  " + "-" * 55)
    for (w1, w2), pmi in top_assoc:
        count = bigram_freq[(w1, w2)]
        marker = "(name)" if is_proper_name_pair(w1, w2) else ""
        print(f"  {w1 + ' → ' + w2:<35} {pmi:>10.4f} {count:>8} {marker}")

    print(f"\n--- Top {top_n} Content-Word Bigram Associations by PMI ---")
    print(f"  {'Bigram':<35} {'PMI':>10} {'Count':>8}")
    print("  " + "-" * 55)
    for (w1, w2), pmi in top_content[:top_n]:
        count = bigram_freq[(w1, w2)]
        print(f"  {w1 + ' → ' + w2:<35} {pmi:>10.4f} {count:>8}")

    print(f"\n--- Top {top_n} Proper Name Bigram Associations by PMI ---")
    print(f"  {'Bigram':<35} {'PMI':>10} {'Count':>8}")
    print("  " + "-" * 55)
    for (w1, w2), pmi in top_names[:top_n]:
        count = bigram_freq[(w1, w2)]
        print(f"  {w1 + ' → ' + w2:<35} {pmi:>10.4f} {count:>8}")

    print(f"\n--- Top {top_n} Proper Name Bigram Associations ---")
    print(f"  {'Bigram':<35} {'P(w2|w1)':>10} {'Count':>8}")
    print("  " + "-" * 55)
    for (w1, w2), prob in top_names[:top_n]:
        count = bigram_freq[(w1, w2)]
        print(f"  {w1 + ' → ' + w2:<35} {prob:>10.4f} {count:>8}")

    # Cross-sentence bigrams
    sentences = sent_tokenize(text)
    cross_sentence = []
    for i in range(len(sentences) - 1):
        tokens_curr = [t.lower() for t in word_tokenize(sentences[i]) if t.isalpha()]
        tokens_next = [
            t.lower() for t in word_tokenize(sentences[i + 1]) if t.isalpha()
        ]
        if tokens_curr and tokens_next:
            cross_sentence.append((tokens_curr[-1], tokens_next[0], i))

    print(f"\n--- Cross-Sentence Associative Links (sample) ---")
    cross_freq = Counter((w1, w2) for w1, w2, _ in cross_sentence)
    for (w1, w2), count in cross_freq.most_common(15):
        print(f"  {w1} → {w2}  ({count} times)")

    # Show specific cross-sentence transitions
    print(f"\n--- Sample Cross-Sentence Transitions ---")
    # Set seed for reproducibility
    random.seed(42)
    sample_indices = random.sample(
        range(len(cross_sentence)), min(10, len(cross_sentence))
    )
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

if __name__ == "__main__":
    lestry = load_episode("08lestrygonians.txt")

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
