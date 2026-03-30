"""
Week 01: Telemachus
====================
Tokenization, text normalization, and basic corpus exploration.

NLTK Focus: nltk.tokenize, nltk.text.Text, concordance, frequency distributions

Exercises:
  1. Tokenize and profile
  2. Concordance as close reading
  3. Frequency and stopwords
"""

import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.corpus import stopwords, gutenberg
import matplotlib.pyplot as plt
import math

# Ensure NLTK data is available
for resource in ['punkt', 'punkt_tab', 'stopwords', 'gutenberg']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    """Load raw text for an episode."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Tokenize and Profile
# ---------------------------------------------------------------------------

def tokenize_and_profile(text, label="Episode"):
    """Compute basic token statistics for a text.

    Returns a dict with:
      - total_tokens: number of word tokens
      - total_types: number of unique word types (lowercased)
      - type_token_ratio: types / tokens
      - total_sentences: number of sentences
      - avg_sentence_length: mean tokens per sentence
    """
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Lowercase for type counting
    types = set(t.lower() for t in tokens if t.isalpha())
    alpha_tokens = [t for t in tokens if t.isalpha()]

    stats = {
        'label': label,
        'total_tokens': len(tokens),
        'total_alpha_tokens': len(alpha_tokens),
        'total_types': len(types),
        'type_token_ratio': len(types) / len(alpha_tokens) if alpha_tokens else 0,
        'total_sentences': len(sentences),
        'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
        'hapax_legomena': sum(1 for w, c in FreqDist(t.lower() for t in alpha_tokens).items() if c == 1),
    }
    stats['hapax_ratio'] = stats['hapax_legomena'] / stats['total_types'] if stats['total_types'] else 0
    return stats


def compare_profiles():
    """Compare Telemachus with a reference Gutenberg text (Austen's Emma ch.1 equivalent)."""
    telemachus = load_episode('01telemachus.txt')
    telem_stats = tokenize_and_profile(telemachus, label="Telemachus")

    # Use a comparable-length passage from Austen's Emma via Gutenberg
    emma_text = gutenberg.raw('austen-emma.txt')
    # Take roughly the same number of characters as Telemachus
    emma_excerpt = emma_text[:len(telemachus)]
    emma_stats = tokenize_and_profile(emma_excerpt, label="Emma (excerpt)")

    print(f"{'Metric':<30} {'Telemachus':>15} {'Emma (excerpt)':>15}")
    print("-" * 62)
    for key in ['total_tokens', 'total_alpha_tokens', 'total_types',
                'type_token_ratio', 'total_sentences', 'avg_sentence_length',
                'hapax_legomena', 'hapax_ratio']:
        tv = telem_stats[key]
        ev = emma_stats[key]
        fmt = '.4f' if isinstance(tv, float) else 'd'
        print(f"{key:<30} {tv:>15{fmt}} {ev:>15{fmt}}")

    return telem_stats, emma_stats


# ---------------------------------------------------------------------------
# Exercise 2: Concordance as Close Reading
# ---------------------------------------------------------------------------

def concordance_analysis(text, words=None):
    """Build concordance views for thematic keywords.

    Returns a dict mapping each keyword to its concordance lines.
    """
    if words is None:
        words = ['mother', 'sea', 'key', 'tower', 'God']

    tokens = word_tokenize(text)
    t = Text(tokens)

    results = {}
    for word in words:
        print(f"\n--- Concordance for '{word}' ---")
        lines = t.concordance_list(word, width=80, lines=25)
        results[word] = lines
        for line in lines:
            print(line.line)

    return results


# ---------------------------------------------------------------------------
# Exercise 3: Frequency and Stopwords
# ---------------------------------------------------------------------------

def frequency_analysis(text, top_n=50):
    """Generate frequency distributions with and without stopwords.

    Produces two plots: top-N words raw, and top-N words after stopword removal.
    Returns (raw_fdist, filtered_fdist).
    """
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]

    raw_fdist = FreqDist(alpha_tokens)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in alpha_tokens if t not in stop_words]
    filtered_fdist = FreqDist(filtered_tokens)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Raw frequency plot
    raw_top = raw_fdist.most_common(top_n)
    axes[0].bar([w for w, _ in raw_top], [c for _, c in raw_top], color='steelblue')
    axes[0].set_title(f'Top {top_n} Words (with stopwords)')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=60)

    # Filtered frequency plot
    filt_top = filtered_fdist.most_common(top_n)
    axes[1].bar([w for w, _ in filt_top], [c for _, c in filt_top], color='coral')
    axes[1].set_title(f'Top {top_n} Words (stopwords removed)')
    axes[1].set_ylabel('Frequency')
    axes[1].tick_params(axis='x', rotation=60)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week01_frequency.png'), dpi=150)
    plt.close()

    print("\nTop 20 content words (stopwords removed):")
    for word, count in filtered_fdist.most_common(20):
        print(f"  {word:<20} {count:>5}")

    return raw_fdist, filtered_fdist


def zipf_plot(text):
    """Plot rank-frequency on log-log scale to test Zipf's Law."""
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    fdist = FreqDist(alpha_tokens)

    ranks = range(1, len(fdist) + 1)
    freqs = [count for _, count in fdist.most_common()]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(list(ranks), freqs, 'b.', markersize=3, alpha=0.6)

    # Ideal Zipf line: f = C / r
    C = freqs[0]
    ideal = [C / r for r in ranks]
    ax.loglog(list(ranks), ideal, 'r--', alpha=0.5, label="Ideal Zipf (C/r)")

    ax.set_xlabel('Rank (log)')
    ax.set_ylabel('Frequency (log)')
    ax.set_title("Zipf's Law: Telemachus")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week01_zipf.png'), dpi=150)
    plt.close()
    print("Zipf plot saved to solutions/week01_zipf.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    text = load_episode('01telemachus.txt')

    print("=" * 62)
    print("EXERCISE 1: Tokenize and Profile")
    print("=" * 62)
    compare_profiles()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Concordance as Close Reading")
    print("=" * 62)
    concordance_analysis(text)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Frequency and Stopwords")
    print("=" * 62)
    frequency_analysis(text)

    print("\n" + "=" * 62)
    print("DIVING DEEPER: Zipf's Law")
    print("=" * 62)
    zipf_plot(text)
