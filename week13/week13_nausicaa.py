"""
Week 13: Nausicaa
==================
Stylometry and authorship attribution.

NLTK Focus: nltk.probability, function word analysis, sentence length
            distributions, vocabulary richness, Burrows' Delta

Exercises:
  1. The split test
  2. Burrows' Delta
  3. The cliché detector
"""

import os
import math
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords, gutenberg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for resource in ["punkt", "punkt_tab", "stopwords", "gutenberg"]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")

# Top 50 function words for stylometry
FUNCTION_WORDS = [
    "the",
    "and",
    "of",
    "to",
    "a",
    "in",
    "that",
    "is",
    "was",
    "it",
    "for",
    "as",
    "with",
    "his",
    "he",
    "on",
    "be",
    "at",
    "by",
    "not",
    "this",
    "but",
    "had",
    "are",
    "from",
    "or",
    "an",
    "they",
    "which",
    "her",
    "she",
    "were",
    "all",
    "their",
    "been",
    "have",
    "has",
    "would",
    "will",
    "what",
    "if",
    "so",
    "no",
    "when",
    "who",
    "him",
    "my",
    "than",
    "its",
    "could",
]


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_nausicaa(text):
    """Split Nausicaa into Gerty's half and Bloom's half.

    The split occurs roughly at the midpoint — after the fireworks climax.
    We use a textual marker: the shift from Gerty's perspective to Bloom's.
    """
    sentences = sent_tokenize(text)
    total = len(sentences)

    # Find the transition point using textual markers
    # Look for "Tight boots? No. She's lame!" followed by "Mr Bloom"
    transition_marker = "Tight boots? No. She's lame!"
    bloom_marker = "Mr Bloom"

    split_idx = total // 2  # default to midpoint

    # Search for the transition marker in the text
    full_text = " ".join(sentences)
    transition_pos = full_text.find(transition_marker)

    if transition_pos != -1:
        # Find the sentence containing the transition marker
        text_so_far = 0
        for i, sentence in enumerate(sentences):
            if text_so_far <= transition_pos < text_so_far + len(sentence):
                # Found the sentence with the transition marker
                # Now look for the next occurrence of "Mr Bloom" to confirm transition
                for j in range(i, min(i + 20, total)):  # Check next 20 sentences
                    if bloom_marker in sentences[j]:
                        split_idx = j
                        break
                break
            text_so_far += len(sentence) + 1  # +1 for space

    gerty_text = " ".join(sentences[:split_idx])
    bloom_text = " ".join(sentences[split_idx:])

    return gerty_text, bloom_text, split_idx


# ---------------------------------------------------------------------------
# Stylometric Profile
# ---------------------------------------------------------------------------


def stylometric_profile(text, label="Text"):
    """Compute a stylometric profile for a text."""
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    sentences = sent_tokenize(text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]

    fdist = FreqDist(alpha_tokens)

    # Function word frequencies
    total = len(alpha_tokens)
    fw_freqs = {w: fdist.get(w, 0) / total for w in FUNCTION_WORDS}

    # Sentence length stats
    mean_sl = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    median_sl = sorted(sent_lengths)[len(sent_lengths) // 2] if sent_lengths else 0
    std_sl = (
        (sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)) ** 0.5
        if sent_lengths
        else 0
    )

    # Vocabulary richness
    types = set(alpha_tokens)
    ttr = len(types) / total if total else 0
    hapax = sum(1 for w, c in fdist.items() if c == 1)
    hapax_ratio = hapax / len(types) if types else 0

    # Yule's K (measure of lexical diversity)
    # K = 10^4 * (sum(i^2 * Vi) - N) / N^2
    # where Vi is the number of words occurring i times, N is total tokens
    freq_counts = Counter(
        fdist.values()
    )  # Count how many words occur with each frequency
    yule_k_numerator = sum(i * i * freq_counts[i] for i in freq_counts) - total
    yule_k = 10000 * yule_k_numerator / (total * total) if total else 0

    # Punctuation
    exclamation = text.count("!") / len(sentences) if sentences else 0
    question = text.count("?") / len(sentences) if sentences else 0
    ellipsis = text.count("...") / len(sentences) if sentences else 0
    comma = text.count(",") / len(sentences) if sentences else 0
    # Include all dash types (em-dash, en-dash, and double-hyphen) for consistency
    em_dash = (
        (text.count("—") + text.count("–") + text.count("--")) / len(sentences)
        if sentences
        else 0
    )

    profile = {
        "label": label,
        "total_tokens": total,
        "total_types": len(types),
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "yule_k": yule_k,
        "mean_sent_len": mean_sl,
        "median_sent_len": median_sl,
        "std_sent_len": std_sl,
        "exclamation_per_sent": exclamation,
        "question_per_sent": question,
        "ellipsis_per_sent": ellipsis,
        "comma_per_sent": comma,
        "em_dash_per_sent": em_dash,
        "fw_freqs": fw_freqs,
    }
    return profile


def print_profile_comparison(profiles):
    """Print side-by-side comparison of stylometric profiles."""
    labels = [p["label"] for p in profiles]
    header = f"{'Metric':<25}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * (25 + 15 * len(labels)))

    keys = [
        "total_tokens",
        "total_types",
        "ttr",
        "hapax_ratio",
        "yule_k",
        "mean_sent_len",
        "median_sent_len",
        "std_sent_len",
        "exclamation_per_sent",
        "question_per_sent",
        "ellipsis_per_sent",
        "comma_per_sent",
        "em_dash_per_sent",
    ]

    for key in keys:
        row = f"  {key:<23}"
        for p in profiles:
            val = p[key]
            if isinstance(val, float):
                row += f"{val:>15.4f}"
            else:
                row += f"{val:>15}"
        print(row)


# ---------------------------------------------------------------------------
# Exercise 1: The Split Test
# ---------------------------------------------------------------------------


def split_test():
    """Compare stylometric profiles of Gerty and Bloom halves."""
    nausicaa = load_episode("13nausicaa.txt")
    gerty, bloom, split_idx = split_nausicaa(nausicaa)

    calypso = load_episode("04calypso.txt")
    lestry = load_episode("08lestrygonians.txt")

    profiles = [
        stylometric_profile(gerty, "Gerty"),
        stylometric_profile(bloom, "Bloom (Naus)"),
        stylometric_profile(calypso, "Calypso"),
        stylometric_profile(lestry, "Lestrygonians"),
    ]

    print(f"--- The Split Test ---")
    print(f"  Nausicaa split at sentence {split_idx}\n")
    print_profile_comparison(profiles)

    # Function word comparison for top-10 most variable
    print(f"\n--- Function Word Comparison (top differences) ---")
    gerty_fw = profiles[0]["fw_freqs"]
    bloom_fw = profiles[1]["fw_freqs"]
    diffs = [(w, abs(gerty_fw[w] - bloom_fw[w])) for w in FUNCTION_WORDS]
    diffs.sort(key=lambda x: -x[1])
    print(f"  {'Word':<12} {'Gerty':>10} {'Bloom':>10} {'Diff':>10}")
    for w, d in diffs[:15]:
        print(f"  {w:<12} {gerty_fw[w]:>10.5f} {bloom_fw[w]:>10.5f} {d:>10.5f}")

    # Visualize the profiles
    print(f"\n--- Visualization ---")
    visualize_profiles(profiles)

    return profiles


# ---------------------------------------------------------------------------
# Exercise 2: Burrows' Delta
# ---------------------------------------------------------------------------


def burrows_delta(test_profile, corpus_profiles):
    """Compute Burrows' Delta between a test text and reference corpus profiles.

    Delta = (1/n) * sum(|z_test_i - z_corpus_i|) for each function word i
    """
    # Compute corpus mean and std for each function word
    n = len(FUNCTION_WORDS)
    corpus_fw = {w: [] for w in FUNCTION_WORDS}
    for profile in corpus_profiles:
        for w in FUNCTION_WORDS:
            corpus_fw[w].append(profile["fw_freqs"].get(w, 0))

    means = {w: sum(vals) / len(vals) for w, vals in corpus_fw.items()}
    stds = {}
    for w, vals in corpus_fw.items():
        m = means[w]
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        stds[w] = max(var**0.5, 1e-10)

    # Z-scores for test text
    test_z = {
        w: (test_profile["fw_freqs"].get(w, 0) - means[w]) / stds[w]
        for w in FUNCTION_WORDS
    }

    # Delta against each corpus text
    deltas = []
    for cp in corpus_profiles:
        cp_z = {
            w: (cp["fw_freqs"].get(w, 0) - means[w]) / stds[w] for w in FUNCTION_WORDS
        }
        delta = sum(abs(test_z[w] - cp_z[w]) for w in FUNCTION_WORDS) / n
        deltas.append((cp["label"], delta))

    return sorted(deltas, key=lambda x: x[1])


def run_burrows_delta():
    """Compute Burrows' Delta for Gerty half against reference corpora."""
    nausicaa = load_episode("13nausicaa.txt")
    gerty, bloom, _ = split_nausicaa(nausicaa)

    # Build reference corpus
    bloom_episodes = [
        stylometric_profile(load_episode("04calypso.txt"), "Calypso"),
        stylometric_profile(load_episode("05lotuseaters.txt"), "Lotus Eaters"),
        stylometric_profile(load_episode("08lestrygonians.txt"), "Lestrygonians"),
        stylometric_profile(bloom, "Bloom (Nausicaa)"),
    ]
    stephen_episodes = [
        stylometric_profile(load_episode("01telemachus.txt"), "Telemachus"),
        stylometric_profile(load_episode("03proteus.txt"), "Proteus"),
        stylometric_profile(load_episode("09scyllacharybdis.txt"), "Scylla"),
    ]
    cyclops_barfly = stylometric_profile(load_episode("12cyclops.txt"), "Cyclops")

    all_corpus = bloom_episodes + stephen_episodes + [cyclops_barfly]
    gerty_profile = stylometric_profile(gerty, "Gerty")

    deltas = burrows_delta(gerty_profile, all_corpus)

    print("--- Burrows' Delta: Gerty Half vs. Reference Corpus ---")
    print(f"  {'Reference Text':<25} {'Delta':>10}")
    print("  " + "-" * 37)
    for label, delta in deltas:
        print(f"  {label:<25} {delta:>10.4f}")
    print(f"\n  Lower delta = more stylistically similar")
    print(f"  Note: Ideally would include Victorian sentimental fiction (e.g.,")
    print(f"        Maria Cummins' 'The Lamplighter') for better comparison")
    print(f"        to Gerty's romantic prose style.")

    return deltas


# ---------------------------------------------------------------------------
# Exercise 3: The Cliché Detector
# ---------------------------------------------------------------------------


def extract_ngrams(text, n_range=(3, 5)):
    """Extract n-grams from text."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    all_ngrams = Counter()
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            all_ngrams[ngram] += 1
    return all_ngrams


def cliche_detector():
    """Detect clichés by comparing n-grams to a reference corpus."""
    nausicaa = load_episode("13nausicaa.txt")
    gerty, bloom, _ = split_nausicaa(nausicaa)

    # Build reference n-grams from Gutenberg
    # Use texts more likely to contain sentimental/romantic clichés
    nltk.download("gutenberg", quiet=True)
    # Focus on Austen novels and other literature with romantic themes
    romantic_texts = [f for f in gutenberg.fileids() if "austen" in f or "stories" in f]
    if not romantic_texts:
        # Fallback to first few texts if no romantic texts found
        romantic_texts = gutenberg.fileids()[:5]

    ref_texts = [gutenberg.raw(f) for f in romantic_texts]
    ref_combined = " ".join(ref_texts)
    ref_ngrams = extract_ngrams(ref_combined)

    # High-frequency reference n-grams (cliché candidates)
    # Lower the threshold to catch more potential clichés
    common_ref = {ng for ng, c in ref_ngrams.items() if c >= 2}

    gerty_ngrams = extract_ngrams(gerty)
    bloom_ngrams = extract_ngrams(bloom)

    gerty_cliches = {
        ng: c for ng, c in gerty_ngrams.items() if ng in common_ref and c >= 1
    }
    bloom_cliches = {
        ng: c for ng, c in bloom_ngrams.items() if ng in common_ref and c >= 1
    }

    gerty_tokens = len([t for t in word_tokenize(gerty) if t.isalpha()])
    bloom_tokens = len([t for t in word_tokenize(bloom) if t.isalpha()])

    gerty_density = len(gerty_cliches) / gerty_tokens * 1000 if gerty_tokens else 0
    bloom_density = len(bloom_cliches) / bloom_tokens * 1000 if bloom_tokens else 0

    print("--- Cliché Detector ---")
    print(
        f"  Gerty: {len(gerty_cliches)} cliché n-grams, "
        f"density: {gerty_density:.2f} per 1000 tokens"
    )
    print(
        f"  Bloom: {len(bloom_cliches)} cliché n-grams, "
        f"density: {bloom_density:.2f} per 1000 tokens"
    )
    print(
        f"  Ratio (Gerty/Bloom): {gerty_density / bloom_density:.2f}x"
        if bloom_density
        else ""
    )

    print(f"\n--- Sample Gerty Clichés ---")
    sorted_gc = sorted(gerty_cliches.items(), key=lambda x: -x[1])
    for ng, c in sorted_gc[:15]:
        print(f"  {c:>3}  {' '.join(ng)}")

    print(f"\n--- Sample Bloom Clichés ---")
    sorted_bc = sorted(bloom_cliches.items(), key=lambda x: -x[1])
    for ng, c in sorted_bc[:15]:
        print(f"  {c:>3}  {' '.join(ng)}")

    return gerty_density, bloom_density


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_profiles(profiles):
    """Visualize stylometric profiles side by side."""
    # Extract labels and metrics
    labels = [p["label"] for p in profiles]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Stylometric Profiles Comparison", fontsize=16)

    # Plot 1: Sentence Length Metrics
    sentence_metrics = ["mean_sent_len", "median_sent_len"]
    x = range(len(labels))
    width = 0.35

    ax1 = axes[0, 0]
    mean_vals = [p["mean_sent_len"] for p in profiles]
    median_vals = [p["median_sent_len"] for p in profiles]
    ax1.bar(x, mean_vals, width, label="Mean", alpha=0.7)
    ax1.bar([i + width for i in x], median_vals, width, label="Median", alpha=0.7)
    ax1.set_xlabel("Texts")
    ax1.set_ylabel("Sentence Length")
    ax1.set_title("Sentence Length Comparison")
    ax1.set_xticks([i + width / 2 for i in x])
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Vocabulary Richness
    ax2 = axes[0, 1]
    ttr_vals = [p["ttr"] for p in profiles]
    hapax_vals = [p["hapax_ratio"] for p in profiles]
    yule_vals = [p["yule_k"] for p in profiles]

    # Normalize Yule's K for better visualization (scale down by 100)
    yule_vals_scaled = [y / 100 for y in yule_vals]

    ax2.plot(labels, ttr_vals, "o-", label="TTR", linewidth=2, markersize=8)
    ax2.plot(labels, hapax_vals, "s-", label="Hapax Ratio", linewidth=2, markersize=8)
    ax2.plot(
        labels,
        yule_vals_scaled,
        "^-",
        label="Yule's K (scaled)",
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel("Texts")
    ax2.set_ylabel("Vocabulary Richness Metrics")
    ax2.set_title("Vocabulary Richness Comparison")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)

    # Plot 3: Punctuation Features
    ax3 = axes[1, 0]
    excl_vals = [p["exclamation_per_sent"] for p in profiles]
    quest_vals = [p["question_per_sent"] for p in profiles]
    ellipsis_vals = [p["ellipsis_per_sent"] for p in profiles]

    ax3.bar(x, excl_vals, width, label="Exclamation", alpha=0.7)
    ax3.bar([i + width for i in x], quest_vals, width, label="Question", alpha=0.7)
    ax3.set_xlabel("Texts")
    ax3.set_ylabel("Punctuation per Sentence")
    ax3.set_title("Punctuation Comparison")
    ax3.set_xticks([i + width / 2 for i in x])
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.legend()

    # Plot 4: Function Word Differences (Top 5 most different between Gerty and Bloom)
    ax4 = axes[1, 1]
    if len(profiles) >= 2:
        gerty_fw = profiles[0]["fw_freqs"]
        bloom_fw = profiles[1]["fw_freqs"]
        diffs = [(w, abs(gerty_fw[w] - bloom_fw[w])) for w in FUNCTION_WORDS]
        diffs.sort(key=lambda x: -x[1])
        top_diffs = diffs[:5]

        words = [d[0] for d in top_diffs]
        gerty_vals = [gerty_fw[w] for w in words]
        bloom_vals = [bloom_fw[w] for w in words]

        x_words = range(len(words))
        ax4.bar(x_words, gerty_vals, width, label="Gerty", alpha=0.7)
        ax4.bar(
            [i + width for i in x_words], bloom_vals, width, label="Bloom", alpha=0.7
        )
        ax4.set_xlabel("Function Words")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Top Function Word Differences")
        ax4.set_xticks([i + width / 2 for i in x_words])
        ax4.set_xticklabels(words, rotation=45, ha="right")
        ax4.legend()
    else:
        ax4.text(
            0.5,
            0.5,
            "Need at least 2 profiles",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "stylometric_profiles.png"), dpi=300, bbox_inches="tight")
    print("Visualization saved as 'week13/stylometric_profiles.png'")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("EXERCISE 1: The Split Test")
    print("=" * 62)
    split_test()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Burrows' Delta")
    print("=" * 62)
    run_burrows_delta()

    print("\n" + "=" * 62)
    print("EXERCISE 3: The Cliché Detector")
    print("=" * 62)
    cliche_detector()
