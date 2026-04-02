"""
Week 18: Penelope
==================
Unsupervised text segmentation, topic modeling, and the discovery
of latent structure.

NLTK Focus: nltk.tokenize.texttiling, topic modeling concepts,
            the return to tokenization

Exercises:
  1. Segment the unsegmentable
  2. Topic modeling Molly's mind
  3. The return to tokenization
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import TextTilingTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math

for resource in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")
STOP_WORDS = set(stopwords.words("english"))


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Segment the Unsegmentable
# ---------------------------------------------------------------------------


def prepare_for_texttiling(text):
    """Prepare Penelope's unpunctuated text for TextTiling.

    TextTiling expects paragraph-structured text. Since Penelope has
    minimal structure, we create artificial paragraph breaks every ~1000 words.
    """
    tokens = word_tokenize(text)
    paragraphs = []
    chunk_size = 1000  # Increased from 200 to 1000 to avoid artificial boundaries

    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i : i + chunk_size])
        paragraphs.append(chunk)

    return "\n\n".join(paragraphs)


def segment_penelope():
    """Apply TextTiling and compare to Joyce's 8 sentences."""
    penelope = load_episode("18penelope.txt")

    # Joyce's 8 sentences: Try to detect actual structural markers first
    total_words = len(word_tokenize(penelope))
    words = word_tokenize(penelope)

    # Look for potential sentence breaks in the original text
    # Penelope has very few actual periods, but we can look for paragraph-like breaks
    # or other structural markers if available
    joyce_boundaries = []

    # Try to find actual structural markers (paragraph breaks or rare punctuation)
    # If none found, fall back to equal division
    periods = [i for i, word in enumerate(words) if "." in word]
    if len(periods) >= 2:
        # Use actual periods if we have at least 2
        step = len(periods) // 8
        if step > 0:
            for i in range(1, min(8, len(periods))):
                joyce_boundaries.append(periods[i * step])
        else:
            # Evenly distribute if not enough periods
            eighth = total_words // 8
            for i in range(1, 8):
                joyce_boundaries.append(i * eighth)
    else:
        # Fall back to equal division approximation
        eighth = total_words // 8
        for i in range(1, 8):
            joyce_boundaries.append(i * eighth)

    # TextTiling segmentation
    prepared = prepare_for_texttiling(penelope)
    try:
        # Tune TextTilingTokenizer parameters to better detect genuine topic shifts
        tiler = TextTilingTokenizer(
            w=20, k=10
        )  # Increased window size and smoothing block size
        segments = tiler.tokenize(prepared)
        tt_boundaries = []
        pos = 0
        for seg in segments[:-1]:
            pos += len(word_tokenize(seg))
            tt_boundaries.append(pos)
    except Exception as e:
        print(f"  TextTiling failed: {e}")
        # Fallback: use vocabulary shift detection
        segments = vocabulary_shift_segmentation(penelope)
        tt_boundaries = []
        pos = 0
        for seg in segments[:-1]:
            pos += len(word_tokenize(seg))
            tt_boundaries.append(pos)

    print(f"--- Segmentation Results ---")
    print(f"  Total words: {total_words}")
    print(f"  Joyce's 8 sentences: boundaries at word positions {joyce_boundaries}")
    print(f"  TextTiling segments: {len(segments) if segments else 0}")
    print(f"  TextTiling boundaries: {tt_boundaries[:20]}")

    # Visualize both segmentations
    fig, axes = plt.subplots(2, 1, figsize=(14, 4))

    # Joyce's divisions
    for b in joyce_boundaries:
        axes[0].axvline(x=b, color="blue", alpha=0.7, linewidth=2)
    axes[0].set_xlim(0, total_words)
    axes[0].set_title("Joyce's 8 Sentence Boundaries")
    axes[0].set_yticks([])

    # TextTiling divisions
    for b in tt_boundaries:
        axes[1].axvline(x=b, color="red", alpha=0.7, linewidth=1)
    axes[1].set_xlim(0, total_words)
    axes[1].set_title(f"TextTiling Boundaries ({len(tt_boundaries)} segments)")
    axes[1].set_xlabel("Word Position")
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week18_segmentation.png"), dpi=150
    )
    plt.close()
    print("  Segmentation plot saved to week18/week18_segmentation.png")

    return segments, joyce_boundaries, tt_boundaries


def vocabulary_shift_segmentation(text, window=200, threshold=0.3):
    """Simple vocabulary-shift segmenter as fallback.

    Detects points where the vocabulary changes significantly between
    adjacent windows.
    """
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    segments = []
    segment_start = 0

    for i in range(window, len(words) - window, window // 2):
        left = set(words[i - window : i])
        right = set(words[i : i + window])
        # Jaccard distance
        intersection = len(left & right)
        union = len(left | right)
        similarity = intersection / union if union else 1
        if similarity < threshold:
            segments.append(" ".join(words[segment_start:i]))
            segment_start = i

    segments.append(" ".join(words[segment_start:]))
    return segments


# ---------------------------------------------------------------------------
# Exercise 2: Topic Modeling Molly's Mind
# ---------------------------------------------------------------------------


def simple_topic_model(text, num_topics=6, window_size=200):
    """Simple keyword-based topic detection (no gensim dependency).

    Uses seed word lists for known Penelope themes and computes
    topic proportions per window.
    """
    all_topic_seeds = {
        "Gibraltar/girlhood": {
            "gibraltar",
            "mulvey",
            "girl",
            "garden",
            "flower",
            "mountain",
            "spanish",
            "moor",
            "sun",
            "rock",
        },
        "Bloom/marriage": {
            "bloom",
            "leopold",
            "poldy",
            "husband",
            "marry",
            "howth",
            "proposal",
            "eccles",
            "house",
            "home",
        },
        "Boylan/desire": {
            "boylan",
            "blazes",
            "afternoon",
            "bed",
            "kiss",
            "love",
            "want",
            "body",
            "man",
            "handsome",
        },
        "Body/physicality": {
            "body",
            "breast",
            "blood",
            "skin",
            "hair",
            "dress",
            "clothes",
            "bath",
            "perfume",
            "beauty",
        },
        "Other women/judgment": {
            "woman",
            "women",
            "mrs",
            "jealous",
            "pretty",
            "hat",
            "fashion",
            "better",
            "worse",
            "dress",
        },
        "Memory/reflection": {
            "remember",
            "time",
            "years",
            "ago",
            "first",
            "always",
            "never",
            "used",
            "once",
            "old",
        },
    }

    # Select topics based on requested number
    topic_keys = list(all_topic_seeds.keys())
    if num_topics <= len(topic_keys):
        selected_keys = topic_keys[:num_topics]
    else:
        # If requesting more topics than available, repeat the existing ones
        selected_keys = topic_keys + topic_keys[: num_topics - len(topic_keys)]

    topic_seeds = {key: all_topic_seeds[key] for key in selected_keys}

    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and len(w) > 2]
    num_windows = len(words) // window_size

    topic_trajectory = {topic: [] for topic in topic_seeds}

    for i in range(num_windows):
        window = set(words[i * window_size : (i + 1) * window_size])
        for topic, seeds in topic_seeds.items():
            overlap = len(window & seeds)
            topic_trajectory[topic].append(overlap)

    # Normalize per window to ensure proportions sum to 1.0 per window
    for i in range(num_windows):
        total = sum(topic_trajectory[topic][i] for topic in topic_trajectory)
        if total > 0:
            for topic in topic_trajectory:
                topic_trajectory[topic][i] = topic_trajectory[topic][i] / total
        else:
            # If no seed words found in window, distribute evenly
            for topic in topic_trajectory:
                topic_trajectory[topic][i] = 1.0 / len(topic_trajectory)

    # Verify normalization (debug info)
    # for i in range(min(5, num_windows)):  # Check first 5 windows
    #     window_total = sum(topic_trajectory[topic][i] for topic in topic_trajectory)
    #     print(f"  Window {i} total: {window_total:.6f}")

    print(
        f"--- Topic Model (keyword-based, {num_topics} topics, {num_windows} windows) ---"
    )
    for topic, vals in topic_trajectory.items():
        mean_prop = sum(vals) / len(vals) if vals else 0
        peak_window = vals.index(max(vals)) if vals else 0
        print(f"  {topic:<25} mean: {mean_prop:.3f}  peak at window {peak_window}")

    # Stacked area chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(num_windows)
    bottom = np.zeros(num_windows)
    colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]

    for i, (topic, vals) in enumerate(topic_trajectory.items()):
        vals_arr = np.array(vals)
        ax.fill_between(
            x,
            bottom,
            bottom + vals_arr,
            label=topic,
            alpha=0.7,
            color=colors[i % len(colors)],
        )
        bottom += vals_arr

    ax.set_xlabel("Window Position")
    ax.set_ylabel("Topic Proportion")
    ax.set_title("Molly's Mind: Topic Distribution Across Penelope")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "week18_topics.png"), dpi=150)
    plt.close()
    print("  Topic distribution plot saved to week18/week18_topics.png")

    return topic_trajectory


# ---------------------------------------------------------------------------
# Exercise 3: The Return to Tokenization
# ---------------------------------------------------------------------------


def return_to_tokenization():
    """Compare Penelope to Telemachus using Week 1 metrics."""
    penelope = load_episode("18penelope.txt")
    telemachus = load_episode("01telemachus.txt")

    def profile(text, label):
        tokens = word_tokenize(text)
        alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
        fdist = FreqDist(alpha_tokens)
        content_tokens = [t for t in alpha_tokens if t not in STOP_WORDS]
        content_fdist = FreqDist(content_tokens)
        types = set(alpha_tokens)

        # "Sentence" segmentation — for Penelope, use approximate methods
        sentences = sent_tokenize(text)
        if len(sentences) < 10:
            # Penelope has almost no sentence boundaries; use TextTiling segments as proxy
            # First try to get segments from TextTiling
            try:
                prepared = prepare_for_texttiling(text)
                tiler = TextTilingTokenizer(w=20, k=10)
                segments = tiler.tokenize(prepared)
                sentences = segments
            except:
                # Fallback to chunking if TextTiling fails
                words = word_tokenize(text)
                chunk_size = 50  # Treat every 50 words as a "sentence"
                sentences = [
                    " ".join(words[i : i + chunk_size])
                    for i in range(0, len(words), chunk_size)
                ]

        return {
            "label": label,
            "total_tokens": len(tokens),
            "total_alpha": len(alpha_tokens),
            "total_types": len(types),
            "ttr": len(types) / len(alpha_tokens) if alpha_tokens else 0,
            "hapax": sum(1 for w, c in fdist.items() if c == 1),
            "hapax_ratio": sum(1 for w, c in fdist.items() if c == 1) / len(types)
            if types
            else 0,
            "approx_sentences": len(sentences),
            "avg_sent_len": len(tokens) / len(sentences) if sentences else 0,
            "top_20_content": [w for w, _ in content_fdist.most_common(20)],
        }

    pen_p = profile(penelope, "Penelope")
    tel_p = profile(telemachus, "Telemachus")

    print(f"\n--- Parallel Profile: Telemachus vs. Penelope ---")
    print(f"{'Metric':<25} {'Telemachus':>15} {'Penelope':>15}")
    print("-" * 57)
    for key in [
        "total_tokens",
        "total_alpha",
        "total_types",
        "ttr",
        "hapax",
        "hapax_ratio",
        "approx_sentences",
        "avg_sent_len",
    ]:
        tv = tel_p[key]
        pv = pen_p[key]
        fmt = ".4f" if isinstance(tv, float) else "d"
        print(f"  {key:<23} {tv:>15{fmt}} {pv:>15{fmt}}")

    print(f"\n--- Top 20 Content Words ---")
    print(f"  {'Telemachus':<25} {'Penelope':<25}")
    print("  " + "-" * 50)
    for i in range(20):
        tw = tel_p["top_20_content"][i] if i < len(tel_p["top_20_content"]) else ""
        pw = pen_p["top_20_content"][i] if i < len(pen_p["top_20_content"]) else ""
        print(f"  {tw:<25} {pw:<25}")

    # Track 'yes' distribution
    words = word_tokenize(penelope.lower())
    yes_positions = [i for i, w in enumerate(words) if w.strip(".,;!?") == "yes"]
    no_positions = [i for i, w in enumerate(words) if w.strip(".,;!?") == "no"]
    and_positions = [i for i, w in enumerate(words) if w.strip(".,;!?") == "and"]

    print(f"\n--- Structural Particles ---")
    print(f"  'yes' occurrences: {len(yes_positions)}")
    print(f"  'no' occurrences:  {len(no_positions)}")
    print(f"  'and' occurrences: {len(and_positions)}")

    # Plot 'yes', 'no', and 'and' distribution
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(
        yes_positions,
        [2] * len(yes_positions),
        s=15,
        alpha=0.7,
        label=f"yes ({len(yes_positions)})",
        color="green",
    )
    ax.scatter(
        no_positions,
        [1] * len(no_positions),
        s=15,
        alpha=0.7,
        label=f"no ({len(no_positions)})",
        color="red",
    )
    ax.scatter(
        and_positions,
        [0] * len(and_positions),
        s=15,
        alpha=0.7,
        label=f"and ({len(and_positions)})",
        color="blue",
    )
    ax.set_xlim(0, len(words))
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["and", "no", "yes"])
    ax.set_xlabel("Word Position")
    ax.set_title("Distribution of 'yes', 'no', and 'and' in Penelope")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week18_structural_particles.png"),
        dpi=150,
    )
    plt.close()
    print(
        "  yes/no/and distribution plot saved to week18/week18_structural_particles.png"
    )

    return pen_p, tel_p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    penelope = load_episode("18penelope.txt")

    print("=" * 62)
    print("EXERCISE 1: Segment the Unsegmentable")
    print("=" * 62)
    segment_penelope()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Topic Modeling Molly's Mind")
    print("=" * 62)
    # Experiment with different numbers of topics (k=4, 6, 8, 10)
    for k in [4, 6, 8, 10]:
        print(f"\n--- Topic Model with {k} topics ---")
        simple_topic_model(penelope, num_topics=k)

    print("\n" + "=" * 62)
    print("EXERCISE 3: The Return to Tokenization")
    print("=" * 62)
    return_to_tokenization()
