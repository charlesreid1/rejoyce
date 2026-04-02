"""
Week 06: Hades
===============
Sentiment analysis and affective lexicons.

NLTK Focus: nltk.sentiment, VADER, SentiWordNet, opinion lexicons

Exercises:
  1. Sentiment trajectory
  2. Bloom's stoicism vs. the narrator's register
  3. Building a death lexicon
"""

import os
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import matplotlib.pyplot as plt
import numpy as np

for resource in [
    "punkt",
    "punkt_tab",
    "vader_lexicon",
    "sentiwordnet",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Sentiment Trajectory
# ---------------------------------------------------------------------------


def sentiment_trajectory(text, window_size=50, label="Hades"):
    """Compute VADER sentiment scores in sliding sentence windows.

    Returns list of (window_midpoint, compound_score) tuples.
    """
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)

    # Score each sentence
    sentence_scores = []
    for sent in sentences:
        scores = sia.polarity_scores(sent)
        sentence_scores.append(scores["compound"])

    # Sliding window average
    trajectory = []
    for i in range(0, len(sentence_scores), window_size // 2):
        window = sentence_scores[i : i + window_size]
        if window:
            avg = sum(window) / len(window)
            trajectory.append((i + len(window) // 2, avg))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Per-sentence scores
    axes[0].plot(
        range(len(sentence_scores)), sentence_scores, "b-", alpha=0.3, linewidth=0.5
    )
    axes[0].set_title(f"VADER Compound Score per Sentence: {label}")
    axes[0].set_ylabel("Compound Score")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Add annotations for key narrative events
    # Approximate sentence indices for key events:
    # 1. Passing Child's murder house (~sentence 300)
    # 2. Arriving at cemetery (~sentence 900)
    # 3. Burial scene (~sentence 1200)
    # 4. Rudy memory (~sentence 1350)
    event_annotations = [
        (300, "Passing Child's\nmurder house"),
        (900, "Arriving at\ncemetery"),
        (1200, "Burial"),
        (1350, "Rudy memory"),
    ]

    for idx, label_text in event_annotations:
        if idx < len(sentence_scores):
            axes[0].axvline(x=idx, color="orange", linestyle=":", alpha=0.7)
            axes[0].text(
                idx,
                0.8,
                label_text,
                rotation=90,
                verticalalignment="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

    # Smoothed trajectory
    if trajectory:
        xs, ys = zip(*trajectory)
        axes[1].plot(xs, ys, "r-", linewidth=2)
        axes[1].fill_between(xs, ys, alpha=0.2, color="red")
    axes[1].set_title(f"Sentiment Trajectory (window={window_size}): {label}")
    axes[1].set_xlabel("Sentence Index")
    axes[1].set_ylabel("Mean Compound Score")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Add annotations to the trajectory plot as well
    for idx, label_text in event_annotations:
        if idx < len(sentence_scores):
            axes[1].axvline(x=idx, color="orange", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week06_sentiment.png"), dpi=150
    )
    plt.close()

    # Summary statistics
    mean_score = sum(sentence_scores) / len(sentence_scores)
    variance = sum((s - mean_score) ** 2 for s in sentence_scores) / len(
        sentence_scores
    )
    pos_count = sum(1 for s in sentence_scores if s > 0.05)
    neg_count = sum(1 for s in sentence_scores if s < -0.05)
    neutral_count = len(sentence_scores) - pos_count - neg_count

    print(f"--- Sentiment Summary: {label} ---")
    print(f"  Total sentences:     {len(sentence_scores)}")
    print(f"  Mean compound:       {mean_score:.4f}")
    print(f"  Variance:            {variance:.4f}")
    print(
        f"  Positive sentences:  {pos_count} ({100 * pos_count / len(sentence_scores):.1f}%)"
    )
    print(
        f"  Negative sentences:  {neg_count} ({100 * neg_count / len(sentence_scores):.1f}%)"
    )
    print(
        f"  Neutral sentences:   {neutral_count} ({100 * neutral_count / len(sentence_scores):.1f}%)"
    )

    # Find mismatches — sentences VADER scores very positively in a funeral chapter
    print(f"\n--- Likely VADER Misfires (positive scores in funeral context) ---")
    scored = list(zip(sentences, sentence_scores))
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    for sent, score in scored_sorted[:5]:
        print(f"  [{score:+.3f}] {sent[:100]}...")

    # Most negative
    print(f"\n--- Most Negative Sentences ---")
    scored_sorted_neg = sorted(scored, key=lambda x: x[1])
    for sent, score in scored_sorted_neg[:5]:
        print(f"  [{score:+.3f}] {sent[:100]}...")

    return sentence_scores, mean_score, variance


# ---------------------------------------------------------------------------
# Exercise 2: Bloom's Stoicism vs. Narrator's Register
# ---------------------------------------------------------------------------


def split_interior_exterior(text):
    """Heuristic split of interior monologue from external narration.

    Interior: Bloom's internal thoughts - lines without em-dashes that
    show characteristics of free indirect discourse: sentence fragments,
    questions without question marks, practical/domestic vocabulary.
    Exterior: dialogue (em-dash prefixed) and descriptive third-person narration.
    """
    lines = text.split("\n")
    dialogue = []
    interior = []
    exterior_narration = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Dialogue lines start with em-dashes
        if stripped.startswith("—") or stripped.startswith("--"):
            dialogue.append(stripped)
        # Look for Bloom's internal thoughts
        elif is_bloom_interior(stripped):
            interior.append(stripped)
        # Otherwise it's external narration
        else:
            exterior_narration.append(stripped)

    return " ".join(dialogue), " ".join(interior), " ".join(exterior_narration)


def is_bloom_interior(line):
    """Heuristic to identify Bloom's interior thoughts.

    Looks for characteristics of free indirect discourse:
    - Short sentence fragments
    - Questions without question marks
    - Practical/domestic vocabulary
    - Lack of full grammatical structure
    """
    # Very short lines are often interior thoughts
    if len(line) < 30:
        return True

    # Lines with incomplete sentences or fragments
    fragment_indicators = [".", ",", ";"]
    if line[-1] in fragment_indicators and not line.endswith((".", "?", "!")):
        return True

    # Practical/domestic vocabulary often indicates interior thoughts
    domestic_words = ["soap", "pocket", "bed", "hair", "nails", "envelope", "unclean"]
    if any(word in line.lower() for word in domestic_words):
        return True

    # Questions without question marks (internal questioning)
    if "?" not in line and (
        line.startswith(("Why", "How", "What", "When", "Where", "Who"))
        or "wonder" in line.lower()
        or "suppose" in line.lower()
    ):
        return True

    return False


def compare_registers(text):
    """Compare VADER performance on dialogue vs. interior vs. exterior narration."""
    dialogue, interior, exterior = split_interior_exterior(text)
    sia = SentimentIntensityAnalyzer()

    def score_text(t, label):
        sents = sent_tokenize(t)
        scores = [sia.polarity_scores(s)["compound"] for s in sents]
        mean_s = sum(scores) / len(scores) if scores else 0
        var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores) if scores else 0
        print(f"\n  {label}:")
        print(f"    Sentences: {len(sents)}")
        print(f"    Mean compound: {mean_s:.4f}")
        print(f"    Variance: {var_s:.4f}")
        return scores, mean_s, var_s

    print("--- Dialogue vs. Interior vs. Exterior Sentiment ---")
    d_scores, d_mean, d_var = score_text(dialogue, "Dialogue")
    i_scores, i_mean, i_var = score_text(interior, "Bloom's Interior Thoughts")
    e_scores, e_mean, e_var = score_text(exterior, "External Narration")

    if d_var > 0:
        print(f"\n  Variance ratio (interior/dialogue): {i_var / d_var:.2f}")
        print(f"  Variance ratio (exterior/dialogue): {e_var / d_var:.2f}")
    else:
        print("  (dialogue variance is 0)")

    return d_scores, i_scores, e_scores


# ---------------------------------------------------------------------------
# Exercise 3: Building a Death Lexicon
# ---------------------------------------------------------------------------

DEATH_WORDS = [
    "coffin",
    "cemetery",
    "grave",
    "corpse",
    "funeral",
    "mourning",
    "burial",
    "decay",
    "death",
    "dead",
    "dying",
    "hearse",
    "tomb",
    "skeleton",
    "ashes",
    "widow",
    "grief",
    "loss",
    "weep",
    "sorrow",
]

PROXIMITY_WORDS = [
    "warm",
    "quiet",
    "home",
    "garden",
    "rest",
    "sleep",
    "peace",
    "gentle",
    "soft",
    "clean",
    "white",
    "green",
    "light",
    "bloom",
]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return "a"  # adjective
    elif treebank_tag.startswith("V"):
        return "v"  # verb
    elif treebank_tag.startswith("N"):
        return "n"  # noun
    elif treebank_tag.startswith("R"):
        return "r"  # adverb
    return None


def death_lexicon_analysis():
    """Look up death-related and proximity words in SentiWordNet."""
    print("--- SentiWordNet: Death Words ---")
    print(f"{'Word':<15} {'Pos':>6} {'Neg':>6} {'Obj':>6} {'Best Synset'}")
    print("-" * 55)

    for word in DEATH_WORDS:
        # Try to get the most relevant synset using POS tagging
        tagged = pos_tag([word])
        wordnet_pos = get_wordnet_pos(tagged[0][1])

        synsets = list(swn.senti_synsets(word))
        if synsets:
            # If we have POS info, filter synsets by POS
            if wordnet_pos:
                pos_filtered = [
                    ss for ss in synsets if str(ss.synset.pos()) == wordnet_pos
                ]
                if pos_filtered:
                    synsets = pos_filtered

            # Show the best synset (highest sentiment score)
            best_ss = max(synsets, key=lambda ss: ss.pos_score() + ss.neg_score())
            print(
                f"  {word:<13} {best_ss.pos_score():>6.3f} {best_ss.neg_score():>6.3f} "
                f"{best_ss.obj_score():>6.3f} {best_ss.synset.name()}"
            )

    print("\n--- SentiWordNet: Proximity Words (context-dependent valence) ---")
    print(f"{'Word':<15} {'Pos':>6} {'Neg':>6} {'Obj':>6} {'Best Synset'}")
    print("-" * 55)

    for word in PROXIMITY_WORDS:
        # Try to get the most relevant synset using POS tagging
        tagged = pos_tag([word])
        wordnet_pos = get_wordnet_pos(tagged[0][1])

        synsets = list(swn.senti_synsets(word))
        if synsets:
            # If we have POS info, filter synsets by POS
            if wordnet_pos:
                pos_filtered = [
                    ss for ss in synsets if str(ss.synset.pos()) == wordnet_pos
                ]
                if pos_filtered:
                    synsets = pos_filtered

            # Show the best synset (highest sentiment score)
            best_ss = max(synsets, key=lambda ss: ss.pos_score() + ss.neg_score())
            print(
                f"  {word:<13} {best_ss.pos_score():>6.3f} {best_ss.neg_score():>6.3f} "
                f"{best_ss.obj_score():>6.3f} {best_ss.synset.name()}"
            )

    # Compute average sentiment for each word group using all synsets
    death_neg = []
    death_pos = []
    for word in DEATH_WORDS:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            # Average across all synsets
            avg_neg = sum(ss.neg_score() for ss in synsets) / len(synsets)
            avg_pos = sum(ss.pos_score() for ss in synsets) / len(synsets)
            death_neg.append(avg_neg)
            death_pos.append(avg_pos)

    prox_pos = []
    prox_neg = []
    for word in PROXIMITY_WORDS:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            # Average across all synsets
            avg_pos = sum(ss.pos_score() for ss in synsets) / len(synsets)
            avg_neg = sum(ss.neg_score() for ss in synsets) / len(synsets)
            prox_pos.append(avg_pos)
            prox_neg.append(avg_neg)

    print(f"\n  Avg negativity of death words: {sum(death_neg) / len(death_neg):.3f}")
    print(f"  Avg positivity of proximity words: {sum(prox_pos) / len(prox_pos):.3f}")

    # Calculate how many actually have positive scores
    positive_proximity_count = sum(1 for score in prox_pos if score > 0)
    print(
        f"  Count of proximity words with positive scores: {positive_proximity_count}/{len(PROXIMITY_WORDS)}"
    )

    # Updated key insight based on actual data
    print(
        f"\n  Key insight: Only {positive_proximity_count} of {len(PROXIMITY_WORDS)} proximity words"
    )
    print(f"  score positively in SentiWordNet. Words like 'rest', 'sleep', 'home'")
    if positive_proximity_count < len(PROXIMITY_WORDS) / 2:
        print(f"  are mostly objective (0.000), showing that context-free sentiment")
        print(f"  analysis misses the contextual meaning in a funeral setting.")
    else:
        print(f"  have some positive sentiment, but context-free sentiment analysis")
        print(f"  still fails to capture their meaning in a funeral context.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hades = load_episode("06hades.txt")

    print("=" * 62)
    print("EXERCISE 1: Sentiment Trajectory")
    print("=" * 62)
    sentiment_trajectory(hades)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Bloom's Stoicism vs. Narrator")
    print("=" * 62)
    compare_registers(hades)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Death Lexicon")
    print("=" * 62)
    death_lexicon_analysis()
