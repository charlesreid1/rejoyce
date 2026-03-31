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

for resource in ['punkt', 'punkt_tab', 'vader_lexicon',
                 'sentiwordnet', 'wordnet', 'omw-1.4',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
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
        sentence_scores.append(scores['compound'])

    # Sliding window average
    trajectory = []
    for i in range(0, len(sentence_scores), window_size // 2):
        window = sentence_scores[i:i + window_size]
        if window:
            avg = sum(window) / len(window)
            trajectory.append((i + len(window) // 2, avg))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Per-sentence scores
    axes[0].plot(range(len(sentence_scores)), sentence_scores,
                 'b-', alpha=0.3, linewidth=0.5)
    axes[0].set_title(f'VADER Compound Score per Sentence: {label}')
    axes[0].set_ylabel('Compound Score')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Smoothed trajectory
    if trajectory:
        xs, ys = zip(*trajectory)
        axes[1].plot(xs, ys, 'r-', linewidth=2)
        axes[1].fill_between(xs, ys, alpha=0.2, color='red')
    axes[1].set_title(f'Sentiment Trajectory (window={window_size}): {label}')
    axes[1].set_xlabel('Sentence Index')
    axes[1].set_ylabel('Mean Compound Score')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week06_sentiment.png'), dpi=150)
    plt.close()

    # Summary statistics
    mean_score = sum(sentence_scores) / len(sentence_scores)
    variance = sum((s - mean_score)**2 for s in sentence_scores) / len(sentence_scores)
    pos_count = sum(1 for s in sentence_scores if s > 0.05)
    neg_count = sum(1 for s in sentence_scores if s < -0.05)
    neutral_count = len(sentence_scores) - pos_count - neg_count

    print(f"--- Sentiment Summary: {label} ---")
    print(f"  Total sentences:     {len(sentence_scores)}")
    print(f"  Mean compound:       {mean_score:.4f}")
    print(f"  Variance:            {variance:.4f}")
    print(f"  Positive sentences:  {pos_count} ({100*pos_count/len(sentence_scores):.1f}%)")
    print(f"  Negative sentences:  {neg_count} ({100*neg_count/len(sentence_scores):.1f}%)")
    print(f"  Neutral sentences:   {neutral_count} ({100*neutral_count/len(sentence_scores):.1f}%)")

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

    Interior: lines without em-dashes that contain short fragments,
    questions, or lack full sentence structure.
    Exterior: dialogue (em-dash prefixed) and descriptive narration.
    """
    lines = text.split('\n')
    dialogue = []
    narration = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('—') or stripped.startswith('--'):
            dialogue.append(stripped)
        else:
            narration.append(stripped)

    return ' '.join(dialogue), ' '.join(narration)


def compare_registers(text):
    """Compare VADER performance on dialogue vs. narration."""
    dialogue, narration = split_interior_exterior(text)
    sia = SentimentIntensityAnalyzer()

    def score_text(t, label):
        sents = sent_tokenize(t)
        scores = [sia.polarity_scores(s)['compound'] for s in sents]
        mean_s = sum(scores) / len(scores) if scores else 0
        var_s = sum((s - mean_s)**2 for s in scores) / len(scores) if scores else 0
        print(f"\n  {label}:")
        print(f"    Sentences: {len(sents)}")
        print(f"    Mean compound: {mean_s:.4f}")
        print(f"    Variance: {var_s:.4f}")
        return scores, mean_s, var_s

    print("--- Dialogue vs. Narration Sentiment ---")
    d_scores, d_mean, d_var = score_text(dialogue, "Dialogue")
    n_scores, n_mean, n_var = score_text(narration, "Narration/Interior")

    print(f"\n  Variance ratio (interior/dialogue): "
          f"{n_var/d_var:.2f}" if d_var > 0 else "  (dialogue variance is 0)")

    return d_scores, n_scores


# ---------------------------------------------------------------------------
# Exercise 3: Building a Death Lexicon
# ---------------------------------------------------------------------------

DEATH_WORDS = [
    'coffin', 'cemetery', 'grave', 'corpse', 'funeral', 'mourning',
    'burial', 'decay', 'death', 'dead', 'dying', 'hearse', 'tomb',
    'skeleton', 'ashes', 'widow', 'grief', 'loss', 'weep', 'sorrow',
]

PROXIMITY_WORDS = [
    'warm', 'quiet', 'home', 'garden', 'rest', 'sleep', 'peace',
    'gentle', 'soft', 'clean', 'white', 'green', 'light', 'bloom',
]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    return None


def death_lexicon_analysis():
    """Look up death-related and proximity words in SentiWordNet."""
    print("--- SentiWordNet: Death Words ---")
    print(f"{'Word':<15} {'Pos':>6} {'Neg':>6} {'Obj':>6} {'Synset'}")
    print("-" * 55)

    for word in DEATH_WORDS:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            ss = synsets[0]
            print(f"  {word:<13} {ss.pos_score():>6.3f} {ss.neg_score():>6.3f} "
                  f"{ss.obj_score():>6.3f} {ss.synset.name()}")

    print("\n--- SentiWordNet: Proximity Words (context-dependent valence) ---")
    print(f"{'Word':<15} {'Pos':>6} {'Neg':>6} {'Obj':>6} {'Synset'}")
    print("-" * 55)

    for word in PROXIMITY_WORDS:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            ss = synsets[0]
            print(f"  {word:<13} {ss.pos_score():>6.3f} {ss.neg_score():>6.3f} "
                  f"{ss.obj_score():>6.3f} {ss.synset.name()}")

    # Compute average sentiment for each word group
    death_neg = []
    for word in DEATH_WORDS:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            death_neg.append(synsets[0].neg_score())

    prox_pos = []
    for word in PROXIMITY_WORDS:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            prox_pos.append(synsets[0].pos_score())

    print(f"\n  Avg negativity of death words: {sum(death_neg)/len(death_neg):.3f}")
    print(f"  Avg positivity of proximity words: {sum(prox_pos)/len(prox_pos):.3f}")
    print(f"\n  Key insight: proximity words like 'rest', 'sleep', 'peace' score")
    print(f"  positively in SentiWordNet, but in a funeral context they carry")
    print(f"  the weight of death — context-free sentiment fails here.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    hades = load_episode('06hades.txt')

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
