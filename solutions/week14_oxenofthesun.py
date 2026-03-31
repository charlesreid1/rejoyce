"""
Week 14: Oxen of the Sun
==========================
Diachronic corpus analysis, historical style classification,
and comparative corpus profiling.

NLTK Focus: nltk.corpus.gutenberg, period-specific feature extraction,
            classification across time periods

Exercises:
  1. Period profiling
  2. The style dating game
  3. The arc of English
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import gutenberg
from nltk.classify import NaiveBayesClassifier, accuracy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

for resource in ['punkt', 'punkt_tab', 'gutenberg',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Oxen Section Segmentation
# ---------------------------------------------------------------------------

def segment_oxen(text, num_sections=9):
    """Divide Oxen of the Sun into roughly equal sections representing
    the chronological style periods.

    Since precise section boundaries require literary annotation,
    we divide into N roughly equal parts as an approximation.
    """
    sentences = sent_tokenize(text)
    section_size = len(sentences) // num_sections
    sections = []
    period_labels = [
        "Anglo-Saxon/Medieval",
        "Malory/Middle English",
        "Elizabethan",
        "King James/Bunyan",
        "Addison/18th Century",
        "Sterne/Romantic",
        "Gibbon/Augustan",
        "Victorian/Dickens",
        "Modern/Slang",
    ]

    for i in range(num_sections):
        start = i * section_size
        end = start + section_size if i < num_sections - 1 else len(sentences)
        section_text = ' '.join(sentences[start:end])
        label = period_labels[i] if i < len(period_labels) else f"Section {i+1}"
        sections.append((label, section_text))

    return sections


# ---------------------------------------------------------------------------
# Feature Extraction for Historical Profiling
# ---------------------------------------------------------------------------

def period_features(text):
    """Extract features diagnostic of historical period."""
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    sentences = sent_tokenize(text)

    if not alpha_tokens or not sentences:
        return {}

    tagged = pos_tag(tokens)
    tag_counts = Counter(tag for _, tag in tagged)
    total_tags = sum(tag_counts.values())

    features = {}

    # Average sentence length
    sent_lens = [len(word_tokenize(s)) for s in sentences]
    features['avg_sent_len'] = sum(sent_lens) / len(sent_lens)

    # Type-token ratio
    types = set(alpha_tokens)
    features['ttr'] = len(types) / len(alpha_tokens)

    # Average word length (proxy for Latinate vocabulary)
    features['avg_word_len'] = sum(len(w) for w in alpha_tokens) / len(alpha_tokens)

    # Long word proportion (>= 8 chars, more Latinate)
    features['long_word_prop'] = sum(1 for w in alpha_tokens if len(w) >= 8) / len(alpha_tokens)

    # Function word proportion
    func_words = {'the', 'a', 'an', 'of', 'to', 'in', 'and', 'is', 'was',
                  'for', 'that', 'with', 'it', 'as', 'be', 'on', 'at', 'by'}
    features['func_word_prop'] = sum(1 for t in alpha_tokens if t in func_words) / len(alpha_tokens)

    # POS-based features
    nouns = sum(c for t, c in tag_counts.items() if t.startswith('NN'))
    verbs = sum(c for t, c in tag_counts.items() if t.startswith('VB'))
    adjs = sum(c for t, c in tag_counts.items() if t.startswith('JJ'))
    features['adj_density'] = adjs / total_tags if total_tags else 0
    features['noun_verb_ratio'] = nouns / verbs if verbs else 0

    # Comma density (periodic sentence indicator)
    features['comma_per_sent'] = text.count(',') / len(sentences) if sentences else 0

    # Semicolons (more formal/classical)
    features['semicolon_per_sent'] = text.count(';') / len(sentences) if sentences else 0

    return features


def discretize_features(features, num_bins=5):
    """Convert continuous feature values into discrete bins for NaiveBayes."""
    discretized = {}
    for key, val in features.items():
        if key == 'avg_sent_len':
            boundaries = [10, 20, 30, 50]
        elif key == 'avg_word_len':
            boundaries = [3.5, 4.0, 4.5, 5.0]
        elif key == 'long_word_prop':
            boundaries = [0.03, 0.06, 0.10, 0.15]
        elif key == 'ttr':
            boundaries = [0.3, 0.45, 0.55, 0.7]
        elif key == 'func_word_prop':
            boundaries = [0.15, 0.20, 0.25, 0.30]
        elif key == 'adj_density':
            boundaries = [0.03, 0.05, 0.07, 0.10]
        elif key == 'noun_verb_ratio':
            boundaries = [1.0, 1.5, 2.0, 3.0]
        elif key == 'comma_per_sent':
            boundaries = [1.0, 2.0, 3.0, 5.0]
        elif key == 'semicolon_per_sent':
            boundaries = [0.05, 0.15, 0.30, 0.50]
        else:
            boundaries = [0.25, 0.50, 0.75, 1.0]

        bin_idx = sum(1 for b in boundaries if val > b)
        discretized[key] = f"{key}_bin{bin_idx}"
    return discretized


# ---------------------------------------------------------------------------
# Exercise 1: Period Profiling
# ---------------------------------------------------------------------------

def period_profiling():
    """Build stylistic profiles for Gutenberg reference texts and Oxen sections."""
    oxen = load_episode('14oxenofthesun.txt')
    sections = segment_oxen(oxen)

    # Gutenberg reference texts (rough period mapping)
    gutenberg_refs = [
        ("Bible (KJV)", gutenberg.raw('bible-kjv.txt')[:20000]),
        ("Shakespeare", gutenberg.raw('shakespeare-hamlet.txt')[:20000]),
        ("Austen (early 19th)", gutenberg.raw('austen-emma.txt')[:20000]),
        ("Melville (mid 19th)", gutenberg.raw('melville-moby_dick.txt')[:20000]),
        ("Whitman (late 19th)", gutenberg.raw('whitman-leaves.txt')[:20000]),
    ]

    print("--- Reference Period Profiles ---")
    all_features = ['avg_sent_len', 'avg_word_len', 'long_word_prop',
                    'ttr', 'adj_density', 'comma_per_sent', 'noun_verb_ratio']

    header = f"{'Text':<25}" + ''.join(f"{f[:12]:>14}" for f in all_features)
    print(header)
    print("-" * (25 + 14 * len(all_features)))

    ref_profiles = []
    for label, text in gutenberg_refs:
        feats = period_features(text)
        ref_profiles.append((label, feats))
        row = f"  {label:<23}"
        for f in all_features:
            row += f"{feats.get(f, 0):>14.4f}"
        print(row)

    print(f"\n--- Oxen of the Sun Section Profiles ---")
    print(header)
    print("-" * (25 + 14 * len(all_features)))

    section_profiles = []
    for label, text in sections:
        feats = period_features(text)
        section_profiles.append((label, feats))
        row = f"  {label:<23}"
        for f in all_features:
            row += f"{feats.get(f, 0):>14.4f}"
        print(row)

    return ref_profiles, section_profiles


# ---------------------------------------------------------------------------
# Exercise 2: The Style Dating Game
# ---------------------------------------------------------------------------

def style_dating_game():
    """Train a period classifier and ask it to 'date' Oxen's sections."""
    oxen = load_episode('14oxenofthesun.txt')
    sections = segment_oxen(oxen)

    # Training data from Gutenberg (simple period labels)
    training_texts = [
        ("early", gutenberg.raw('bible-kjv.txt')[:15000]),
        ("early", gutenberg.raw('shakespeare-hamlet.txt')[:15000]),
        ("middle", gutenberg.raw('austen-emma.txt')[:15000]),
        ("middle", gutenberg.raw('austen-persuasion.txt')[:15000]),
        ("late", gutenberg.raw('melville-moby_dick.txt')[:15000]),
        ("late", gutenberg.raw('whitman-leaves.txt')[:15000]),
    ]

    # Create training feature sets from chunks
    train_labeled = []
    for period, text in training_texts:
        sents = sent_tokenize(text)
        chunk_size = 20
        for i in range(0, len(sents) - chunk_size, chunk_size):
            chunk = ' '.join(sents[i:i+chunk_size])
            feats = period_features(chunk)
            if feats:
                train_labeled.append((discretize_features(feats), period))

    random.seed(42)
    random.shuffle(train_labeled)
    split = int(len(train_labeled) * 0.8)
    train_set = train_labeled[:split]
    test_set = train_labeled[split:]

    classifier = NaiveBayesClassifier.train(train_set)
    acc = accuracy(classifier, test_set)
    print(f"\n--- Style Dating Game ---")
    print(f"  Training accuracy: {acc:.3f}")
    print(f"  Most informative features:")
    classifier.show_most_informative_features(10)

    # Classify Oxen sections
    print(f"\n--- Oxen Sections 'Dated' by Classifier ---")
    print(f"  {'Section':<25} {'Predicted':>12} {'Confidence':>12}")
    print("  " + "-" * 51)

    for label, text in sections:
        feats = period_features(text)
        if feats:
            disc_feats = discretize_features(feats)
            predicted = classifier.classify(disc_feats)
            probs = classifier.prob_classify(disc_feats)
            conf = probs.prob(predicted)
            print(f"  {label:<25} {predicted:>12} {conf:>12.3f}")

    return classifier


# ---------------------------------------------------------------------------
# Exercise 3: The Arc of English
# ---------------------------------------------------------------------------

def arc_of_english():
    """Plot feature trajectories across Oxen's sections as a time series."""
    oxen = load_episode('14oxenofthesun.txt')
    sections = segment_oxen(oxen)

    features_to_plot = ['avg_sent_len', 'avg_word_len', 'long_word_prop',
                        'adj_density', 'comma_per_sent', 'ttr']
    labels = [s[0][:15] for s in sections]

    trajectories = {f: [] for f in features_to_plot}
    for label, text in sections:
        feats = period_features(text)
        for f in features_to_plot:
            trajectories[f].append(feats.get(f, 0))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        ax.plot(range(len(labels)), trajectories[feat], 'bo-', markersize=6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7, ha='right')
        ax.set_title(feat)
        ax.set_ylabel('Value')

    plt.suptitle('The Arc of English: Feature Trajectories Across Oxen of the Sun', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week14_arc.png'), dpi=150)
    plt.close()

    print("\n--- Feature Trajectory Summary ---")
    for feat in features_to_plot:
        vals = trajectories[feat]
        trend = "increasing" if vals[-1] > vals[0] else "decreasing"
        print(f"  {feat:<20}: {vals[0]:.4f} → {vals[-1]:.4f} ({trend})")

    print("\n  Feature trajectory plot saved to solutions/week14_arc.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 62)
    print("EXERCISE 1: Period Profiling")
    print("=" * 62)
    period_profiling()

    print("\n" + "=" * 62)
    print("EXERCISE 2: The Style Dating Game")
    print("=" * 62)
    style_dating_game()

    print("\n" + "=" * 62)
    print("EXERCISE 3: The Arc of English")
    print("=" * 62)
    arc_of_english()
