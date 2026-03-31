"""
Week 12: Cyclops
=================
Text classification, feature engineering, and genre/register detection.

NLTK Focus: nltk.classify, NaiveBayesClassifier, DecisionTreeClassifier,
            feature extraction, evaluation metrics

Exercises:
  1. Annotate and classify
  2. The barfly's fingerprint
  3. Gigantism as feature amplification
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

for resource in ['punkt', 'punkt_tab', 'stopwords',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')
STOP_WORDS = set(stopwords.words('english'))


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Segmentation: Barfly Narration vs. Interpolations
# ---------------------------------------------------------------------------

def segment_cyclops(text):
    """Segment Cyclops into barfly narration and gigantist interpolations.

    Heuristic: The barfly's voice is colloquial, first-person, starts with
    lowercase or 'I'. Interpolations tend to be longer paragraphs with
    formal/archaic/parodic register, often starting with definite articles
    and employing elaborate syntax.

    We use paragraph length and register markers as signals.
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    barfly_segments = []
    interpolation_segments = []

    # Simple heuristic markers
    barfly_markers = {'says', 'begob', 'bloody', 'begad', 'arrah',
                      'damn', 'blazes', 'cripes', 'gob'}
    formal_markers = {'whereas', 'aforementioned', 'hereinafter', 'thereof',
                      'notwithstanding', 'pursuant', 'hitherto', 'whereupon'}

    for para in paragraphs:
        words = word_tokenize(para.lower())
        word_set = set(words)

        # Count register markers
        barfly_score = len(word_set & barfly_markers)
        formal_score = len(word_set & formal_markers)

        # Average sentence length as proxy
        sents = sent_tokenize(para)
        avg_sent_len = len(words) / len(sents) if sents else 0

        # Long paragraphs with high average sentence length → interpolation
        if (formal_score > 0 or
            (avg_sent_len > 40 and len(words) > 100) or
            (para.startswith('And') and avg_sent_len > 30)):
            interpolation_segments.append(para)
        elif barfly_score > 0 or para.startswith('—') or avg_sent_len < 25:
            barfly_segments.append(para)
        else:
            # Default: shorter = barfly, longer = interpolation
            if len(words) < 80:
                barfly_segments.append(para)
            else:
                interpolation_segments.append(para)

    return barfly_segments, interpolation_segments


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(text):
    """Extract classification features from a text segment."""
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    sentences = sent_tokenize(text)

    if not alpha_tokens or not sentences:
        return {}

    tagged = pos_tag(tokens)
    tag_counts = Counter(tag for _, tag in tagged)
    total_tags = sum(tag_counts.values())

    # Features
    features = {}

    # Average sentence length
    features['avg_sent_len'] = len(tokens) / len(sentences)

    # Type-token ratio
    types = set(alpha_tokens)
    features['ttr'] = len(types) / len(alpha_tokens) if alpha_tokens else 0

    # Average word length (proxy for latinate vocabulary)
    features['avg_word_len'] = sum(len(w) for w in alpha_tokens) / len(alpha_tokens)

    # POS proportions
    nouns = sum(c for t, c in tag_counts.items() if t.startswith('NN'))
    verbs = sum(c for t, c in tag_counts.items() if t.startswith('VB'))
    adjs = sum(c for t, c in tag_counts.items() if t.startswith('JJ'))
    features['noun_prop'] = nouns / total_tags if total_tags else 0
    features['verb_prop'] = verbs / total_tags if total_tags else 0
    features['adj_prop'] = adjs / total_tags if total_tags else 0

    # Passive voice approximation: VBN preceded by VB* (be/was/were + past participle)
    passive_count = 0
    for i in range(1, len(tagged)):
        if (tagged[i][1] == 'VBN' and
            tagged[i-1][1] in ('VBD', 'VBZ', 'VBP', 'VB') and
            tagged[i-1][0].lower() in ('was', 'were', 'is', 'be', 'been', 'being')):
            passive_count += 1
    features['passive_rate'] = passive_count / len(sentences) if sentences else 0

    # First-person pronouns
    first_person = sum(1 for t in alpha_tokens if t in ('i', 'me', 'my', 'we', 'us'))
    features['first_person_rate'] = first_person / len(alpha_tokens)

    # Discourse markers
    discourse = sum(1 for t in alpha_tokens
                    if t in ('says', 'begob', 'bloody', 'damn', 'arrah', 'gob'))
    features['discourse_markers'] = discourse / len(alpha_tokens)

    # Exclamation marks per sentence
    features['exclamation_rate'] = text.count('!') / len(sentences) if sentences else 0

    return features


# ---------------------------------------------------------------------------
# Exercise 1: Annotate and Classify
# ---------------------------------------------------------------------------

def classify_segments():
    """Train a Naive Bayes classifier on barfly vs. interpolation segments."""
    cyclops = load_episode('12cyclops.txt')
    barfly, interp = segment_cyclops(cyclops)

    print(f"--- Segmentation ---")
    print(f"  Barfly segments:        {len(barfly)}")
    print(f"  Interpolation segments: {len(interp)}")

    # Create labeled feature sets
    labeled = []
    for seg in barfly:
        feats = extract_features(seg)
        if feats:
            labeled.append((feats, 'barfly'))
    for seg in interp:
        feats = extract_features(seg)
        if feats:
            labeled.append((feats, 'interpolation'))

    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled)
    split = int(len(labeled) * 0.7)
    train_set = labeled[:split]
    test_set = labeled[split:]

    # Train
    classifier = NaiveBayesClassifier.train(train_set)
    acc = accuracy(classifier, test_set)

    print(f"\n--- Classification Results ---")
    print(f"  Training samples: {len(train_set)}")
    print(f"  Test samples:     {len(test_set)}")
    print(f"  Accuracy:         {acc:.3f}")

    print(f"\n--- Most Informative Features ---")
    classifier.show_most_informative_features(15)

    return classifier, acc


# ---------------------------------------------------------------------------
# Exercise 2: The Barfly's Fingerprint
# ---------------------------------------------------------------------------

def barfly_fingerprint():
    """Profile the barfly's voice and scan for it across other episodes."""
    cyclops = load_episode('12cyclops.txt')
    barfly_segs, _ = segment_cyclops(cyclops)
    barfly_text = ' '.join(barfly_segs)

    # Compute barfly profile
    profile = extract_features(barfly_text)
    print("--- Barfly's Stylistic Profile ---")
    for feat, val in sorted(profile.items()):
        print(f"  {feat:<25} {val:.4f}")

    # Scan other episodes
    episodes = [
        ('Telemachus', '01telemachus.txt'),
        ('Hades', '06hades.txt'),
        ('Aeolus', '07aeolus.txt'),
        ('Wandering Rocks', '10wanderingrocks.txt'),
        ('Nausicaa', '13nausicaa.txt'),
        ('Circe', '15circe.txt'),
    ]

    print(f"\n--- Barfly Similarity Scan (discourse_markers + first_person_rate) ---")
    for label, filename in episodes:
        ep_text = load_episode(filename)
        ep_feats = extract_features(ep_text)
        # Barfly similarity = sum of key feature similarities
        sim = (1 - abs(profile.get('first_person_rate', 0) -
                       ep_feats.get('first_person_rate', 0))) * 0.5 + \
              (1 - abs(profile.get('discourse_markers', 0) -
                       ep_feats.get('discourse_markers', 0))) * 0.5
        print(f"  {label:<20} similarity: {sim:.4f}  "
              f"(1st person: {ep_feats.get('first_person_rate', 0):.4f}, "
              f"discourse: {ep_feats.get('discourse_markers', 0):.4f})")


# ---------------------------------------------------------------------------
# Exercise 3: Gigantism as Feature Amplification
# ---------------------------------------------------------------------------

def gigantism_analysis():
    """Compare interpolation feature values to real-world genre baselines."""
    cyclops = load_episode('12cyclops.txt')
    _, interp = segment_cyclops(cyclops)

    interp_text = ' '.join(interp)
    interp_feats = extract_features(interp_text)

    # Baselines from other episodes (as proxy for "normal" prose)
    calypso = load_episode('04calypso.txt')
    baseline_feats = extract_features(calypso)

    print("--- Gigantism: Feature Amplification ---")
    print(f"{'Feature':<25} {'Interpolations':>15} {'Baseline':>15} {'Ratio':>10}")
    print("-" * 67)
    for feat in sorted(interp_feats.keys()):
        iv = interp_feats[feat]
        bv = baseline_feats.get(feat, 0)
        ratio = iv / bv if bv > 0 else float('inf')
        print(f"  {feat:<23} {iv:>15.4f} {bv:>15.4f} {ratio:>10.2f}x")

    print(f"\n  Key insight: gigantism should manifest as feature values")
    print(f"  systematically MORE EXTREME than baseline prose —")
    print(f"  longer sentences, more adjectives, higher latinate vocabulary.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 62)
    print("EXERCISE 1: Annotate and Classify")
    print("=" * 62)
    classify_segments()

    print("\n" + "=" * 62)
    print("EXERCISE 2: The Barfly's Fingerprint")
    print("=" * 62)
    barfly_fingerprint()

    print("\n" + "=" * 62)
    print("EXERCISE 3: Gigantism as Feature Amplification")
    print("=" * 62)
    gigantism_analysis()
