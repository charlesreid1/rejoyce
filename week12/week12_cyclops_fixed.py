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
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier, accuracy
from collections import defaultdict
from nltk.corpus import stopwords
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")
STOP_WORDS = set(stopwords.words("english"))


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
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
    # Split on blank lines to get actual paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    barfly_segments = []
    interpolation_segments = []

    # Simple heuristic markers
    barfly_markers = {
        "says",
        "begob",
        "bloody",
        "begad",
        "arrah",
        "damn",
        "blazes",
        "cripes",
        "gob",
    }
    formal_markers = {
        "whereas",
        "aforementioned",
        "hereinafter",
        "thereof",
        "notwithstanding",
        "pursuant",
        "hitherto",
        "whereupon",
    }

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
        if (
            formal_score > 0
            or (avg_sent_len > 40 and len(words) > 100)
            or (para.startswith("And") and avg_sent_len > 30)
        ):
            interpolation_segments.append(para)
        elif barfly_score > 0 or para.startswith("—") or avg_sent_len < 25:
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

    # Average sentence length (binned)
    avg_sent_len = len(tokens) / len(sentences)
    if avg_sent_len < 15:
        features["avg_sent_len"] = "short"
    elif avg_sent_len < 30:
        features["avg_sent_len"] = "medium"
    else:
        features["avg_sent_len"] = "long"

    # Type-token ratio (binned)
    ttr = len(set(alpha_tokens)) / len(alpha_tokens) if alpha_tokens else 0
    if ttr < 0.3:
        features["ttr"] = "low"
    elif ttr < 0.6:
        features["ttr"] = "medium"
    else:
        features["ttr"] = "high"

    # Average word length (binned)
    avg_word_len = sum(len(w) for w in alpha_tokens) / len(alpha_tokens)
    if avg_word_len < 4:
        features["avg_word_len"] = "short"
    elif avg_word_len < 5:
        features["avg_word_len"] = "medium"
    else:
        features["avg_word_len"] = "long"

    # POS proportions (binned)
    nouns = sum(c for t, c in tag_counts.items() if t.startswith("NN"))
    verbs = sum(c for t, c in tag_counts.items() if t.startswith("VB"))
    adjs = sum(c for t, c in tag_counts.items() if t.startswith("JJ"))
    noun_prop = nouns / total_tags if total_tags else 0
    verb_prop = verbs / total_tags if total_tags else 0
    adj_prop = adjs / total_tags if total_tags else 0

    if noun_prop < 0.2:
        features["noun_prop"] = "low"
    elif noun_prop < 0.4:
        features["noun_prop"] = "medium"
    else:
        features["noun_prop"] = "high"

    if verb_prop < 0.1:
        features["verb_prop"] = "low"
    elif verb_prop < 0.2:
        features["verb_prop"] = "medium"
    else:
        features["verb_prop"] = "high"

    if adj_prop < 0.05:
        features["adj_prop"] = "low"
    elif adj_prop < 0.1:
        features["adj_prop"] = "medium"
    else:
        features["adj_prop"] = "high"

    # Passive voice approximation (binned)
    passive_count = 0
    for i in range(1, len(tagged)):
        if (
            tagged[i][1] == "VBN"
            and tagged[i - 1][1] in ("VBD", "VBZ", "VBP", "VB")
            and tagged[i - 1][0].lower() in ("was", "were", "is", "be", "been", "being")
        ):
            passive_count += 1
    passive_rate = passive_count / len(sentences) if sentences else 0
    if passive_rate < 0.01:
        features["passive_rate"] = "low"
    elif passive_rate < 0.05:
        features["passive_rate"] = "medium"
    else:
        features["passive_rate"] = "high"

    # First-person pronouns (binned)
    first_person = sum(1 for t in alpha_tokens if t in ("i", "me", "my", "we", "us"))
    first_person_rate = first_person / len(alpha_tokens)
    if first_person_rate < 0.01:
        features["first_person_rate"] = "low"
    elif first_person_rate < 0.03:
        features["first_person_rate"] = "medium"
    else:
        features["first_person_rate"] = "high"

    # Discourse markers (binned)
    discourse = sum(
        1
        for t in alpha_tokens
        if t in ("says", "begob", "bloody", "damn", "arrah", "gob")
    )
    discourse_markers = discourse / len(alpha_tokens)
    if discourse_markers < 0.01:
        features["discourse_markers"] = "low"
    elif discourse_markers < 0.03:
        features["discourse_markers"] = "medium"
    else:
        features["discourse_markers"] = "high"

    # Exclamation marks per sentence (binned)
    exclamation_rate = text.count("!") / len(sentences) if sentences else 0
    if exclamation_rate < 0.01:
        features["exclamation_rate"] = "low"
    elif exclamation_rate < 0.05:
        features["exclamation_rate"] = "medium"
    else:
        features["exclamation_rate"] = "high"

    return features


# ---------------------------------------------------------------------------
# Exercise 1: Annotate and Classify
# ---------------------------------------------------------------------------


def classify_segments():
    """Train a Naive Bayes classifier on barfly vs. interpolation segments."""
    cyclops = load_episode("12cyclops.txt")
    barfly, interp = segment_cyclops(cyclops)

    print(f"--- Segmentation ---")
    print(f"  Barfly segments:        {len(barfly)}")
    print(f"  Interpolation segments: {len(interp)}")

    # Create labeled feature sets
    labeled = []
    for seg in barfly:
        feats = extract_features(seg)
        if feats:
            labeled.append((feats, "barfly"))
    for seg in interp:
        feats = extract_features(seg)
        if feats:
            labeled.append((feats, "interpolation"))

    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled)
    split = int(len(labeled) * 0.7)
    train_set = labeled[:split]
    test_set = labeled[split:]

    # Train Naive Bayes Classifier
    nb_classifier = NaiveBayesClassifier.train(train_set)

    # Train Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier.train(train_set)

    # Predict on test set with Naive Bayes
    nb_predictions = [nb_classifier.classify(fs) for fs, _ in test_set]
    true_labels = [label for _, label in test_set]

    # Calculate metrics for Naive Bayes
    nb_acc = accuracy(nb_classifier, test_set)

    # Calculate per-class metrics for Naive Bayes
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    classes = set(true_labels)

    for pred, true in zip(nb_predictions, true_labels):
        for cls in classes:
            if pred == cls and true == cls:
                class_metrics[cls]["tp"] += 1
            elif pred == cls and true != cls:
                class_metrics[cls]["fp"] += 1
            elif pred != cls and true == cls:
                class_metrics[cls]["fn"] += 1
            else:
                class_metrics[cls]["tn"] += 1

    # Calculate precision, recall, and F1 for each class
    class_scores = {}
    for cls in classes:
        tp = class_metrics[cls]["tp"]
        fp = class_metrics[cls]["fp"]
        fn = class_metrics[cls]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        class_scores[cls] = {"precision": precision, "recall": recall, "f1": f1}

    # Calculate balanced accuracy
    balanced_acc = sum(class_scores[cls]["recall"] for cls in classes) / len(classes)

    print(f"\n--- Naive Bayes Classification Results ---")
    print(f"  Training samples: {len(train_set)}")
    print(f"  Test samples:     {len(test_set)}")
    print(f"  Accuracy:         {nb_acc:.3f}")
    print(f"  Balanced Accuracy:{balanced_acc:.3f}")

    print(f"\n--- Per-Class Metrics ---")
    for cls in sorted(classes):
        metrics = class_scores[cls]
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1-Score:  {metrics['f1']:.3f}")

    print(f"\n--- Most Informative Features ---")
    nb_classifier.show_most_informative_features(15)

    # Evaluate Decision Tree Classifier
    dt_predictions = [dt_classifier.classify(fs) for fs, _ in test_set]
    dt_acc = accuracy(dt_classifier, test_set)

    # Calculate per-class metrics for Decision Tree
    dt_class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for pred, true in zip(dt_predictions, true_labels):
        for cls in classes:
            if pred == cls and true == cls:
                dt_class_metrics[cls]["tp"] += 1
            elif pred == cls and true != cls:
                dt_class_metrics[cls]["fp"] += 1
            elif pred != cls and true == cls:
                dt_class_metrics[cls]["fn"] += 1
            else:
                dt_class_metrics[cls]["tn"] += 1

    # Calculate precision, recall, and F1 for each class for Decision Tree
    dt_class_scores = {}
    for cls in classes:
        tp = dt_class_metrics[cls]["tp"]
        fp = dt_class_metrics[cls]["fp"]
        fn = dt_class_metrics[cls]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        dt_class_scores[cls] = {"precision": precision, "recall": recall, "f1": f1}

    # Calculate balanced accuracy for Decision Tree
    dt_balanced_acc = sum(dt_class_scores[cls]["recall"] for cls in classes) / len(
        classes
    )

    print(f"\n--- Decision Tree Classification Results ---")
    print(f"  Training samples: {len(train_set)}")
    print(f"  Test samples:     {len(test_set)}")
    print(f"  Accuracy:         {dt_acc:.3f}")
    print(f"  Balanced Accuracy:{dt_balanced_acc:.3f}")

    print(f"\n--- Per-Class Metrics ---")
    for cls in sorted(classes):
        metrics = dt_class_scores[cls]
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1-Score:  {metrics['f1']:.3f}")

    return nb_classifier, nb_acc


# ---------------------------------------------------------------------------
# Exercise 2: The Barfly's Fingerprint
# ---------------------------------------------------------------------------


def barfly_fingerprint(classifier):
    """Profile the barfly's voice and scan for it across other episodes using classifier probabilities."""
    cyclops = load_episode("12cyclops.txt")
    barfly_segs, _ = segment_cyclops(cyclops)

    # Extract features for all barfly segments to create a profile
    barfly_features_list = []
    for seg in barfly_segs:
        feats = extract_features(seg)
        if feats:
            barfly_features_list.append(feats)

    # Calculate average feature values for barfly profile
    barfly_profile = {}
    if barfly_features_list:
        for feat in barfly_features_list[0]:
            # For categorical features, we'll calculate the most common value
            values = [f.get(feat, "unknown") for f in barfly_features_list]
            barfly_profile[feat] = max(set(values), key=values.count)

    # Scan other episodes using classifier probabilities
    episodes = [
        ("Telemachus", "01telemachus.txt"),
        ("Hades", "06hades.txt"),
        ("Aeolus", "07aeolus.txt"),
        ("Wandering Rocks", "10wanderingrocks.txt"),
        ("Nausicaa", "13nausicaa.txt"),
        ("Circe", "15circe.txt"),
    ]

    print(f"\n--- Barfly Probability Scores by Episode ---")
    print(f"Computing P(barfly) for each episode using trained classifier...")
    episode_scores = []
    for label, filename in episodes:
        ep_text = load_episode(filename)
        # Segment episode into paragraphs
        paragraphs = [p.strip() for p in ep_text.split("\n\n") if p.strip()]

        # Compute average probability of barfly classification for all paragraphs
        total_prob = 0
        valid_paragraphs = 0

        for para in paragraphs[:100]:  # Limit to first 100 paragraphs for efficiency
            feats = extract_features(para)
            if feats:
                try:
                    prob_dist = classifier.prob_classify(feats)
                    barfly_prob = prob_dist.prob("barfly")
                    total_prob += barfly_prob
                    valid_paragraphs += 1
                except:
                    # Skip paragraphs that cause issues with probability calculation
                    pass

        avg_barfly_prob = total_prob / valid_paragraphs if valid_paragraphs > 0 else 0
        episode_scores.append((label, avg_barfly_prob))
        print(f"  {label:<20} P(barfly): {avg_barfly_prob:.4f}")

    # Improved similarity using feature matching
    print(f"\n--- Barfly Similarity by Feature Matching ---")
    print(f"Comparing episode features to barfly profile...")
    for label, filename in episodes:
        ep_text = load_episode(filename)
        # Segment episode into paragraphs
        paragraphs = [p.strip() for p in ep_text.split("\n\n") if p.strip()]

        # Compute similarity based on feature matching
        total_similarity = 0
        valid_paragraphs = 0

        for para in paragraphs[:50]:  # Limit for efficiency
            feats = extract_features(para)
            if feats and barfly_profile:
                # Calculate similarity as percentage of matching features
                matching_features = sum(
                    1 for feat in feats if feats[feat] == barfly_profile.get(feat, None)
                )
                similarity = (
                    matching_features / len(barfly_profile) if barfly_profile else 0
                )
                total_similarity += similarity
                valid_paragraphs += 1

        avg_similarity = (
            total_similarity / valid_paragraphs if valid_paragraphs > 0 else 0
        )
        print(f"  {label:<20} similarity: {avg_similarity:.4f}")


# ---------------------------------------------------------------------------
# Interpolation Genre Classification
# ---------------------------------------------------------------------------


def classify_interpolation_genre(text):
    """Classify interpolation segments by genre: legal, epic, journalistic, biblical."""
    words = text.lower().split()
    word_set = set(words)

    # Legal terms
    legal_terms = {
        "whereas",
        "aforementioned",
        "hereinafter",
        "thereof",
        "notwithstanding",
        "pursuant",
        "hitherto",
        "whereupon",
        "vendor",
        "merchant",
        "sold",
        "delivered",
    }

    # Epic/biblical terms
    epic_terms = {
        "warriors",
        "princes",
        "mighty",
        "beheld",
        "shining",
        "crystal",
        "glittering",
        "mariners",
        "barks",
        "thither",
        "herds",
        "innumerable",
        "bellwethers",
        "ewes",
    }

    # Biblical terms
    biblical_terms = {"land", "lies", "sleep", "dead", "life", "arose", "watchtower"}

    # Journalistic terms
    journalistic_terms = {"ward", "city", "dublin", "street", "merchant"}

    # Score each genre
    legal_score = len(word_set & legal_terms)
    epic_score = len(word_set & epic_terms)
    biblical_score = len(word_set & biblical_terms)
    journalistic_score = len(word_set & journalistic_terms)

    # Determine primary genre
    scores = {
        "legal": legal_score,
        "epic": epic_score,
        "biblical": biblical_score,
        "journalistic": journalistic_score,
    }
    primary_genre = max(scores, key=scores.get)

    return primary_genre if scores[primary_genre] > 0 else "unknown"


# ---------------------------------------------------------------------------
# Exercise 3: Gigantism as Feature Amplification
# ---------------------------------------------------------------------------


def gigantism_analysis():
    """Compare interpolation feature values to real-world genre baselines."""
    cyclops = load_episode("12cyclops.txt")
    _, interp = segment_cyclops(cyclops)

    interp_text = " ".join(interp)
    interp_feats = extract_features(interp_text)

    # Baselines from other episodes (as proxy for "normal" prose)
    calypso = load_episode("04calypso.txt")
    baseline_feats = extract_features(calypso)

    # Real-world genre baseline from Reuters corpus (legal/financial texts)
    nltk.download("reuters", quiet=True)
    from nltk.corpus import reuters

    # Get a sample of Reuters documents for comparison
    reuters_docs = [reuters.raw(fileid) for fileid in reuters.fileids()[:10]]
    reuters_text = " ".join(reuters_docs)
    reuters_feats = extract_features(reuters_text)

    print("--- Gigantism: Feature Amplification ---")
    print(
        f"{'Feature':<25} {'Interpolations':>15} {'Calypso Baseline':>20} {'Reuters Baseline':>20}"
    )
    print("-" * 87)
    for feat in sorted(interp_feats.keys()):
        iv = interp_feats[feat]
        bv = baseline_feats.get(feat, "N/A")
        rv = reuters_feats.get(feat, "N/A")
        print(f"  {feat:<23} {iv:>15} {bv:>20} {rv:>20}")

    print(f"\n  Key insight: gigantism should manifest as feature values")
    print(f"  systematically MORE EXTREME than baseline prose —")
    print(f"  longer sentences, more adjectives, higher latinate vocabulary.")

    # Analyze genre distribution
    print(f"\n--- Interpolation Genre Analysis ---")
    genre_counts = {}
    for segment in interp:
        genre = classify_interpolation_genre(segment)
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    print("Genre distribution of interpolation segments:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre:<15}: {count}")

    # Compare features by genre
    if genre_counts:
        print(f"\n--- Genre-Specific Feature Analysis ---")
        for genre in genre_counts:
            if genre_counts[genre] > 2:  # Only analyze genres with sufficient samples
                genre_segments = [
                    seg for seg in interp if classify_interpolation_genre(seg) == genre
                ]
                genre_text = " ".join(genre_segments)
                genre_feats = extract_features(genre_text)

                print(f"\n{genre.capitalize()} genre features vs Reuters baseline:")
                for feat in sorted(genre_feats.keys()):
                    gv = genre_feats[feat]
                    rv = reuters_feats.get(feat, "N/A")
                    print(f"  {feat:<23} {gv:>15} {rv:>20}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("EXERCISE 1: Annotate and Classify")
    print("=" * 62)
    classifier, acc = classify_segments()

    print("\n" + "=" * 62)
    print("EXERCISE 2: The Barfly's Fingerprint")
    print("=" * 62)
    barfly_fingerprint(classifier)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Gigantism as Feature Amplification")
    print("=" * 62)
    gigantism_analysis()
