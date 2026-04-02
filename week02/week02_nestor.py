"""
Week 02: Nestor
================
Part-of-speech tagging and basic morphological analysis.

NLTK Focus: nltk.pos_tag, nltk.corpus.wordnet (lemmatization), tagged corpus exploration

Exercises:
  1. Tag and tabulate
  2. Deasy vs. Stephen
  3. Lemmatization and the weight of history
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import brown, wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "brown",
    "wordnet",
    "omw-1.4",
    "universal_tagset",
]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Tag and Tabulate
# ---------------------------------------------------------------------------


def tag_and_tabulate(text, label="Nestor"):
    """POS-tag a text and compute tag frequency statistics.

    Returns:
        tagged: list of (word, tag) tuples
        tag_freq: Counter of tag frequencies
        ratios: dict of key ratios (noun/verb, adj/adv)
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tag_freq = Counter(tag for _, tag in tagged)

    # Noun tags: NN, NNS, NNP, NNPS
    nouns = sum(c for t, c in tag_freq.items() if t.startswith("NN"))
    # Verb tags: VB, VBD, VBG, VBN, VBP, VBZ
    verbs = sum(c for t, c in tag_freq.items() if t.startswith("VB"))
    # Adjectives: JJ, JJR, JJS
    adjs = sum(c for t, c in tag_freq.items() if t.startswith("JJ"))
    # Adverbs: RB, RBR, RBS
    advs = sum(c for t, c in tag_freq.items() if t.startswith("RB"))

    ratios = {
        "noun_verb_ratio": nouns / verbs if verbs else float("inf"),
        "adj_adv_ratio": adjs / advs if advs else float("inf"),
        "noun_count": nouns,
        "verb_count": verbs,
        "adj_count": adjs,
        "adv_count": advs,
    }

    print(f"\n--- POS Tag Distribution: {label} ---")
    for tag, count in tag_freq.most_common(15):
        print(f"  {tag:<6} {count:>6}")
    print(f"\n  Noun/Verb ratio:  {ratios['noun_verb_ratio']:.3f}")
    print(f"  Adj/Adv ratio:    {ratios['adj_adv_ratio']:.3f}")

    # Create a bar chart of the top 10 POS tags
    top_tags = tag_freq.most_common(10)
    tags, counts = zip(*top_tags)

    plt.figure(figsize=(10, 6))
    plt.bar(tags, counts)
    plt.title(f"Top 10 POS Tags: {label}")
    plt.xlabel("POS Tags")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Sanitize label for filename (remove problematic characters)
    sanitized_label = label.lower().replace("/", "_")
    plt.savefig(os.path.join(os.path.dirname(__file__), f"pos_distribution_{sanitized_label}.png"))
    plt.close()

    return tagged, tag_freq, ratios


def compare_to_brown(tag_freq):
    """Compare episode POS distribution to the Brown Corpus."""
    brown_tags = Counter(tag for _, tag in brown.tagged_words(tagset="universal"))
    # Map Penn Treebank to simplified for comparison using NLTK's built-in mapping
    episode_universal = Counter()
    for tag, count in tag_freq.items():
        try:
            univ = nltk.tag.map_tag("en-ptb", "universal", tag)
            episode_universal[univ] += count
        except KeyError:
            # If mapping fails, default to X
            episode_universal["X"] += count

    total_ep = sum(episode_universal.values())
    total_brown = sum(brown_tags.values())

    print("\n--- Nestor vs. Brown Corpus (Universal Tags) ---")
    print(f"{'Tag':<8} {'Nestor %':>10} {'Brown %':>10} {'Diff':>10}")
    print("-" * 40)
    for tag in sorted(set(list(episode_universal.keys()) + list(brown_tags.keys()))):
        ep_pct = 100.0 * episode_universal.get(tag, 0) / total_ep
        br_pct = 100.0 * brown_tags.get(tag, 0) / total_brown
        diff = ep_pct - br_pct
        if ep_pct > 0.5 or br_pct > 0.5:
            print(f"{tag:<8} {ep_pct:>9.2f}% {br_pct:>9.2f}% {diff:>+9.2f}%")

    # Create a bar chart comparing Nestor vs. Brown Corpus
    # Filter tags with significant presence
    filtered_tags = []
    nestor_pcts = []
    brown_pcts = []

    for tag in sorted(set(list(episode_universal.keys()) + list(brown_tags.keys()))):
        ep_pct = 100.0 * episode_universal.get(tag, 0) / total_ep
        br_pct = 100.0 * brown_tags.get(tag, 0) / total_brown
        if ep_pct > 0.5 or br_pct > 0.5:
            filtered_tags.append(tag)
            nestor_pcts.append(ep_pct)
            brown_pcts.append(br_pct)

    x = range(len(filtered_tags))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], nestor_pcts, width, label="Nestor")
    plt.bar([i + width / 2 for i in x], brown_pcts, width, label="Brown Corpus")
    plt.xlabel("Universal POS Tags")
    plt.ylabel("Percentage")
    plt.title("POS Tag Distribution: Nestor vs. Brown Corpus")
    plt.xticks(x, filtered_tags)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "pos_comparison.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Exercise 2: Deasy vs. Stephen
# ---------------------------------------------------------------------------


def split_deasy_stephen(text):
    """Heuristic split of Deasy's dialogue from Stephen's interior monologue.

    Deasy's dialogue is identified by lines starting with an em-dash that
    contain characteristic topic words. Everything else (other dialogue,
    narration, interior monologue) is grouped as Stephen's voice.
    """
    lines = text.split("\n")

    # Collect dialogue lines (starting with em-dash) and interior lines
    deasy_dialogue_lines = []
    other_dialogue_lines = []
    interior_lines = []

    # Track context to distinguish Deasy's speech from other characters
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("—") or stripped.startswith("--"):
            # Extract content after the dash
            content = (
                stripped[1:].strip()
                if stripped.startswith("—")
                else stripped[2:].strip()
            )

            # Deasy's characteristic topics and phrases
            deasy_indicators = [
                "money",
                "pence",
                "history",
                "england",
                "empire",
                "jews",
                "jew",
                "school",
                "foot",
                "mouth",
                "letter",
                "key",
                "keys",
                "greek",
                "latin",
                "rome",
                "roman",
                "authority",
                "power",
                "king",
                "queen",
                "prince",
                "princess",
                "royal",
                "crown",
                "government",
                "state",
                "nation",
                "country",
                "british",
                "english",
                "irish",
                "ireland",
                "teacher",
                "headmaster",
                "head",
                "master",
                "boys",
                "boy",
                "student",
                "students",
                "class",
                "lesson",
                "teach",
                "education",
            ]

            # Check if this is likely Deasy speaking
            if any(indicator in content.lower() for indicator in deasy_indicators):
                deasy_dialogue_lines.append(content)
            else:
                # Other characters' dialogue (boys, etc.)
                other_dialogue_lines.append(content)
        elif stripped:
            # Non-dialogue lines are interior/narration (Stephen's thoughts)
            interior_lines.append(stripped)

    # Combine Deasy's dialogue separately from other dialogue
    deasy_dialogue_text = " ".join(deasy_dialogue_lines)
    # For this exercise, we'll consider other dialogue as part of interior monologue
    # since the focus is on comparing Deasy's dialogue to Stephen's interior thoughts
    other_content = " ".join(other_dialogue_lines + interior_lines)

    return deasy_dialogue_text, other_content


def compare_voices(text):
    """Compare POS distributions of dialogue vs. interior monologue."""
    dialogue, interior = split_deasy_stephen(text)

    print("\n--- Dialogue vs. Interior Monologue ---")
    print(f"  Dialogue tokens:  {len(word_tokenize(dialogue))}")
    print(f"  Interior tokens:  {len(word_tokenize(interior))}")

    _, dtags, dratios = tag_and_tabulate(dialogue, "Dialogue")
    _, itags, iratios = tag_and_tabulate(interior, "Interior/Narration")

    print("\n--- Voice Comparison ---")
    print(f"{'Metric':<25} {'Dialogue':>12} {'Interior':>12}")
    print("-" * 50)
    for key in ["noun_verb_ratio", "adj_adv_ratio", "noun_count", "verb_count"]:
        print(f"{key:<25} {dratios[key]:>12.3f} {iratios[key]:>12.3f}")

    return dratios, iratios


# ---------------------------------------------------------------------------
# Exercise 3: Lemmatization and the Weight of History
# ---------------------------------------------------------------------------


def get_wordnet_pos(treebank_tag):
    """Map Penn Treebank POS tag to WordNet POS."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def lemmatize_and_compare(text_a, text_b, label_a="Nestor", label_b="Telemachus"):
    """Lemmatize two episode texts and find distinctive lemmas.

    Returns the top lemmas more frequent in text_a than text_b (normalized).
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def get_lemma_freq(text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        lemmas = []
        for word, tag in tagged:
            if word.isalpha():
                wn_pos = get_wordnet_pos(tag)
                if word.lower() == "stared" and tag in ["NN", "NNS", "NNP", "NNPS"]:
                    wn_pos = wordnet.VERB
                lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
                lemmas.append(lemma)
        return FreqDist(lemmas), len(lemmas)

    freq_a, total_a = get_lemma_freq(text_a)
    freq_b, total_b = get_lemma_freq(text_b)

    # Normalized difference: (freq_a/total_a) - (freq_b/total_b)
    distinctive = {}
    for lemma in freq_a:
        if lemma in stop_words:
            continue

        norm_a = freq_a[lemma] / total_a
        norm_b = freq_b.get(lemma, 0) / total_b
        if freq_a[lemma] >= 3:  # minimum frequency threshold
            distinctive[lemma] = norm_a - norm_b

    top_distinctive = sorted(distinctive.items(), key=lambda x: -x[1])[:20]

    print(f"\n--- Top 20 Lemmas More Frequent in {label_a} than {label_b} ---")
    print(
        f"{'Lemma':<20} {label_a + ' (norm)':>15} {label_b + ' (norm)':>15} {'Diff':>10}"
    )
    print("-" * 62)
    for lemma, diff in top_distinctive:
        na = freq_a[lemma] / total_a
        nb = freq_b.get(lemma, 0) / total_b
        print(f"{lemma:<20} {na:>15.5f} {nb:>15.5f} {diff:>+10.5f}")

    return top_distinctive


def lemmatization_loss_examples(text):
    """Show cases where lemmatization collapses meaningful distinctions."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Find words that lemmatize to the same form but may carry different meanings
    lemma_groups = {}
    for word, tag in tagged:
        if word.isalpha() and len(word) > 3:
            wn_pos = get_wordnet_pos(tag)
            if word.lower() == "stared" and tag in ["NN", "NNS", "NNP", "NNPS"]:
                wn_pos = wordnet.VERB
            lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            if lemma != word.lower():
                if lemma not in lemma_groups:
                    lemma_groups[lemma] = set()
                lemma_groups[lemma].add((word.lower(), tag, wn_pos))

    print("\n--- Lemmatization Collapses (surface forms → single lemma) ---")
    for lemma, forms in sorted(lemma_groups.items(), key=lambda x: -len(x[1])):
        if len(forms) > 1:
            forms_list = sorted([f[0] for f in forms])
            print(f"  {lemma:<20} ← {', '.join(forms_list)}")

    # Targeted check for 'riddles' vs 'riddled'
    print("\n--- Specific Test Case: 'riddles' vs 'riddled' ---")
    riddles_lemma = lemmatizer.lemmatize("riddles", pos=wordnet.NOUN)
    riddled_lemma = lemmatizer.lemmatize("riddled", pos=wordnet.VERB)
    print(f"  'riddles' (noun) → '{riddles_lemma}'")
    print(f"  'riddled' (verb) → '{riddled_lemma}'")
    if riddles_lemma == riddled_lemma:
        print(
            "  Note: Both forms lemmatize to the same base form, showing the loss of distinction."
        )
    else:
        print("  Note: Forms maintain their distinction after lemmatization.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nestor = load_episode("02nestor.txt")
    telemachus = load_episode("01telemachus.txt")

    print("=" * 62)
    print("EXERCISE 1: Tag and Tabulate")
    print("=" * 62)
    tagged, tag_freq, ratios = tag_and_tabulate(nestor)
    compare_to_brown(tag_freq)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Deasy vs. Stephen")
    print("=" * 62)
    compare_voices(nestor)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Lemmatization and the Weight of History")
    print("=" * 62)
    lemmatize_and_compare(nestor, telemachus)
    lemmatization_loss_examples(nestor)
