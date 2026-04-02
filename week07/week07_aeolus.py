"""
Week 07: Aeolus
================
TF-IDF, keyword extraction, and extractive summarization heuristics.

NLTK Focus: nltk.text, nltk.FreqDist, TF-IDF (manual computation),
            collocation measures as salience proxies

Exercises:
  1. TF-IDF from scratch
  2. Rhetoric detection
  3. Headline generation
"""

import os
import re
import math
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import matplotlib.pyplot as plt

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


def split_aeolus_sections(text):
    """Split Aeolus into sections using ALL-CAPS headlines as delimiters.

    Returns list of (headline, section_text) tuples.
    """
    # Aeolus headlines are lines that are mostly uppercase
    lines = text.split("\n")
    sections = []
    current_headline = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect headline: short-ish line, mostly uppercase letters
        alpha_chars = [c for c in stripped if c.isalpha()]
        # Lower threshold slightly and add more stringent checks
        if (
            len(alpha_chars) > 3
            and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.6
            and len(stripped) < 120
            and
            # Additional checks to avoid false positives:
            # 1. Should not end with common sentence-ending punctuation followed by more text
            not (stripped.endswith(".") and ". " in stripped)
            and not (stripped.endswith("!") and "! " in stripped)
            and not (stripped.endswith("?") and "? " in stripped)
            and
            # 2. Should not contain obvious dialogue markers
            "---" not in stripped
            and
            # 3. Specific checks for known problematic patterns
            not ("own way. Sllt." in stripped)
            and not ("nonsense. AND IT WAS" in stripped)
            and not ("jerkily. RAISING" in stripped)
            and not ("focus. IMPROMPTU" in stripped)
        ):
            # Save previous section
            if current_headline is not None or current_lines:
                sections.append(
                    (current_headline or "[OPENING]", " ".join(current_lines))
                )
            current_headline = stripped
            current_lines = []
        else:
            current_lines.append(stripped)

    # Don't forget last section
    if current_headline is not None or current_lines:
        sections.append((current_headline or "[CLOSING]", " ".join(current_lines)))

    return sections


# ---------------------------------------------------------------------------
# Exercise 1: TF-IDF from Scratch
# ---------------------------------------------------------------------------


# Custom filter for Joyce-specific artifacts and onomatopoeia
JOYCE_ARTIFACTS = {"sllt", "psha", "deus", "nobis", "hæc", "otia"}


def compute_tfidf(sections):
    """Compute TF-IDF scores for every term in every section.

    TF(t,d) = count(t in d) / len(d)
    IDF(t) = log(N / df(t))
    TF-IDF(t,d) = TF * IDF

    Returns dict mapping section_index -> list of (term, tfidf_score).
    """
    N = len(sections)

    # Tokenize each section
    section_tokens = []
    for headline, text in sections:
        tokens = [
            t.lower()
            for t in word_tokenize(text)
            if t.isalpha()
            and t.lower() not in STOP_WORDS
            and len(t) > 2
            and t.lower() not in JOYCE_ARTIFACTS
        ]
        section_tokens.append(tokens)

    # Document frequency
    df = Counter()
    for tokens in section_tokens:
        unique = set(tokens)
        for term in unique:
            df[term] += 1

    # TF-IDF per section
    tfidf_results = {}
    for i, tokens in enumerate(section_tokens):
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        scores = {}
        for term, count in tf.items():
            tf_score = count / total
            idf_score = math.log(N / df[term]) if df[term] > 0 else 0
            scores[term] = tf_score * idf_score
        tfidf_results[i] = sorted(scores.items(), key=lambda x: -x[1])

    return tfidf_results


def tfidf_vs_headlines(sections, tfidf_results, top_k=5):
    """Compare TF-IDF keywords to Joyce's actual headlines."""
    print("--- TF-IDF Keywords vs. Joyce's Headlines ---\n")

    overlap_count = 0
    total_sections = 0

    for i, (headline, text) in enumerate(sections):
        if i not in tfidf_results or not tfidf_results[i]:
            continue

        keywords = [term for term, score in tfidf_results[i][:top_k]]
        print(f"  Section {i + 1}:")
        print(f"    Joyce's headline: {headline}")
        print(f"    TF-IDF keywords:  {', '.join(keywords)}")

        # Compute overlap between keywords and headline words
        headline_words = set(
            word.lower().strip(".,!?;:") for word in headline.split() if word.isalpha()
        )
        keyword_set = set(keywords)
        overlap = headline_words.intersection(keyword_set)

        if overlap:
            overlap_count += 1
            print(f"    Overlap: {', '.join(overlap)}")
        else:
            print("    Overlap: None")

        total_sections += 1
        print()

    # Compute and display the overlap percentage
    if total_sections > 0:
        overlap_percentage = (overlap_count / total_sections) * 100
        print(
            f"Keyword-headline overlap: {overlap_count}/{total_sections} sections ({overlap_percentage:.1f}%)"
        )

    return


# ---------------------------------------------------------------------------
# Exercise 2: Rhetoric Detection
# ---------------------------------------------------------------------------


def detect_anaphora(text, min_repeat=2):
    """Detect sequences of sentences/clauses beginning with the same word(s).

    Returns list of (repeated_prefix, sentences) tuples.
    """
    sentences = sent_tokenize(text)
    # Track first 1-3 words of each sentence
    prefix_groups = defaultdict(list)

    for sent in sentences:
        tokens = word_tokenize(sent)
        if len(tokens) >= 2:
            prefix = tokens[0].lower()
            # Filter out punctuation-only openers and non-alphabetic prefixes
            if prefix.isalpha() and len(prefix) > 1:
                prefix_groups[prefix].append(sent)

    anaphora = [
        (prefix, sents)
        for prefix, sents in prefix_groups.items()
        if len(sents) >= min_repeat and prefix not in STOP_WORDS
    ]

    anaphora.sort(key=lambda x: -len(x[1]))

    print("--- Anaphora Detection ---")
    for prefix, sents in anaphora[:10]:
        print(f"\n  Repeated opener: '{prefix}' ({len(sents)} sentences)")
        for s in sents[:3]:
            print(f"    → {s[:80]}...")

    return anaphora


def is_speech_attribution(phrase):
    """Check if a phrase is a speech attribution tag (e.g., 'the professor said').

    Returns True if the phrase matches common speech attribution patterns.
    """
    # Convert to lowercase and tokenize
    tokens = word_tokenize(phrase.lower())

    # Common speech verbs
    speech_verbs = {
        "said",
        "asked",
        "cried",
        "shouted",
        "whispered",
        "murmured",
        "declared",
        "announced",
        "added",
        "continued",
        "replied",
        "answered",
        "exclaimed",
        "called",
        "remarked",
        "observed",
        "commented",
    }

    # Check if the phrase ends with a speech verb
    if tokens and tokens[-1] in speech_verbs:
        return True

    # Check for common patterns like "J. J. O'Molloy said"
    if len(tokens) >= 2 and tokens[-1] in speech_verbs:
        return True

    return False


def detect_tricolon(text):
    """Detect sequences of three parallel phrases of similar structure and length.

    Approximation: three consecutive comma-separated or semicolon-separated
    phrases of similar token count.
    """
    sentences = sent_tokenize(text)
    tricolons = []

    for sent in sentences:
        # Split by commas or semicolons
        parts = re.split(r"[,;]", sent)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 3:
            continue

        # Sliding window of 3
        for i in range(len(parts) - 2):
            triple = parts[i : i + 3]

            # Filter out triples containing speech attribution tags
            if any(is_speech_attribution(p) for p in triple):
                continue

            lengths = [len(word_tokenize(p)) for p in triple]

            # Similar length (within 50% of mean)
            mean_len = sum(lengths) / 3
            if mean_len < 3:
                continue
            if all(abs(l - mean_len) / mean_len < 0.5 for l in lengths):
                # Check POS similarity
                tags = [
                    tuple(t for _, t in pos_tag(word_tokenize(p))[:3]) for p in triple
                ]
                # At least 2 of 3 should share opening POS pattern
                if tags[0][:2] == tags[1][:2] or tags[1][:2] == tags[2][:2]:
                    tricolons.append((triple, lengths))

    print("\n--- Tricolon Detection ---")
    for triple, lengths in tricolons[:8]:
        print(f"\n  Lengths: {lengths}")
        for p in triple:
            print(f"    → {p}")

    print(f"\n  Total tricolons detected: {len(tricolons)}")
    return tricolons


# ---------------------------------------------------------------------------
# Exercise 3: Headline Generation
# ---------------------------------------------------------------------------


def generate_headlines(sections, tfidf_results, top_k=3):
    """Generate headlines from TF-IDF keywords and compare to Joyce's.

    Simple template: join top-k keywords in uppercase.
    """
    print("--- Generated Headlines vs. Joyce ---\n")
    print(f"{'#':<4} {'Joyce':<50} {'Generated':<50}")
    print("-" * 105)

    for i, (headline, text) in enumerate(sections):
        if i not in tfidf_results or not tfidf_results[i]:
            continue
        keywords = [term.upper() for term, score in tfidf_results[i][:top_k]]
        generated = " ".join(keywords)

        joyce_short = headline[:48] if headline else "[none]"
        gen_short = generated[:48]
        print(f"  {i + 1:<2}  {joyce_short:<50} {gen_short:<50}")

    return


def process_episode(episode_name, filename, has_headlines=True):
    """Process an episode and generate TF-IDF analysis."""
    text = load_episode(filename)

    if has_headlines:
        print("=" * 62)
        print(f"PARSING {episode_name.upper()} SECTIONS")
        print("=" * 62)
        sections = split_aeolus_sections(text)
        print(f"  Found {len(sections)} sections")
        for i, (h, t) in enumerate(sections[:5]):
            print(f"  [{i + 1}] {h[:60]}")
    else:
        # For episodes without headlines, treat the whole text as one section
        print("=" * 62)
        print(f"PROCESSING {episode_name.upper()} TEXT")
        print("=" * 62)
        sections = [(f"[{episode_name.upper()} TEXT]", text)]
        print(f"  Treating entire episode as one section")

    print("\n" + "=" * 62)
    print(f"EXERCISE 1: TF-IDF from Scratch - {episode_name}")
    print("=" * 62)
    tfidf_results = compute_tfidf(sections)
    tfidf_vs_headlines(sections, tfidf_results)

    if has_headlines:
        print("\n" + "=" * 62)
        print(f"EXERCISE 2: Rhetoric Detection - {episode_name}")
        print("=" * 62)
        detect_anaphora(text)
        detect_tricolon(text)

    print("\n" + "=" * 62)
    print(f"EXERCISE 3: Headline Generation - {episode_name}")
    print("=" * 62)
    generate_headlines(sections, tfidf_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Process Aeolus (has headlines)
    process_episode("Aeolus", "07aeolus.txt", has_headlines=True)

    print("\n" + "=" * 62)
    print("CROSS-EPISODE ANALYSIS: HADES")
    print("=" * 62)
    print("Applying the same TF-IDF method to Hades (which has no headlines)")

    # Process Hades (no headlines)
    hades = load_episode("06hades.txt")
    hades_sections = [("[HADES TEXT]", hades)]
    hades_tfidf = compute_tfidf(hades_sections)
    generate_headlines(hades_sections, hades_tfidf)
