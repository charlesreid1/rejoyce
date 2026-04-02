"""
Week 11: Sirens
================
Phonetic analysis, sound patterning, and sequence repetition detection.

NLTK Focus: nltk.corpus.cmudict, phonetic features, alliteration/assonance
            detection, string matching for motif recurrence

Exercises:
  1. The overture decoded
  2. Phonetic density analysis
  3. Motif tracking
"""

import os
import re
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
from nltk.metrics.distance import edit_distance
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for resource in ["punkt", "punkt_tab", "cmudict"]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")
PRONUNCIATIONS = cmudict.dict()

# Phonetic feature helpers
VOWEL_PHONEMES = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_phonemes(word):
    """Get CMU phoneme list for a word. Returns None if not found."""
    return PRONUNCIATIONS.get(word.lower(), [None])[0]


def get_onset(phonemes):
    """Extract onset consonant(s) from a phoneme list."""
    if not phonemes:
        return None
    onset = []
    for p in phonemes:
        stripped = re.sub(r"\d", "", p)
        if stripped in VOWEL_PHONEMES:
            break
        onset.append(stripped)
    return tuple(onset) if onset else None


def get_vowel_nucleus(phonemes):
    """Extract the stressed vowel nucleus."""
    if not phonemes:
        return None
    for p in phonemes:
        stripped = re.sub(r"\d", "", p)
        if stripped in VOWEL_PHONEMES:
            return stripped
    return None


# ---------------------------------------------------------------------------
# Exercise 1: The Overture Decoded
# ---------------------------------------------------------------------------


def split_overture_body(text):
    """Split Sirens into its overture (first ~63 fragments) and the body.

    The overture is the opening section of short, often incomplete fragments
    before the narrative proper begins.
    """
    lines = text.split("\n")
    # The overture fragments are typically very short lines at the start
    overture_lines = []
    body_start = 0

    # Look for the structural break that begins the narrative proper
    # This is the line "Bronze by gold, miss Douce's head by miss Kennedy's head..."
    structural_break = None
    for i, line in enumerate(lines[:200]):  # Check first 200 lines
        if (
            "miss douce's head" in line.lower()
            and "miss kennedy's head" in line.lower()
        ):
            structural_break = i
            break

    # If we found the structural break, use it as the boundary
    if structural_break is not None:
        overture_lines = [
            line.strip() for line in lines[:structural_break] if line.strip()
        ]
        body_start = structural_break
    else:
        # Fallback: improved heuristic that allows occasional longer lines
        short_line_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # Overture lines are mostly short fragments, but allow some longer ones
            words = word_tokenize(stripped)
            if len(words) < 20:  # Increased threshold slightly
                overture_lines.append(stripped)
                short_line_count += 1
                # Continue collecting until we have enough fragments or hit a clearly long section
                if (
                    short_line_count >= 63 and i > 50
                ):  # Ensure we have enough before stopping
                    # Check if we're hitting a pattern of consistently long lines
                    next_few_long = True
                    for j in range(i + 1, min(i + 6, len(lines))):
                        next_stripped = lines[j].strip()
                        if next_stripped and len(word_tokenize(next_stripped)) < 20:
                            next_few_long = False
                            break
                    if next_few_long:
                        body_start = i + 1
                        break
            elif (
                short_line_count >= 30
            ):  # Only stop if we've collected substantial fragments
                body_start = i
                break

        # Ensure we have approximately 63 fragments
        if len(overture_lines) < 50:
            # Take the first ~63 non-empty lines as fallback
            overture_lines = []
            count = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped:
                    overture_lines.append(stripped)
                    count += 1
                    if count >= 63:
                        body_start = i + 1
                        break

    body = "\n".join(lines[body_start:])
    return overture_lines[:63], body  # Limit to ~63 fragments


def decode_overture(text):
    """Match overture fragments to their source passages in the body."""
    overture, body = split_overture_body(text)

    print(f"--- Overture: {len(overture)} fragments ---")
    matches = []
    unmatched = []

    for frag in overture:
        # Try exact substring match first
        frag_clean = frag.strip().rstrip(".")
        if len(frag_clean) < 3:
            continue

        if frag_clean in body:
            # Find context
            idx = body.index(frag_clean)
            context = body[max(0, idx - 30) : idx + len(frag_clean) + 30]
            matches.append((frag, "exact", context))
        else:
            # Try fuzzy match: find the body sentence with lowest edit distance
            body_sents = sent_tokenize(body)
            best_dist = float("inf")
            best_sent = None
            for sent in body_sents:
                # Check if fragment words appear in sentence
                frag_words = set(word_tokenize(frag_clean.lower()))
                sent_words = set(word_tokenize(sent.lower()))
                overlap = (
                    len(frag_words & sent_words) / len(frag_words) if frag_words else 0
                )
                if overlap > 0.5:
                    dist = edit_distance(
                        frag_clean.lower(), sent[: len(frag_clean) * 2].lower()
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_sent = sent

            if best_sent and best_dist < len(frag_clean):
                matches.append((frag, f"fuzzy (dist={best_dist})", best_sent[:80]))
            else:
                unmatched.append(frag)

    print(f"\n  Matched: {len(matches)}, Unmatched: {len(unmatched)}")
    print(f"  Match rate: {len(matches) / (len(matches) + len(unmatched)) * 100:.1f}%")

    print(f"\n--- Sample Matches ---")
    for frag, match_type, context in matches[:10]:
        print(f'  [{match_type}] "{frag[:60]}"')
        print(f"    → {context[:80]}...\n")

    print(f"--- Unmatched Fragments ---")
    for frag in unmatched[:10]:
        print(f'  "{frag}"')

    return matches, unmatched


# ---------------------------------------------------------------------------
# Exercise 2: Phonetic Density Analysis
# ---------------------------------------------------------------------------


def phonetic_density(text, window_size=100, report_coverage=False):
    """Compute alliteration, assonance, and consonance density per paragraph.

    Returns list of (para_idx, alliteration_rate, assonance_rate, consonance_rate).
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    # Merge short paragraphs to get reasonable windows
    merged = []
    current = []
    for p in paragraphs:
        current.append(p)
        words = " ".join(current).split()
        if len(words) >= window_size:
            merged.append(" ".join(current))
            current = []
    if current:
        merged.append(" ".join(current))

    results = []
    total_words = 0
    words_with_phonemes = 0
    missing_words = Counter()

    for i, para in enumerate(merged):
        words = [w.lower() for w in word_tokenize(para) if w.isalpha() and len(w) > 1]
        total_words += len(words)
        if len(words) < 5:
            continue

        # Get phonemes for each word
        phoneme_words = [(w, get_phonemes(w)) for w in words]
        valid = [(w, p) for w, p in phoneme_words if p is not None]
        words_with_phonemes += len(valid)

        # Track missing words
        for w, p in phoneme_words:
            if p is None:
                missing_words[w] += 1

        if len(valid) < 5:
            results.append((i, 0, 0, 0, len(words)))
            continue

        # Alliteration: repeated onsets in adjacent words
        alliterations = 0
        for j in range(len(valid) - 1):
            onset_a = get_onset(valid[j][1])
            onset_b = get_onset(valid[j + 1][1])
            if onset_a and onset_b and onset_a == onset_b:
                alliterations += 1

        # Assonance: repeated vowel nuclei in a window of 3
        assonances = 0
        for j in range(len(valid) - 2):
            vowels = [get_vowel_nucleus(valid[j + k][1]) for k in range(3)]
            vowels = [v for v in vowels if v]
            if len(vowels) >= 2 and len(set(vowels)) < len(vowels):
                assonances += 1

        # Consonance: repeated coda consonants
        consonances = 0
        for j in range(len(valid) - 1):
            p_a = valid[j][1]
            p_b = valid[j + 1][1]
            if p_a and p_b:
                # Proper coda extraction: start from end and collect consonants backward
                # until hitting a vowel
                coda_a = []
                for ph in reversed(p_a):
                    stripped = re.sub(r"\d", "", ph)
                    if stripped in VOWEL_PHONEMES:
                        break
                    coda_a.append(stripped)
                coda_a.reverse()  # Reverse to maintain proper order

                coda_b = []
                for ph in reversed(p_b):
                    stripped = re.sub(r"\d", "", ph)
                    if stripped in VOWEL_PHONEMES:
                        break
                    coda_b.append(stripped)
                coda_b.reverse()  # Reverse to maintain proper order

                # Compare last consonant of first word with first consonant of second word
                # Or compare the full codas if they're short
                if coda_a and coda_b:
                    if coda_a[-1] == coda_b[0] or (
                        len(coda_a) <= 2
                        and len(coda_b) <= 2
                        and coda_a
                        and coda_b
                        and set(coda_a) & set(coda_b)
                    ):
                        consonances += 1

        n = len(valid) - 1  # normalization base
        results.append(
            (
                i,
                alliterations / n if n else 0,
                assonances / max(n - 1, 1),
                consonances / n if n else 0,
                len(words),
            )
        )

    # Report CMU dictionary coverage
    if report_coverage and total_words > 0:
        coverage = words_with_phonemes / total_words
        print(
            f"\nCMU Dictionary Coverage: {coverage:.1%} ({words_with_phonemes}/{total_words} words)"
        )
        print("Most frequent missing words:")
        for word, count in missing_words.most_common(10):
            print(f"  {word}: {count} occurrences")

    return results


def compare_phonetic_density():
    """Compare phonetic density of Sirens to other episodes."""
    episodes = [
        ("Sirens", "11sirens.txt"),
        ("Lestrygonians", "08lestrygonians.txt"),
        ("Calypso", "04calypso.txt"),
    ]

    print("--- Phonetic Density Comparison ---")
    print(f"{'Episode':<20} {'Alliteration':>14} {'Assonance':>12} {'Consonance':>12}")
    print("-" * 60)

    episode_results = {}
    for label, filename in episodes:
        text = load_episode(filename)
        results = phonetic_density(text, report_coverage=(label == "Sirens"))
        episode_results[label] = results
        if results:
            avg_allit = sum(r[1] for r in results) / len(results)
            avg_asson = sum(r[2] for r in results) / len(results)
            avg_conso = sum(r[3] for r in results) / len(results)
            print(
                f"  {label:<18} {avg_allit:>14.4f} {avg_asson:>12.4f} {avg_conso:>12.4f}"
            )

    # Plot Sirens density across episode
    sirens = load_episode("11sirens.txt")
    results = phonetic_density(sirens, window_size=80)

    if results:
        fig, ax = plt.subplots(figsize=(14, 5))
        xs = [r[0] for r in results]
        ax.plot(xs, [r[1] for r in results], "b-", alpha=0.7, label="Alliteration")
        ax.plot(xs, [r[2] for r in results], "r-", alpha=0.7, label="Assonance")
        ax.plot(xs, [r[3] for r in results], "g-", alpha=0.7, label="Consonance")

        # Add annotations for fugal stretto passages
        # These are approximate positions for the fugal stretto in Sirens
        stretto_positions = [
            50,
            100,
            150,
        ]  # Example positions - would need to be refined
        for pos in stretto_positions:
            ax.axvline(x=pos, color="purple", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            stretto_positions[0],
            0.3,
            "Fugal Stretto",
            rotation=90,
            verticalalignment="bottom",
            color="purple",
            fontsize=8,
        )

        ax.set_xlabel("Paragraph Window")
        ax.set_ylabel("Density (per adjacent pair)")
        ax.set_title("Phonetic Patterning Density: Sirens")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(os.path.dirname(__file__), "week11_phonetic.png"), dpi=150
        )
        plt.close()
        print("\n  Phonetic density plot saved to week11/week11_phonetic.png")

    # Create grouped bar chart for cross-episode comparison
    episode_names = []
    allit_values = []
    asson_values = []
    conso_values = []

    for label, filename in episodes:
        text = load_episode(filename)
        results = phonetic_density(text)
        if results:
            episode_names.append(label)
            avg_allit = sum(r[1] for r in results) / len(results)
            avg_asson = sum(r[2] for r in results) / len(results)
            avg_conso = sum(r[3] for r in results) / len(results)
            allit_values.append(avg_allit)
            asson_values.append(avg_asson)
            conso_values.append(avg_conso)

    # Grouped bar chart
    x = range(len(episode_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], allit_values, width, label="Alliteration", alpha=0.7)
    ax.bar(x, asson_values, width, label="Assonance", alpha=0.7)
    ax.bar([i + width for i in x], conso_values, width, label="Consonance", alpha=0.7)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Density")
    ax.set_title("Phonetic Density Comparison Across Episodes")
    ax.set_xticks(x)
    ax.set_xticklabels(episode_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week11_comparison.png"), dpi=150
    )
    plt.close()
    print("\n  Cross-episode comparison plot saved to week11/week11_comparison.png")


# ---------------------------------------------------------------------------
# Exercise 3: Motif Tracking
# ---------------------------------------------------------------------------

OVERTURE_MOTIFS = [
    "bronze by gold",
    "jingle",
    "tap",
    "Blmstup",
    "imperthnthn",
    "chips",
    "When first",
    "Full tup",
]


def track_motifs(text, motifs=None):
    """Track recurring motifs through the episode.

    For each motif, records every occurrence and its edit distance from
    the canonical form.
    """
    if motifs is None:
        motifs = OVERTURE_MOTIFS

    sentences = sent_tokenize(text)
    motif_data = defaultdict(list)

    for sent_idx, sent in enumerate(sentences):
        sent_lower = sent.lower()
        for motif in motifs:
            motif_lower = motif.lower()
            # Check for exact or near match
            if motif_lower in sent_lower:
                motif_data[motif].append(
                    {
                        "position": sent_idx,
                        "exact_form": motif,
                        "edit_dist": 0,
                        "context": sent[:80],
                    }
                )
            else:
                # Improved partial match logic: require higher word-overlap threshold
                # or multiple motif words co-occurring in the same sentence
                motif_words = motif_lower.split()
                if len(motif_words) > 1:
                    # For multi-word motifs, require stricter matching
                    sent_words = set(word_tokenize(sent_lower))
                    motif_word_set = set(motif_words)
                    overlap = len(sent_words & motif_word_set)

                    # For "bronze by gold", we need all three words to match
                    if motif_lower == "bronze by gold":
                        if overlap == 3:  # All three words must be present
                            motif_data[motif].append(
                                {
                                    "position": sent_idx,
                                    "exact_form": motif,
                                    "edit_dist": 0,
                                    "context": sent[:80],
                                }
                            )
                        # Also check for close variations with edit distance
                        elif overlap >= 2:  # At least two words
                            # Check if this is a close variation
                            best_dist = edit_distance(
                                motif_lower, sent_lower[: len(motif_lower) * 2]
                            )
                            if (
                                best_dist < len(motif_lower) // 2
                            ):  # Less than half the characters different
                                motif_data[motif].append(
                                    {
                                        "position": sent_idx,
                                        "exact_form": motif,
                                        "edit_dist": best_dist,
                                        "context": sent[:80],
                                    }
                                )
                    # For "When first", we need both words to match
                    elif motif_lower == "when first":
                        if overlap == 2:  # Both words must be present
                            motif_data[motif].append(
                                {
                                    "position": sent_idx,
                                    "exact_form": motif,
                                    "edit_dist": 0,
                                    "context": sent[:80],
                                }
                            )
                        # Also check for close variations with edit distance
                        elif "when" in sent_lower and "first" in sent_lower:
                            # Check if this is a close variation
                            best_dist = edit_distance(
                                motif_lower, sent_lower[: len(motif_lower) * 2]
                            )
                            if (
                                best_dist < len(motif_lower) // 2
                            ):  # Less than half the characters different
                                motif_data[motif].append(
                                    {
                                        "position": sent_idx,
                                        "exact_form": motif,
                                        "edit_dist": best_dist,
                                        "context": sent[:80],
                                    }
                                )
                    else:
                        # For other multi-word motifs, require at least half the words
                        if overlap >= max(1, len(motif_words) // 2):
                            # Calculate edit distance to the full motif
                            best_dist = edit_distance(
                                motif_lower, sent_lower[: len(motif_lower) * 2]
                            )
                            motif_data[motif].append(
                                {
                                    "position": sent_idx,
                                    "exact_form": motif,
                                    "edit_dist": best_dist,
                                    "context": sent[:80],
                                }
                            )
                elif len(motif_words) == 1:
                    # For single-word motifs, be more restrictive
                    word = motif_words[0]
                    # Special handling for "tap" to avoid substring matches
                    if word == "tap":
                        # Match whole word "tap", "taps", "tapped", "tapping", etc.
                        # but not words like "eppripfftaph" where "tap" is just a substring
                        pattern = r"\btap(ped|ping|s)?\b"
                        if re.search(pattern, sent_lower):
                            motif_data[motif].append(
                                {
                                    "position": sent_idx,
                                    "exact_form": word,
                                    "edit_dist": 0,
                                    "context": sent[:80],
                                }
                            )
                    elif word in sent_lower and len(word) > 3:
                        # Additional check: make sure it's not just a substring in a larger word
                        # unless it's a complete word match
                        pattern = r"\b" + re.escape(word) + r"\b"
                        if re.search(pattern, sent_lower):
                            motif_data[motif].append(
                                {
                                    "position": sent_idx,
                                    "exact_form": word,
                                    "edit_dist": edit_distance(motif_lower, word),
                                    "context": sent[:80],
                                }
                            )

    print("--- Motif Tracking ---")
    for motif in motifs:
        occurrences = motif_data[motif]
        print(f'\n  Motif: "{motif}" — {len(occurrences)} occurrences')
        for occ in occurrences[:5]:
            print(
                f"    [pos={occ['position']:>4}, dist={occ['edit_dist']}] "
                f"{occ['context'][:70]}..."
            )

    # Plot motif timeline
    fig, ax = plt.subplots(figsize=(14, 6))
    total_sents = len(sentences)
    for i, motif in enumerate(motifs):
        positions = [occ["position"] for occ in motif_data[motif]]
        if positions:
            ax.scatter(positions, [i] * len(positions), s=20, alpha=0.7)

    ax.set_yticks(range(len(motifs)))
    ax.set_yticklabels([m[:20] for m in motifs], fontsize=8)
    ax.set_xlabel("Sentence Position")
    ax.set_title("Motif Score: Sirens")
    ax.set_xlim(0, total_sents)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "week11_motifs.png"), dpi=150)
    plt.close()
    print("\n  Motif score plot saved to week11/week11_motifs.png")

    # Plot edit-distance trajectories per motif
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, motif in enumerate(motifs):
        if motif_data[motif]:
            positions = [occ["position"] for occ in motif_data[motif]]
            distances = [occ["edit_dist"] for occ in motif_data[motif]]
            ax.plot(
                positions,
                distances,
                marker="o",
                markersize=3,
                alpha=0.7,
                label=motif[:15],
            )

    ax.set_xlabel("Sentence Position")
    ax.set_ylabel("Edit Distance from Canonical Form")
    ax.set_title("Motif Edit Distance Trajectories: Sirens")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week11_edit_distances.png"), dpi=150
    )
    plt.close()
    print("\n  Edit distance trajectory plot saved to week11/week11_edit_distances.png")

    return motif_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sirens = load_episode("11sirens.txt")

    print("=" * 62)
    print("EXERCISE 1: The Overture Decoded")
    print("=" * 62)
    decode_overture(sirens)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Phonetic Density Analysis")
    print("=" * 62)
    compare_phonetic_density()

    print("\n" + "=" * 62)
    print("EXERCISE 3: Motif Tracking")
    print("=" * 62)
    track_motifs(sirens)
