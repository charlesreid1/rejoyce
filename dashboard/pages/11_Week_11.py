"""
Week 11 — Sirens
Phonetic/acoustic analysis: overture decoding, phonetic density, motif tracking.
"""

import contextlib
import io
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.metrics.distance import edit_distance

from week11.week11_sirens import (
    split_overture_body,
    phonetic_density,
    track_motifs,
    get_phonemes,
    get_onset,
    get_vowel_nucleus,
    OVERTURE_MOTIFS,
    VOWEL_PHONEMES,
    PRONUNCIATIONS,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
)

st.set_page_config(page_title="Week 11 — Sirens", page_icon="📖", layout="wide")
st.title("Week 11 — Sirens")
st.caption("Phonetic Analysis, Sound Patterning & Motif Tracking")


# ============================================================================
# Helpers
# ============================================================================


def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ============================================================================
# Cached computations
# ============================================================================


@st.cache_data
def cached_match_fragments(text):
    """Reimplement overture fragment matching without print output.

    Returns (overture_fragments, matches, unmatched) where matches is a list
    of (fragment, match_type, edit_dist, context) tuples.
    """
    overture, body = split_overture_body(text)
    body_sents = sent_tokenize(body)

    matches = []
    unmatched = []

    for frag in overture:
        frag_clean = frag.strip().rstrip(".")
        if len(frag_clean) < 3:
            continue

        if frag_clean in body:
            idx = body.index(frag_clean)
            context = body[max(0, idx - 30): idx + len(frag_clean) + 30]
            matches.append((frag, "exact", 0, context))
        else:
            best_dist = float("inf")
            best_sent = None
            for sent in body_sents:
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
                matches.append((frag, "fuzzy", best_dist, best_sent[:120]))
            else:
                unmatched.append(frag)

    return overture, matches, unmatched


@st.cache_data
def cached_phonetic_density(text, window_size=100):
    """Compute phonetic density and also return coverage info."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
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

        phoneme_words = [(w, get_phonemes(w)) for w in words]
        valid = [(w, p) for w, p in phoneme_words if p is not None]
        words_with_phonemes += len(valid)

        for w, p in phoneme_words:
            if p is None:
                missing_words[w] += 1

        if len(valid) < 5:
            results.append((i, 0, 0, 0, len(words)))
            continue

        # Alliteration
        alliterations = 0
        for j in range(len(valid) - 1):
            onset_a = get_onset(valid[j][1])
            onset_b = get_onset(valid[j + 1][1])
            if onset_a and onset_b and onset_a == onset_b:
                alliterations += 1

        # Assonance
        assonances = 0
        for j in range(len(valid) - 2):
            vowels = [get_vowel_nucleus(valid[j + k][1]) for k in range(3)]
            vowels = [v for v in vowels if v]
            if len(vowels) >= 2 and len(set(vowels)) < len(vowels):
                assonances += 1

        # Consonance
        consonances = 0
        for j in range(len(valid) - 1):
            p_a = valid[j][1]
            p_b = valid[j + 1][1]
            if p_a and p_b:
                coda_a = []
                for ph in reversed(p_a):
                    stripped = re.sub(r"\d", "", ph)
                    if stripped in VOWEL_PHONEMES:
                        break
                    coda_a.append(stripped)
                coda_a.reverse()

                coda_b = []
                for ph in reversed(p_b):
                    stripped = re.sub(r"\d", "", ph)
                    if stripped in VOWEL_PHONEMES:
                        break
                    coda_b.append(stripped)
                coda_b.reverse()

                if coda_a and coda_b:
                    if coda_a[-1] == coda_b[0] or (
                        len(coda_a) <= 2
                        and len(coda_b) <= 2
                        and set(coda_a) & set(coda_b)
                    ):
                        consonances += 1

        n = len(valid) - 1
        results.append((
            i,
            alliterations / n if n else 0,
            assonances / max(n - 1, 1),
            consonances / n if n else 0,
            len(words),
        ))

    coverage = words_with_phonemes / total_words if total_words else 0
    top_missing = missing_words.most_common(20)

    return results, coverage, total_words, words_with_phonemes, top_missing


@st.cache_data
def cached_track_motifs(text, motifs_tuple):
    """Track motifs with stdout suppression. Returns motif_data dict."""
    motifs = list(motifs_tuple)
    motif_data = suppress_stdout(track_motifs, text, motifs)
    return dict(motif_data)


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=10,  # Sirens
    caption="Week 11: Phonetic Analysis & Motif Tracking",
)

is_sirens = episode_file == "11sirens.txt"

with st.sidebar:
    window_size = st.slider("Phonetic density window size", 50, 200, 100, key="window_size")
    st.divider()
    st.markdown(
        "**Sirens** is Joyce's fugue — language subordinates meaning to sound. "
        "This week we analyze the acoustic texture computationally: decoding the "
        "overture's fragments, measuring phonetic patterning density, and tracking "
        "motifs through the episode like notes through a musical score."
    )

# Load data
episode_text = cached_load_episode(episode_file)


# ============================================================================
# Section 1: The Overture Decoded
# ============================================================================

st.header("1. The Overture Decoded")

if is_sirens:
    overture, matches, unmatched = cached_match_fragments(episode_text)

    total_frags = len(overture)
    exact_count = sum(1 for _, mt, _, _ in matches if mt == "exact")
    fuzzy_count = sum(1 for _, mt, _, _ in matches if mt == "fuzzy")
    unmatched_count = len(unmatched)
    total_processed = len(matches) + unmatched_count
    match_rate = len(matches) / total_processed * 100 if total_processed else 0

    # --- Metrics row ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Fragments", total_frags)
    m2.metric("Match Rate", f"{match_rate:.1f}%")
    m3.metric("Unmatched", unmatched_count)

    # --- Match type breakdown donut ---
    st.subheader("Match Type Breakdown")

    fig_donut, ax_donut = plt.subplots(figsize=(6, 4))
    sizes = [exact_count, fuzzy_count, unmatched_count]
    labels = [f"Exact ({exact_count})", f"Fuzzy ({fuzzy_count})", f"Unmatched ({unmatched_count})"]
    colors = ["#4A9D8E", "#DAA520", "#C05555"]
    wedges, texts, autotexts = ax_donut.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
    )
    centre = plt.Circle((0, 0), 0.55, fc="white")
    ax_donut.add_artist(centre)
    ax_donut.set_title("Overture Fragment Matching")
    plt.tight_layout()
    st.pyplot(fig_donut)
    plt.close(fig_donut)

    # --- Fragment match table ---
    st.subheader("Fragment Match Table")

    table_rows = []
    for frag, match_type, dist, context in matches:
        table_rows.append({
            "Fragment": frag[:80],
            "Match Type": match_type,
            "Edit Distance": dist,
            "Body Context": context[:100],
        })
    for frag in unmatched:
        table_rows.append({
            "Fragment": frag[:80],
            "Match Type": "unmatched",
            "Edit Distance": "—",
            "Body Context": "—",
        })

    df_frags = pd.DataFrame(table_rows)
    st.dataframe(df_frags, use_container_width=True, hide_index=True)

    # --- Fragment explorer ---
    st.subheader("Fragment Explorer")

    all_frags = [(frag, match_type, dist, context) for frag, match_type, dist, context in matches]
    all_frags += [(frag, "unmatched", None, None) for frag in unmatched]
    frag_labels = [f[:60] for f, _, _, _ in all_frags]

    if frag_labels:
        selected_frag_label = st.selectbox("Select a fragment", frag_labels, key="frag_explorer")
        sel_idx = frag_labels.index(selected_frag_label)
        sel_frag, sel_type, sel_dist, sel_context = all_frags[sel_idx]

        st.markdown(f"**Fragment:** {sel_frag}")
        st.markdown(f"**Match type:** {sel_type}")
        if sel_type != "unmatched":
            st.markdown(f"**Edit distance:** {sel_dist}")
            st.markdown(f"**Body context:** ...{sel_context}...")
        else:
            st.info("No match found in the episode body for this fragment.")

    # --- Cross-episode overture test ---
    with st.expander("Cross-Episode Overture Test"):
        st.markdown(
            "Apply fragment-matching to another episode to show the overture/body "
            "structure is unique to Sirens."
        )
        test_label = st.selectbox(
            "Test episode",
            [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
            key="cross_ep_overture",
        )
        test_file = EPISODE_FILES[EPISODE_LABELS.index(test_label)]
        test_text = cached_load_episode(test_file)

        # Try matching Sirens overture fragments against the other episode
        test_matches = 0
        for frag in overture:
            frag_clean = frag.strip().rstrip(".")
            if len(frag_clean) < 3:
                continue
            if frag_clean in test_text:
                test_matches += 1

        st.metric(
            f"Sirens overture fragments found in {test_label}",
            f"{test_matches}/{len(overture)}",
        )
        st.markdown(
            "The overture's fragments are drawn from the *body* of Sirens specifically. "
            "Near-zero matches in other episodes confirms the fugal structure is unique."
        )

else:
    st.info(
        "The overture decoding analysis is specific to Sirens — the only episode "
        "with a fugal overture structure. Select **11 — Sirens** to explore this section."
    )


# ============================================================================
# Section 2: Phonetic Density Analysis
# ============================================================================

st.header("2. Phonetic Density Analysis")

results, coverage, total_words, words_with_phonemes, top_missing = cached_phonetic_density(
    episode_text, window_size
)

# --- CMU coverage metric ---
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("CMU Dictionary Coverage", f"{coverage:.1%}", help=f"{words_with_phonemes}/{total_words} words")

with c2:
    with st.expander("Top missing words"):
        if top_missing:
            missing_df = pd.DataFrame(top_missing, columns=["Word", "Occurrences"])
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
            st.caption(
                "Missing words are often Joyce's onomatopoeia and invented language "
                "— the most phonetically interesting material the dictionary can't capture."
            )
        else:
            st.write("All words found in CMU dictionary.")

# --- Density line chart ---
st.subheader("Phonetic Density Across Episode")

if results:
    fig_density, ax_density = plt.subplots(figsize=(14, 5))
    xs = [r[0] for r in results]
    ax_density.plot(xs, [r[1] for r in results], "b-", alpha=0.7, label="Alliteration")
    ax_density.plot(xs, [r[2] for r in results], "r-", alpha=0.7, label="Assonance")
    ax_density.plot(xs, [r[3] for r in results], "g-", alpha=0.7, label="Consonance")
    ax_density.set_xlabel("Paragraph Window")
    ax_density.set_ylabel("Density (per adjacent pair)")
    ax_density.set_title(f"Phonetic Patterning Density: {episode_label}")
    ax_density.legend()
    plt.tight_layout()
    st.pyplot(fig_density)
    plt.close(fig_density)
else:
    st.info("Not enough text to compute phonetic density.")

# --- Cross-episode comparison ---
st.subheader("Cross-Episode Comparison")

default_compare = []
for lbl in ["11 — Sirens", "08 — Lestrygonians", "04 — Calypso"]:
    if lbl in EPISODE_LABELS and lbl != episode_label:
        default_compare.append(lbl)

compare_episodes = st.multiselect(
    "Select episodes to compare",
    [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
    default=default_compare,
    key="phonetic_compare",
)

all_compare_labels = [episode_label] + compare_episodes
all_compare_files = [episode_file] + [
    EPISODE_FILES[EPISODE_LABELS.index(lbl)] for lbl in compare_episodes
]

if len(all_compare_labels) >= 2:
    ep_names = []
    allit_vals = []
    asson_vals = []
    conso_vals = []

    for lbl, fname in zip(all_compare_labels, all_compare_files):
        text = cached_load_episode(fname)
        res, _, _, _, _ = cached_phonetic_density(text, window_size)
        if res:
            ep_names.append(lbl.split(" — ")[1])
            allit_vals.append(sum(r[1] for r in res) / len(res))
            asson_vals.append(sum(r[2] for r in res) / len(res))
            conso_vals.append(sum(r[3] for r in res) / len(res))

    if ep_names:
        x = np.arange(len(ep_names))
        width = 0.25

        fig_comp, ax_comp = plt.subplots(figsize=(max(8, len(ep_names) * 2), 5))
        ax_comp.bar(x - width, allit_vals, width, label="Alliteration", alpha=0.7, color="#4A90D9")
        ax_comp.bar(x, asson_vals, width, label="Assonance", alpha=0.7, color="#E07A5F")
        ax_comp.bar(x + width, conso_vals, width, label="Consonance", alpha=0.7, color="#81B29A")
        ax_comp.set_xlabel("Episode")
        ax_comp.set_ylabel("Average Density")
        ax_comp.set_title("Phonetic Density Comparison Across Episodes")
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(ep_names)
        ax_comp.legend()
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close(fig_comp)

    # --- Phonetic type deep-dive ---
    st.subheader("Phonetic Type Deep-Dive")

    measure = st.radio(
        "Select measure",
        ["Alliteration", "Assonance", "Consonance"],
        horizontal=True,
        key="phonetic_measure",
    )
    measure_idx = {"Alliteration": 1, "Assonance": 2, "Consonance": 3}[measure]

    fig_deep, ax_deep = plt.subplots(figsize=(14, 5))
    for lbl, fname in zip(all_compare_labels, all_compare_files):
        text = cached_load_episode(fname)
        res, _, _, _, _ = cached_phonetic_density(text, window_size)
        if res:
            xs = [r[0] for r in res]
            ys = [r[measure_idx] for r in res]
            ax_deep.plot(xs, ys, alpha=0.7, label=lbl.split(" — ")[1])

    ax_deep.set_xlabel("Paragraph Window")
    ax_deep.set_ylabel("Density")
    ax_deep.set_title(f"{measure} Density Overlay")
    ax_deep.legend()
    plt.tight_layout()
    st.pyplot(fig_deep)
    plt.close(fig_deep)

# --- Hottest passages table ---
st.subheader("Hottest Passages")

if results:
    scored = [
        (r[0], r[1] + r[2] + r[3], r[1], r[2], r[3], r[4])
        for r in results
    ]
    scored.sort(key=lambda x: -x[1])

    # Rebuild merged paragraphs to get text
    paragraphs = [p.strip() for p in episode_text.split("\n") if p.strip()]
    merged_paras = []
    current = []
    for p in paragraphs:
        current.append(p)
        words = " ".join(current).split()
        if len(words) >= window_size:
            merged_paras.append(" ".join(current))
            current = []
    if current:
        merged_paras.append(" ".join(current))

    for rank, (idx, total, allit, asson, conso, wc) in enumerate(scored[:5]):
        st.markdown(
            f"**#{rank+1}** — Window {idx} | "
            f"Total: {total:.3f} | "
            f"Allit: {allit:.3f} | Asson: {asson:.3f} | Conso: {conso:.3f}"
        )
        if idx < len(merged_paras):
            with st.expander(f"View passage text ({wc} words)"):
                st.write(merged_paras[idx][:500])


# ============================================================================
# Section 3: Motif Tracking
# ============================================================================

st.header("3. Motif Tracking")

# --- Motif selector ---
default_motifs = list(OVERTURE_MOTIFS)

selected_motifs = st.multiselect(
    "Select motifs to track",
    default_motifs,
    default=default_motifs,
    key="motif_select",
)

custom_motif = st.text_input(
    "Add a custom motif",
    value="",
    key="custom_motif",
)

all_motifs = list(selected_motifs)
if custom_motif.strip():
    all_motifs.append(custom_motif.strip())

if all_motifs:
    motif_data = cached_track_motifs(episode_text, tuple(all_motifs))
    sentences = sent_tokenize(episode_text)
    total_sents = len(sentences)

    # --- Motif occurrence metrics ---
    st.subheader("Motif Occurrences")

    cols = st.columns(min(len(all_motifs), 4))
    for i, motif in enumerate(all_motifs):
        occurrences = motif_data.get(motif, [])
        cols[i % len(cols)].metric(f'"{motif[:15]}"', len(occurrences))

    # --- Motif score timeline ---
    st.subheader("Motif Score Timeline")

    fig_timeline, ax_timeline = plt.subplots(figsize=(14, max(4, len(all_motifs) * 0.6)))
    for i, motif in enumerate(all_motifs):
        occurrences = motif_data.get(motif, [])
        if occurrences:
            positions = [occ["position"] for occ in occurrences]
            distances = [max(5, occ["edit_dist"] * 8 + 15) for occ in occurrences]
            ax_timeline.scatter(positions, [i] * len(positions), s=distances, alpha=0.6)

    ax_timeline.set_yticks(range(len(all_motifs)))
    ax_timeline.set_yticklabels([m[:20] for m in all_motifs], fontsize=8)
    ax_timeline.set_xlabel("Sentence Position")
    ax_timeline.set_title(f"Motif Score: {episode_label}")
    ax_timeline.set_xlim(0, total_sents)
    plt.tight_layout()
    st.pyplot(fig_timeline)
    plt.close(fig_timeline)

    st.caption("Dot size proportional to edit distance from canonical form (larger = more distorted).")

    # --- Edit distance trajectory ---
    st.subheader("Edit Distance Trajectory")

    fig_edit, ax_edit = plt.subplots(figsize=(14, 5))
    has_data = False
    for motif in all_motifs:
        occurrences = motif_data.get(motif, [])
        if occurrences:
            positions = [occ["position"] for occ in occurrences]
            distances = [occ["edit_dist"] for occ in occurrences]
            ax_edit.plot(positions, distances, marker="o", markersize=3, alpha=0.7, label=motif[:15])
            has_data = True

    if has_data:
        ax_edit.set_xlabel("Sentence Position")
        ax_edit.set_ylabel("Edit Distance from Canonical Form")
        ax_edit.set_title(f"Motif Edit Distance Trajectories: {episode_label}")
        ax_edit.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_edit)
    plt.close(fig_edit)

    # --- Motif convergence analysis ---
    st.subheader("Motif Convergence (Stretto Test)")

    convergence_rows = []
    for motif in all_motifs:
        occurrences = motif_data.get(motif, [])
        if len(occurrences) >= 3:
            positions = sorted(occ["position"] for occ in occurrences)
            midpoint = total_sents // 2

            first_half = [p for p in positions if p < midpoint]
            second_half = [p for p in positions if p >= midpoint]

            def avg_gap(pos_list):
                if len(pos_list) < 2:
                    return None
                gaps = [pos_list[i + 1] - pos_list[i] for i in range(len(pos_list) - 1)]
                return sum(gaps) / len(gaps)

            gap_1 = avg_gap(first_half)
            gap_2 = avg_gap(second_half)

            convergence_rows.append({
                "Motif": motif,
                "1st Half Occurrences": len(first_half),
                "2nd Half Occurrences": len(second_half),
                "Avg Gap (1st Half)": f"{gap_1:.1f}" if gap_1 is not None else "—",
                "Avg Gap (2nd Half)": f"{gap_2:.1f}" if gap_2 is not None else "—",
                "Converging?": "Yes" if gap_1 and gap_2 and gap_2 < gap_1 else "No" if gap_1 and gap_2 else "—",
            })

    if convergence_rows:
        st.dataframe(pd.DataFrame(convergence_rows), use_container_width=True, hide_index=True)
        st.markdown(
            "If motifs cluster tighter in the second half (smaller avg gap), that's evidence "
            "of *stretto* — the fugal technique where voices overlap more closely as the piece "
            "builds to its climax."
        )

        # Paired bar chart
        conv_motifs = [r["Motif"] for r in convergence_rows if r["Avg Gap (1st Half)"] != "—" and r["Avg Gap (2nd Half)"] != "—"]
        if conv_motifs:
            gap1_vals = [float(r["Avg Gap (1st Half)"]) for r in convergence_rows if r["Motif"] in conv_motifs]
            gap2_vals = [float(r["Avg Gap (2nd Half)"]) for r in convergence_rows if r["Motif"] in conv_motifs]

            x = np.arange(len(conv_motifs))
            width = 0.35

            fig_conv, ax_conv = plt.subplots(figsize=(max(8, len(conv_motifs) * 1.5), 5))
            ax_conv.bar(x - width / 2, gap1_vals, width, label="1st Half Avg Gap", color="#4A90D9")
            ax_conv.bar(x + width / 2, gap2_vals, width, label="2nd Half Avg Gap", color="#E07A5F")
            ax_conv.set_xticks(x)
            ax_conv.set_xticklabels([m[:15] for m in conv_motifs], rotation=45, ha="right", fontsize=8)
            ax_conv.set_ylabel("Average Gap (sentences)")
            ax_conv.set_title("Motif Convergence: 1st vs 2nd Half")
            ax_conv.legend()
            plt.tight_layout()
            st.pyplot(fig_conv)
            plt.close(fig_conv)
    else:
        st.info("Not enough occurrences to compute convergence statistics.")

    # --- Motif context viewer ---
    st.subheader("Motif Context Viewer")

    motifs_with_data = [m for m in all_motifs if motif_data.get(m)]
    if motifs_with_data:
        view_motif = st.selectbox("Select a motif", motifs_with_data, key="motif_viewer")
        occurrences = motif_data.get(view_motif, [])

        for occ in occurrences:
            pos = occ["position"]
            dist = occ["edit_dist"]
            context = occ["context"]
            # Get fuller context from sentences
            if pos < total_sents:
                full_sent = sentences[pos]
            else:
                full_sent = context

            with st.expander(f"Position {pos} (edit dist: {dist})"):
                # Bold the motif within the sentence
                highlighted = full_sent
                motif_lower = view_motif.lower()
                idx = highlighted.lower().find(motif_lower)
                if idx >= 0:
                    highlighted = (
                        highlighted[:idx]
                        + "**" + highlighted[idx:idx + len(view_motif)] + "**"
                        + highlighted[idx + len(view_motif):]
                    )
                st.markdown(highlighted)
    else:
        st.info("No occurrences found for selected motifs.")

else:
    st.info("Select at least one motif to track.")


st.markdown("""
---

**What this week reveals:** Sirens is Joyce's most explicitly musical episode — and
computational analysis can partially decode its structure. The overture's fragments are
recoverable via string matching, phonetic density measures capture the acoustic texture
that makes Sirens *sound* different from other episodes, and motif tracking reveals the
fugal architecture of repetition and variation. But the limits are instructive: CMU dict
misses Joyce's invented words, edit distance can't capture semantic distortion, and our
motif tracker can't hear the music. The gap between what we measure and what we hear is
precisely what makes this episode a masterclass in the limits of computational text analysis.
""")
