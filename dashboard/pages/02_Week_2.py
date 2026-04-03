"""
Week 02 — Nestor
Part-of-speech tagging and morphological analysis.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import brown, wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

for resource in [
    "punkt", "punkt_tab", "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng", "brown", "wordnet",
    "omw-1.4", "universal_tagset", "stopwords",
]:
    nltk.download(resource, quiet=True)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)
from week02.week02_nestor import get_wordnet_pos

st.set_page_config(page_title="Week 02 — Nestor", page_icon="📖", layout="wide")
st.title("Week 02 — Nestor")
st.caption("Part-of-Speech Tagging & Morphological Analysis")

# --- Sidebar ---
episode_file, episode_label = episode_sidebar(
    default_index=1,  # Nestor
    caption="Week 2: POS Tagging & Morphological Analysis",
    description=(
        "**Week 2** explores part-of-speech tagging, dialogue vs. narration voice analysis, "
        "and lemmatization on Joyce's *Ulysses*."
    ),
)

episode_text = cached_load_episode(episode_file)

# ============================================================================
# Caching helpers
# ============================================================================

POS_CATEGORIES = {
    "Noun": lambda t: t.startswith("NN"),
    "Verb": lambda t: t.startswith("VB"),
    "Adjective": lambda t: t.startswith("JJ"),
    "Adverb": lambda t: t.startswith("RB"),
}

POS_COLORS = {
    "Noun": "#4A90D9",
    "Verb": "#50C878",
    "Adjective": "#E8913A",
    "Adverb": "#9B59B6",
    "Other": "#AAAAAA",
}


def _pos_category(tag):
    for cat, test in POS_CATEGORIES.items():
        if test(tag):
            return cat
    return "Other"


@st.cache_data
def cached_pos_analysis(text):
    """POS-tag a text and compute tag frequencies and grammatical ratios."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tag_freq = Counter(tag for _, tag in tagged)

    nouns = sum(c for t, c in tag_freq.items() if t.startswith("NN"))
    verbs = sum(c for t, c in tag_freq.items() if t.startswith("VB"))
    adjs = sum(c for t, c in tag_freq.items() if t.startswith("JJ"))
    advs = sum(c for t, c in tag_freq.items() if t.startswith("RB"))

    return {
        "tag_freq": dict(tag_freq),
        "total_tokens": len(tokens),
        "noun_count": nouns,
        "verb_count": verbs,
        "adj_count": adjs,
        "adv_count": advs,
        "noun_verb_ratio": nouns / verbs if verbs else float("inf"),
        "adj_adv_ratio": adjs / advs if advs else float("inf"),
    }


@st.cache_data
def cached_universal_tags(tag_freq_dict):
    """Convert Penn Treebank tag frequencies to universal tagset."""
    episode_universal = Counter()
    for tag, count in tag_freq_dict.items():
        try:
            univ = nltk.tag.map_tag("en-ptb", "universal", tag)
            episode_universal[univ] += count
        except KeyError:
            episode_universal["X"] += count
    return dict(episode_universal)


@st.cache_data
def cached_brown_universal():
    """Get Brown Corpus POS distribution (universal tagset)."""
    return dict(Counter(tag for _, tag in brown.tagged_words(tagset="universal")))


@st.cache_data
def split_dialogue_narration(text):
    """Split text into dialogue (em-dash lines) and narration."""
    lines = text.split("\n")
    dialogue_lines = []
    narration_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("\u2014") or stripped.startswith("--"):
            content = stripped[1:].strip() if stripped.startswith("\u2014") else stripped[2:].strip()
            if content:
                dialogue_lines.append(content)
        else:
            narration_lines.append(stripped)
    return " ".join(dialogue_lines), " ".join(narration_lines)


@st.cache_data
def cached_lemma_analysis(text):
    """Lemmatize text and return lemma frequencies and collapse groups."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    lemma_freq = Counter()
    word_freq = Counter()
    lemma_groups = {}  # lemma -> set of (surface_form, tag)

    for word, tag in tagged:
        if not word.isalpha():
            continue
        lower = word.lower()
        word_freq[lower] += 1
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(lower, pos=wn_pos)
        lemma_freq[lemma] += 1

        if lemma not in lemma_groups:
            lemma_groups[lemma] = {}
        if lower != lemma:
            lemma_groups[lemma][lower] = lemma_groups[lemma].get(lower, 0) + 1

    # Only keep groups where something actually collapsed
    collapse_groups = {
        lemma: forms for lemma, forms in lemma_groups.items() if forms
    }

    return {
        "lemma_freq": dict(lemma_freq),
        "word_freq": dict(word_freq),
        "collapse_groups": collapse_groups,
        "total_lemmas": sum(lemma_freq.values()),
        "total_words": sum(word_freq.values()),
        "unique_words": len(word_freq),
        "unique_lemmas": len(lemma_freq),
    }


# ============================================================================
# Section 1: POS Tag Explorer & Cross-Chapter Comparison
# ============================================================================

st.header("1. POS Tag Explorer")

pos_data = cached_pos_analysis(episode_text)

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Noun/Verb Ratio", f"{pos_data['noun_verb_ratio']:.3f}")
c2.metric("Adj/Adv Ratio", f"{pos_data['adj_adv_ratio']:.3f}")
c3.metric("Nouns", f"{pos_data['noun_count']:,}")
c4.metric("Verbs", f"{pos_data['verb_count']:,}")

# POS tag reference
with st.expander("POS Tag Reference (Penn Treebank)"):
    tag_ref = {
        "CC": "Coordinating conjunction",
        "CD": "Cardinal number",
        "DT": "Determiner",
        "EX": "Existential there",
        "FW": "Foreign word",
        "IN": "Preposition/subordinating conjunction",
        "JJ": "Adjective",
        "JJR": "Adjective, comparative",
        "JJS": "Adjective, superlative",
        "MD": "Modal",
        "NN": "Noun, singular",
        "NNP": "Proper noun, singular",
        "NNPS": "Proper noun, plural",
        "NNS": "Noun, plural",
        "PRP": "Personal pronoun",
        "PRP$": "Possessive pronoun",
        "RB": "Adverb",
        "RBR": "Adverb, comparative",
        "RBS": "Adverb, superlative",
        "TO": "to",
        "UH": "Interjection",
        "VB": "Verb, base form",
        "VBD": "Verb, past tense",
        "VBG": "Verb, gerund/present participle",
        "VBN": "Verb, past participle",
        "VBP": "Verb, non-3rd person singular present",
        "VBZ": "Verb, 3rd person singular present",
        "WDT": "Wh-determiner",
        "WP": "Wh-pronoun",
        "WRB": "Wh-adverb",
    }
    ref_df = pd.DataFrame(
        [{"Tag": k, "Description": v} for k, v in tag_ref.items()]
    )
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

# Top POS tags bar chart
st.subheader("POS Tag Frequencies")
top_n_tags = st.slider("Top N tags", 5, 30, 15, key="top_n_tags")

tag_freq = Counter(pos_data["tag_freq"])
top_tags = tag_freq.most_common(top_n_tags)
tags = [t for t, _ in top_tags]
counts = [c for _, c in top_tags]
bar_colors = [POS_COLORS[_pos_category(t)] for t in tags]

fig_tags, ax_tags = plt.subplots(figsize=(12, 5))
bars = ax_tags.barh(range(len(tags)), counts, color=bar_colors)
ax_tags.set_yticks(range(len(tags)))
ax_tags.set_yticklabels(tags)
ax_tags.invert_yaxis()
ax_tags.set_xlabel("Frequency")
ax_tags.set_title(f"Top {top_n_tags} POS Tags — {episode_label}")

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cat) for cat, c in POS_COLORS.items()]
ax_tags.legend(handles=legend_elements, loc="lower right", fontsize=8)
plt.tight_layout()
st.pyplot(fig_tags)
plt.close(fig_tags)

# Brown Corpus comparison
st.subheader("Comparison to Brown Corpus (Universal Tags)")

ep_universal = cached_universal_tags(pos_data["tag_freq"])
brown_universal = cached_brown_universal()

total_ep = sum(ep_universal.values())
total_brown = sum(brown_universal.values())

all_univ_tags = sorted(set(list(ep_universal.keys()) + list(brown_universal.keys())))
# Filter to tags with >0.5% presence in either
filtered_tags = []
ep_pcts = []
brown_pcts = []
for tag in all_univ_tags:
    ep_pct = 100.0 * ep_universal.get(tag, 0) / total_ep
    br_pct = 100.0 * brown_universal.get(tag, 0) / total_brown
    if ep_pct > 0.5 or br_pct > 0.5:
        filtered_tags.append(tag)
        ep_pcts.append(ep_pct)
        brown_pcts.append(br_pct)

x = np.arange(len(filtered_tags))
width = 0.35

fig_brown, ax_brown = plt.subplots(figsize=(12, 5))
ax_brown.bar(x - width / 2, ep_pcts, width, label=episode_label, color="#4A90D9")
ax_brown.bar(x + width / 2, brown_pcts, width, label="Brown Corpus", color="#E8913A")
ax_brown.set_xticks(x)
ax_brown.set_xticklabels(filtered_tags)
ax_brown.set_ylabel("Percentage of tokens")
ax_brown.set_title(f"POS Distribution: {episode_label} vs. Brown Corpus (Universal Tags)")
ax_brown.legend()
plt.tight_layout()
st.pyplot(fig_brown)
plt.close(fig_brown)

# Multi-chapter POS profile comparison
st.subheader("Cross-Chapter Grammatical Profile")

other_labels = [l for l in EPISODE_LABELS if l != episode_label]
compare_chapters = st.multiselect(
    "Compare with other chapters",
    other_labels,
    default=[],
    key="pos_compare",
)

all_selected = [episode_label] + compare_chapters

if len(all_selected) > 1:
    cat_names = list(POS_CATEGORIES.keys())
    chapter_data = {}

    for label in all_selected:
        fname = EPISODE_FILES[EPISODE_LABELS.index(label)]
        text = cached_load_episode(fname)
        data = cached_pos_analysis(text)
        total = data["total_tokens"]
        chapter_data[label] = {
            "Noun": 100.0 * data["noun_count"] / total,
            "Verb": 100.0 * data["verb_count"] / total,
            "Adjective": 100.0 * data["adj_count"] / total,
            "Adverb": 100.0 * data["adv_count"] / total,
        }

    x_ch = np.arange(len(cat_names))
    n = len(all_selected)
    bar_width = 0.8 / n

    fig_multi, ax_multi = plt.subplots(figsize=(10, 5))
    ch_colors = plt.cm.Set2(np.linspace(0, 1, n))
    for idx, label in enumerate(all_selected):
        vals = [chapter_data[label][cat] for cat in cat_names]
        offset = (idx - n / 2 + 0.5) * bar_width
        ax_multi.bar(x_ch + offset, vals, bar_width, label=label, color=ch_colors[idx])

    ax_multi.set_xticks(x_ch)
    ax_multi.set_xticklabels(cat_names)
    ax_multi.set_ylabel("% of tokens")
    ax_multi.set_title("Grammatical Profile Comparison")
    ax_multi.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_multi)
    plt.close(fig_multi)

st.markdown("""
**What POS distributions reveal about literary style:**

The **noun/verb ratio** captures the balance between description and action. A high ratio (noun-heavy) suggests static, descriptive, or cataloguing prose — think of *Ithaca*'s catechistic inventories. A low ratio (verb-heavy) suggests dynamic, action-oriented, or conversational passages.

The **adjective/adverb ratio** reflects how a writer modifies: adjectives qualify nouns (descriptive detail), adverbs qualify verbs (manner of action). Joyce's stylistic shifts across episodes show up clearly here.

Comparing to the **Brown Corpus** — a balanced sample of 1960s American English — reveals how far Joyce departs from "standard" written English. Fiction already differs from news or academic prose, but Joyce pushes further, especially in experimental episodes.
""")

# ============================================================================
# Section 2: Voice Analysis — Dialogue vs. Narration
# ============================================================================

st.header("2. Voice Analysis — Dialogue vs. Narration")

dialogue_text, narration_text = split_dialogue_narration(episode_text)

dial_tokens = len(word_tokenize(dialogue_text)) if dialogue_text.strip() else 0
narr_tokens = len(word_tokenize(narration_text)) if narration_text.strip() else 0
total_voice_tokens = dial_tokens + narr_tokens

# Voice split metrics
col_d, col_n = st.columns(2)

with col_d:
    st.metric("Dialogue Tokens", f"{dial_tokens:,}")
    if total_voice_tokens > 0:
        st.caption(f"{100 * dial_tokens / total_voice_tokens:.1f}% of episode")

with col_n:
    st.metric("Narration/Interior Tokens", f"{narr_tokens:,}")
    if total_voice_tokens > 0:
        st.caption(f"{100 * narr_tokens / total_voice_tokens:.1f}% of episode")

# Donut chart showing proportion
if total_voice_tokens > 0 and dial_tokens > 0 and narr_tokens > 0:
    fig_donut, ax_donut = plt.subplots(figsize=(4, 4))
    sizes = [dial_tokens, narr_tokens]
    labels = ["Dialogue", "Narration"]
    donut_colors = ["#4A90D9", "#E8913A"]
    wedges, texts, autotexts = ax_donut.pie(
        sizes, labels=labels, colors=donut_colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax_donut.add_artist(centre_circle)
    ax_donut.set_title(f"Voice Split — {episode_label}", fontsize=11)
    st.pyplot(fig_donut)
    plt.close(fig_donut)

    # POS comparison between voices
    st.subheader("POS Comparison: Dialogue vs. Narration")

    dial_pos = cached_pos_analysis(dialogue_text)
    narr_pos = cached_pos_analysis(narration_text)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(
        "Noun/Verb — Dialogue",
        f"{dial_pos['noun_verb_ratio']:.3f}",
        delta=f"{dial_pos['noun_verb_ratio'] - narr_pos['noun_verb_ratio']:+.3f} vs narration",
    )
    mc2.metric(
        "Noun/Verb — Narration",
        f"{narr_pos['noun_verb_ratio']:.3f}",
    )
    mc3.metric(
        "Adj/Adv — Dialogue",
        f"{dial_pos['adj_adv_ratio']:.3f}",
        delta=f"{dial_pos['adj_adv_ratio'] - narr_pos['adj_adv_ratio']:+.3f} vs narration",
    )
    mc4.metric(
        "Adj/Adv — Narration",
        f"{narr_pos['adj_adv_ratio']:.3f}",
    )

    # Grouped bar chart: top POS tags in each voice
    voice_toggle = st.toggle("Show as percentages", value=True, key="voice_pct")

    dial_freq = Counter(dial_pos["tag_freq"])
    narr_freq = Counter(narr_pos["tag_freq"])
    combined_top = (dial_freq + narr_freq).most_common(10)
    top_voice_tags = [t for t, _ in combined_top]

    if voice_toggle:
        dial_total = sum(dial_freq.values())
        narr_total = sum(narr_freq.values())
        dial_vals = [100.0 * dial_freq.get(t, 0) / dial_total for t in top_voice_tags]
        narr_vals = [100.0 * narr_freq.get(t, 0) / narr_total for t in top_voice_tags]
        ylabel = "% of tokens"
    else:
        dial_vals = [dial_freq.get(t, 0) for t in top_voice_tags]
        narr_vals = [narr_freq.get(t, 0) for t in top_voice_tags]
        ylabel = "Count"

    x_v = np.arange(len(top_voice_tags))
    fig_voice, ax_voice = plt.subplots(figsize=(12, 5))
    ax_voice.bar(x_v - 0.175, dial_vals, 0.35, label="Dialogue", color="#4A90D9")
    ax_voice.bar(x_v + 0.175, narr_vals, 0.35, label="Narration", color="#E8913A")
    ax_voice.set_xticks(x_v)
    ax_voice.set_xticklabels(top_voice_tags)
    ax_voice.set_ylabel(ylabel)
    ax_voice.set_title(f"Top POS Tags by Voice — {episode_label}")
    ax_voice.legend()
    plt.tight_layout()
    st.pyplot(fig_voice)
    plt.close(fig_voice)

    # Stacked composition bar
    st.subheader("POS Composition by Voice")
    cat_names = ["Noun", "Verb", "Adjective", "Adverb", "Other"]

    def _cat_pcts(pos_result):
        total = pos_result["total_tokens"]
        n = pos_result["noun_count"]
        v = pos_result["verb_count"]
        a = pos_result["adj_count"]
        r = pos_result["adv_count"]
        o = total - n - v - a - r
        return [100.0 * x / total for x in [n, v, a, r, o]]

    dial_pcts = _cat_pcts(dial_pos)
    narr_pcts = _cat_pcts(narr_pos)

    fig_stack, ax_stack = plt.subplots(figsize=(10, 3))
    cat_colors = [POS_COLORS[c] for c in cat_names]
    y_pos = [0, 1]
    y_labels = ["Dialogue", "Narration"]

    for voice_idx, pcts in enumerate([dial_pcts, narr_pcts]):
        left = 0
        for cat_idx, (pct, color) in enumerate(zip(pcts, cat_colors)):
            bar = ax_stack.barh(voice_idx, pct, left=left, color=color, height=0.6)
            if pct > 4:
                ax_stack.text(
                    left + pct / 2, voice_idx, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                )
            left += pct

    ax_stack.set_yticks(y_pos)
    ax_stack.set_yticklabels(y_labels)
    ax_stack.set_xlabel("% of tokens")
    ax_stack.set_title(f"Grammatical Composition — {episode_label}")
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in zip(cat_names, cat_colors)]
    ax_stack.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=5)
    plt.tight_layout()
    st.pyplot(fig_stack)
    plt.close(fig_stack)

else:
    st.info("This episode has no detectable dialogue (em-dash lines) to compare with narration.")

# Cross-chapter dialogue ratio
with st.expander("Dialogue vs. Narration Across All 18 Chapters"):
    st.caption("Click to compute — shows the % of each chapter that is dialogue vs. narration.")

    @st.cache_data
    def compute_all_dialogue_ratios():
        rows = []
        for fname, elabel in EPISODE_MAP.items():
            text = cached_load_episode(fname)
            d, n = split_dialogue_narration(text)
            dt = len(word_tokenize(d)) if d.strip() else 0
            nt = len(word_tokenize(n)) if n.strip() else 0
            total = dt + nt
            rows.append({
                "Episode": elabel,
                "Dialogue %": 100.0 * dt / total if total else 0,
                "Narration %": 100.0 * nt / total if total else 0,
                "Dialogue tokens": dt,
                "Narration tokens": nt,
            })
        return rows

    if st.button("Compute all chapters", key="compute_dialogue"):
        ratio_rows = compute_all_dialogue_ratios()
        labels_all = [r["Episode"] for r in ratio_rows]
        dial_pcts_all = [r["Dialogue %"] for r in ratio_rows]
        narr_pcts_all = [r["Narration %"] for r in ratio_rows]

        fig_all, ax_all = plt.subplots(figsize=(14, 6))
        x_all = np.arange(len(labels_all))
        ax_all.bar(x_all, dial_pcts_all, label="Dialogue", color="#4A90D9")
        ax_all.bar(x_all, narr_pcts_all, bottom=dial_pcts_all, label="Narration", color="#E8913A")
        ax_all.set_xticks(x_all)
        ax_all.set_xticklabels([l.split(" — ")[1] for l in labels_all], rotation=45, ha="right")
        ax_all.set_ylabel("% of tokens")
        ax_all.set_title("Dialogue vs. Narration Across Ulysses")
        ax_all.legend()
        plt.tight_layout()
        st.pyplot(fig_all)
        plt.close(fig_all)

        st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)

st.markdown("""
**What voice splits reveal:**

In *Ulysses*, Joyce signals dialogue with an em-dash (—) instead of quotation marks. Lines beginning with an em-dash are spoken aloud; everything else is narration, interior monologue, or stage direction.

**Dialogue** tends to be pronoun- and verb-heavy — people talk about doing things, referring to each other. **Narration and interior monologue** tend toward nouns and adjectives — describing the world and its qualities.

The balance shifts dramatically across the novel. Early episodes (Telemachus through Scylla) are conversation-heavy; middle episodes increasingly favor interior monologue; Penelope is pure unbroken interior monologue with no dialogue at all.
""")

# ============================================================================
# Section 3: Lemmatization Explorer
# ============================================================================

st.header("3. Lemmatization Explorer")

# Comparison episode selector
st.subheader("Distinctive Lemmas Between Chapters")

compare_label = st.selectbox(
    "Compare against",
    [l for l in EPISODE_LABELS if l != episode_label],
    index=0,
    key="lemma_compare",
)
compare_file = EPISODE_FILES[EPISODE_LABELS.index(compare_label)]
compare_text = cached_load_episode(compare_file)

min_freq = st.slider("Minimum lemma frequency", 1, 10, 3, key="min_freq")
top_n_lemmas = st.slider("Top N distinctive lemmas", 5, 40, 20, key="top_n_lemmas")

lemma_data = cached_lemma_analysis(episode_text)
compare_lemma_data = cached_lemma_analysis(compare_text)

# Compute distinctive lemmas
stop_words = set(stopwords.words("english"))
distinctive = {}
for lemma, count in lemma_data["lemma_freq"].items():
    if lemma in stop_words or count < min_freq:
        continue
    norm_a = count / lemma_data["total_lemmas"]
    norm_b = compare_lemma_data["lemma_freq"].get(lemma, 0) / compare_lemma_data["total_lemmas"]
    distinctive[lemma] = norm_a - norm_b

top_distinctive = sorted(distinctive.items(), key=lambda x: -x[1])[:top_n_lemmas]

if top_distinctive:
    lemmas_d = [l for l, _ in top_distinctive]
    diffs_d = [d * 1000 for _, d in top_distinctive]  # scale to per-1000 for readability

    fig_dist, ax_dist = plt.subplots(figsize=(12, max(4, 0.35 * len(lemmas_d))))
    bar_colors_d = ["#4A90D9" if d > 0 else "#E8913A" for d in diffs_d]
    ax_dist.barh(range(len(lemmas_d)), diffs_d, color=bar_colors_d)
    ax_dist.set_yticks(range(len(lemmas_d)))
    ax_dist.set_yticklabels(lemmas_d)
    ax_dist.invert_yaxis()
    ax_dist.set_xlabel("Normalized frequency difference (per 1,000 lemmas)")
    ax_dist.set_title(
        f"Lemmas more distinctive in {episode_label} vs. {compare_label}"
    )
    ax_dist.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)

# Before/after: raw words vs lemmas
st.subheader("Before & After Lemmatization")

lemma_toggle = st.toggle("Show lemmatized frequencies", value=False, key="lemma_toggle")

top_n_ba = st.slider("Top N words/lemmas", 10, 50, 25, key="top_n_ba")

if lemma_toggle:
    freq_items = sorted(lemma_data["lemma_freq"].items(), key=lambda x: -x[1])[:top_n_ba]
    title_ba = f"Top {top_n_ba} Lemma Frequencies — {episode_label}"
else:
    freq_items = sorted(lemma_data["word_freq"].items(), key=lambda x: -x[1])[:top_n_ba]
    title_ba = f"Top {top_n_ba} Word Frequencies — {episode_label}"

words_ba = [w for w, _ in freq_items]
counts_ba = [c for _, c in freq_items]

fig_ba, ax_ba = plt.subplots(figsize=(12, 5))
ax_ba.bar(words_ba, counts_ba, color="#50C878" if lemma_toggle else "#4A90D9")
ax_ba.set_ylabel("Frequency")
ax_ba.set_title(title_ba)
ax_ba.tick_params(axis="x", rotation=60)
plt.tight_layout()
st.pyplot(fig_ba)
plt.close(fig_ba)

vcol1, vcol2 = st.columns(2)
vcol1.metric("Unique word forms", f"{lemma_data['unique_words']:,}")
vcol2.metric(
    "Unique lemmas",
    f"{lemma_data['unique_lemmas']:,}",
    delta=f"{lemma_data['unique_lemmas'] - lemma_data['unique_words']:,} ({100 * (lemma_data['unique_lemmas'] - lemma_data['unique_words']) / lemma_data['unique_words']:.1f}%)",
)

# Lemmatization collapse browser
st.subheader("Lemmatization Collapse Browser")
st.caption("Surface forms that collapse to the same lemma — sorted by number of forms.")

collapse = lemma_data["collapse_groups"]

# Sort by number of collapsed forms (most collapses first)
sorted_collapses = sorted(collapse.items(), key=lambda x: -len(x[1]))

collapse_filter = st.text_input("Filter by lemma (leave blank for all)", key="collapse_filter")

collapse_rows = []
for lemma, forms in sorted_collapses:
    if collapse_filter and collapse_filter.lower() not in lemma.lower():
        continue
    forms_str = ", ".join(
        f"{form} ({count})" for form, count in sorted(forms.items(), key=lambda x: -x[1])
    )
    collapse_rows.append({
        "Lemma": lemma,
        "Surface Forms": forms_str,
        "# Forms": len(forms),
        "Lemma Freq": lemma_data["lemma_freq"].get(lemma, 0),
    })

if collapse_rows:
    st.dataframe(
        pd.DataFrame(collapse_rows[:100]),
        use_container_width=True,
        hide_index=True,
    )
    if len(collapse_rows) > 100:
        st.caption(f"Showing first 100 of {len(collapse_rows)} collapse groups. Use the filter to narrow results.")

# Interactive word lemmatizer
st.subheader("Word Lemmatizer Lookup")

lookup_word = st.text_input("Type a word to see its lemma and siblings", key="lemma_lookup")
st.caption(
    'Try **"said"** — it appears 64 times but collapses to the lemma "say". '
    'Or try **"went"** to see how "went", "gone", and "going" all consolidate under "go".'
)

if lookup_word:
    w = lookup_word.strip().lower()
    lemmatizer = WordNetLemmatizer()

    # Find what POS tags this word has in the text
    tokens = word_tokenize(episode_text)
    tagged = pos_tag(tokens)
    word_tags = set()
    for word, tag in tagged:
        if word.lower() == w:
            word_tags.add(tag)

    if word_tags:
        for tag in sorted(word_tags):
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(w, pos=wn_pos)
            pos_name = {wordnet.NOUN: "noun", wordnet.VERB: "verb", wordnet.ADJ: "adjective", wordnet.ADV: "adverb"}.get(wn_pos, "noun")

            st.markdown(f"**{w}** (tagged `{tag}` → {pos_name}) **→ lemma: {lemma}**")

            # Find siblings — other surface forms that share this lemma
            if lemma in collapse:
                siblings = collapse[lemma]
                sibling_strs = [f"**{form}** ({count}x)" for form, count in sorted(siblings.items(), key=lambda x: -x[1])]
                st.markdown(f"Siblings under `{lemma}`: {', '.join(sibling_strs)}")

            freq = lemma_data["lemma_freq"].get(lemma, 0)
            word_freq_val = lemma_data["word_freq"].get(w, 0)
            lc1, lc2 = st.columns(2)
            lc1.metric(f"'{w}' frequency", f"{word_freq_val:,}")
            lc2.metric(f"'{lemma}' lemma frequency", f"{freq:,}")
    else:
        st.warning(f"'{w}' not found in this episode.")

st.markdown("""
**What lemmatization reveals and conceals:**

Lemmatization reduces inflected forms to their dictionary base form: "running", "runs", "ran" all become "run". This consolidates the vocabulary, making frequency comparisons between texts fairer — but at a cost.

**What's gained:** Clearer signal for thematic analysis. If one chapter uses "teaches", "taught", and "teaching" while another uses "teach" once, raw frequency would miss the pattern, but lemma frequency catches it.

**What's lost:** Morphological nuance. "Riddles" (noun, a puzzle) and "riddled" (verb/adjective, perforated) collapse to the same lemma despite carrying very different meanings. In Joyce's wordplay-heavy prose, these distinctions often matter — a word's inflection can carry thematic weight that lemmatization erases.

The **distinctive lemmas** chart shows each chapter's conceptual fingerprint — the ideas that dominate one chapter relative to another. For Nestor, expect history, school, money; for Telemachus, tower, sea, mother.
""")
