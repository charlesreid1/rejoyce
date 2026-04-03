"""
Week 03 — Proteus
Stemming, multilingual detection, and morphological analysis.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.metrics.distance import edit_distance
from nltk.corpus import stopwords, wordnet
import nltk

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

from week03.week03_proteus import (
    stemmers_struggle,
    detect_languages,
    morphological_analysis,
    hypothesize_parse,
    LATIN_STOPWORDS,
    LATIN_VERB_ENDINGS,
)
from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 03 — Proteus", page_icon="📖", layout="wide")
st.title("Week 03 — Proteus")
st.caption("Stemming, Multilingual Detection & Morphological Analysis")

# --- Sidebar ---
episode_file, episode_label = episode_sidebar(
    default_index=2,  # Proteus
    caption="Week 3: Stemming, Language Detection & Morphology",
    description=(
        "**Week 3** explores how stemming algorithms struggle with Joyce's vocabulary, "
        "detects multilingual passages using stopword overlap, "
        "and dissects neologisms and compound words via WordNet."
    ),
)

episode_text = cached_load_episode(episode_file)

# ============================================================================
# Section 1: The Stemmer's Struggle
# ============================================================================

st.header("1. The Stemmer's Struggle")

st.markdown(
    "Stemmers reduce words to a root form, but they disagree on how aggressively to cut. "
    "Joyce's unusual vocabulary — neologisms, compounds, foreign borrowings — "
    "pushes these algorithms to their limits."
)


@st.cache_data
def compute_stemmer_results(text, top_n):
    """Run all three stemmers and compute reductions + disagreement."""
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    tokens = word_tokenize(text)
    alpha_tokens = list(set(t.lower() for t in tokens if t.isalpha() and len(t) > 2))

    stemmers = {"Porter": porter, "Lancaster": lancaster, "Snowball": snowball}

    results = {}
    all_reductions = {}
    for name, stemmer in stemmers.items():
        reductions = []
        for word in alpha_tokens:
            stem = stemmer.stem(word)
            dist = edit_distance(word, stem)
            reductions.append((word, stem, dist))
        reductions.sort(key=lambda x: -x[2])
        results[name] = reductions[:top_n]
        all_reductions[name] = reductions

    # Disagreement rate
    disagree_count = 0
    for word in alpha_tokens:
        stems = set(s.stem(word) for s in stemmers.values())
        if len(stems) > 1:
            disagree_count += 1
    disagree_rate = disagree_count / len(alpha_tokens) if alpha_tokens else 0

    return results, disagree_rate, all_reductions, len(alpha_tokens)


top_n = st.slider("Top N most aggressive reductions", 5, 30, 10, key="stem_top_n")

results, disagree_rate, all_reductions, vocab_size = compute_stemmer_results(episode_text, top_n)

# --- Metrics row ---
m1, m2, m3 = st.columns(3)
m1.metric("Unique Words Analyzed", f"{vocab_size:,}")
m2.metric("Stemmer Disagreement Rate", f"{disagree_rate:.1%}")
m3.metric(
    "Max Edit Distance (Lancaster)",
    max(d for _, _, d in all_reductions["Lancaster"]) if all_reductions["Lancaster"] else 0,
)

# --- Side-by-side stemmer tables ---
st.subheader("Most Aggressive Reductions by Stemmer")

cols = st.columns(3)
for col, name in zip(cols, ["Porter", "Lancaster", "Snowball"]):
    with col:
        st.markdown(f"**{name}**")
        rows = [
            {"Word": w, "Stem": s, "Edit Dist": d}
            for w, s, d in results[name]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Edit distance distribution chart ---
st.subheader("Edit Distance Distribution")

fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
stemmer_colors = {"Porter": "#4C78A8", "Lancaster": "#E45756", "Snowball": "#72B7B2"}
max_dist = max(
    max(d for _, _, d in reds) for reds in all_reductions.values()
)
bins = np.arange(0, max_dist + 2) - 0.5

for name in ["Porter", "Lancaster", "Snowball"]:
    distances = [d for _, _, d in all_reductions[name]]
    ax_dist.hist(
        distances, bins=bins, alpha=0.5, label=name,
        color=stemmer_colors[name], edgecolor="white",
    )

ax_dist.set_xlabel("Edit Distance (characters removed/changed)")
ax_dist.set_ylabel("Number of Words")
ax_dist.set_title("How aggressively does each stemmer cut?")
ax_dist.legend()
ax_dist.set_xticks(range(0, int(max_dist) + 1))
plt.tight_layout()
st.pyplot(fig_dist)
plt.close(fig_dist)

st.markdown(
    "Lancaster is the most aggressive stemmer — it frequently reduces words by 4+ characters. "
    "Porter and Snowball are nearly identical (Snowball English is a reimplementation of Porter)."
)

# --- Interactive word stemmer ---
st.subheader("Stem Any Word")

stem_input = st.text_input(
    "Enter a word to see how each stemmer handles it",
    value="contransmagnificandjewbangtantiality",
)

if stem_input:
    word = stem_input.strip().lower()
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    stem_results = {
        "Porter": porter.stem(word),
        "Lancaster": lancaster.stem(word),
        "Snowball": snowball.stem(word),
    }

    sc1, sc2, sc3 = st.columns(3)
    for col, (name, stem) in zip([sc1, sc2, sc3], stem_results.items()):
        dist = edit_distance(word, stem)
        col.metric(name, stem, delta=f"-{dist} chars", delta_color="inverse")

    # Show which stemmers agree
    unique_stems = set(stem_results.values())
    if len(unique_stems) == 1:
        st.success("All three stemmers agree on this word.")
    elif len(unique_stems) == 2:
        st.warning("The stemmers disagree — two agree, one diverges.")
    else:
        st.error("All three stemmers produce different stems!")

# --- Scatter: word length vs edit distance ---
st.subheader("Word Length vs. Stemming Aggressiveness")

scatter_stemmer = st.selectbox("Stemmer", ["Porter", "Lancaster", "Snowball"], index=1)

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
lengths = [len(w) for w, _, _ in all_reductions[scatter_stemmer]]
dists = [d for _, _, d in all_reductions[scatter_stemmer]]
ax_scatter.scatter(lengths, dists, alpha=0.15, s=12, color=stemmer_colors[scatter_stemmer])

# Highlight the top-N most aggressive
for w, s, d in results[scatter_stemmer][:5]:
    ax_scatter.annotate(
        w, (len(w), d), fontsize=7, alpha=0.9,
        textcoords="offset points", xytext=(5, 5),
    )

ax_scatter.set_xlabel("Original Word Length")
ax_scatter.set_ylabel("Edit Distance After Stemming")
ax_scatter.set_title(f"{scatter_stemmer}: Do longer words get cut more?")
plt.tight_layout()
st.pyplot(fig_scatter)
plt.close(fig_scatter)


# ============================================================================
# Section 2: Multilingual Detection
# ============================================================================

st.header("2. Multilingual Detection")

st.markdown(
    "Proteus is the most multilingual episode of *Ulysses* — Stephen's interior monologue "
    "drifts through English, French, Italian, Latin, and German. "
    "This detector uses stopword overlap to classify each sentence by language."
)


@st.cache_data
def compute_language_detection(text):
    """Detect languages per sentence using stopword overlap."""
    lang_stops = {
        "English": set(stopwords.words("english")),
        "French": set(stopwords.words("french")),
        "German": set(stopwords.words("german")),
        "Italian": set(stopwords.words("italian")),
        "Latin": LATIN_STOPWORDS,
    }

    sentences = sent_tokenize(text)
    results = []

    for sent in sentences:
        tokens = [t.lower() for t in word_tokenize(sent) if t.isalpha()]
        if not tokens:
            continue

        scores = {}
        for lang, stops in lang_stops.items():
            overlap = sum(1 for t in tokens if t in stops)
            scores[lang] = overlap / len(tokens)

        # Boost Latin when verb morphology supports
        if scores["Latin"] > 0:
            latin_endings = sum(
                1 for t in tokens
                if any(t.endswith(ending) for ending in LATIN_VERB_ENDINGS)
            )
            if latin_endings > 0:
                scores["Latin"] += 0.1

        best_lang = max(scores, key=scores.get)
        if best_lang != "English" and scores[best_lang] < 0.1:
            best_lang = "English"

        results.append({
            "sentence": sent,
            "language": best_lang,
            "scores": scores,
            "tokens": len(tokens),
        })

    return results


detection_results = compute_language_detection(episode_text)

# --- Summary metrics ---
from collections import Counter
lang_counts = Counter(r["language"] for r in detection_results)
total_sents = len(detection_results)
non_english_count = total_sents - lang_counts.get("English", 0)

lm1, lm2, lm3, lm4 = st.columns(4)
lm1.metric("Total Sentences", total_sents)
lm2.metric("Non-English Sentences", non_english_count)
lm3.metric("Non-English %", f"{non_english_count / total_sents:.1%}" if total_sents else "0%")
lm4.metric("Languages Detected", len(lang_counts))

# --- Language proportion pie chart ---
st.subheader("Language Distribution")

fig_lang, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

lang_colors = {
    "English": "#4C78A8",
    "French": "#E45756",
    "German": "#F58518",
    "Italian": "#72B7B2",
    "Latin": "#B279A2",
}

# Pie chart — collapse to English vs Other (the bar chart shows the detail)
eng_count = lang_counts.get("English", 0)
other_count = total_sents - eng_count
pie_labels = ["English", "Other"]
pie_sizes = [eng_count, other_count]
pie_cols = [lang_colors["English"], "#AAAAAA"]

ax_pie.pie(
    pie_sizes,
    labels=pie_labels,
    colors=pie_cols,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=0.5),
    textprops=dict(fontsize=9),
)
ax_pie.set_title("Sentences by Detected Language")

# Bar chart of non-English languages, sorted by count
non_eng = {k: v for k, v in lang_counts.items() if k != "English"}
if non_eng:
    sorted_ne = sorted(non_eng.items(), key=lambda x: x[1])
    bar_labels = [k for k, v in sorted_ne]
    bar_vals = [v for k, v in sorted_ne]
    bar_colors = [lang_colors.get(l, "#999") for l in bar_labels]
    ax_bar.barh(bar_labels, bar_vals, color=bar_colors)
    ax_bar.set_xlabel("Number of Sentences")
    ax_bar.set_title("Non-English Breakdown")
else:
    ax_bar.text(0.5, 0.5, "No non-English\nsentences detected", ha="center", va="center")
    ax_bar.set_title("Non-English Breakdown")

plt.tight_layout()
st.pyplot(fig_lang)
plt.close(fig_lang)

# --- Language dispersion through the text ---
st.subheader("Language Dispersion Through the Episode")

fig_ldisp, ax_ldisp = plt.subplots(figsize=(12, 2.5))
for i, r in enumerate(detection_results):
    color = lang_colors.get(r["language"], "#999")
    position = i / total_sents * 100
    if r["language"] != "English":
        ax_ldisp.axvline(position, color=color, alpha=0.7, linewidth=2)
        ax_ldisp.annotate(
            r["language"][:2], (position, 0.5), fontsize=6, rotation=90,
            ha="center", va="center", color=color, fontweight="bold",
        )

ax_ldisp.set_xlim(0, 100)
ax_ldisp.set_xlabel("Position in episode (%)")
ax_ldisp.set_title("Where non-English sentences appear")
ax_ldisp.set_yticks([])

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=lang_colors[l], label=l)
    for l in lang_counts if l != "English"
]
if legend_elements:
    ax_ldisp.legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
st.pyplot(fig_ldisp)
plt.close(fig_ldisp)

# --- Non-English sentence browser ---
st.subheader("Non-English Sentence Browser")

filter_lang = st.multiselect(
    "Filter by language",
    [l for l in lang_counts if l != "English"],
    default=[l for l in lang_counts if l != "English"],
    key="lang_filter",
)

non_english_sents = [
    r for r in detection_results if r["language"] in filter_lang
]

if non_english_sents:
    browser_rows = []
    for r in non_english_sents:
        scores_str = " | ".join(f"{l}: {r['scores'][l]:.2f}" for l, _ in sorted(r["scores"].items(), key=lambda x: -x[1]) if r["scores"][l] > 0)
        browser_rows.append({
            "Language": r["language"],
            "Sentence": r["sentence"][:120] + ("..." if len(r["sentence"]) > 120 else ""),
            "Confidence Scores": scores_str,
        })
    st.dataframe(pd.DataFrame(browser_rows), use_container_width=True, hide_index=True)
else:
    st.info("No non-English sentences detected with the current filter.")

st.markdown(
    "**Caveats:** This stopword-overlap method is a heuristic. "
    "Dialectal English like *'De boys up in de hayloft'* can be misclassified as Latin "
    "(because *de* is a Latin stopword). Cross-language homographs like *hat* (German) "
    "also cause false positives. These are useful failure cases to discuss."
)

# --- Compare multilingual density across chapters ---
st.subheader("Multilingual Density Across All Chapters")

with st.expander("Compute for all 18 episodes (slow on first run)"):
    if st.button("Compute All", key="compute_multilingual_all"):
        progress = st.progress(0)
        all_lang_data = []
        for idx, (fname, elabel) in enumerate(EPISODE_MAP.items()):
            text = cached_load_episode(fname)
            results_ep = compute_language_detection(text)
            lc = Counter(r["language"] for r in results_ep)
            total = len(results_ep)
            non_eng_pct = (total - lc.get("English", 0)) / total if total else 0
            all_lang_data.append({
                "Episode": elabel,
                "Sentences": total,
                "Non-English %": f"{non_eng_pct:.1%}",
                "French": lc.get("French", 0),
                "German": lc.get("German", 0),
                "Italian": lc.get("Italian", 0),
                "Latin": lc.get("Latin", 0),
            })
            progress.progress((idx + 1) / 18)

        df_all = pd.DataFrame(all_lang_data)
        # Highlight current episode
        st.dataframe(df_all, use_container_width=True, hide_index=True)


# ============================================================================
# Section 3: Morphological Analysis & Neologisms
# ============================================================================

st.header("3. Morphological Analysis & Neologisms")

st.markdown(
    "Joyce's Proteus is packed with unusual words — German borrowings, Greek loanwords, "
    "and massive compound coinages. WordNet knows some of them; for the rest, "
    "we attempt morphological decomposition."
)


@st.cache_data
def find_unusual_words(text, n=20):
    """Extract the N most unusual words from episode text.

    Unusual = not in WordNet AND not a common English stopword, sorted by length
    (longer words tend to be more interesting neologisms/compounds).
    Falls back to rare WordNet words (fewest synsets) if fewer than n are found.
    """
    tokens = word_tokenize(text)
    # Unique alphabetic tokens, lowercased, length > 4 to skip trivial words
    unique_words = set(t.lower() for t in tokens if t.isalpha() and len(t) > 4)
    stop = set(stopwords.words("english"))
    unique_words -= stop

    not_in_wn = []
    rare_in_wn = []  # (word, synset_count)
    for w in unique_words:
        syns = wordnet.synsets(w)
        if not syns:
            not_in_wn.append(w)
        elif len(syns) <= 2:
            rare_in_wn.append((w, len(syns)))

    # Sort by length descending — longer words are more interesting
    not_in_wn.sort(key=lambda w: -len(w))
    result = not_in_wn[:n]

    # Backfill with rare WordNet words if needed
    if len(result) < n:
        rare_in_wn.sort(key=lambda x: (x[1], -len(x[0])))
        for w, _ in rare_in_wn:
            if w not in result:
                result.append(w)
            if len(result) >= n:
                break

    return result[:n]


@st.cache_data
def analyze_words(words):
    """Analyze a list of words via WordNet + morphological decomposition."""
    results = []
    for word in words:
        synsets = wordnet.synsets(word.lower())
        if synsets:
            top = synsets[0]
            definition = top.definition()
            pos = top.pos()
            results.append({
                "word": word,
                "in_wordnet": True,
                "synset_count": len(synsets),
                "definition": definition,
                "pos": pos,
                "analysis": f"In WordNet ({len(synsets)} synsets): {definition}",
                "category": "In WordNet",
            })
        else:
            parse = hypothesize_parse(word)
            if "Compound" in parse:
                category = "Compound"
            elif "Prefix" in parse:
                category = "Prefixed"
            elif "German" in parse:
                category = "Foreign"
            else:
                category = "Sui Generis"
            results.append({
                "word": word,
                "in_wordnet": False,
                "synset_count": 0,
                "definition": "",
                "pos": "",
                "analysis": parse,
                "category": category,
            })
    return results


# --- The 20 most unusual words from this episode ---
st.subheader("Joyce's 20 Most Unusual Words")

unusual_words = find_unusual_words(episode_text, n=20)
word_analyses = analyze_words(unusual_words)

# Summary metrics
in_wn = sum(1 for w in word_analyses if w["in_wordnet"])
not_in_wn = len(word_analyses) - in_wn
categories = Counter(w["category"] for w in word_analyses)

wm1, wm2, wm3, wm4 = st.columns(4)
wm1.metric("In WordNet", f"{in_wn}/{len(word_analyses)}")
wm2.metric("Compounds Found", categories.get("Compound", 0))
wm3.metric("Foreign Words", categories.get("Foreign", 0))
wm4.metric("Sui Generis", categories.get("Sui Generis", 0))

# Color-coded table
category_colors = {
    "In WordNet": "#d4edda",
    "Compound": "#cce5ff",
    "Prefixed": "#fff3cd",
    "Foreign": "#f8d7da",
    "Sui Generis": "#e2e3e5",
}

table_rows = []
for w in word_analyses:
    table_rows.append({
        "Word": w["word"],
        "Category": w["category"],
        "Analysis": w["analysis"],
    })

df_words = pd.DataFrame(table_rows)
st.dataframe(df_words, use_container_width=True, hide_index=True)

# --- Category breakdown chart ---
fig_cat, ax_cat = plt.subplots(figsize=(8, 3))
all_categories = ["In WordNet", "Compound", "Prefixed", "Foreign", "Sui Generis"]
cat_labels = all_categories
cat_vals = [categories.get(c, 0) for c in all_categories]
cat_colors = [category_colors.get(c, "#e2e3e5") for c in cat_labels]
bars = ax_cat.barh(cat_labels, cat_vals, color=cat_colors, edgecolor="#666")
for bar, val in zip(bars, cat_vals):
    ax_cat.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontweight="bold")
ax_cat.set_xlabel("Number of Words")
ax_cat.set_title("How Joyce's unusual words break down")
plt.tight_layout()
st.pyplot(fig_cat)
plt.close(fig_cat)

# --- Interactive word analyzer ---
st.subheader("Analyze Any Word")

custom_word = st.text_input(
    "Enter a word to check WordNet coverage and attempt decomposition",
    value="metempsychosis",
)

if custom_word:
    word = custom_word.strip().lower()
    synsets = wordnet.synsets(word)

    if synsets:
        st.success(f"**{word}** is in WordNet with {len(synsets)} synset(s).")
        for i, syn in enumerate(synsets[:5]):
            st.markdown(f"- **{syn.name()}** ({syn.pos()}): {syn.definition()}")
            examples = syn.examples()
            if examples:
                st.markdown(f"  *Example:* {examples[0]}")
            # Show hypernym path
            paths = syn.hypernym_paths()
            if paths:
                chain = " > ".join(h.name().split(".")[0] for h in paths[0][-4:])
                st.markdown(f"  *Hypernym chain:* ...{chain}")
    else:
        st.warning(f"**{word}** is NOT in WordNet.")
        parse = hypothesize_parse(word)
        st.markdown(f"**Morphological analysis:** {parse}")

    # Always show stemmer results for context
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Porter stem", porter.stem(word))
    sc2.metric("Lancaster stem", lancaster.stem(word))
    sc3.metric("Snowball stem", snowball.stem(word))

# --- Compound word visualizer ---
st.subheader("Compound Word Splitter")

st.markdown(
    "Joyce coins compound words by fusing recognizable English parts. "
    "This tool tries every possible split point and checks both halves against WordNet."
)

compound_word = st.text_input(
    "Enter a compound to split",
    value="scrotumtightening",
    key="compound_input",
)

if compound_word and len(compound_word) >= 6:
    word_lower = compound_word.strip().lower()
    splits = []
    for i in range(3, len(word_lower) - 2):
        left = word_lower[:i]
        right = word_lower[i:]
        left_in = bool(wordnet.synsets(left))
        right_in = bool(wordnet.synsets(right))
        score = int(left_in) + int(right_in)
        splits.append({
            "Split Point": i,
            "Left": left,
            "Left in WN": left_in,
            "Right": right,
            "Right in WN": right_in,
            "Score": score,
        })

    df_splits = pd.DataFrame(splits)

    # Visualize split quality as a horizontal bar chart
    split_labels = [f"{s['Left']} | {s['Right']}" for s in splits]
    scores = [s["Score"] for s in splits]
    split_colors = ["#d4edda" if s == 2 else "#fff3cd" if s == 1 else "#f8d7da" for s in scores]

    fig_height = max(3, len(splits) * 0.35 + 1)
    fig_split, ax_split = plt.subplots(figsize=(8, fig_height))
    y_pos = range(len(splits))
    ax_split.barh(y_pos, scores, color=split_colors, edgecolor="#ccc")
    ax_split.set_yticks(y_pos)
    ax_split.set_yticklabels(split_labels, fontsize=8, family="monospace")
    ax_split.set_xticks([0, 1, 2])
    ax_split.set_xticklabels(["Neither", "One half", "Both halves"])
    ax_split.set_xlim(-0.1, 2.5)
    ax_split.set_title(f"All possible splits of '{word_lower}'")
    ax_split.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_split)
    plt.close(fig_split)

    # Show best splits as a table
    best = [s for s in splits if s["Score"] >= 1]
    if best:
        best.sort(key=lambda s: -s["Score"])
        st.dataframe(
            pd.DataFrame(best[:10]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No split produced a recognizable half. This word may be foreign or truly sui generis.")

st.markdown("""
---

**What this week reveals:** NLP tools assume linguistic stability — standardized spelling,
monolingual text, words that exist in dictionaries. Joyce deliberately violates all three
assumptions. Proteus is about metamorphosis, and Joyce's language is itself protean:
shifting between languages mid-sentence, coining compounds that no dictionary contains,
and using archaic or foreign forms that confuse even sophisticated stemmers.
The exercises teach students to see where rule-based NLP breaks — and why that breaking
is itself informative about the text.
""")
