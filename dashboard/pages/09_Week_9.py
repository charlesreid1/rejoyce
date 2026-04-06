"""
Week 09 — Scylla and Charybdis
Context-free grammars and syntactic parsing.
"""

import contextlib
import io
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.grammar import CFG

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "treebank",
]:
    nltk.download(resource, quiet=True)

from week09.week09_scyllacharybdis import (
    ARGUMENT_GRAMMAR,
    find_argument_sentences,
    expand_cfg_lexicon,
    parse_with_cfg,
    create_ambiguous_test_sentences,
    treebank_statistics,
    episode_complexity,
    extract_quotations,
    compare_quotation_syntax,
)

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

st.set_page_config(page_title="Week 09 — Scylla & Charybdis", page_icon="📖", layout="wide")
st.title("Week 09 — Scylla & Charybdis")
st.caption("Context-Free Grammars and Syntactic Parsing")


# ============================================================================
# Helpers
# ============================================================================

def suppress_stdout(func, *args, **kwargs):
    """Call a function that prints to stdout and suppress its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def capture_tree_string(tree):
    """Capture tree.pretty_print() output as a string."""
    buf = io.StringIO()
    tree.pretty_print(stream=buf)
    return buf.getvalue()


def classify_quotation(text_content, quotation):
    """Classify how a quotation was extracted."""
    # Check for em-dash dialogue
    lines = text_content.split("\n")
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith("\u2014") or stripped.startswith("—")) and quotation in stripped:
            return "Em-dash dialogue"

    # Check for various quote types
    if f'"{quotation}"' in text_content:
        return "Double-quoted"
    if f"'{quotation}'" in text_content:
        return "Single-quoted"
    if f"\u201c{quotation}\u201d" in text_content:
        return "Double-quoted"
    if f"\u2018{quotation}\u2019" in text_content:
        return "Single-quoted"
    if f"*{quotation}*" in text_content:
        return "Italicized"

    return "Other"


# ============================================================================
# Cached computations
# ============================================================================

@st.cache_data
def cached_find_argument_sentences(text, connectives_tuple):
    connectives = set(connectives_tuple) if connectives_tuple else None
    return find_argument_sentences(text, connectives=connectives)


@st.cache_data
def cached_treebank_statistics():
    return suppress_stdout(treebank_statistics)


@st.cache_data
def cached_episode_complexity(text, label="Episode"):
    return suppress_stdout(episode_complexity, text, label=label)


@st.cache_data
def cached_extract_quotations(text):
    return extract_quotations(text)


@st.cache_data
def cached_expand_cfg_lexicon(text):
    return expand_cfg_lexicon(text, ARGUMENT_GRAMMAR)


@st.cache_data
def cached_parse_with_cfg(sentence, _grammar_str, text):
    """Parse a sentence. _grammar_str is used as cache key."""
    grammar = CFG.fromstring(_grammar_str)
    expanded = expand_cfg_lexicon(text, grammar)
    tokens = word_tokenize(sentence.lower())
    terminals = set()
    for prod in expanded.productions():
        if prod.is_lexical():
            terminals.add(prod.rhs()[0])
    filtered = [t for t in tokens if t in terminals]
    if len(filtered) < 3:
        return [], tokens, filtered
    from nltk.parse.chart import ChartParser
    parser = ChartParser(expanded)
    try:
        trees = list(parser.parse(filtered))
    except Exception:
        trees = []
    return trees, tokens, filtered


# ============================================================================
# Sidebar
# ============================================================================

episode_file, episode_label = episode_sidebar(
    default_index=8,  # Scylla & Charybdis
    caption="Week 9: CFGs & Syntactic Parsing",
)

with st.sidebar:
    compare_label = st.selectbox(
        "Compare to (Section 2)",
        [lbl for lbl in EPISODE_LABELS if lbl != episode_label],
        index=EPISODE_LABELS.index("04 — Calypso") if episode_label != "04 — Calypso" else 0,
        key="compare_episode",
    )
    compare_file = EPISODE_FILES[EPISODE_LABELS.index(compare_label)]
    st.divider()
    st.markdown(
        "**Scylla and Charybdis** is the dialectical library episode where Stephen "
        "presents his Shakespeare theory. Its dense, argumentative prose makes it ideal "
        "for exploring how computational parsing reveals the hierarchical structure "
        "of complex literary language."
    )

# Load data
episode_text = cached_load_episode(episode_file)
compare_text = cached_load_episode(compare_file)


# ============================================================================
# Section 1: Parsing the Argument
# ============================================================================

st.header("1. Parsing the Argument")

st.markdown(
    "A hand-written context-free grammar (CFG) specifies rules like `S -> NP VP` "
    "to describe how sentences decompose into constituents. Here we explore how such "
    "a grammar handles (or fails to handle) Joyce's argumentative prose — and what "
    "structural ambiguity looks like when multiple valid parse trees exist."
)

# --- Connective customization ---
DEFAULT_CONNECTIVES = [
    "therefore", "because", "if", "but", "yet",
    "however", "thus", "hence", "since", "although",
]

selected_connectives = st.multiselect(
    "Logical connectives to search for",
    DEFAULT_CONNECTIVES + ["nevertheless", "moreover", "whereas", "unless", "so", "still"],
    default=DEFAULT_CONNECTIVES,
    key="connectives",
)

connectives_tuple = tuple(sorted(selected_connectives)) if selected_connectives else None
argument_sents = cached_find_argument_sentences(episode_text, connectives_tuple)

st.metric("Argument sentences found", len(argument_sents))

# --- Argument sentence table ---
if argument_sents:
    sent_data = []
    for s in argument_sents:
        tokens = word_tokenize(s)
        sent_data.append({
            "Sentence": s[:120] + ("..." if len(s) > 120 else ""),
            "Tokens": len(tokens),
        })
    st.dataframe(pd.DataFrame(sent_data), use_container_width=True, hide_index=True)

# --- Grammar inspector ---
with st.expander("View/Edit CFG Grammar"):
    # Convert grammar to a round-trippable string (str() adds an unparseable header)
    grammar_str = "\n".join(str(p) for p in ARGUMENT_GRAMMAR.productions())
    st.code(grammar_str, language=None)

    custom_rules = st.text_area(
        "Add custom lexical rules (one per line, e.g. `NN -> 'theory' | 'argument'`)",
        value="",
        key="custom_grammar",
        height=100,
    )

# Build the effective grammar string
if custom_rules.strip():
    effective_grammar_str = grammar_str + "\n" + custom_rules.strip()
else:
    effective_grammar_str = grammar_str

# --- Sentence selector + parser ---
if argument_sents:
    st.subheader("Parse a Sentence")

    sent_options = [s[:80] + ("..." if len(s) > 80 else "") for s in argument_sents[:30]]
    selected_sent_label = st.selectbox("Select an argument sentence", sent_options, key="parse_sent")
    selected_sent_idx = sent_options.index(selected_sent_label)
    selected_sent = argument_sents[selected_sent_idx]

    st.markdown(f"**Full sentence:** {selected_sent}")

    trees, tokens, filtered = cached_parse_with_cfg(
        selected_sent, effective_grammar_str, episode_text
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total tokens", len(tokens))
    m2.metric("Grammar-covered tokens", len(filtered))
    coverage = len(filtered) / len(tokens) * 100 if tokens else 0
    m3.metric("Coverage", f"{coverage:.1f}%")

    if trees:
        if len(trees) > 1:
            st.warning(f"{len(trees)} valid parses — ambiguity!")
        else:
            st.success("1 valid parse found")

        # Show first tree
        tree_str = capture_tree_string(trees[0])
        st.code(tree_str, language=None)

        if len(trees) > 1:
            with st.expander(f"View all {len(trees)} parse trees"):
                for i, tree in enumerate(trees[:5]):
                    st.markdown(f"**Parse {i+1}**")
                    st.code(capture_tree_string(tree), language=None)
    else:
        # Show covered vs uncovered tokens
        terminals = set()
        try:
            grammar = CFG.fromstring(effective_grammar_str)
            expanded = expand_cfg_lexicon(episode_text, grammar)
            for prod in expanded.productions():
                if prod.is_lexical():
                    terminals.add(prod.rhs()[0])
        except Exception:
            pass

        st.info(
            "No complete parse tree found — the grammar cannot fully "
            "describe this sentence's structure. This is expected: our "
            "hand-written CFG only covers common English constructions, "
            "while Joyce's prose pushes far beyond standard syntax.\n\n"
            "The token display below shows which individual words the "
            "grammar recognises (**green**) versus words it has no rule "
            "for (**red**). Adding lexical rules in the grammar editor "
            "above can increase coverage."
        )

        colored_tokens = []
        for t in tokens:
            if t.lower() in terminals:
                colored_tokens.append(f'<span style="color: green; font-weight: bold;">{t}</span>')
            else:
                colored_tokens.append(f'<span style="color: red;">{t}</span>')
        st.markdown(
            "**Token coverage** (green = recognised by grammar, red = no matching rule): "
            + " ".join(colored_tokens),
            unsafe_allow_html=True,
        )

# --- Grammar coverage heatmap ---
if argument_sents:
    st.subheader("Grammar Coverage Heatmap")

    top_sents = argument_sents[:20]
    coverage_data = []
    for s in top_sents:
        _, toks, filt = cached_parse_with_cfg(s, effective_grammar_str, episode_text)
        n_tok = len(toks)
        n_cov = len(filt)
        coverage_data.append({
            "sentence": s[:60] + ("..." if len(s) > 60 else ""),
            "covered": n_cov,
            "uncovered": n_tok - n_cov,
            "total": n_tok,
        })

    if coverage_data:
        fig_cov, ax_cov = plt.subplots(figsize=(10, max(4, len(coverage_data) * 0.35)))
        labels = [d["sentence"] for d in coverage_data]
        covered = [d["covered"] for d in coverage_data]
        uncovered = [d["uncovered"] for d in coverage_data]
        y_pos = range(len(labels))

        ax_cov.barh(y_pos, covered, color="#4A9D8E", label="Covered")
        ax_cov.barh(y_pos, uncovered, left=covered, color="#CCCCCC", label="Uncovered")
        ax_cov.set_yticks(y_pos)
        ax_cov.set_yticklabels(labels, fontsize=7)
        ax_cov.invert_yaxis()
        ax_cov.set_xlabel("Tokens")
        ax_cov.set_title("Grammar Coverage — Top 20 Argument Sentences")
        ax_cov.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cov)
        plt.close(fig_cov)

# --- Ambiguity explorer ---
st.subheader("Ambiguity Explorer")

st.markdown(
    "These test sentences demonstrate structural ambiguity — a single string can "
    "have multiple valid syntactic structures under the same grammar."
)

test_sentences = create_ambiguous_test_sentences()
custom_test = st.text_input(
    "Add a custom test sentence",
    value="",
    key="custom_test_sent",
)
if custom_test.strip():
    test_sentences = test_sentences + [custom_test.strip()]

for sent in test_sentences:
    trees, tokens, filtered = cached_parse_with_cfg(sent, effective_grammar_str, episode_text)

    if trees:
        if len(trees) > 1:
            st.warning(f'**"{sent}"** — {len(trees)} competing readings (structural ambiguity)')
            cols = st.columns(min(len(trees), 3))
            for i, tree in enumerate(trees[:3]):
                with cols[i]:
                    st.markdown(f"**Parse {i+1}**")
                    st.code(capture_tree_string(tree), language=None)
        else:
            st.success(f'**"{sent}"** — 1 parse')
            st.code(capture_tree_string(trees[0]), language=None)
    else:
        st.info(f'**"{sent}"** — no parse (coverage: {len(filtered)}/{len(tokens)} tokens)')


# ============================================================================
# Section 2: Syntactic Complexity — Treebank vs. Joyce
# ============================================================================

st.header("2. Syntactic Complexity — Treebank vs. Joyce")

st.markdown(
    "The Penn Treebank provides baseline statistics for 'normal' English syntax. "
    "Comparing these against Joyce's episodes reveals how his dialectical prose "
    "differs measurably from standard English."
)

# --- Treebank baseline ---
tb_stats = cached_treebank_statistics()

st.subheader("Penn Treebank Baseline")
tb1, tb2, tb3, tb4 = st.columns(4)
tb1.metric("Avg Tree Depth", f"{tb_stats['avg_depth']:.2f}")
tb2.metric("Max Tree Depth", tb_stats["max_depth"])
tb3.metric("Avg Branching Factor", f"{tb_stats['avg_branching']:.2f}")
tb4.metric("SBAR / Sentence", f"{tb_stats['sbar_per_sentence']:.2f}")

# --- Episode complexity ---
ep_stats = cached_episode_complexity(episode_text, label=episode_label)
cmp_stats = cached_episode_complexity(compare_text, label=compare_label)

st.subheader(f"Episode Complexity: {episode_label}")
ec1, ec2, ec3, ec4 = st.columns(4)
ec1.metric(
    "Mean Sentence Length",
    f"{ep_stats['mean_sent_len']:.1f}",
    delta=f"{ep_stats['mean_sent_len'] - cmp_stats['mean_sent_len']:+.1f} vs {compare_label.split(' — ')[1]}",
)
ec2.metric(
    "Max Sentence Length",
    ep_stats["max_sent_len"],
    delta=f"{ep_stats['max_sent_len'] - cmp_stats['max_sent_len']:+d} vs {compare_label.split(' — ')[1]}",
)
ec3.metric(
    "Sub Conj / Sentence",
    f"{ep_stats['sub_conj_per_sent']:.2f}",
    delta=f"{ep_stats['sub_conj_per_sent'] - cmp_stats['sub_conj_per_sent']:+.2f}",
)
ec4.metric(
    "Commas / Sentence",
    f"{ep_stats['comma_per_sent']:.2f}",
    delta=f"{ep_stats['comma_per_sent'] - cmp_stats['comma_per_sent']:+.2f}",
)

# --- Sentence length distribution ---
st.subheader("Sentence Length Distribution")

# Tab20 palette reordered: bold shades first (even indices), then light shades (odd)
_tab20 = plt.cm.tab20.colors
TAB20_BOLD_FIRST = [_tab20[i] for i in range(0, 20, 2)] + [_tab20[i] for i in range(1, 20, 2)]

_default_dist_labels = [episode_label, compare_label]
selected_dist_labels = st.multiselect(
    "Episodes to compare",
    EPISODE_LABELS,
    default=_default_dist_labels,
    key="dist_episodes",
)

if selected_dist_labels:
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    all_max = 0
    episode_lengths = []
    for idx, lbl in enumerate(selected_dist_labels):
        fname = EPISODE_FILES[EPISODE_LABELS.index(lbl)]
        txt = cached_load_episode(fname)
        lengths = [len(word_tokenize(s)) for s in sent_tokenize(txt)]
        episode_lengths.append((lbl, lengths))
        if lengths:
            all_max = max(all_max, max(lengths))

    bins = np.arange(0, min(all_max + 5, 150), 5)
    for idx, (lbl, lengths) in enumerate(episode_lengths):
        color = TAB20_BOLD_FIRST[idx % len(TAB20_BOLD_FIRST)]
        short = lbl.split(" — ")[1] if " — " in lbl else lbl
        ax_hist.hist(lengths, bins=bins, alpha=0.55, color=color, label=short)
        if lengths:
            ax_hist.axvline(np.mean(lengths), color=color, linestyle="--", alpha=0.8,
                            label=f"Mean ({short})")

    ax_hist.set_xlabel("Sentence Length (tokens)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Sentence Length Distribution")
    ax_hist.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)
else:
    st.info("Select at least one episode above.")

# --- All-episodes complexity table ---
with st.expander("Compare all 18 episodes"):
    if st.button("Compute all episodes", key="compute_all"):
        all_rows = []
        for fname, lbl in EPISODE_MAP.items():
            txt = cached_load_episode(fname)
            stats = cached_episode_complexity(txt, label=lbl)
            all_rows.append({
                "Episode": lbl,
                "Mean Sent Len": f"{stats['mean_sent_len']:.1f}",
                "Max Sent Len": stats["max_sent_len"],
                "Sub Conj Rate": f"{stats['sub_conj_per_sent']:.2f}",
                "Comma Density": f"{stats['comma_per_sent']:.2f}",
            })

        df_all = pd.DataFrame(all_rows)
        st.dataframe(df_all, use_container_width=True, hide_index=True)
    else:
        st.info("Click the button to compute complexity metrics for all 18 episodes.")


# ============================================================================
# Section 3: The Quotation Problem
# ============================================================================

st.header("3. The Quotation Problem")

st.markdown(
    "Scylla and Charybdis is saturated with quoted material — Shakespeare quotations, "
    "dialogue, attributed speech. This section extracts and analyzes the syntactic "
    "differences between quoted material and Stephen's framing prose."
)

quotations = cached_extract_quotations(episode_text)

st.metric("Quotations found", len(quotations))

# --- Quotation table ---
if quotations:
    q_data = []
    type_counts = Counter()
    for q in quotations:
        q_type = classify_quotation(episode_text, q)
        type_counts[q_type] += 1
        q_data.append({
            "Quotation": q[:100] + ("..." if len(q) > 100 else ""),
            "Words": len(q.split()),
            "Type": q_type,
        })
    st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)

    # --- Quotation type breakdown ---
    st.subheader("Quotation Type Breakdown")
    fig_types, ax_types = plt.subplots(figsize=(8, 4))
    types = list(type_counts.keys())
    counts = list(type_counts.values())
    colors = ["#E07A5F", "#4A90D9", "#81B29A", "#F2CC8F", "#CCCCCC"]
    ax_types.barh(types, counts, color=colors[:len(types)])
    ax_types.set_xlabel("Count")
    ax_types.set_title("Quotation Extraction Methods")
    plt.tight_layout()
    st.pyplot(fig_types)
    plt.close(fig_types)

    # --- POS comparison: Quoted vs. Framing ---
    st.subheader("POS Comparison: Quoted vs. Framing Prose")

    quote_text = " ".join(quotations)
    frame_text = episode_text
    for q in quotations:
        frame_text = frame_text.replace(q, "")

    if quote_text.strip() and frame_text.strip():
        quote_tags = Counter(tag for _, tag in pos_tag(word_tokenize(quote_text)))
        frame_tags = Counter(tag for _, tag in pos_tag(word_tokenize(frame_text)))

        qt = sum(quote_tags.values())
        ft = sum(frame_tags.values())

        # Focus on major POS tags
        major_tags = ["NN", "NNS", "NNP", "VB", "VBD", "VBZ", "VBN", "VBG",
                      "JJ", "RB", "IN", "DT", "PRP", "CC"]
        tag_data = []
        for tag in major_tags:
            qp = 100 * quote_tags.get(tag, 0) / qt if qt else 0
            fp = 100 * frame_tags.get(tag, 0) / ft if ft else 0
            diff = qp - fp
            tag_data.append({"tag": tag, "quoted": qp, "framing": fp, "diff": diff})

        # Key metrics
        nn_tags = ["NN", "NNS", "NNP"]
        vb_tags = ["VB", "VBD", "VBZ", "VBN", "VBG"]

        noun_q = sum(100 * quote_tags.get(t, 0) / qt for t in nn_tags) if qt else 0
        noun_f = sum(100 * frame_tags.get(t, 0) / ft for t in nn_tags) if ft else 0
        verb_q = sum(100 * quote_tags.get(t, 0) / qt for t in vb_tags) if qt else 0
        verb_f = sum(100 * frame_tags.get(t, 0) / ft for t in vb_tags) if ft else 0
        prp_q = 100 * quote_tags.get("PRP", 0) / qt if qt else 0
        prp_f = 100 * frame_tags.get("PRP", 0) / ft if ft else 0

        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Noun % diff", f"{noun_q - noun_f:+.1f}%")
        pm2.metric("Verb % diff", f"{verb_q - verb_f:+.1f}%")
        pm3.metric("Pronoun % diff", f"{prp_q - prp_f:+.1f}%")

        # Grouped bar chart
        fig_pos, ax_pos = plt.subplots(figsize=(12, 6))
        x = np.arange(len(tag_data))
        width = 0.35

        quoted_vals = [d["quoted"] for d in tag_data]
        framing_vals = [d["framing"] for d in tag_data]
        tag_labels = [d["tag"] for d in tag_data]
        diffs = [abs(d["diff"]) for d in tag_data]

        ax_pos.bar(x - width / 2, quoted_vals, width, color="#E07A5F", label="Quoted")
        ax_pos.bar(x + width / 2, framing_vals, width, color="#4A9D8E", label="Framing")

        # Star tags with >2pp difference
        for i, diff in enumerate(diffs):
            if diff > 2:
                ax_pos.annotate(
                    "*",
                    (x[i], max(quoted_vals[i], framing_vals[i]) + 0.3),
                    ha="center", fontsize=14, fontweight="bold",
                )

        ax_pos.set_xticks(x)
        ax_pos.set_xticklabels(tag_labels)
        ax_pos.set_ylabel("Percentage")
        ax_pos.set_title("POS Distribution: Quoted vs. Framing Prose (* = >2pp difference)")
        ax_pos.legend()
        plt.tight_layout()
        st.pyplot(fig_pos)
        plt.close(fig_pos)

    # --- Quotation browser ---
    st.subheader("Quotation Browser")

    q_options = [q[:80] + ("..." if len(q) > 80 else "") for q in quotations[:50]]
    selected_q_label = st.selectbox("Select a quotation", q_options, key="q_browser")
    selected_q_idx = q_options.index(selected_q_label)
    selected_q = quotations[selected_q_idx]

    # POS-tag and display with color coding
    q_tokens = word_tokenize(selected_q)
    q_tagged = pos_tag(q_tokens)

    POS_COLORS = {
        "NN": "#4A90D9", "NNS": "#4A90D9", "NNP": "#4A90D9", "NNPS": "#4A90D9",
        "VB": "#E07A5F", "VBD": "#E07A5F", "VBZ": "#E07A5F", "VBN": "#E07A5F",
        "VBG": "#E07A5F", "VBP": "#E07A5F",
        "JJ": "#81B29A", "JJR": "#81B29A", "JJS": "#81B29A",
        "RB": "#F2CC8F", "RBR": "#F2CC8F", "RBS": "#F2CC8F",
    }

    colored_q = []
    for word, tag in q_tagged:
        color = POS_COLORS.get(tag, "#888888")
        colored_q.append(f'<span style="color: {color}; font-weight: bold;" title="{tag}">{word}</span>')
    st.markdown(" ".join(colored_q), unsafe_allow_html=True)

    st.caption(
        '<span style="color: #4A90D9;">Nouns</span> | '
        '<span style="color: #E07A5F;">Verbs</span> | '
        '<span style="color: #81B29A;">Adjectives</span> | '
        '<span style="color: #F2CC8F;">Adverbs</span> | '
        '<span style="color: #888888;">Other</span>',
        unsafe_allow_html=True,
    )

    # Mini POS frequency table for this quotation
    q_pos_counts = Counter(tag for _, tag in q_tagged)
    q_pos_df = pd.DataFrame([
        {"POS": tag, "Count": count}
        for tag, count in q_pos_counts.most_common()
    ])
    st.dataframe(q_pos_df, use_container_width=True, hide_index=True)

    # --- Word clouds / frequency bars ---
    st.subheader("Vocabulary: Quoted vs. Framing")

    try:
        from wordcloud import WordCloud

        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            st.markdown("**Quoted Material**")
            wc_q = WordCloud(width=400, height=300, background_color="white", colormap="Reds")
            wc_q.generate(quote_text)
            fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 4))
            ax_wc1.imshow(wc_q, interpolation="bilinear")
            ax_wc1.axis("off")
            st.pyplot(fig_wc1)
            plt.close(fig_wc1)

        with wc_col2:
            st.markdown("**Framing Prose**")
            wc_f = WordCloud(width=400, height=300, background_color="white", colormap="Blues")
            wc_f.generate(frame_text)
            fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 4))
            ax_wc2.imshow(wc_f, interpolation="bilinear")
            ax_wc2.axis("off")
            st.pyplot(fig_wc2)
            plt.close(fig_wc2)

    except ImportError:
        # Fallback: top-20 frequency bars
        q_words = [w.lower() for w in word_tokenize(quote_text) if w.isalpha() and len(w) > 2]
        f_words = [w.lower() for w in word_tokenize(frame_text) if w.isalpha() and len(w) > 2]

        q_freq = Counter(q_words).most_common(20)
        f_freq = Counter(f_words).most_common(20)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Quoted Material — Top 20 Words**")
            if q_freq:
                fig_fq, ax_fq = plt.subplots(figsize=(6, 5))
                ax_fq.barh(
                    [w for w, _ in reversed(q_freq)],
                    [c for _, c in reversed(q_freq)],
                    color="#E07A5F",
                )
                ax_fq.set_xlabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig_fq)
                plt.close(fig_fq)

        with fc2:
            st.markdown("**Framing Prose — Top 20 Words**")
            if f_freq:
                fig_ff, ax_ff = plt.subplots(figsize=(6, 5))
                ax_ff.barh(
                    [w for w, _ in reversed(f_freq)],
                    [c for _, c in reversed(f_freq)],
                    color="#4A9D8E",
                )
                ax_ff.set_xlabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig_ff)
                plt.close(fig_ff)

else:
    st.info("No quotations found in this episode.")


st.markdown("""
---

**What this week reveals:** A hand-written CFG can parse simple English sentences, but Joyce's
argumentative prose quickly exposes its limits — most tokens fall outside the grammar's vocabulary,
and structurally complex sentences resist decomposition into clean constituent trees. The gap
between Treebank norms and Joyce's syntax quantifies what makes *Scylla and Charybdis* feel
dense: longer sentences, more subordination, higher comma density. And the quotation problem
shows that even extracting quoted material is non-trivial — different extraction methods find
different material, and the syntactic profile of Shakespeare's language differs measurably
from Stephen's modernist framing prose.
""")
