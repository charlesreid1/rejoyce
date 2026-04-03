"""
Week 04 — Calypso
Named entity recognition, noun phrase chunking, and entity co-occurrence.
"""

import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
from nltk.tree import Tree

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]:
    nltk.download(resource, quiet=True)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from dashboard.shared import (
    cached_load_episode,
    episode_sidebar,
    EPISODE_FILES,
    EPISODE_LABELS,
    EPISODE_MAP,
)

# --- Entity type color palette (consistent across all visualizations) ---
ENTITY_COLORS = {
    "PERSON": "#E07A5F",
    "GPE": "#4A90D9",
    "ORGANIZATION": "#F2CC8F",
    "FACILITY": "#81B29A",
    "GSP": "#4A90D9",
    "LOCATION": "#3D9970",
    "EVENT": "#B279A2",
}
ENTITY_COLOR_DEFAULT = "#B0B0B0"

st.set_page_config(page_title="Week 04 — Calypso", page_icon="📖", layout="wide")
st.title("Week 04 — Calypso")
st.caption("Named Entity Recognition, Noun Phrase Chunking & Entity Co-occurrence")

# --- Sidebar ---
episode_file, episode_label = episode_sidebar(
    default_index=3,  # Calypso
    caption="Week 4: NER, Chunking & Co-occurrence",
)

with st.sidebar:
    compare_label = st.selectbox(
        "Comparison episode",
        EPISODE_LABELS,
        index=0,  # Telemachus
        key="compare_episode",
    )
    compare_file = EPISODE_FILES[EPISODE_LABELS.index(compare_label)]
    st.divider()
    st.markdown(
        "**Week 4** explores how Named Entity Recognition reveals Bloom's "
        "concrete, cataloguing consciousness — tagging every person, street, "
        "and price he encounters. Chunking extracts the noun phrases that "
        "furnish his world, and co-occurrence analysis reconstructs "
        "narrative structure from entity associations."
    )
    st.divider()
    full_text = st.toggle(
        "Process full text",
        value=False,
        help="Off = first 100 sentences (fast). On = all sentences (slower, more accurate).",
    )

sentence_limit = None if full_text else 100

episode_text = cached_load_episode(episode_file)
compare_text = cached_load_episode(compare_file)


# ============================================================================
# Shared NLP pipeline — tokenize, POS-tag, NE-chunk ONCE per text
# ============================================================================


@st.cache_data
def _nlp_pipeline(text, limit=None):
    """Run the expensive tokenize → POS-tag → NE-chunk pipeline once.

    Returns (sentences, tagged_list, entities_per_sent, token_count).
    - sentences: list of sentence strings
    - tagged_list: list of [(word, tag), ...] per sentence
    - entities_per_sent: list of [(entity_text, entity_type), ...] per sentence
    - token_count: total word tokens
    """
    sentences = sent_tokenize(text)
    if limit:
        sentences = sentences[:limit]

    tagged_list = []
    entities_per_sent = []
    token_count = 0

    for sent in sentences:
        tokens = word_tokenize(sent)
        token_count += len(tokens)
        tagged = pos_tag(tokens)
        tagged_list.append(tagged)
        tree = ne_chunk(tagged)
        sent_ents = []
        for subtree in tree:
            if isinstance(subtree, Tree):
                ent_text = " ".join(w for w, t in subtree.leaves())
                ent_type = subtree.label()
                sent_ents.append((ent_text, ent_type))
        entities_per_sent.append(sent_ents)

    return sentences, tagged_list, entities_per_sent, token_count


# ============================================================================
# Cached NER / Chunking / Co-occurrence (all reuse _nlp_pipeline)
# ============================================================================


@st.cache_data
def extract_entities(text, limit=None):
    """Extract named entities using NLTK ne_chunk.

    Returns (entities, type_counts, token_count) where entities is a list
    of (entity_text, entity_type) tuples.
    """
    sentences, tagged_list, entities_per_sent, token_count = _nlp_pipeline(text, limit)
    entities = [e for sent_ents in entities_per_sent for e in sent_ents]
    type_counts = Counter(etype for _, etype in entities)
    return entities, type_counts, token_count


@st.cache_data
def extract_chunks(text, grammar_str, limit=None):
    """Extract noun phrases and prepositional phrases using a chunking grammar.

    Returns (np_freq, pp_freq) as Counters.
    """
    sentences, tagged_list, entities_per_sent, token_count = _nlp_pipeline(text, limit)
    parser = RegexpParser(grammar_str)

    np_freq = Counter()
    pp_freq = Counter()

    for tagged in tagged_list:
        tree = parser.parse(tagged)

        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join(w.lower() for w, t in subtree.leaves())
                if (
                    len(phrase) >= 2
                    and phrase[0].isalpha()
                    and phrase not in {"t", "s", "d", "ll", "ve", "re"}
                ):
                    np_freq[phrase] += 1
            elif subtree.label() == "PP":
                phrase = " ".join(w.lower() for w, t in subtree.leaves())
                if len(phrase) >= 3 and any(c.isalpha() for c in phrase):
                    pp_freq[phrase] += 1

    return np_freq, pp_freq


@st.cache_data
def extract_cooccurrence(text, limit=None):
    """Build entity co-occurrence matrix by paragraph.

    Reuses the shared NLP pipeline and maps pre-tagged sentences back to
    paragraphs by character position, avoiding redundant pos_tag/ne_chunk calls.

    Returns (cooccurrence Counter, entity_paragraphs dict, paragraph_count).
    """
    sentences, tagged_list, entities_per_sent, token_count = _nlp_pipeline(text, limit)

    # Build paragraph structure
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 5:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Precompute paragraph character ranges
    para_ranges = []
    search = 0
    for para in paragraphs:
        idx = text.find(para, search)
        if idx >= 0:
            para_ranges.append((idx, idx + len(para)))
            search = idx + len(para)
        else:
            para_ranges.append((-1, -1))

    # Map each sentence to its paragraph by character position
    sent_para = []
    search = 0
    for sent in sentences:
        idx = text.find(sent, search)
        if idx < 0:
            idx = text.find(sent)
        para_idx = -1
        if idx >= 0:
            for pi, (ps, pe) in enumerate(para_ranges):
                if ps <= idx < pe:
                    para_idx = pi
                    break
            search = idx + len(sent)
        sent_para.append(para_idx)

    # Group entities by paragraph
    para_entity_sets = defaultdict(set)
    for si, sent_ents in enumerate(entities_per_sent):
        pi = sent_para[si]
        if pi >= 0:
            for ent_text, ent_type in sent_ents:
                para_entity_sets[pi].add(ent_text)

    entity_paragraphs = defaultdict(list)
    cooccurrence = Counter()

    for pi, entities_in_para in para_entity_sets.items():
        for entity in entities_in_para:
            entity_paragraphs[entity].append(pi)

        entity_list = sorted(entities_in_para)
        for j in range(len(entity_list)):
            for k in range(j + 1, len(entity_list)):
                pair = (entity_list[j], entity_list[k])
                cooccurrence[pair] += 1

    # Use the effective paragraph range (highest paragraph index reached),
    # not the full text paragraph count, so trajectories scale correctly
    # when sentence_limit restricts processing to early text.
    max_para_idx = max(sent_para) if sent_para else 0
    effective_para_count = max_para_idx + 1 if max_para_idx >= 0 else len(paragraphs)

    return dict(cooccurrence), dict(entity_paragraphs), effective_para_count


# ============================================================================
# Section 1: NER as Characterization
# ============================================================================

st.header("1. NER as Characterization")

st.markdown(
    "Named Entity Recognition is the technology of noticing what Bloom notices: "
    "proper nouns, places, organizations — the specific, nameable furniture of the world. "
    "Compare NER density across episodes to test the hypothesis that Bloom's chapter "
    "is entity-dense (places/things) while Stephen's is entity-sparse (persons)."
)

entities_primary, types_primary, tokens_primary = extract_entities(
    episode_text, limit=sentence_limit
)
entities_compare, types_compare, tokens_compare = extract_entities(
    compare_text, limit=sentence_limit
)

density_primary = len(entities_primary) / tokens_primary * 1000 if tokens_primary else 0
density_compare = len(entities_compare) / tokens_compare * 1000 if tokens_compare else 0

# --- Metrics row ---
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Named Entities",
    len(entities_primary),
    delta=f"{len(entities_primary) - len(entities_compare):+d} vs {compare_label.split(' — ')[1]}",
)
m2.metric(
    "Entities / 1k Tokens",
    f"{density_primary:.1f}",
    delta=f"{density_primary - density_compare:+.1f}",
)
m3.metric("Entity Types", len(types_primary))
dominant = types_primary.most_common(1)[0][0] if types_primary else "—"
m4.metric("Dominant Type", dominant)

# --- Entity type distribution: grouped bar chart ---
st.subheader("Entity Type Distribution")

all_types = sorted(set(list(types_primary.keys()) + list(types_compare.keys())))

fig_types, ax_types = plt.subplots(figsize=(10, max(4, len(all_types) * 0.5)))
y = np.arange(len(all_types))
bar_h = 0.35
vals_primary = [types_primary.get(t, 0) for t in all_types]
vals_compare = [types_compare.get(t, 0) for t in all_types]

bars1 = ax_types.barh(y - bar_h / 2, vals_primary, bar_h, label=episode_label, color="#E07A5F")
bars2 = ax_types.barh(y + bar_h / 2, vals_compare, bar_h, label=compare_label, color="#4A90D9")

for bar, val in zip(bars1, vals_primary):
    if val > 0:
        ax_types.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      str(val), va="center", fontsize=8)
for bar, val in zip(bars2, vals_compare):
    if val > 0:
        ax_types.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      str(val), va="center", fontsize=8)

ax_types.set_yticks(y)
ax_types.set_yticklabels(all_types)
ax_types.set_xlabel("Count")
ax_types.set_title("Entity Type Distribution: Two Modes of Consciousness")
ax_types.legend()
plt.tight_layout()
st.pyplot(fig_types)
plt.close(fig_types)

# --- Entity type radar chart ---
st.subheader("Entity Type Fingerprint")

if len(all_types) >= 3:
    angles = np.linspace(0, 2 * np.pi, len(all_types), endpoint=False).tolist()
    angles += angles[:1]

    # Normalize each dimension independently so no single high-count type dominates
    norm_primary = []
    norm_compare = []
    for vp, vc in zip(vals_primary, vals_compare):
        dim_max = max(vp, vc, 1)
        norm_primary.append(vp / dim_max)
        norm_compare.append(vc / dim_max)
    norm_primary += norm_primary[:1]
    norm_compare += norm_compare[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, norm_primary, "o-", linewidth=2, label=episode_label, color="#E07A5F")
    ax_radar.fill(angles, norm_primary, alpha=0.15, color="#E07A5F")
    ax_radar.plot(angles, norm_compare, "o-", linewidth=2, label=compare_label, color="#4A90D9")
    ax_radar.fill(angles, norm_compare, alpha=0.15, color="#4A90D9")

    # Label each axis with the type name and the per-dimension max
    dim_labels = [
        f"{t}\n(max {max(vp, vc)})"
        for t, vp, vc in zip(all_types, vals_primary, vals_compare)
    ]
    ax_radar.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], dim_labels, fontsize=9)
    ax_radar.set_ylim(0, 1.1)
    ax_radar.set_yticklabels([])
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    ax_radar.set_title("Entity Fingerprint (per-dimension normalization)", pad=20)
    st.pyplot(fig_radar)
    plt.close(fig_radar)

st.markdown(
    "Each axis is normalized independently — 1.0 = whichever episode has more of that type. "
    "This reveals the *shape* of each episode's entity profile without high-count types "
    "like GPE drowning out the rest. Bloom's Calypso should bulge toward GPE and FACILITY "
    "(places, buildings), while Stephen's Telemachus should lean toward PERSON."
)

# --- Entity browser ---
st.subheader("Entity Browser")

filter_type = st.selectbox(
    "Filter by entity type",
    ["All"] + all_types,
    key="entity_filter",
)

browser_entities = entities_primary
if filter_type != "All":
    browser_entities = [(e, t) for e, t in entities_primary if t == filter_type]

if browser_entities:
    # Deduplicate and count
    entity_counter = Counter(browser_entities)
    browser_rows = [
        {"Entity": e, "Type": t, "Count": c}
        for (e, t), c in entity_counter.most_common()
    ]
    st.dataframe(pd.DataFrame(browser_rows), use_container_width=True, hide_index=True)
else:
    st.info("No entities of that type found.")

# ============================================================================
# Section 2: Noun Phrase Chunking
# ============================================================================

st.header("2. Noun Phrase Chunking")

st.markdown(
    "Chunking extracts meaningful multi-word phrases from tagged text. "
    "Bloom's thoughts organize the world into concrete noun phrases — "
    "*the kidney*, *the cat*, *the bed*, *the letter* — rather than "
    "Stephen's abstract verb-heavy meditations. "
    "The top NPs capture the episode's domestic texture."
)

# --- Controls ---
chunk_ctrl1, chunk_ctrl2 = st.columns([1, 1])
with chunk_ctrl1:
    chunk_top_n = st.slider("Top N phrases", 10, 50, 25, key="chunk_top_n")
with chunk_ctrl2:
    show_pp = st.toggle("Show prepositional phrases", value=True, key="show_pp")

phrase_search = st.text_input(
    "Filter phrases containing...",
    value="",
    key="phrase_search",
    help="Type a word to show only NPs/PPs containing that word (e.g., 'cat', 'bed')",
)

# --- Grammar explorer ---
st.subheader("Chunking Grammar")

DEFAULT_GRAMMAR = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    PP: {<IN><NP>}
"""

grammar_str = st.text_area(
    "Edit the chunking grammar (RegexpParser format)",
    value=DEFAULT_GRAMMAR.strip(),
    height=100,
    key="grammar_input",
    help="NP captures noun phrases, PP captures prepositional phrases. Modify and re-run!",
)

try:
    np_freq, pp_freq = extract_chunks(episode_text, grammar_str, limit=sentence_limit)
    grammar_valid = True
except Exception as e:
    st.error(f"Grammar error: {e}")
    grammar_valid = False
    np_freq, pp_freq = Counter(), Counter()

if grammar_valid:
    # Also compute with default grammar for comparison if user modified it
    if grammar_str.strip() != DEFAULT_GRAMMAR.strip():
        np_freq_default, pp_freq_default = extract_chunks(
            episode_text, DEFAULT_GRAMMAR, limit=sentence_limit
        )
        delta_np = len(np_freq) - len(np_freq_default)
        delta_pp = len(pp_freq) - len(pp_freq_default)
        gm1, gm2 = st.columns(2)
        gm1.metric("Unique NPs", len(np_freq), delta=f"{delta_np:+d} vs default")
        gm2.metric("Unique PPs", len(pp_freq), delta=f"{delta_pp:+d} vs default")
    else:
        gm1, gm2 = st.columns(2)
        gm1.metric("Unique NPs", len(np_freq))
        gm2.metric("Unique PPs", len(pp_freq))

    # Apply phrase search filter
    if phrase_search:
        search_lower = phrase_search.strip().lower()
        np_filtered = Counter({p: c for p, c in np_freq.items() if search_lower in p})
        pp_filtered = Counter({p: c for p, c in pp_freq.items() if search_lower in p})
    else:
        np_filtered = np_freq
        pp_filtered = pp_freq

    # --- NP bar chart ---
    st.subheader("Top Noun Phrases")

    top_nps = np_filtered.most_common(chunk_top_n)

    if top_nps:
        # Color-code: entity-bearing NPs in gold, others in teal
        entity_texts_lower = {e.lower() for e, _ in entities_primary}

        fig_np, ax_np = plt.subplots(figsize=(10, max(4, len(top_nps) * 0.3)))
        np_labels = [p for p, _ in top_nps]
        np_counts = [c for _, c in top_nps]
        np_colors = [
            "#F2CC8F" if any(ent in phrase for ent in entity_texts_lower)
            else "#4A9D8E"
            for phrase in np_labels
        ]
        ax_np.barh(range(len(np_labels)), np_counts, color=np_colors)
        ax_np.set_yticks(range(len(np_labels)))
        ax_np.set_yticklabels(np_labels, fontsize=8)
        ax_np.invert_yaxis()
        ax_np.set_xlabel("Frequency")
        ax_np.set_title(f"Top {len(top_nps)} Noun Phrases — {episode_label}")

        # Legend
        from matplotlib.patches import Patch

        ax_np.legend(
            handles=[
                Patch(facecolor="#F2CC8F", label="Contains named entity"),
                Patch(facecolor="#4A9D8E", label="Descriptive"),
            ],
            loc="lower right",
            fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig_np)
        plt.close(fig_np)
    else:
        st.info("No noun phrases match your filter.")

    # --- PP bar chart ---
    if show_pp:
        st.subheader("Top Prepositional Phrases")

        top_pps = pp_filtered.most_common(chunk_top_n)

        if top_pps:
            SPATIAL_PREPS = {"in", "on", "at", "to", "from", "into", "through", "across",
                             "along", "over", "under", "beside", "near", "between"}
            TEMPORAL_PREPS = {"after", "before", "during", "until", "since"}

            fig_pp, ax_pp = plt.subplots(figsize=(10, max(4, len(top_pps) * 0.3)))
            pp_labels = [p for p, _ in top_pps]
            pp_counts = [c for _, c in top_pps]

            def pp_color(phrase):
                first_word = phrase.split()[0] if phrase.split() else ""
                if first_word in SPATIAL_PREPS:
                    return "#4A90D9"
                elif first_word in TEMPORAL_PREPS:
                    return "#50C878"
                return "#B0B0B0"

            pp_colors = [pp_color(p) for p in pp_labels]
            ax_pp.barh(range(len(pp_labels)), pp_counts, color=pp_colors)
            ax_pp.set_yticks(range(len(pp_labels)))
            ax_pp.set_yticklabels(pp_labels, fontsize=8)
            ax_pp.invert_yaxis()
            ax_pp.set_xlabel("Frequency")
            ax_pp.set_title(f"Top {len(top_pps)} Prepositional Phrases — {episode_label}")

            ax_pp.legend(
                handles=[
                    Patch(facecolor="#4A90D9", label="Spatial (in, on, at, to, ...)"),
                    Patch(facecolor="#50C878", label="Temporal (after, before, ...)"),
                    Patch(facecolor="#B0B0B0", label="Other"),
                ],
                loc="lower right",
                fontsize=8,
            )
            plt.tight_layout()
            st.pyplot(fig_pp)
            plt.close(fig_pp)

            st.markdown(
                "Bloom's Dublin is a world of things *in* places. "
                "Spatial prepositions (blue) should dominate, "
                "reflecting his concrete, locating consciousness."
            )
        else:
            st.info("No prepositional phrases match your filter.")

    # --- Phrase-in-context browser ---
    st.subheader("Phrase in Context")

    phrase_options = [p for p, _ in np_filtered.most_common(50)]
    if phrase_options:
        selected_phrase = st.selectbox("Select a noun phrase to see in context", phrase_options)

        @st.cache_data
        def find_phrase_contexts(text, phrase, max_contexts=5):
            """Find sentences containing the given phrase."""
            sentences = sent_tokenize(text)
            matches = []
            phrase_lower = phrase.lower()
            for sent in sentences:
                if phrase_lower in sent.lower():
                    # Bold the phrase in the sentence
                    import re

                    highlighted = re.sub(
                        re.escape(phrase),
                        f"**{phrase}**",
                        sent,
                        flags=re.IGNORECASE,
                    )
                    matches.append(highlighted)
                    if len(matches) >= max_contexts:
                        break
            return matches

        contexts = find_phrase_contexts(episode_text, selected_phrase)
        if contexts:
            for ctx in contexts:
                st.markdown(f"- {ctx}")
        else:
            st.info("Phrase not found in sentence-level context.")

    # --- NP comparison across episodes ---
    with st.expander("Compare NP inventory with another episode"):
        np_compare, pp_compare = extract_chunks(compare_text, grammar_str, limit=sentence_limit)
        top_np_primary = np_filtered.most_common(chunk_top_n)
        top_np_compare = np_compare.most_common(chunk_top_n)

        if top_np_primary and top_np_compare:
            # Overlay chart
            all_phrases = list(dict.fromkeys(
                [p for p, _ in top_np_primary[:15]] + [p for p, _ in top_np_compare[:15]]
            ))

            fig_cmp, ax_cmp = plt.subplots(figsize=(10, max(4, len(all_phrases) * 0.35)))
            y_pos = np.arange(len(all_phrases))
            bar_h = 0.35

            vals_ep = [np_filtered.get(p, 0) for p in all_phrases]
            vals_cmp = [np_compare.get(p, 0) for p in all_phrases]

            ax_cmp.barh(y_pos - bar_h / 2, vals_ep, bar_h, label=episode_label, color="#E07A5F")
            ax_cmp.barh(y_pos + bar_h / 2, vals_cmp, bar_h, label=compare_label, color="#4A90D9",
                        alpha=0.7)

            ax_cmp.set_yticks(y_pos)
            ax_cmp.set_yticklabels(all_phrases, fontsize=7)
            ax_cmp.invert_yaxis()
            ax_cmp.set_xlabel("Frequency")
            ax_cmp.set_title("NP Inventory Comparison")
            ax_cmp.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

            st.markdown(
                "Compare Calypso's domestic NPs (*the cat*, *the kitchen*) with "
                "Telemachus's social NPs (*the old milkwoman*, *Buck Mulligan*). "
                "The NP inventory is a fingerprint of each character's consciousness."
            )


# ============================================================================
# Section 3: Entity Co-occurrence & Narrative Structure
# ============================================================================

st.header("3. Entity Co-occurrence & Narrative Structure")

st.markdown(
    "Which named entities appear together in the same paragraphs? "
    "Co-occurrence analysis reconstructs narrative structure from entity associations — "
    "Molly linked to the bed and the letter, Dlugacz to the porkbutcher and the kidney. "
    "Track entity trajectories to trace Bloom's movement through space."
)

cooc_dict, entity_paras, para_count = extract_cooccurrence(
    episode_text, limit=sentence_limit
)
cooccurrence = Counter(cooc_dict)
entity_paragraphs = entity_paras

n_entities = st.slider("Top entities to track", 5, 15, 8, key="n_entities")
min_cooc = st.slider("Min co-occurrence for network", 1, 5, 1, key="min_cooc")

# Sorted entities by paragraph frequency
sorted_entities = sorted(entity_paragraphs.items(), key=lambda x: -len(x[1]))
top_entity_names = [e for e, _ in sorted_entities[:n_entities]]

# --- Entity frequency table ---
st.subheader("Entity Frequency by Paragraph")

freq_rows = []
for entity, paras in sorted_entities[:20]:
    freq_rows.append({
        "Entity": entity,
        "Paragraphs": len(paras),
        "% of Episode": f"{len(paras) / para_count * 100:.1f}%" if para_count else "0%",
    })
st.dataframe(pd.DataFrame(freq_rows), use_container_width=True, hide_index=True)

# --- Entity trajectory plot ---
st.subheader("Entity Trajectory")

entity_select = st.multiselect(
    "Entities to highlight",
    [e for e, _ in sorted_entities[:30]],
    default=top_entity_names,
    key="entity_select",
)

if entity_select:
    fig_traj, ax_traj = plt.subplots(figsize=(14, max(4, len(entity_select) * 0.6)))

    rng = np.random.default_rng(42)
    for i, entity in enumerate(entity_select):
        paras = entity_paragraphs.get(entity, [])
        positions = [p / para_count * 100 for p in paras] if para_count else []
        jitter = rng.uniform(-0.25, 0.25, size=len(positions))
        ax_traj.scatter(
            positions, [i + j for j in jitter],
            c="#333333", s=60, alpha=0.9, edgecolors="none", zorder=3,
        )

    ax_traj.set_yticks(range(len(entity_select)))
    ax_traj.set_yticklabels(entity_select, fontsize=10)
    ax_traj.set_xlabel("Position in episode (%)")
    ax_traj.set_xlim(0, 100)
    ax_traj.set_title(f"Entity Trajectory — {episode_label}")
    plt.tight_layout()
    st.pyplot(fig_traj)
    plt.close(fig_traj)

    st.markdown(
        "Each tick mark shows a paragraph where the entity appears. "
        "Trace Bloom's morning: Molly at the start (bed), Dlugacz in the middle "
        "(trip to butcher), and the letter's anxiety arriving later."
    )

# --- Co-occurrence heatmap ---
st.subheader("Entity Co-occurrence Heatmap")

heatmap_entities = entity_select if entity_select else top_entity_names

if len(heatmap_entities) >= 2:
    n = len(heatmap_entities)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            pair = tuple(sorted([heatmap_entities[i], heatmap_entities[j]]))
            count = cooccurrence.get(pair, 0)
            matrix[i][j] = count
            matrix[j][i] = count

    fig_heat, ax_heat = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    im = ax_heat.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(heatmap_entities, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(heatmap_entities, fontsize=8)

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                ax_heat.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=9)

    fig_heat.colorbar(im, ax=ax_heat, label="Shared paragraphs")
    ax_heat.set_title("Entity Co-occurrence (shared paragraphs)")
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)


# --- Top co-occurring pairs table ---
st.subheader("Top Co-occurring Entity Pairs")

top_pairs = cooccurrence.most_common(15)
if top_pairs:
    pair_rows = [
        {"Entity A": e1, "Entity B": e2, "Shared Paragraphs": count}
        for (e1, e2), count in top_pairs
    ]
    st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

# --- Entity network graph ---
st.subheader("Entity Network Graph")

if not NETWORKX_AVAILABLE:
    st.warning("Install `networkx` for the network graph: `pip install networkx`")
else:
    filtered_cooc = {pair: count for pair, count in cooccurrence.items() if count >= min_cooc}

    if filtered_cooc:
        G = nx.Graph()
        for (e1, e2), count in filtered_cooc.items():
            G.add_edge(e1, e2, weight=count)

        if len(G.nodes()) > 0:
            fig_net, ax_net = plt.subplots(figsize=(10, 8))

            # Lay out each connected component separately, then tile them
            components = list(nx.connected_components(G))
            # Sort largest first
            components.sort(key=len, reverse=True)

            pos = {}
            n_comp = len(components)
            cols = max(1, int(np.ceil(np.sqrt(n_comp))))
            rows = max(1, int(np.ceil(n_comp / cols)))

            for idx, comp in enumerate(components):
                subgraph = G.subgraph(comp)
                # Layout within a unit box
                if len(comp) == 1:
                    sub_pos = {list(comp)[0]: np.array([0.0, 0.0])}
                elif len(comp) == 2:
                    nodes = list(comp)
                    sub_pos = {nodes[0]: np.array([-0.3, 0.0]),
                               nodes[1]: np.array([0.3, 0.0])}
                else:
                    sub_pos = nx.kamada_kawai_layout(subgraph)

                # Offset to grid cell
                grid_row = idx // cols
                grid_col = idx % cols
                spacing = 2.5
                offset = np.array([grid_col * spacing, -grid_row * spacing])

                for node, p in sub_pos.items():
                    pos[node] = p + offset

            node_sizes = [
                len(entity_paragraphs.get(node, [])) * 80 + 200
                for node in G.nodes()
            ]

            entity_type_map = {}
            for ent_text, ent_type in entities_primary:
                entity_type_map[ent_text] = ent_type
            node_colors = [
                ENTITY_COLORS.get(entity_type_map.get(node, ""), ENTITY_COLOR_DEFAULT)
                for node in G.nodes()
            ]

            edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1

            nx.draw_networkx_edges(G, pos,
                                   width=[w / max_weight * 3 + 0.5 for w in edge_weights],
                                   alpha=0.25, edge_color="#999999", ax=ax_net)
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                   alpha=0.9, edgecolors="#333333", linewidths=1, ax=ax_net)
            nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif",
                                    bbox=dict(facecolor="white", edgecolor="none",
                                              alpha=0.8, pad=1.5),
                                    ax=ax_net)

            edge_labels = {(u, v): str(G[u][v]["weight"])
                           for u, v in G.edges() if G[u][v]["weight"] >= 2}
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                             font_size=8, ax=ax_net)

            ax_net.set_title(f"Entity Co-occurrence Network — {episode_label}")
            ax_net.axis("off")

            from matplotlib.patches import Patch
            legend_types = sorted(set(entity_type_map.get(n, "") for n in G.nodes()) - {""})
            if legend_types:
                legend_handles = [
                    Patch(facecolor=ENTITY_COLORS.get(t, ENTITY_COLOR_DEFAULT), label=t)
                    for t in legend_types
                ]
                ax_net.legend(handles=legend_handles, loc="upper left", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_net)
            plt.close(fig_net)

            st.markdown(
                "Node size reflects how many paragraphs an entity appears in. "
                "Edge width and labels show co-occurrence strength. "
                "Node color indicates entity type: "
                "PERSON (coral), GPE (blue), ORGANIZATION (gold), FACILITY (green)."
            )
        else:
            st.info("No entities meet the minimum co-occurrence threshold.")
    else:
        st.info("No co-occurring pairs found at this threshold. Try lowering the minimum.")

# --- Compare narrative structures ---
with st.expander(f"Compare narrative structure with {compare_label}"):
    cooc_cmp_dict, entity_paras_cmp, para_count_cmp = extract_cooccurrence(
        compare_text, limit=sentence_limit
    )
    # Only include entities appearing in 2+ paragraphs (single occurrences are noise)
    recurring_cmp = [(e, p) for e, p in entity_paras_cmp.items() if len(p) >= 2]
    recurring_cmp.sort(key=lambda x: -len(x[1]))
    top_cmp = [e for e, _ in recurring_cmp[:15]]

    if top_cmp:
        fig_cmp_traj, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5))

        # Primary episode trajectory
        rng_l = np.random.default_rng(42)
        for i, entity in enumerate(entity_select[:15]):
            paras = entity_paragraphs.get(entity, [])
            positions = [p / para_count * 100 for p in paras] if para_count else []
            jitter = rng_l.uniform(-0.25, 0.25, size=len(positions))
            ax_left.scatter(
                positions, [i + j for j in jitter],
                c="#333333", s=60, alpha=0.9, edgecolors="none", zorder=3,
            )
        ax_left.set_yticks(range(min(15, len(entity_select))))
        ax_left.set_yticklabels(entity_select[:15], fontsize=9)
        ax_left.set_xlabel("Position (%)")
        ax_left.set_xlim(0, 100)
        ax_left.set_title(f"{episode_label}")

        # Comparison episode trajectory
        rng_r = np.random.default_rng(42)
        for i, entity in enumerate(top_cmp):
            paras = entity_paras_cmp.get(entity, [])
            positions = [p / para_count_cmp * 100 for p in paras] if para_count_cmp else []
            jitter = rng_r.uniform(-0.25, 0.25, size=len(positions))
            ax_right.scatter(
                positions, [i + j for j in jitter],
                c="#333333", s=60, alpha=0.9, edgecolors="none", zorder=3,
            )
        ax_right.set_yticks(range(len(top_cmp)))
        ax_right.set_yticklabels(top_cmp, fontsize=9)
        ax_right.set_xlabel("Position (%)")
        ax_right.set_xlim(0, 100)
        ax_right.set_title(f"{compare_label}")

        plt.tight_layout()
        st.pyplot(fig_cmp_traj)
        plt.close(fig_cmp_traj)

        st.markdown(
            "Entities are ranked by paragraph frequency (most frequent at top). "
            "Look for the difference in spread: entities that recur across the full "
            "episode vs. those that cluster in a single scene."
        )

st.markdown("""
---

**What this week reveals:** Named Entity Recognition is the technology of noticing what
Bloom notices. Where Stephen's Telemachus is entity-sparse and person-heavy — his opening
chapter orbits around Buck Mulligan and Haines — Bloom's Calypso is entity-dense, tagging
every person, street, and price. The chunking grammar captures his domestic inventory
(*the kidney*, *the cat*, *the bed*), and the co-occurrence network reveals narrative
structure through association: Molly linked to the bed and the letter, Dlugacz to the
porkbutcher. Bloom's mind is an NER engine running at full tilt.
""")
