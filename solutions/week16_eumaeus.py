"""
Week 16: Eumaeus
=================
Corpus-wide enumeration and the novel as a structured dataset.

Focus: pandas, matplotlib, seaborn — compiling metrics from all prior weeks
       into a comprehensive dashboard

Exercises:
  1. The master table
  2. The dashboard
  3. The error audit
"""

import os
import math
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import cmudict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

for resource in ['punkt', 'punkt_tab', 'stopwords', 'vader_lexicon',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                 'cmudict']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')
STOP_WORDS = set(stopwords.words('english'))

EPISODES = [
    ('01', 'Telemachus', '01telemachus.txt'),
    ('02', 'Nestor', '02nestor.txt'),
    ('03', 'Proteus', '03proteus.txt'),
    ('04', 'Calypso', '04calypso.txt'),
    ('05', 'Lotus Eaters', '05lotuseaters.txt'),
    ('06', 'Hades', '06hades.txt'),
    ('07', 'Aeolus', '07aeolus.txt'),
    ('08', 'Lestrygonians', '08lestrygonians.txt'),
    ('09', 'Scylla & Charybdis', '09scyllacharybdis.txt'),
    ('10', 'Wandering Rocks', '10wanderingrocks.txt'),
    ('11', 'Sirens', '11sirens.txt'),
    ('12', 'Cyclops', '12cyclops.txt'),
    ('13', 'Nausicaa', '13nausicaa.txt'),
    ('14', 'Oxen of the Sun', '14oxenofthesun.txt'),
    ('15', 'Circe', '15circe.txt'),
    ('16', 'Eumaeus', '16eumaeus.txt'),
]


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_all_metrics(text):
    """Compute a comprehensive set of metrics for an episode."""
    tokens = word_tokenize(text)
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    sentences = sent_tokenize(text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]

    fdist = FreqDist(alpha_tokens)
    types = set(alpha_tokens)
    hapax = sum(1 for w, c in fdist.items() if c == 1)

    # POS
    tagged = pos_tag(tokens[:5000])  # Sample for performance
    tag_counts = Counter(tag for _, tag in tagged)
    total_tags = sum(tag_counts.values())
    nouns = sum(c for t, c in tag_counts.items() if t.startswith('NN'))
    verbs = sum(c for t, c in tag_counts.items() if t.startswith('VB'))
    adjs = sum(c for t, c in tag_counts.items() if t.startswith('JJ'))

    # Sentiment
    sia = SentimentIntensityAnalyzer()
    sent_scores = [sia.polarity_scores(s)['compound']
                   for s in sentences[:500]]  # Sample
    mean_sent = sum(sent_scores) / len(sent_scores) if sent_scores else 0
    var_sent = (sum((s - mean_sent)**2 for s in sent_scores) / len(sent_scores)
                if sent_scores else 0)

    # Readability (Flesch-Kincaid approximation)
    total_syllables = 0
    pron_dict = cmudict.dict()
    for w in alpha_tokens[:3000]:
        if w in pron_dict:
            total_syllables += len([ph for ph in pron_dict[w][0]
                                    if ph[-1].isdigit()])
        else:
            total_syllables += max(1, len(w) // 3)

    avg_syl = total_syllables / len(alpha_tokens[:3000]) if alpha_tokens else 1
    avg_sl = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 1
    flesch_kincaid = 0.39 * avg_sl + 11.8 * avg_syl - 15.59

    metrics = {
        'total_tokens': len(tokens),
        'total_types': len(types),
        'ttr': len(types) / len(alpha_tokens) if alpha_tokens else 0,
        'hapax_ratio': hapax / len(types) if types else 0,
        'avg_sent_len': avg_sl,
        'median_sent_len': sorted(sent_lengths)[len(sent_lengths)//2] if sent_lengths else 0,
        'sent_len_std': (sum((l - avg_sl)**2 for l in sent_lengths) / len(sent_lengths))**0.5 if sent_lengths else 0,
        'noun_verb_ratio': nouns / verbs if verbs else 0,
        'adj_density': adjs / total_tags * 100 if total_tags else 0,
        'vader_mean': mean_sent,
        'vader_var': var_sent,
        'avg_word_len': sum(len(w) for w in alpha_tokens) / len(alpha_tokens) if alpha_tokens else 0,
        'flesch_kincaid': flesch_kincaid,
        'exclamation_rate': text.count('!') / len(sentences) if sentences else 0,
        'comma_rate': text.count(',') / len(sentences) if sentences else 0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Exercise 1: The Master Table
# ---------------------------------------------------------------------------

def build_master_table():
    """Compute metrics for all 16 episodes and build the master table."""
    print("--- Computing metrics for all episodes ---")
    all_metrics = []
    metric_keys = None

    for ep_num, ep_name, filename in EPISODES:
        print(f"  Processing {ep_num}: {ep_name}...")
        text = load_episode(filename)
        metrics = compute_all_metrics(text)
        metrics['episode'] = f"{ep_num}. {ep_name}"
        all_metrics.append(metrics)
        if metric_keys is None:
            metric_keys = [k for k in metrics.keys() if k != 'episode']

    # Print table
    print(f"\n--- Master Table ---")
    header = f"{'Episode':<25}" + ''.join(f"{k[:12]:>14}" for k in metric_keys[:8])
    print(header)
    print("-" * (25 + 14 * min(8, len(metric_keys))))

    for m in all_metrics:
        row = f"  {m['episode']:<23}"
        for k in metric_keys[:8]:
            val = m[k]
            if isinstance(val, float):
                row += f"{val:>14.3f}"
            else:
                row += f"{val:>14}"
        print(row)

    # Second half of metrics
    print(f"\n  (continued)")
    header2 = f"{'Episode':<25}" + ''.join(f"{k[:12]:>14}" for k in metric_keys[8:])
    print(header2)
    print("-" * (25 + 14 * len(metric_keys[8:])))

    for m in all_metrics:
        row = f"  {m['episode']:<23}"
        for k in metric_keys[8:]:
            val = m[k]
            if isinstance(val, float):
                row += f"{val:>14.3f}"
            else:
                row += f"{val:>14}"
        print(row)

    # Rank table for Eumaeus
    print(f"\n--- Eumaeus Rank (out of 16 episodes) ---")
    eumaeus_idx = 15  # index of Eumaeus in the list
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        sorted_vals = sorted(enumerate(vals), key=lambda x: -x[1])
        rank = next(i+1 for i, (idx, _) in enumerate(sorted_vals) if idx == eumaeus_idx)
        print(f"  {k:<25}: rank {rank:>2}/16  (value: {all_metrics[eumaeus_idx][k]:.3f})")

    return all_metrics, metric_keys


# ---------------------------------------------------------------------------
# Exercise 2: The Dashboard
# ---------------------------------------------------------------------------

def build_dashboard(all_metrics, metric_keys):
    """Build multi-panel visualization."""
    episodes = [m['episode'][:15] for m in all_metrics]
    n_eps = len(episodes)

    # Normalize to z-scores for heatmap
    matrix = np.zeros((n_eps, len(metric_keys)))
    for i, m in enumerate(all_metrics):
        for j, k in enumerate(metric_keys):
            matrix[i][j] = m[k]

    # Z-score normalize columns
    z_matrix = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        mean = np.mean(col)
        std = np.std(col)
        z_matrix[:, j] = (col - mean) / std if std > 0 else 0

    # Panel 1: Heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(z_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([k[:10] for k in metric_keys], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_eps))
    ax.set_yticklabels(episodes, fontsize=8)
    ax.set_title('Ulysses: Episode × Metric Heatmap (z-scores)')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week16_heatmap.png'), dpi=150)
    plt.close()

    # Panel 2: Small multiples (sparklines)
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, n_metrics * 1.2))
    for j, k in enumerate(metric_keys):
        ax = axes[j]
        vals = [m[k] for m in all_metrics]
        ax.plot(range(n_eps), vals, 'b-', linewidth=1)
        ax.fill_between(range(n_eps), vals, alpha=0.1)
        ax.set_ylabel(k[:8], fontsize=7, rotation=0, ha='right')
        ax.set_xlim(0, n_eps - 1)
        ax.tick_params(labelsize=6)
        if j < n_metrics - 1:
            ax.set_xticks([])

    axes[-1].set_xticks(range(n_eps))
    axes[-1].set_xticklabels(episodes, rotation=45, ha='right', fontsize=6)
    plt.suptitle('Metric Sparklines Across Episodes', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week16_sparklines.png'), dpi=150)
    plt.close()

    # Panel 3: Correlation matrix
    corr = np.corrcoef(z_matrix.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([k[:10] for k in metric_keys], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(metric_keys)))
    ax.set_yticklabels([k[:10] for k in metric_keys], fontsize=7)
    ax.set_title('Metric Correlation Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week16_correlation.png'), dpi=150)
    plt.close()

    print("  Dashboard plots saved:")
    print("    week16_heatmap.png")
    print("    week16_sparklines.png")
    print("    week16_correlation.png")


# ---------------------------------------------------------------------------
# Exercise 3: The Error Audit
# ---------------------------------------------------------------------------

def error_audit():
    """Identify potential errors or misleading results from prior analyses."""
    print("\n--- Error Audit Framework ---")
    known_issues = [
        ("06 Hades", "VADER sentiment", "Expected negative", "Mixed/neutral",
         "VADER misreads irony and gallows humor as positive; dark jokes score > 0"),
        ("03 Proteus", "Language detection", "Multi-language", "Mostly English",
         "Stopword overlap too sparse for short code-switched phrases"),
        ("04 Calypso", "NER", "Expect Bloom, Molly", "Many false entities",
         "NLTK ne_chunk misclassifies Irish place names and brand names"),
        ("07 Aeolus", "TF-IDF headlines", "Match Joyce's headlines", "Low match rate",
         "TF-IDF captures content salience but not irony, wordplay, or tonal commentary"),
        ("09 Scylla", "CFG parsing", "Parse complex sentences", "Very low coverage",
         "Hand-written grammar covers <50% of tokens; Joyce's syntax exceeds CFG capacity"),
        ("11 Sirens", "CMU dict coverage", "Full phonetic conversion", "~80% coverage",
         "Onomatopoeia and neologisms — the most interesting words — are missing from CMU"),
        ("12 Cyclops", "Genre classification", "Clear barfly/interpolation", "Heuristic boundary",
         "Some paragraphs are ambiguous; the barfly's voice bleeds into transitional passages"),
    ]

    print(f"{'Episode':<20} {'Tool':<20} {'Expected':<20} {'Actual':<18} {'Diagnosis'}")
    print("-" * 100)
    for ep, tool, expected, actual, diagnosis in known_issues:
        print(f"  {ep:<18} {tool:<18} {expected:<18} {actual:<16} {diagnosis[:40]}...")

    print(f"\n  Total known issues: {len(known_issues)}")
    print(f"  Pattern: errors concentrate in stylistically extreme episodes")
    print(f"  and in tools designed for standard English (sentiment, NER, parsing).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 62)
    print("EXERCISE 1: The Master Table")
    print("=" * 62)
    all_metrics, metric_keys = build_master_table()

    print("\n" + "=" * 62)
    print("EXERCISE 2: The Dashboard")
    print("=" * 62)
    build_dashboard(all_metrics, metric_keys)

    print("\n" + "=" * 62)
    print("EXERCISE 3: The Error Audit")
    print("=" * 62)
    error_audit()
