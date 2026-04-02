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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import pi

for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "vader_lexicon",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "cmudict",
]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")
STOP_WORDS = set(stopwords.words("english"))

EPISODES = [
    ("01", "Telemachus", "01telemachus.txt"),
    ("02", "Nestor", "02nestor.txt"),
    ("03", "Proteus", "03proteus.txt"),
    ("04", "Calypso", "04calypso.txt"),
    ("05", "Lotus Eaters", "05lotuseaters.txt"),
    ("06", "Hades", "06hades.txt"),
    ("07", "Aeolus", "07aeolus.txt"),
    ("08", "Lestrygonians", "08lestrygonians.txt"),
    ("09", "Scylla & Charybdis", "09scyllacharybdis.txt"),
    ("10", "Wandering Rocks", "10wanderingrocks.txt"),
    ("11", "Sirens", "11sirens.txt"),
    ("12", "Cyclops", "12cyclops.txt"),
    ("13", "Nausicaa", "13nausicaa.txt"),
    ("14", "Oxen of the Sun", "14oxenofthesun.txt"),
    ("15", "Circe", "15circe.txt"),
    ("16", "Eumaeus", "16eumaeus.txt"),
]


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
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
    tagged = pos_tag(tokens)  # Process full text
    tag_counts = Counter(tag for _, tag in tagged)
    total_tags = sum(tag_counts.values())
    nouns = sum(c for t, c in tag_counts.items() if t.startswith("NN"))
    verbs = sum(c for t, c in tag_counts.items() if t.startswith("VB"))
    adjs = sum(c for t, c in tag_counts.items() if t.startswith("JJ"))

    # Sentiment
    sia = SentimentIntensityAnalyzer()
    sent_scores = [
        sia.polarity_scores(s)["compound"] for s in sentences
    ]  # Process all sentences
    mean_sent = sum(sent_scores) / len(sent_scores) if sent_scores else 0
    var_sent = (
        sum((s - mean_sent) ** 2 for s in sent_scores) / len(sent_scores)
        if sent_scores
        else 0
    )

    # Readability (Flesch-Kincaid approximation)
    total_syllables = 0
    pron_dict = cmudict.dict()
    for w in alpha_tokens:
        if w in pron_dict:
            total_syllables += len([ph for ph in pron_dict[w][0] if ph[-1].isdigit()])
        else:
            total_syllables += max(1, len(w) // 3)

    avg_syl = total_syllables / len(alpha_tokens) if alpha_tokens else 1
    avg_sl = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 1
    flesch_kincaid = 0.39 * avg_sl + 11.8 * avg_syl - 15.59

    # Named entity density approximation via POS tags (NNP/NNPS)
    proper_nouns = sum(c for t, c in tag_counts.items() if t in ("NNP", "NNPS"))
    entity_density = (proper_nouns / len(tokens)) * 1000 if tokens else 0

    metrics = {
        "total_tokens": len(tokens),
        "total_types": len(types),
        "ttr": len(types) / len(alpha_tokens) if alpha_tokens else 0,
        "hapax_ratio": hapax / len(tokens)
        if tokens
        else 0,  # Using total_tokens as denominator
        "avg_sent_len": avg_sl,
        "median_sent_len": (
            (
                sorted(sent_lengths)[len(sent_lengths) // 2 - 1]
                + sorted(sent_lengths)[len(sent_lengths) // 2]
            )
            / 2
            if len(sent_lengths) % 2 == 0 and len(sent_lengths) > 0
            else sorted(sent_lengths)[len(sent_lengths) // 2]
        )
        if sent_lengths
        else 0,
        "sent_len_std": (
            sum((l - avg_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
        )
        ** 0.5
        if sent_lengths
        else 0,
        "noun_verb_ratio": nouns / verbs if verbs else 0,
        "adj_density": adjs / total_tags * 100 if total_tags else 0,
        "vader_mean": mean_sent,
        "vader_var": var_sent,
        "avg_word_len": sum(len(w) for w in alpha_tokens) / len(alpha_tokens)
        if alpha_tokens
        else 0,
        "flesch_kincaid": flesch_kincaid,
        "exclamation_rate": text.count("!") / len(sentences) if sentences else 0,
        "comma_rate": text.count(",") / len(sentences) if sentences else 0,
        "entity_density": entity_density,  # Named entities per 1000 tokens
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
        metrics["episode"] = f"{ep_num}. {ep_name}"
        all_metrics.append(metrics)
        if metric_keys is None:
            metric_keys = [k for k in metrics.keys() if k != "episode"]

    # Print table
    print(f"\n--- Master Table ---")
    header = f"{'Episode':<25}" + "".join(f"{k[:12]:>14}" for k in metric_keys[:8])
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
    header2 = f"{'Episode':<25}" + "".join(f"{k[:12]:>14}" for k in metric_keys[8:])
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
        rank = next(
            i + 1 for i, (idx, _) in enumerate(sorted_vals) if idx == eumaeus_idx
        )
        print(
            f"  {k:<25}: rank {rank:>2}/16  (value: {all_metrics[eumaeus_idx][k]:.3f})"
        )

    return all_metrics, metric_keys


# ---------------------------------------------------------------------------
# Radar Chart Function
# ---------------------------------------------------------------------------


def create_radar_chart(all_metrics, metric_keys):
    """Create a radar chart comparing Telemachus, Sirens, Cyclops, and Eumaeus."""
    # Select episodes by name
    target_episodes = ["01. Telemachus", "11. Sirens", "12. Cyclops", "16. Eumaeus"]
    selected_metrics = [
        "ttr",
        "avg_sent_len",
        "flesch_kincaid",
        "vader_var",
        "noun_verb_ratio",
        "adj_density",
    ]

    # Prepare data
    angles = [
        n / float(len(selected_metrics)) * 2 * pi for n in range(len(selected_metrics))
    ]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, episode in enumerate(target_episodes):
        try:
            # Find the episode data
            episode_data = next(m for m in all_metrics if m["episode"] == episode)

            # Extract values for selected metrics
            values = [episode_data[metric] for metric in selected_metrics]
            values += values[:1]  # Complete the circle

            # Normalize values to 0-1 range for better comparison
            normalized_values = []
            for j, val in enumerate(values[:-1]):  # Exclude the duplicated last value
                metric = selected_metrics[j]
                all_vals = [m[metric] for m in all_metrics]
                min_val, max_val = min(all_vals), max(all_vals)
                norm_val = (
                    (val - min_val) / (max_val - min_val) if max_val != min_val else 0
                )
                normalized_values.append(norm_val)
            normalized_values.append(normalized_values[0])  # Complete the circle

            # Plot data
            ax.plot(
                angles,
                normalized_values,
                "o-",
                linewidth=2,
                label=episode.split(". ")[1],
                color=colors[i],
            )
            ax.fill(angles, normalized_values, alpha=0.25, color=colors[i])
        except Exception as e:
            print(f"Warning: Could not plot {episode}: {e}")
            continue

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected_metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    ax.set_title(
        "Episode Comparison: Telemachus, Sirens, Cyclops, Eumaeus", size=14, pad=20
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week16_radar_chart.png"), dpi=150
    )
    plt.close()


# ---------------------------------------------------------------------------
# Exercise 2: The Dashboard
# ---------------------------------------------------------------------------


def build_dashboard(all_metrics, metric_keys):
    """Build multi-panel visualization."""
    episodes = [m["episode"][:15] for m in all_metrics]
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
    im = ax.imshow(z_matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(
        [k[:10] for k in metric_keys], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(n_eps))
    ax.set_yticklabels(episodes, fontsize=8)
    ax.set_title("Ulysses: Episode × Metric Heatmap (z-scores)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "week16_heatmap.png"), dpi=150)
    plt.close()

    # Panel 2: Small multiples (sparklines)
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, n_metrics * 1.2))
    for j, k in enumerate(metric_keys):
        ax = axes[j]
        vals = [m[k] for m in all_metrics]
        ax.plot(range(n_eps), vals, "b-", linewidth=1)
        ax.fill_between(range(n_eps), vals, alpha=0.1)
        ax.set_ylabel(k[:8], fontsize=7, rotation=0, ha="right")
        ax.set_xlim(0, n_eps - 1)
        ax.tick_params(labelsize=6)
        if j < n_metrics - 1:
            ax.set_xticks([])

    axes[-1].set_xticks(range(n_eps))
    axes[-1].set_xticklabels(episodes, rotation=45, ha="right", fontsize=6)
    plt.suptitle("Metric Sparklines Across Episodes", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week16_sparklines.png"), dpi=150
    )
    plt.close()

    # Panel 3: Correlation matrix
    corr = np.corrcoef(z_matrix.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(
        [k[:10] for k in metric_keys], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(metric_keys)))
    ax.set_yticklabels([k[:10] for k in metric_keys], fontsize=7)
    ax.set_title("Metric Correlation Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "week16_correlation.png"), dpi=150
    )
    plt.close()

    # Create radar chart
    create_radar_chart(all_metrics, metric_keys)

    print("  Dashboard plots saved:")
    print("    week16_heatmap.png")
    print("    week16_sparklines.png")
    print("    week16_correlation.png")
    print("    week16_radar_chart.png")


# ---------------------------------------------------------------------------
# Exercise 3: The Error Audit
# ---------------------------------------------------------------------------


def error_audit(all_metrics, metric_keys):
    """Identify potential errors or misleading results from outlier detection."""
    print("\n--- Error Audit Framework ---")

    # Identify outliers using z-scores (>2 standard deviations from mean)
    outliers = []
    for metric in metric_keys:
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Skip metrics with no variation
        if std_val == 0:
            continue

        # Find episodes with extreme values
        for i, m in enumerate(all_metrics):
            z_score = abs((m[metric] - mean_val) / std_val)
            if z_score > 2.0:  # Outlier threshold
                outliers.append(
                    {
                        "episode": m["episode"],
                        "metric": metric,
                        "value": m[metric],
                        "z_score": z_score,
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

    # Group outliers by episode to identify problematic episodes
    episode_issues = {}
    for outlier in outliers:
        ep_name = outlier["episode"]
        if ep_name not in episode_issues:
            episode_issues[ep_name] = []
        episode_issues[ep_name].append(outlier)

    # Sort episodes by number of issues
    sorted_episodes = sorted(
        episode_issues.items(), key=lambda x: len(x[1]), reverse=True
    )

    print(f"{'Episode':<20} {'Issues Count':<15} {'Top Issue Metric'}")
    print("-" * 60)
    for ep_name, issues in sorted_episodes[:7]:  # Top 7 episodes with issues
        top_issue = max(issues, key=lambda x: x["z_score"])
        print(f"  {ep_name:<18} {len(issues):<15} {top_issue['metric']}")

    print(f"\n  Total detected outliers: {len(outliers)}")
    print(f"  Pattern: Errors concentrate in episodes with extreme metric values")
    print(f"  These episodes may require special consideration in analysis")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
    error_audit(all_metrics, metric_keys)
