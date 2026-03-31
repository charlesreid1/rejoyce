# Week 16: Eumaeus
### *"Preparatory to anything else Mr Bloom brushed off the greater bulk of the shavings and handed Stephen the hat and ashplant and bucked him up generally in orthodox Samaritan fashion, which he very badly needed" — Exhausted prose, exhaustive data, and the novel as spreadsheet.*

After the hallucinatory violence of Circe, Bloom and Stephen stumble into a cabman's shelter near Butt Bridge for coffee and a roll. They are exhausted. The prose is exhausted too — deliberately, magnificently so. The narrator of Eumaeus writes like a man fighting sleep: sentences lose their syntax halfway through, clichés pile up unexamined ("the cup that cheers but not inebriates"), malapropisms breed ("the idol with feet of clay"), mixed metaphors crash into each other, qualifications and hedges smother every claim ("so to speak," "as it were," "in a manner of speaking," "to put it mildly"). It is the worst-written episode in Ulysses, and every sentence is perfectly calculated in its imperfection. The technique is *narrative (old)*; the art is *navigation*; the organ is the *nerves*. The prose has the nervy, fumbling quality of someone trying to tell a story when they can barely keep their eyes open.

**This Week's Focus:** Corpus-wide enumeration and the novel as a structured dataset — compiling, tabulating, and visualizing 15 episodes' worth of metrics into a comprehensive analytical dashboard (`pandas`, `matplotlib`, `seaborn`, `plotly`; NLTK as the extraction layer)

**Pairing Rationale:**
Eumaeus is the episode of *navigation* — Bloom steering the wrecked Stephen homeward through the small hours — and by Week 16, students need navigation too. They have processed 15 episodes. They have computed token counts, type-token ratios, POS distributions, named entity inventories, sentiment scores, phonetic densities, stylometric profiles, classification outputs, and network statistics. These numbers exist in scattered notebooks and one-off analyses. Eumaeus is the moment to stop, take inventory, and *enumerate*: to gather every metric from every week into a single structured dataset (one row per episode, one column per metric) and build the visualizations that make the novel's computational anatomy legible at a glance.

The thematic fit is precise. Eumaeus is the episode where Bloom recounts to Stephen the events of the day — a tired, imprecise, error-prone *summary* of the novel so far. The student's dashboard is the computational version of the same act: a summary of everything measured so far, presented not as narrative but as data. And just as Bloom's summary is unreliable (he hedges, misremembers, elides), the dashboard will reveal where the metrics are unreliable too — where sentiment analysis failed (Week 6), where NER missed entities (Week 4), where the style classifier confused genres (Week 12). The dashboard is not a victory lap; it is an honest accounting.

Eumaeus itself, as the novel's measurable trough — the episode where every quality metric bottoms out — becomes the case study that justifies the exercise. If your dashboard doesn't show Eumaeus as an anomaly, something is wrong with your dashboard.

**Core Exercises:**

1. **The master table.** Construct a `pandas` DataFrame with one row per episode (1–15) and at minimum the following columns, drawn from prior weeks' analyses:

   | Metric | Source Week |
   |---|---|
   | Total tokens | 1 |
   | Total types (unique words) | 1 |
   | Type-token ratio | 1 |
   | Hapax legomena ratio | 1 |
   | Average sentence length (words) | 1 |
   | Median sentence length | 1 |
   | Sentence length standard deviation | 1 |
   | Top-3 POS tags by frequency | 2 |
   | Noun-to-verb ratio | 2 |
   | Adjective density (adj per 100 words) | 2 |
   | Unique lemmas count | 2 |
   | Stemmer disagreement rate (Porter vs. Lancaster) | 3 |
   | Non-English token proportion | 3 |
   | Named entities per 1,000 tokens | 4 |
   | Entity type distribution (PERSON/GPE/ORG) | 4 |
   | Average WordNet synset count per content word | 5 |
   | VADER mean compound sentiment | 6 |
   | VADER sentiment variance | 6 |
   | Top-3 TF-IDF keywords | 7 |
   | Bigram perplexity (self-trained) | 8 |
   | Mean parse tree depth | 9 |
   | Graph density (entity co-occurrence network) | 10 |
   | Alliteration density | 11 |
   | Most-informative classification features | 12 |
   | Burrows' Delta from Bloom baseline | 13 |
   | Readability score (Flesch-Kincaid) | new |
   | Cliché density (n-gram overlap with reference) | new |

   Now add Eumaeus as row 16. Compute every metric fresh. Where does Eumaeus rank in each column? Build a **rank table**: for each metric, rank all 16 episodes. Eumaeus should appear near the extremes on multiple measures — longest average sentence length, lowest type-token ratio (the same tired words recycled), highest cliché density, most hedging language. Identify the 5 metrics where Eumaeus is the most extreme outlier. These are the quantitative signatures of Joyce's "bad" prose.

2. **The dashboard.** Build a multi-panel visualization of the master table. At minimum:
   - A **heatmap** of the full episodes × metrics matrix (normalized to z-scores), making the novel's structural rhythm visible: the Bloom/Stephen alternation, the stylistic escalation toward Circe, the Eumaeus trough.
   - A **small multiples** display: one sparkline per metric across all 16 episodes, so you can scan for patterns, correlations, and outliers at a glance.
   - A **radar chart** (spider plot) comparing the metric profiles of 4 representative episodes: Telemachus (conventional), Sirens (musical), Cyclops (multi-register), and Eumaeus (exhausted). The shapes should be visually distinct.
   - A **correlation matrix** across metrics: which measures move together? (Hypothesis: sentiment variance and entity density are correlated — emotionally complex episodes tend to be entity-dense. Test it.)

   Use `plotly` for interactivity or `matplotlib`/`seaborn` for publication-quality statics. The dashboard should be something you'd want to print and hang on a wall.

3. **The error audit.** For each prior week's analysis, identify one metric or result that you now believe was wrong or misleading — a sentiment score that missed irony, an NER extraction that hallucinated entities, a classification that confused registers. Compile these into an **error catalog**: a structured table with columns for episode, metric, expected value, actual value, and diagnosis. This is the most important exercise of the week. Every computational analysis contains errors; the question is whether you can find them. Visualize the error distribution: are errors concentrated in certain episodes (the stylistically extreme ones) or certain tools (sentiment analysis, NER)? Does the error rate increase as the novel gets stranger? Plot it.

**Diving Deeper:**

- Dimensionality reduction (PCA, t-SNE, UMAP) can project the high-dimensional episode × metric matrix into 2D space, placing episodes that are metrically similar near each other. Run PCA on your master table and plot the episodes in the first two principal components. Do the clusters make literary sense? Does PCA separate Bloom episodes from Stephen episodes? Early episodes from late ones? Realistic from experimental?
- The "badness" of Eumaeus is a critical commonplace, but your metrics can make it precise. Compute a composite "prose quality" score by combining readability, TTR, cliché density, and sentence completion rate (proportion of sentences that parse successfully). Rank all episodes. Is Eumaeus really the worst, or does another episode (Oxen's Anglo-Saxon section? Circe's stage directions?) score lower on some measures? Badness is multidimensional.
- Benford's Law (the distribution of leading digits in naturally occurring numerical data) has been applied to literary texts as a test of "naturalness." Extract all numbers from Ulysses and test whether their leading digits follow Benford's distribution. If they do, Joyce's numerical imagination mirrors reality; if they don't, it's a specific kind of departure. See Cerioli et al. (2019) on Benford's Law in fiction.
- The master table is itself a dataset that can be analyzed with machine learning. Train a regression model to predict an episode's position in the novel (its episode number, 1–18) from its metric profile alone. How predictable is the novel's ordering from its quantitative features? If you shuffle the episodes, can the model reconstruct the sequence? This tests whether Ulysses has a measurable *direction* — a quantitative arc that the metrics capture.
- Connection to Week 15 (Circe): the entity graph from Circe should appear as a dramatic outlier in graph density, node count, and connectivity. If it doesn't dominate the graph metrics column of your master table, revisit your Circe extraction. Connection to Week 17 (Ithaca): that episode will demand the opposite of enumeration — not cataloging what you've already computed but extracting new structured knowledge from the text itself. The dashboard from this week is the inventory; Ithaca is the interrogation.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Compile a master dataset** of computational metrics across all episodes analyzed so far, structured as rows (episodes) × columns (metrics).
2. **Build multi-panel visualizations** (heatmap, sparklines, radar chart, correlation matrix) that reveal the novel's structural patterns at a glance.
3. **Rank and compare** episodes by individual metrics to identify outliers (particularly Eumaeus as the "measurable trough").
4. **Conduct an error audit** — identifying, cataloging, and explaining where prior analyses produced misleading results.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected for Eumaeus |
|---|---|---|
| Total tokens | word_tokenize count | among the longest episodes |
| TTR | types / tokens | among the lowest (recycled vocabulary) |
| Average sentence length | tokens / sentences | among the longest |
| VADER mean sentiment | mean compound score | near neutral |
| Flesch-Kincaid grade level | readability formula | high (complex but repetitive) |
| Exclamation rate | exclamations per sentence | low |
| Dashboard panels produced | heatmap + sparklines + correlation | 3+ panels |
| Error catalog entries | documented prior-week errors | 5+ entries |

## Rubric

### Exercise 1: The Master Table (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Completeness** | 15+ metrics from prior weeks computed for all 16 episodes; structured as DataFrame | 10+ metrics for most episodes | Fewer than 10 or incomplete |
| **Eumaeus profiling** | Eumaeus rank computed for every metric; top-5 most extreme positions identified | Some ranking done | No ranking |
| **Interpretation** | Quantitative signatures of Joyce's "bad" prose identified and discussed | Brief discussion | No interpretation |

### Exercise 2: The Dashboard (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Heatmap** | Z-scored episode × metric heatmap; structural rhythm visible (Bloom/Stephen, escalation, Eumaeus trough) | Heatmap produced | No heatmap |
| **Sparklines** | One sparkline per metric across all episodes; patterns and outliers scannable | Some per-metric plots | No sparklines |
| **Correlation matrix** | Metric correlation computed and visualized; correlated metric pairs discussed | Correlation shown | No correlation analysis |

### Exercise 3: The Error Audit (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Error catalog** | 5+ documented errors with episode, metric, expected vs. actual, and diagnosis | 3+ errors documented | Fewer than 3 |
| **Error distribution** | Errors mapped to episodes and tools; pattern identified (extreme episodes, specific tools) | Some distribution noted | No pattern analysis |
| **Reflection** | Honest assessment of where computational analysis of literary text systematically fails | Brief reflection | No reflection |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| PCA/t-SNE/UMAP projection of episodes into 2D with interpretation | +3 |
| Composite "prose quality" score ranking all episodes | +3 |
| Benford's Law test on numbers in Ulysses | +2 |
| Regression model predicting episode order from metric profiles | +2 |

## Reference Implementation

See [`week16_eumaeus.py`](week16_eumaeus.py)
