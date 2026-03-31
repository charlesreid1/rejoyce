# Week 12 Writeup: Cyclops -- Text Classification and Genre Detection

## Overview

This week applies text classification and feature engineering to Episode 12, "Cyclops," which alternates between a colloquial barfly narrator and a series of gigantist interpolations -- elaborate parodic set-pieces that inflate ordinary events into the registers of legal, journalistic, epic, biblical, and other genres. The script segments the episode using heuristic rules, trains a Naive Bayes classifier to distinguish the two text types, profiles the barfly's distinctive voice, and attempts to quantify Joyce's gigantism as measurable feature amplification.

---

## Exercise 1: Annotate and Classify

### What the code does

The function `segment_cyclops()` splits the raw text of Cyclops into paragraphs (splitting on newlines), then assigns each paragraph to either "barfly" or "interpolation" based on three heuristic signals:

- **Register markers**: Barfly markers include colloquial words like *says*, *begob*, *bloody*, *damn*. Formal markers include latinate/legal words like *whereas*, *aforementioned*, *hereinafter*.
- **Average sentence length**: Long average sentence length (>40 words) combined with long paragraph length (>100 words) suggests an interpolation; short average sentence length (<25 words) suggests the barfly.
- **Opening patterns**: Paragraphs starting with an em-dash are classified as barfly (dialogue); paragraphs starting with "And" plus long sentences lean toward interpolation.

Once labeled, `extract_features()` computes ten features per segment using NLTK: average sentence length (`sent_tokenize` + `word_tokenize`), type-token ratio, average word length (proxy for latinate vocabulary), POS-tag proportions for nouns/verbs/adjectives (`pos_tag`), passive voice rate (via a VBN-preceded-by-auxiliary pattern), first-person pronoun rate, discourse marker rate, and exclamation rate.

These labeled feature sets are shuffled (seed 42), split 70/30, and fed to `NaiveBayesClassifier.train()`. Accuracy is computed via `nltk.classify.accuracy()`.

### Output and interpretation

```
Barfly segments:        1963
Interpolation segments: 5
```

The segmentation heuristic identified 1963 barfly segments and only 5 interpolation segments. This is a significant problem. Scholars typically identify roughly 10 to 30 distinct interpolation passages in Cyclops, and the exercise itself expects 60-120 barfly segments and 10-30 interpolations. The script splits on single newlines, which means each line of text becomes its own "paragraph." Since Cyclops' interpolations -- though lengthy in print -- may not always exceed the 100-word or 40-word-average-sentence thresholds on a line-by-line basis, most interpolation lines get misclassified as barfly text. The segmentation granularity is too fine and the thresholds are too aggressive.

```
Accuracy: 0.992
```

The classifier achieves 99.2% accuracy, which sounds impressive but is almost certainly an artifact of class imbalance. With 1963 barfly segments and only 5 interpolation segments, a trivial classifier that always guesses "barfly" would score 1963/1968 = 99.7%. The 0.992 accuracy tells us very little about the classifier's ability to actually detect interpolations.

**Most informative features** (top findings from `show_most_informative_features(15)`):

| Feature | Value | Direction | Ratio |
|---------|-------|-----------|-------|
| ttr = 0.7 | interpolation : barfly | 58.0 : 1 |
| avg_word_len = 4.1 | interpolation : barfly | 3.7 : 1 |
| ttr = 0.818... | interpolation : barfly | 3.7 : 1 |
| adj_prop = 0.0 | barfly : interpolation | 3.6 : 1 |
| noun_prop = 0.454... | interpolation : barfly | 3.3 : 1 |
| first_person_rate = 0.0 | barfly : interpolation | 3.3 : 1 |

The classifier's top feature is **type-token ratio at 0.7**, which favors interpolation at a 58:1 ratio. This makes linguistic sense: the interpolations, being formal parodic prose, use a more varied and specialized vocabulary than the barfly's repetitive colloquial speech. Other informative features include higher average word length (longer, more latinate words in interpolations), higher noun proportion (nominal style typical of formal registers), and the absence of first-person pronouns and discourse markers in interpolations (the barfly speaks in first person; the interpolations are impersonal). These are sensible features, broadly consistent with what a human reader would identify as distinguishing the two registers -- though the extreme class imbalance means the ratios should be treated with caution.

One notable detail: `ttr = 1.0` favors *barfly* at 5:1. This likely reflects very short barfly segments (a single line of dialogue where every word is unique, yielding TTR of 1.0), which are common in the barfly's speech but rare in the longer interpolation passages.

---

## Exercise 2: The Barfly's Fingerprint

### What the code does

`barfly_fingerprint()` concatenates all barfly segments into a single text and runs `extract_features()` to produce a composite stylistic profile. It then loads six other episodes (Telemachus, Hades, Aeolus, Wandering Rocks, Nausicaa, Circe) and computes a similarity score for each, based on two features: `first_person_rate` and `discourse_markers`. The similarity formula is a simple average of (1 - absolute difference) for each feature.

### Output and interpretation

**Barfly Profile:**

| Feature | Value |
|---------|-------|
| avg_sent_len | 18.50 |
| avg_word_len | 4.37 |
| discourse_markers | 0.0232 |
| first_person_rate | 0.0169 |
| ttr | 0.26 |
| noun_prop | 0.30 |
| verb_prop | 0.13 |
| adj_prop | 0.06 |
| passive_rate | 0.07 |
| exclamation_rate | 0.06 |

The barfly's profile shows relatively short sentences (18.5 words on average), a low type-token ratio (0.26, reflecting his repetitive, limited vocabulary), moderate word length (4.37 characters -- everyday Anglo-Saxon vocabulary rather than latinate formality), and the presence of discourse markers (0.0232) and first-person pronouns (0.0169). The low TTR is particularly characteristic: the barfly repeats himself, uses stock phrases, and circles around the same words. His passive rate of 0.07 is notable -- some passive constructions appear even in colloquial speech ("he was asked," "it was said"). The exclamation rate of 0.06 reflects the performative nature of his narration.

However, the first-person rate of 0.0169 seems surprisingly low for a first-person narrator. This is likely because the barfly segments include a great deal of reported dialogue from other characters (which would not use "I" from the barfly's perspective), and because the segmentation problem means many interpolation lines were incorrectly lumped into the barfly text, diluting the first-person signal.

**Cross-Episode Similarity Scan:**

| Episode | Similarity | 1st Person Rate | Discourse Markers |
|---------|-----------|----------------|-------------------|
| Telemachus | 0.9838 | 0.0279 | 0.0019 |
| Hades | 0.9876 | 0.0195 | 0.0010 |
| Aeolus | 0.9878 | 0.0190 | 0.0011 |
| Wandering Rocks | 0.9878 | 0.0144 | 0.0014 |
| Nausicaa | 0.9868 | 0.0134 | 0.0002 |
| Circe | 0.9846 | 0.0254 | 0.0009 |

All six episodes score between 0.9838 and 0.9878 in similarity to the barfly -- essentially indistinguishable from each other and all registering as "very similar." This is a flaw in the similarity metric: because `first_person_rate` and `discourse_markers` are both very small numbers (0.01-0.03 and 0.0002-0.0023 respectively), the absolute differences are tiny, and `1 - tiny_number` is always close to 1.0. The metric lacks discriminative power at this scale.

That said, the raw feature values tell a meaningful story. No other episode comes close to the barfly's discourse marker rate of 0.0232. The closest is Telemachus at 0.0019 -- more than 10x lower. This confirms that the barfly's voice, with its *says I* and *begob* and *bloody*, is genuinely unique in the novel. The exercise's prediction -- that the barfly's register should be "uniquely informal for the novel" -- is borne out by the data, even though the similarity metric fails to surface this finding clearly.

Circe and Telemachus show the highest first-person pronoun rates among the comparison episodes (0.0254 and 0.0279), which makes sense: Telemachus opens with Stephen's perspective in a relatively intimate mode, and Circe's hallucinatory drama includes substantial first-person speech.

---

## Exercise 3: Gigantism as Feature Amplification

### What the code does

`gigantism_analysis()` computes features for the concatenated interpolation text and compares them to features extracted from Episode 4, "Calypso," which serves as a baseline for "normal" Joycean prose. The ratio (interpolation value / baseline value) is meant to quantify how much Joyce exaggerates each feature in the parodic passages.

### Output and interpretation

| Feature | Interpolations | Baseline (Calypso) | Ratio |
|---------|---------------|-------------------|-------|
| avg_sent_len | 32.00 | 8.98 | 3.57x |
| ttr | 0.667 | 0.334 | 2.00x |
| avg_word_len | 4.96 | 4.30 | 1.16x |
| noun_prop | 0.375 | 0.272 | 1.38x |
| adj_prop | 0.047 | 0.059 | 0.80x |
| verb_prop | 0.047 | 0.134 | 0.35x |
| discourse_markers | 0.000 | 0.001 | 0.00x |
| exclamation_rate | 0.000 | 0.024 | 0.00x |
| first_person_rate | 0.000 | 0.013 | 0.00x |
| passive_rate | 0.000 | 0.005 | 0.00x |

**Features that show amplification (ratio > 1):**

- **Average sentence length: 3.57x.** This is the clearest signal of gigantism. The interpolations average 32 words per sentence versus Calypso's 9 words. Formal parodic prose -- legal, epic, journalistic -- employs elaborately nested sentences, and Joyce cranks this up to grotesque proportions. This is exactly what the exercise predicts: "longer sentences than real legal prose."
- **Type-token ratio: 2.00x.** The interpolations use twice the lexical variety of normal Joycean prose. This reflects the specialized vocabularies of the parodied genres -- legal terminology, epic epithets, scientific jargon -- each contributing unique words that inflate the TTR.
- **Noun proportion: 1.38x.** Formal registers are more nominal (noun-heavy) than conversational registers. The interpolations, parodying formal genres, show a higher noun density.
- **Average word length: 1.16x.** A modest increase, reflecting the latinate vocabulary of legal and scientific parody.

**Features that show suppression (ratio < 1):**

- **Discourse markers: 0.00x.** The interpolations contain zero instances of *says*, *begob*, *bloody*, etc. This is the inverse of gigantism -- the parodic passages completely suppress the barfly's colloquial markers.
- **First-person pronouns: 0.00x.** The interpolations are impersonal, written in third person or with collective voice.
- **Exclamation rate: 0.00x.** Formal genres do not exclaim.
- **Verb proportion: 0.35x.** The interpolations are heavily nominal, with far fewer verbs relative to other parts of speech.
- **Passive rate: 0.00x.** Surprisingly, the interpolations show no passive constructions. This may be an artifact of having only 5 short interpolation segments -- with so little text, the passive-voice heuristic (which requires a specific VBN + auxiliary pattern) may simply not trigger.
- **Adjective proportion: 0.80x.** Slightly lower than baseline, which is unexpected -- one might predict that the parodic passages, especially the epic and romantic ones, would be adjective-rich. This again likely reflects the small sample size.

**Interpretation:** The amplification results partially confirm the exercise's hypothesis about gigantism as statistical caricature. Sentence length and lexical variety are dramatically amplified, while colloquial markers are completely suppressed. However, the results are hampered by having only 5 interpolation segments to analyze. With so few data points, some features (passive voice, adjective proportion) that should theoretically show amplification instead show zero or reduction. A better segmentation that captures all 10-30 interpolations would likely produce stronger and more consistent amplification signals across all formal-register features.

The Calypso baseline is a reasonable choice (it represents Bloom's ordinary domestic morning in relatively plain initial-style prose), though the exercise also suggests comparing against actual genre corpora (legal documents from Reuters, news articles, etc.) for a more rigorous test of the "parody exceeds its source" hypothesis.

---

## Summary of Findings

The script demonstrates the core Week 12 concepts -- feature-based text classification, stylistic profiling, and quantitative comparison -- but its results are significantly compromised by the segmentation step, which identifies only 5 interpolations out of what should be 10-30. This cascading error inflates the barfly class, produces misleadingly high classifier accuracy, and weakens the gigantism analysis. The most trustworthy results are: (1) the feature importance rankings from the Naive Bayes classifier, which correctly identify TTR, word length, and discourse markers as key discriminators; (2) the barfly's uniquely high discourse marker rate compared to all other episodes; and (3) the 3.57x sentence-length amplification in the interpolations, which quantitatively captures Joyce's technique of inflating formal prose to grotesque lengths.
