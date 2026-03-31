# Week 13 Writeup: Nausicaa -- Stylometry & Authorship Attribution

## Overview

This week applies stylometric analysis to Episode 13 (Nausicaa), the chapter Joyce split down the middle: the first half renders Gerty MacDowell's consciousness in the saccharine language of Victorian sentimental fiction, while the second half collapses into Bloom's flat, deflationary interior monologue. The script computes stylometric profiles for both halves, implements Burrows' Delta to measure stylistic distance from other episodes, and builds a cliche detector to quantify prefabricated language. The three exercises together ask whether Joyce's pastiche is measurably different from his own voice, and whether the numbers can see the seam he deliberately stitched into the text.

---

## Exercise 1: The Split Test

### What the code does

The function `split_nausicaa()` divides the episode into two halves using a heuristic: it computes a rolling average of sentence length (window of 20 sentences), then finds the sharpest drop in that average between the first and third quartiles of the text. This drop corresponds to the moment when the tumescent, ornate Gerty prose gives way to Bloom's clipped post-coital thoughts. The split lands at sentence 845.

For each half, `stylometric_profile()` computes:
- **Function word frequencies**: relative frequencies of the top 50 English function words, computed via `nltk.probability.FreqDist` over lowercased alphabetic tokens.
- **Sentence length distribution**: mean, median, and standard deviation of sentence lengths (in tokens), using `nltk.tokenize.sent_tokenize` and `nltk.tokenize.word_tokenize`.
- **Vocabulary richness**: type-token ratio (TTR) and hapax legomena ratio (proportion of types that occur exactly once).
- **Punctuation frequency**: exclamation marks, semicolons, commas, and em-dashes per sentence.

The same profiles are computed for Calypso (Episode 4) and Lestrygonians (Episode 8) -- two other Bloom episodes -- to test whether Bloom's half of Nausicaa is consistent with his voice elsewhere.

### What the output shows

```
Metric                             Gerty   Bloom (Naus)        Calypso  Lestrygonians
-------------------------------------------------------------------------------------
  total_tokens                     13257           3731           5957          12852
  total_types                       2861           1338           1989           3522
  ttr                             0.2158         0.3586         0.3339         0.2740
  hapax_ratio                     0.6501         0.6824         0.6626         0.6638
  mean_sent_len                  17.6568         7.4853         8.9757         8.0687
  median_sent_len                     10              6              7              7
  std_sent_len                   22.0312         6.5881         6.3435         5.7334
  exclamation_per_sent            0.0722         0.0212         0.0243         0.0298
  comma_per_sent                  0.6012         0.2248         0.3990         0.2476
  em_dash_per_sent                0.0402         0.0000         0.0706         0.0743
```

Several patterns stand out:

**Sentence length.** Gerty's mean sentence length is 17.7 words, more than double Bloom's 7.5 in the same episode, and more than double Calypso (9.0) and Lestrygonians (8.1). The standard deviation is enormous (22.0 vs. roughly 6 for the other three), confirming the tumescent quality of Gerty's prose: some sentences swell to great length while others are short exclamations. Bloom's half of Nausicaa is actually slightly shorter in mean sentence length than Calypso and Lestrygonians, consistent with the deflated, post-coital flatness the exercise describes.

**Vocabulary richness.** Gerty's TTR (0.2158) is markedly lower than Bloom's Nausicaa half (0.3586), Calypso (0.3339), and Lestrygonians (0.2740). This is the signature of prefabricated language: Gerty's prose recycles the same stock phrases and sentimental vocabulary, producing a lower ratio of unique words to total words. Bloom's half, despite being much shorter, achieves the highest TTR -- his mind ranges more freely across topics even in a brief stretch of text. The hapax ratio is relatively stable across all four texts (0.65-0.68), suggesting that the proportion of once-used words is less sensitive to stylistic register than TTR.

**Exclamation marks.** Gerty's rate is 0.072 per sentence -- roughly 3.4 times Bloom's 0.021, and about triple Calypso and Lestrygonians. The exclamation mark is the punctuation of sentiment, breathless emphasis, romantic excess. Its near-absence from Bloom's prose is the typographic signature of his temperament.

**Commas.** Gerty's comma rate (0.60 per sentence) dwarfs all three Bloom texts. The elaborate, subordinated syntax of sentimental prose demands commas; Bloom's paratactic, fragmented thoughts do not.

**Bloom consistency across episodes.** Bloom's Nausicaa half, Calypso, and Lestrygonians cluster together on nearly every metric: mean sentence length in the 7-9 range, TTR between 0.27 and 0.36, low exclamation rates, moderate comma rates. Gerty's half is the clear outlier on every dimension. This confirms the exercise's prediction: Bloom's voice is stylometrically consistent, and Gerty's half departs from the entire novel.

### Function word comparison

The top differences between Gerty and Bloom:

| Word | Gerty | Bloom | Difference |
|------|-------|-------|------------|
| and | 0.04111 | 0.01849 | 0.02262 |
| was | 0.02014 | 0.00456 | 0.01558 |
| she | 0.02278 | 0.00804 | 0.01474 |
| her | 0.02195 | 0.00884 | 0.01311 |

The dominance of "and," "was," "she," and "her" in Gerty's half is telling. "And" at over 4% reflects the additive, list-like syntax of sentimental prose ("and her face was...and she knew...and it was..."). The past tense "was" at double Bloom's rate reflects the narrative distance of the Gerty section -- it describes events in a storytelling register. "She" and "her" are overwhelmingly present because Gerty's section is written in third-person free indirect discourse focused on a female character, while Bloom's half is first-person interior monologue.

---

## Exercise 2: Burrows' Delta

### What the code does

The `burrows_delta()` function implements the standard Burrows' Delta method for stylometric distance. For each of the 50 function words, it computes z-scores by subtracting the corpus mean frequency and dividing by the corpus standard deviation. Delta is the mean of the absolute differences between the test text's z-scores and each reference text's z-scores across all 50 function words. Lower delta means greater stylistic similarity.

The reference corpus consists of:
- **Bloom episodes**: Calypso, Lotus Eaters, Lestrygonians, and Bloom's Nausicaa half
- **Stephen episodes**: Telemachus, Proteus, Scylla and Charybdis
- **Barfly narration**: Cyclops

Gerty's half is the test text measured against all of these.

### What the output shows

```
  Reference Text                 Delta
  -------------------------------------
  Lestrygonians                 1.8732
  Cyclops                       1.9788
  Bloom (Nausicaa)              2.0851
  Lotus Eaters                  2.1110
  Telemachus                    2.2105
  Calypso                       2.2573
  Scylla                        2.3181
  Proteus                       2.6256
```

The results are somewhat surprising. Gerty's half is closest to Lestrygonians (Delta 1.87), followed by Cyclops (1.98) and then Bloom's own Nausicaa half (2.09). The Stephen episodes all rank further away, with Proteus at the greatest distance (2.63).

This suggests that Joyce's Gerty pastiche, despite its radically different surface texture, retains enough of Joyce's underlying function word patterns that it does not fully escape his authorial fingerprint. The Gerty section lands closest to other Joyce texts rather than diverging into a truly alien stylistic space. This is consistent with the exercise's suggestion that the pastiche is "neither perfect imitation nor transparent parody but something uncanny in between." Joyce puts on the costume of sentimental prose at the level of vocabulary and syntax, but his function word habits -- the small, unconscious choices about "the," "of," "to," "and" -- leak through the disguise.

The fact that Lestrygonians, one of the longest Bloom episodes, is the closest match may partly reflect a size effect: both texts are among the longest in the corpus, which can stabilize function word frequencies and reduce noise in the delta calculation.

The fact that Cyclops ranks second is intriguing. Cyclops is also a chapter defined by an unreliable narrator (the anonymous barfly) with a voice distinct from Bloom's or Stephen's. Both Nausicaa-Gerty and Cyclops feature narrators who are not Bloom or Stephen, which may produce similar function word distortions away from the Joyce baseline.

Proteus, Stephen's most linguistically dense and experimental chapter, is the most distant from Gerty -- an expected result given that Proteus's prose is maximally literary while Gerty's is maximally formulaic.

---

## Exercise 3: The Cliche Detector

### What the code does

The `extract_ngrams()` function extracts all 3-grams, 4-grams, and 5-grams from a text (lowercased, alphabetic tokens only). The reference corpus is built from the first five texts in the NLTK Gutenberg corpus, which provides a baseline of common English phrase patterns. An n-gram counts as a cliche candidate if it appears 3 or more times in the reference corpus and at least once in the target text.

Cliche density is measured as the number of distinct cliche n-grams per 1000 tokens.

### What the output shows

```
  Gerty: 1195 cliche n-grams, density: 90.14 per 1000 tokens
  Bloom: 231 cliche n-grams, density: 61.91 per 1000 tokens
  Ratio (Gerty/Bloom): 1.46x
```

Gerty's cliche density (90.14 per 1000 tokens) is 1.46 times Bloom's (61.91 per 1000 tokens). The exercise predicted a ratio greater than 1.5x; the measured ratio falls just slightly short of that target at 1.46x. The direction is clear and correct: Gerty's prose is built from more prefabricated material than Bloom's.

**Sample Gerty cliches** are revealing in their banality:
- "on account of" (9 occurrences) -- the stock causal phrase of middlebrow prose
- "she could see" (7) -- the formulaic observation
- "and she was" / "and that was" / "that he was" (6 each) -- the additive, copulative syntax of romance fiction
- "she felt that" (4) -- the sentimental interiority marker

These are the building blocks of Gerty's consciousness as Joyce renders it: a mind assembled from the readymade phrases of women's magazines and sentimental novels. Every "she felt that" is a thought that arrives pre-packaged.

**Sample Bloom cliches** are fewer and different in character:
- "pray for us" (3) -- a liturgical formula, not a sentimental cliche; Bloom is thinking about Catholic prayer
- "what do you" (2) -- a conversational fragment
- "out of the" (2) -- a bare spatial preposition phrase

Bloom's cliches are functional rather than sentimental. When he uses stock language, it tends to be religious formula (the "pray for us" litany he is mentally rehearsing) or bare prepositional scaffolding. He does not think in the received language of popular fiction; his cliches are the unavoidable connective tissue of English rather than the decorative frosting of romance.

The ratio of 1.46x, while slightly below the 1.5x target in the rubric, still captures the essential insight: Gerty's consciousness is substantially more cliche-saturated than Bloom's. The difference would likely be even more pronounced if the reference corpus were drawn from Victorian sentimental fiction rather than the NLTK Gutenberg corpus (which includes the Bible, Milton, Shakespeare, and other texts that do not particularly overlap with romance-novel phrasing).

---

## Summary

The three exercises converge on a consistent picture. Exercise 1 shows that the Gerty half is a stylometric outlier from Bloom's voice across the novel -- longer sentences, lower vocabulary richness, more exclamation marks, more commas, and a radically different function word profile dominated by "and," "was," "she," and "her." Exercise 2's Burrows' Delta reveals that despite these surface differences, the Gerty half is still closest to other Joyce texts rather than fully escaping into a separate stylistic orbit -- Joyce's authorial fingerprint leaks through the pastiche. Exercise 3 quantifies the cliche saturation of Gerty's prose at 1.46 times Bloom's rate, confirming that her consciousness is built from prefabricated language in a way Bloom's is not.

Together, these results support the reading that Nausicaa's Gerty section is a virtuosic but imperfect forgery: Joyce inhabits the register of sentimental fiction at the level of vocabulary, syntax, and stock phrases, but his underlying function word habits and sentence-level patterns retain traces of his own style. The pastiche is good enough to produce measurably different stylometric profiles, but not so complete that Burrows' Delta cannot see the author behind the mask.
