# Week 13: Nausicaa
### *"The waxen pallor of her face was almost spiritual in its ivorylike purity" — Two halves, two voices, and the forensics of style.*

Nausicaa is a chapter with a seam running down its center. The first half belongs to Gerty MacDowell, a young woman on Sandymount Strand who watches Bloom watching her, and whose consciousness is rendered in the lavender-scented prose of Victorian sentimental fiction — *The Lamplighter*, *The Princess Novelette*, the language of advertising copy and women's magazines. Every phrase is prefabricated: "the waxen pallor of her face," "a languid queenly hauteur," "her woman's instinct told her." The prose tumesces — builds, swells, reaches toward a climax that is simultaneously Gerty's romantic fantasy and Bloom's orgasm on the beach. Then the fireworks burst, and the episode's second half collapses into Bloom's post-coital interior monologue: flat, practical, deflationary, punctuated by self-awareness and mild self-disgust. The technique is *tumescence/detumescence*; the art is *painting*; the organ is the *eye and nose*. The episode reads as if two different authors — one a hack sentimentalist, the other James Joyce — wrote alternate halves.

**NLTK Focus:** Stylometry and authorship attribution (`nltk.probability`, function word analysis, sentence length distributions, vocabulary richness metrics, Burrows' Delta, classification applied to stylistic features)

**Pairing Rationale:**
Authorship attribution is the forensic science of style: given an anonymous text, can you determine who wrote it by measuring linguistic features below the level of conscious control? Function words, sentence length, hapax legomena ratios, punctuation patterns — these are the fingerprints that writers leave involuntarily. Nausicaa is the ultimate test case because Joyce is *deliberately* writing in someone else's style for half the chapter. The sentimentalist prose is a forgery — but is it a perfect one? Stylometry asks: does Joyce's signal leak through the pastiche? When he imitates the language of *The Lamplighter*, does his own vocabulary, his own sentence rhythm, his own function-word distribution betray him beneath the costume? The two-half structure gives students a clean experimental design: train a style profile on Bloom's monologue (the "real" Joyce), then test whether the Gerty half matches or diverges. The answer is more interesting than a simple yes or no — Joyce's pastiche is neither perfect imitation nor transparent parody but something uncanny in between.

**Core Exercises:**

1. **The split test.** Divide Nausicaa into its two halves (the Gerty section and the Bloom section). For each half, compute a stylometric profile: (a) the 50 most common function words and their relative frequencies, (b) sentence length distribution (mean, median, standard deviation), (c) vocabulary richness (type-token ratio, hapax legomena ratio, Yule's K), (d) punctuation frequency (commas, semicolons, exclamation marks, em-dashes per sentence). Visualize the two profiles side by side. Where are the differences most extreme? The exclamation mark count alone will tell a story. Now compute the same profiles for two other Bloom episodes (Calypso, Lestrygonians). Is Bloom's half of Nausicaa stylometrically consistent with his voice elsewhere? Is Gerty's half an outlier from the entire novel?

2. **Burrows' Delta.** Implement John Burrows' Delta method — the standard stylometric distance measure. Compute Delta between the Gerty half and a reference corpus consisting of (a) other Bloom episodes, (b) other Stephen episodes, (c) the Cyclops barfly narration, and (d) if you can find digitized excerpts, actual Victorian sentimental fiction. Which corpus is Gerty's prose closest to? Burrows' Delta operates on the z-scores of function word frequencies, which means it's measuring how *unusual* a text's function word profile is relative to a corpus mean. Does Joyce's Gerty pastiche land closer to Joyce or closer to the genre he's parodying? What would a "perfect" pastiche look like in Delta space?

3. **The cliché detector.** Gerty's prose is built from stock phrases — prefabricated chunks of sentimental language. Build a simple cliché detector: extract all n-grams (n=3,4,5) from the Gerty half and compare them to a reference corpus of common English phrases (you can use the Google n-gram corpus or, more practically, extract high-frequency n-grams from a large Gutenberg sample). Flag n-grams that appear in both Gerty's text and the reference corpus at high frequency. These are the clichés — the borrowed lumber of Gerty's consciousness. Now run the same detector on Bloom's half. How many clichés does Bloom use? The ratio tells you something precise about the difference between a mind that thinks in received language and a mind that, for all its ordinariness, generates its own.

**Diving Deeper:**

- Computational stylistics has a rich history in Joyce studies. See Rybicki and Eder (2011) on stylometric analysis of literary translations, and explore the `stylo` R package, which implements Burrows' Delta, Craig's Zeta, and other measures. If you want to go deep, run a full stylometric analysis of all 18 episodes and visualize the results as a dendrogram. Does Ulysses cluster by narrator (Bloom vs. Stephen vs. other), by technique, or by position in the novel?
- The tumescence/detumescence structure maps onto a measurable arc in stylistic features. Plot sentence length, exclamation frequency, and adjective density across the episode as a time series. Can you see the "climax" in the data? Does the detumescence appear as a discontinuity or a gradual decline?
- Gerty's voice raises questions about free indirect discourse and ideology. Whose language is this — Gerty's, Joyce's, or the culture's? Ken Kenner and Hugh Kenner have argued that the sentimental prose *is* Gerty's consciousness, shaped by the media she consumes. Others read it as Joyce's satire imposed on her. Stylometry can contribute evidence: if the Gerty half shows traces of Joyce's authorial signal (anomalous function word ratios, sentence structures too complex for the pastiche), that supports the satiric reading. If it's stylometrically clean pastiche, that supports the mimetic reading.
- Modern authorship attribution uses character-level n-gram features, compression-based distance measures (Benedetto et al., 2002), and neural approaches (Boenninghoff et al., 2019). These methods can detect authorial signal even through deliberate style imitation. Test them on Nausicaa — can a neural authorship model see through Joyce's disguise?
- Connection to Week 14 (Oxen of the Sun): that episode extends the pastiche principle from one style to thirty. The stylometric tools from this week become the backbone of next week's analysis.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Compute stylometric profiles** including function word frequencies, sentence length distributions, vocabulary richness, and punctuation patterns.
2. **Implement Burrows' Delta** — the standard stylometric distance measure — and use it to compare texts across a reference corpus.
3. **Split and compare** two halves of a deliberately bifurcated text, using quantitative features to characterize the split.
4. **Build a cliché detector** using n-gram overlap with a reference corpus, and use it to measure the density of prefabricated language.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Nausicaa) |
|---|---|---|
| Nausicaa split point | sentence index of Gerty→Bloom transition | ~40–60% through the episode |
| TTR difference | Gerty TTR vs. Bloom TTR | Bloom higher (more varied vocabulary) |
| Exclamation rate | exclamation marks per sentence | Gerty >> Bloom |
| Mean sentence length difference | Gerty vs. Bloom | Gerty longer (tumescent prose) |
| Burrows' Delta (Gerty → Bloom corpus) | Delta value | higher than Bloom→Bloom Delta |
| Cliché density ratio | Gerty clichés / Bloom clichés per 1000 tokens | > 1.5x |

## Rubric

### Exercise 1: The Split Test (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Split identification** | Nausicaa correctly split; function words, sentence length, TTR, hapax, punctuation all computed for both halves | Split done; most metrics computed | Incomplete split or metrics |
| **Cross-episode comparison** | Bloom half compared to Calypso and Lestrygonians for consistency | Some comparison | No comparison |
| **Visualization** | Side-by-side profiles visualized; most dramatic differences highlighted | Some visualization | No visualization |

### Exercise 2: Burrows' Delta (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Implementation** | Delta computed correctly with z-score normalization over 50 function words | Delta computed with some errors | Not implemented |
| **Reference corpus** | Gerty compared to Bloom episodes, Stephen episodes, and barfly narration | Compared to 3+ references | Fewer than 3 |
| **Interpretation** | Discusses whether Gerty lands closer to Joyce or to pastiche; connects to mimetic vs. satiric reading | Some interpretation | No interpretation |

### Exercise 3: The Cliché Detector (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **N-gram extraction** | 3-gram, 4-gram, 5-gram extracted from both halves; compared to Gutenberg reference | Some n-grams extracted | Extraction broken |
| **Density comparison** | Cliché density computed for both halves; ratio reported; sample clichés shown | Some density comparison | No comparison |
| **Interpretive reflection** | Connects cliché density to the difference between a mind that thinks in received language and one that generates its own | Brief reflection | No reflection |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Full 18-episode stylometric dendrogram | +4 |
| Tumescence/detumescence feature arc plotted across episode | +3 |
| Neural authorship attribution (compression distance, char-ngrams) | +2 |
| Free indirect discourse / ideology discussion | +1 |

## Reference Implementation

See [`week13_nausicaa.py`](week13_nausicaa.py)
