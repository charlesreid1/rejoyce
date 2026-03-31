# Week 14: Oxen of the Sun
### *"Universally that person's acumen is esteemed very little perceptive..." — Nine centuries of English, one chapter, and the embryology of style.*

Oxen of the Sun is the most audacious formal experiment in a novel that is nothing but audacious formal experiments. Set in the Holles Street maternity hospital where Mina Purefoy labors to give birth, the episode recapitulates the entire history of English prose style — from Anglo-Saxon alliterative verse through Middle English romance, Malory, Mandeville, the Elizabethans, the King James Bible, Bunyan, Pepys, Defoe, Addison and Steele, Sterne, Gibbon, Lamb, De Quincey, Landor, Macaulay, Dickens, Newman, Pater, Ruskin, Carlyle — and crashes finally into a cacophony of modern Dublin slang, pidgin, and drunken fragmentation. Each section imitates a different historical period. The technique is *embryonic development*: the nine months of gestation parallel the nine stages of English prose, the language itself being born, growing, maturing, and finally erupting into chaotic modern life. The art is *medicine*; the organ is the *womb*.

It is the single most demanding episode for a computational approach, and the most rewarding.

**NLTK Focus:** Diachronic corpus analysis, historical style classification, and comparative corpus profiling (`nltk.corpus.gutenberg`, period-specific feature extraction, classification across time periods, historical language modeling)

**Pairing Rationale:**
Oxen of the Sun is an experiment in historical linguistics disguised as fiction — or perhaps the reverse. Joyce studied each historical prose style intensively and reproduced its characteristic features: the alliteration and kennings of Anglo-Saxon, the paratactic simplicity of Malory, the balanced Latinate periods of Gibbon, the Germanic compound-heavy density of Carlyle. These are exactly the features that corpus linguists use to study the evolution of English: average sentence length increases then decreases across the centuries; Latinate vocabulary peaks in the 18th century; the ratio of subordinate to coordinate clauses shifts; the passive voice rises and falls. Students can extract these features from NLTK's historical corpora and from Joyce's imitations, and ask: how good a historical linguist was Joyce? Can a classifier trained on real 16th-century prose correctly identify Joyce's pseudo-16th-century section? The episode turns the course into a time machine, and the tools measure the accuracy of the trip.

**Core Exercises:**

1. **Period profiling.** Using NLTK's Gutenberg corpus and any supplementary historical English texts you can access, build stylistic profiles for at least five historical periods represented in Oxen (e.g., Middle English, Elizabethan, 18th-century Augustan, early 19th-century Romantic, late 19th-century Victorian). For each period, compute: average sentence length, type-token ratio, average word length (as a proxy for Latinate vocabulary), proportion of function words, and any POS-tag distribution features that seem diagnostic (e.g., adjective density for Victorian prose, verb-initial constructions for older periods). Now segment Oxen into its stylistic sections (use Stuart Gilbert's or Don Gifford's annotations as a guide) and compute the same features. Plot each section's feature values alongside the corresponding real-period values. Where is Joyce's imitation metrically faithful? Where does he exaggerate or compress?

2. **The style dating game.** Frame this as a classification problem. Train a period classifier on real historical texts — even a simple Naive Bayes or decision tree using the features from Exercise 1. Then feed it the sections of Oxen, unlabeled, and ask it to "date" each one. Does the classifier's chronological ordering match Joyce's intended sequence? Where does it get confused? The errors are as interesting as the successes: if the classifier thinks Joyce's Bunyan sounds more like Defoe, that tells you something about which stylistic features Joyce captured and which he didn't (or chose not to). Produce a confusion matrix and write an analysis of what the misclassifications reveal.

3. **The arc of English.** Treating the sections of Oxen as a time series (ordered by their target historical period), plot the trajectory of each feature across the episode. Does sentence length increase from Anglo-Saxon to Victorian? Does vocabulary richness? Does the Latinate-to-Germanic word ratio shift? Compare these trajectories to the actual historical trends documented in corpus linguistics (see Biber & Finegan, 1989, on the evolution of English style). Joyce was writing in 1920 based on his reading and his ear. The corpus linguists were working in the 1980s–2000s based on massive digital archives. Where do Joyce's intuitions align with the empirical record? This exercise treats Oxen of the Sun as a falsifiable hypothesis about the history of English — and tests it.

**Diving Deeper:**

- The `CLMET` (Corpus of Late Modern English Texts) and `ARCHER` (A Representative Corpus of Historical English Registers) corpora provide much richer historical data than Gutenberg alone. If you can access them, repeat the analysis with larger reference corpora. The `COHA` (Corpus of Historical American English) is another option, though it's American rather than British.
- Word embeddings can be trained on period-specific corpora to create "historical word vectors" — see Hamilton et al. (2016) on diachronic word embeddings. Do words have different nearest neighbors in different centuries? Train period-specific embeddings and visualize how the semantic neighborhood of a word like *wit* or *virtue* or *nature* changes from the 17th to the 19th century. Compare to how Joyce uses those words in the corresponding sections of Oxen.
- Syntactic complexity measures (dependency tree depth, number of subordinate clauses per sentence) can be computed using spaCy or the Stanford Parser. These measures capture aspects of historical style that bag-of-words features miss. The rise and fall of the periodic sentence (long, syntactically suspended structures that delay their main verb) is one of the great arcs of English prose history — and Oxen performs it.
- The episode's final section — the drunken, slangy, fragmented modern coda — has been compared to Eliot's *The Waste Land* and to the Dada movement. Its style is *anti-style*: the rejection of historical form. How does a style classifier handle anti-style? What features characterize the *absence* of period markers? This is a genuinely tricky classification problem: the "modern" class is defined by what it lacks.
- Connection to Week 3 (Proteus): that episode raised the question of synchronic linguistic instability — words that shift meaning in the moment. Oxen extends this to diachronic instability — words and constructions that mean differently across centuries. Together they form the course's complete statement on the mutability of language.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Build period-specific stylistic profiles** from historical reference texts using features diagnostic of prose era (sentence length, Latinate vocabulary, adjective density, comma patterns).
2. **Segment and profile** Oxen of the Sun's chronological style sections, comparing Joyce's imitations to real-period baselines.
3. **Train and evaluate a period classifier** that "dates" text by its stylistic features, and analyze misclassification as evidence of what Joyce captured or missed.
4. **Plot feature trajectories** across the episode as a time series, treating Oxen as a falsifiable hypothesis about the history of English.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Oxen of the Sun) |
|---|---|---|
| Sections segmented | equal divisions of the episode | ~9 sections |
| Period classifier accuracy | train/test on Gutenberg texts | ~0.50–0.70 (3-class problem) |
| Correct chronological ordering | classifier's predicted periods match intended sequence | partial — errors are informative |
| Sentence length arc | early sections → late sections | expect increase then collapse (modern) |
| Latinate vocabulary arc | avg word length across sections | expect peak in middle (Augustan) |
| TTR trajectory | vocabulary richness across sections | varies by period |

## Rubric

### Exercise 1: Period Profiling (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Reference profiles** | 5+ historical periods profiled from Gutenberg; features tabulated | 3+ periods | Fewer than 3 |
| **Oxen profiles** | All sections profiled with same features; side-by-side comparison to reference periods | Some sections profiled | Incomplete |
| **Fidelity assessment** | Discusses where Joyce's imitation is metrically faithful and where it exaggerates or compresses | Brief comparison | No assessment |

### Exercise 2: The Style Dating Game (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Classifier training** | NB or DT trained on period-labeled Gutenberg chunks; accuracy reported | Classifier trained | Classifier broken |
| **Oxen dating** | All sections classified; predicted periods compared to intended sequence | Most sections classified | Incomplete |
| **Misclassification analysis** | Confusion matrix produced; misclassifications interpreted (what features confused the classifier; what this reveals about Joyce's methods) | Some error analysis | No analysis |

### Exercise 3: The Arc of English (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Feature trajectories** | 5+ features plotted across episode sections as time series | 3+ features plotted | Fewer than 3 |
| **Comparison to historical record** | Joyce's feature arcs compared to documented linguistic trends (Biber & Finegan) | Some comparison | No comparison |
| **Anti-style analysis** | Discusses the modern/slang final section as "anti-style" — what happens when features drop out | Brief discussion | No discussion |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Historical word embeddings (Hamilton et al.) comparison | +3 |
| Syntactic complexity measures (dependency depth) across sections | +3 |
| CLMET/ARCHER corpus comparison to Gutenberg | +2 |
| Periodic sentence detection across the arc | +2 |

## Reference Implementation

See [`week14_oxenofthesun.py`](week14_oxenofthesun.py)
