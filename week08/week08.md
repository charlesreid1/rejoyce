# Week 8: Lestrygonians
### *"Sardines on the shelves. Almost taste them by looking" — The peristaltic mind, food, memory, and the body's associative logic.*

It is lunchtime, and Bloom is hungry. He walks through Dublin considering where to eat, thinking about food with an intensity that transforms the episode into a map of appetite. But Bloom's hunger opens onto everything else: the sight of food triggers memories of Molly, of their early courtship on Howth Head (the seedcake kiss — "Ravished over her I lay, full lips full open, kissed her mouth. Yum"), of his dead son, of the economics of restaurants, of the digestive systems of birds. The technique is *peristaltic* — the prose moves in waves, one association swallowed into the next, the rhythm of a mind digesting the world. The art is *architecture*; the organ is the *esophagus*.

**NLTK Focus:** N-gram language models, Markov chains, and probabilistic text generation (`nltk.lm`, `nltk.util.bigrams`, `nltk.util.trigrams`, Maximum Likelihood Estimation, Laplace smoothing)

**Pairing Rationale:**
Peristalsis is the involuntary muscular wave that moves food through the digestive tract — each contraction triggered by the one before it. Bloom's stream of consciousness in Lestrygonians works the same way: each thought is conditioned on the thought that preceded it, producing long chains of association that feel simultaneously inevitable and unpredictable. This is precisely what an n-gram language model captures: the probability of the next word given the preceding *n*−1 words. A bigram model of Bloom's mind would encode the local transitions — *food* → *Molly*, *Molly* → *Howth*, *Howth* → *rhododendrons* — that drive the episode's associative current. And just as peristalsis is involuntary and mechanical (the gut doesn't "choose" its contractions), the Markov property strips language generation of anything resembling intention: the model has no plan, no theme, no memory beyond its window. The results are revealing: Markov-generated "Bloom" is recognizably Bloomian in local texture but globally incoherent, which tells us something important about what stream of consciousness actually is.

**Core Exercises:**

1. **Train and generate.** Build bigram and trigram language models for Lestrygonians using `nltk.lm.MLE` (with appropriate preprocessing and padding). Generate 10 sentences from each model. Read them aloud. Which fragments sound like Bloom? Where does the generated text achieve a convincingly Bloomian local texture, and where does it fall apart? Now train the same models on Proteus (Stephen's interior monologue). Generate sentences and compare. Can you distinguish machine-generated-Bloom from machine-generated-Stephen? What does this tell you about the linguistic fingerprint of each character's consciousness?

2. **Perplexity as style measure.** Compute the perplexity of the bigram model trained on Lestrygonians when evaluated on (a) Lestrygonians itself, (b) Calypso (another Bloom episode), (c) Proteus (a Stephen episode), and (d) a passage of journalistic prose from the Gutenberg corpus. Rank these by perplexity. The model should be least surprised by text that resembles its training data. Does it correctly identify Bloom's other chapter as the closest match? Is Stephen's prose more or less "surprising" to a Bloom-trained model than journalism? What does this reveal about the distinctiveness of each character's language?

3. **Associative chains.** Extract all bigrams from Lestrygonians and rank them by conditional probability: P(word₂ | word₁). Identify the 20 strongest associations. Do they capture the episode's thematic preoccupations — food, body, Molly, memory? Now extract bigrams that cross sentence boundaries (the last word of one sentence and the first word of the next). These capture Bloom's *inter-sentence* associations, the logic by which one thought triggers the next. Map out 5–10 of these cross-sentence transitions and annotate them: what kind of associative link is at work? (Sensory? Thematic? Phonetic? Purely idiosyncratic?)

**Diving Deeper:**

- The Markov property (memorylessness) is both the n-gram model's power and its limitation. Bloom's stream of consciousness has *long-range* dependencies: the Howth memory recurs across episodes, not just across sentences. Explore how RNN and Transformer language models handle long-range dependency. Train a character-level RNN on Ulysses using PyTorch and compare its output to your n-gram model's. (Karpathy's "The Unreasonable Effectiveness of Recurrent Neural Networks" is the classic introduction.)
- The "peristaltic" technique raises a question about the relationship between prose rhythm and content. Do Bloom's sentences about food have measurably different rhythmic properties (sentence length, clause structure, syllable patterns) than his sentences about memory or grief? You can approximate syllable counting with `nltk.corpus.cmudict`.
- Shannon's (1951) experiments on human predictability of English text used exactly the n-gram framework. He asked subjects to predict the next letter in a sequence — a human perplexity measure. Replicate Shannon's experiment informally with classmates using passages from Lestrygonians. Is Joyce more or less predictable than ordinary English? (The answer, interestingly, depends on the scale: word-level Joyce is less predictable; thematic-level Joyce is often deeply patterned.)
- Explore interpolated and Kneser-Ney smoothing (`nltk.lm.KneserNeyInterpolated`). These methods address the sparse data problem that is especially acute in literary language modeling, where the vocabulary is large and many word combinations occur only once. How much does smoothing improve generation quality?
- Connection to Week 11 (Sirens): that episode is structured as a musical fugue, with themes introduced, developed, and recapitulated. The sequential, probabilistic framework of n-gram models will be extended there to capture the quasi-musical repetition structure of fugal prose.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Train n-gram language models** (bigram, trigram) using NLTK's `lm` module with appropriate preprocessing and padding.
2. **Generate text** from trained models and critically evaluate the output's resemblance to the source style.
3. **Compute and interpret perplexity** as a measure of stylistic similarity between texts.
4. **Extract and analyze bigram associations** including cross-sentence transitions that reveal the logic of stream of consciousness.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Lestrygonians) |
|---|---|---|
| Vocabulary size (bigram model) | unique tokens in training data | ~3,000–5,000 |
| Perplexity (self-evaluation) | model on its own training data | lowest of all test texts |
| Perplexity (Calypso) | Bloom model on another Bloom episode | lower than Proteus/reference |
| Perplexity (Proteus) | Bloom model on Stephen's interior | higher than Calypso |
| Top bigram conditional probability | max P(w2\|w1) | varies; expect some ~0.5+ |
| Cross-sentence bigrams | total boundary transitions | ~300–600 |

## Rubric

### Exercise 1: Train and Generate (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Model training** | Bigram and trigram models trained on Lestrygonians and Proteus; preprocessing documented | Models trained on one text | Model training fails or incomplete |
| **Generation quality** | 10+ sentences generated per model; specific fragments identified as "Bloomian" or "Stephenian" | Sentences generated; some commentary | Generation works but no analysis |
| **Character comparison** | Discusses distinguishing linguistic fingerprints between machine-generated Bloom and Stephen | Brief comparison | No comparison |

### Exercise 2: Perplexity (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Computation** | Perplexity computed for all 4 test texts with Laplace smoothing; results tabulated | 3+ texts; smoothing used | Fewer than 3 or no smoothing |
| **Ranking analysis** | Texts ranked by perplexity; ranking discussed in terms of stylistic similarity | Ranking noted | No ranking |
| **Interpretation** | Connects perplexity differences to character voice, prose style, and genre; discusses what "surprise" means for a language model encountering literary text | Some interpretation | Perplexities reported without analysis |

### Exercise 3: Associative Chains (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Bigram extraction** | Top-20 associations by conditional probability; thematic relevance assessed | Top-10 extracted | Basic bigrams without ranking |
| **Cross-sentence analysis** | Boundary bigrams extracted and categorized by association type (sensory, thematic, phonetic) | Some boundary bigrams shown | Not attempted |
| **Annotation** | 5–10 transitions manually annotated with associative logic explanation | 3+ annotated | No annotation |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Character-level RNN trained on Ulysses with output comparison | +4 |
| Prose rhythm analysis (syllable patterns in food vs. memory passages) | +3 |
| Kneser-Ney smoothing comparison to Laplace | +2 |
| Shannon human-prediction experiment on classmates | +1 |

## Reference Implementation

See [`week08_lestrygonians.py`](week08_lestrygonians.py)
