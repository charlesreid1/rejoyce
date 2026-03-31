# Week 3: Proteus
### *"Ineluctable modality of the visible" — The mind alone on Sandymount Strand, language folding back on itself.*

Proteus is the first great shock of the novel. We are entirely inside Stephen's head as he walks along the beach, thinking about Aristotle, about his dead mother, about perception and the nature of reality. The prose is dense, allusive, polyglot, and unstable — words shift their shapes, sentences change direction mid-stride, languages bleed into each other (English, Latin, French, Italian, German). The episode's Homeric parallel is Menelaus wrestling the shape-shifting god Proteus; its technique is *monologue* (male); its art is *philology*. This is Joyce's chapter about what language *is*, and it is accordingly the chapter where language becomes most slippery.

**NLTK Focus:** Stemming, morphological analysis, and language identification (`nltk.stem`, `PorterStemmer`, `SnowballStemmer`, `LancasterStemmer`, `nltk.corpus.wordnet` for derivational morphology, language detection heuristics)

**Pairing Rationale:**
Proteus is the chapter of metamorphosis — of forms that shift, of surfaces that deceive, of the "ineluctable modality" of things that refuse to stay fixed. Stemming is the computational attempt to find the stable root beneath inflected forms: *walking*, *walked*, *walks* → *walk*. It is an act of faith in an underlying identity — exactly the kind of faith that Proteus undermines. Joyce fills this episode with words that resist reduction: neologisms, portmanteaus, foreign borrowings, onomatopoeia, hapax legomena. The multilingual texture of Stephen's thought (switching from English to Latin to French to Italian within a single paragraph) makes language identification itself a non-trivial task. Proteus is where students learn that NLP tools encode assumptions about linguistic stability, and that Joyce is philosophically committed to violating those assumptions.

**Core Exercises:**

1. **The stemmer's struggle.** Apply three different NLTK stemmers (Porter, Lancaster, Snowball) to the full text of Proteus. For each stemmer, identify the 10 most "aggressive" reductions — cases where the stemmed form is most distant from the original token (you might measure this by edit distance using `nltk.metrics.distance.edit_distance`). Which stemmer handles Joyce's vocabulary best? Which produces the most absurd results? Collect your favorite failures.

2. **Multilingual detection.** Proteus contains passages in at least four languages. Using NLTK's stopword lists for English, French, German, Italian, and Latin (you may need to supplement with a simple Latin stopword list), build a rudimentary sliding-window language detector: for each sentence or clause, compute the proportion of tokens that appear in each language's stopword list and assign the most likely language. Where does your detector succeed? Where does it fail, and why? (Hint: Stephen's macaronic sentences are *designed* to defeat this kind of classification.)

3. **Derivational morphology and neologism.** Using WordNet, attempt to trace the derivational history of 15–20 unusual words from Proteus (e.g., *ineluctable*, *nacheinander*, *nebeneinander*, *contransmagnificandjewbangtantiality*). For words that aren't in WordNet, hypothesize a morphological parse. Which of Joyce's coinages follow productive English morphological rules? Which are borrowings? Which are sui generis? What does this tell you about the limits of dictionary-based NLP?

**Diving Deeper:**

- The `langdetect` and `langid` Python libraries use statistical models for language identification. Test them on Proteus passages and compare to your stopword heuristic. Modern multilingual models (XLM-RoBERTa) can identify languages at the token level — try HuggingFace's `xlm-roberta-base` if you want to see the state of the art applied to Joyce's code-switching.
- Proteus is where Joyce's portmanteau words begin in earnest. Lewis Carroll's portmanteau tradition → Joyce → Finnegans Wake is a well-known lineage. Explore computational approaches to portmanteau analysis: can you write a function that, given a portmanteau, proposes its source words? (This is related to the task of compound splitting in German NLP — see Ziering & van der Plas, 2016.)
- The philosophical backdrop of this episode is Aristotle's *De Anima* (on perception) and the Scholastic distinction between *form* and *matter*. Stemming is implicitly Aristotelian — it assumes that words have essential forms beneath accidental inflections. Is this a defensible model of language? (See Blevins, 2016, *Word and Paradigm Morphology* for a theoretical critique.)
- Subword segmentation methods like BPE and Unigram (as used in SentencePiece) offer an alternative to both stemming and lemmatization. They have no linguistic theory — they find statistically optimal segments. Apply SentencePiece to Proteus and compare its segmentation to linguistic morphological analysis. When does the statistical approach outperform the rule-based one?
- Connection to Week 14 (Oxen of the Sun): that episode performs a chronological tour through the history of English prose styles, making the *diachronic* instability of language visible. Proteus makes the *synchronic* instability visible. Together they bracket the course's engagement with the question: what is a word?

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Apply and compare** multiple stemming algorithms (Porter, Lancaster, Snowball) to literary text, and evaluate their aggressiveness using edit distance.
2. **Build a heuristic language detector** using stopword overlap in a sliding window, and assess its performance on multilingual literary text.
3. **Perform morphological analysis** on neologisms and compound words, distinguishing productive English morphology from borrowings and coinages.
4. **Articulate the limits** of rule-based NLP tools when applied to text that deliberately violates linguistic conventions.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Proteus) |
|---|---|---|
| Stemmer disagreement rate | % of unique words where Porter, Lancaster, Snowball disagree | ~25–40% |
| Max edit distance (Lancaster) | largest edit distance between word and stem | > 8 characters |
| Non-English sentence proportion | sentences flagged non-English by stopword detector | ~3–8% |
| Non-English token proportion | tokens not in WordNet / total alpha tokens | ~5–15% |
| WordNet coverage of target words | proportion of 20 selected words found in WordNet | ~40–60% |
| Successful compound decomposition | neologisms decomposable into known morphemes | ~30–50% |

## Rubric

### Exercise 1: The Stemmer's Struggle (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Three stemmers applied** | All three stemmers run correctly; top-10 aggressive reductions identified per stemmer with edit distances | Two stemmers or incomplete ranking | Fewer than two stemmers or broken output |
| **Comparative analysis** | Specific discussion of which stemmer is most/least aggressive; best absurd reductions highlighted with commentary | Some comparison between stemmers | Results listed without comparison |
| **Joyce-specific insight** | Identifies why Joyce's vocabulary is particularly hard for stemmers (neologisms, compounds, foreign words) and connects to the episode's Protean theme | General observation about stemmer limits | No connection to the text's character |

### Exercise 2: Multilingual Detection (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Detector implementation** | Working sliding-window detector using 5+ language stopword lists; outputs per-sentence language labels with scores | Detector works for 3+ languages | Broken or trivial detector |
| **Performance assessment** | Precision/recall estimated against manual annotation of at least 10 non-English passages; failure modes categorized | Some manual verification | No evaluation of detector quality |
| **Macaronic analysis** | Identifies sentences where code-switching defeats classification; explains *why* the mixing is undetectable (shared vocabulary, short spans, Joyce's deliberate blending) | Notes some failures | No analysis of failure cases |

### Exercise 3: Derivational Morphology (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **WordNet lookup** | 15–20 words investigated; synset counts and definitions reported; hypernym paths traced for available words | 10+ words investigated | Fewer than 10 words |
| **Morphological parsing** | For words absent from WordNet, plausible decompositions proposed (prefix + root, compound, borrowing); classified by type | Some parsing attempted | No parsing of unknown words |
| **Reflection on limits** | Discussion of what dictionary-based NLP cannot capture about Joyce's lexical creativity; connects to the episode's theme of metamorphosis | Brief reflection | No reflection |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| `langdetect` or `langid` comparison to stopword heuristic | +3 |
| Portmanteau decomposition function (given compound, propose sources) | +3 |
| BPE/SentencePiece segmentation of Proteus vs. linguistic morphology | +2 |
| Philosophical reflection on stemming as Aristotelian form-finding | +2 |

## Reference Implementation

See [`week03_proteus.py`](week03_proteus.py)
