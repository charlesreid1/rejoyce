# Week 1: Telemachus
### *"Introibo ad altare Dei" — Establishing the world through dialogue, ritual, and the texture of ordinary speech.*

Joyce opens with the initial style: third-person narration interwoven with naturalistic dialogue, interior thought rendered close to the surface, and a dense web of literary and liturgical allusion. The episode establishes voice, place, and character through the way people talk — Buck Mulligan's theatrical bluster, Haines's polite Anglo menace, Stephen's brooding compression. It is a chapter about surfaces, about hearing a world before you understand it.

**NLTK Focus:** Tokenization, text normalization, and basic corpus exploration (`nltk.tokenize`, `nltk.text.Text`, concordance, frequency distributions)

**Pairing Rationale:**
Every NLP pipeline begins where every novel begins: with the raw stream of language, not yet parsed into meaning. Tokenization is the act of deciding where one word ends and another begins — a question that seems trivial until you meet Joyce, who will spend the next seventeen episodes making it progressively harder. Telemachus is the right place to start because its prose is still cooperating with us: sentences parse, dialogue is punctuated, words are (mostly) English. We learn the tools here so we can watch them struggle later. The concordance and frequency distribution are our first instruments for seeing what a text is *about* — and students will discover immediately that what Ulysses is "about" by word frequency (he, the, of, his) is not what it's *about* at all.

**Core Exercises:**

1. **Tokenize and profile.** Load the text of Telemachus using NLTK's `PlaintextCorpusReader` or a manual file load. Tokenize using `word_tokenize` and `sent_tokenize`. Compute basic statistics: total tokens, total types, type-token ratio, average sentence length. Compare these numbers to a reference prose text (e.g., a chapter of *Pride and Prejudice* from the Gutenberg corpus). What do the differences — or surprising similarities — tell you?

2. **Concordance as close reading.** Use `nltk.text.Text` to build concordance views for key thematic words: *mother*, *sea*, *key*, *tower*, *God*. Examine the contexts in which each appears. Write a short paragraph (3–5 sentences) arguing for a connection between two of these words based solely on their concordance patterns — that is, based on the company they keep.

3. **Frequency and stopwords.** Generate a frequency distribution of all tokens. Plot the top 50 words. Now remove stopwords (using `nltk.corpus.stopwords`) and re-plot. What thematic vocabulary emerges? Do the results surprise you, given what happens in the episode? Pay special attention to proper nouns and to words that appear with unexpected frequency.

**Diving Deeper:**

- Zipf's Law and literary language: Plot the rank-frequency distribution for Telemachus on a log-log scale. Does Joyce's prose follow Zipf's Law? How does it compare to non-literary corpora? (See Piantadosi, 2014, "Zipf's word frequency law in natural language.")
- The type-token ratio is a crude measure of lexical richness. Explore more sophisticated measures: Yule's K, Hapax Legomena ratio, and the moving average TTR (MATTR). The `lexicalrichness` Python package implements several of these.
- Subword tokenization (BPE, WordPiece) has largely replaced word-level tokenization in modern NLP. How would a BPE tokenizer handle Mulligan's "Chrysostomos"? Try it with HuggingFace's `tokenizers` library and compare.
- Joyce's use of unmarked interior monologue (free indirect discourse) blurs the boundary between narrator and character. Can you identify passages where the narrative voice shifts by looking at vocabulary or sentence structure alone? This question will return with force in later weeks.
- Connection to Week 10 (Wandering Rocks): that episode's fragmented, multi-perspective structure will test whether the profiling tools learned here can distinguish between voices and locations.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Load and tokenize** a literary text using NLTK's `word_tokenize` and `sent_tokenize`, and explain the decisions tokenizers make at word and sentence boundaries.
2. **Compute and interpret** basic corpus statistics (total tokens, total types, type-token ratio, average sentence length, hapax legomena ratio) and use them to compare texts quantitatively.
3. **Use concordance views** to perform computationally assisted close reading — identifying thematic clusters through the distributional context of keywords.
4. **Generate and interpret frequency distributions**, understanding the effect of stopword removal on what a frequency profile reveals about a text's content.
5. **Evaluate Zipf's Law** on literary text and articulate why rank-frequency regularities do (or don't) hold for a specific author's prose.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Telemachus) |
|---|---|---|
| Total tokens | `len(word_tokenize(text))` | ~15,000–16,000 |
| Total types (unique, lowercased alpha) | `len(set(...))` | ~3,500–4,500 |
| Type-token ratio (TTR) | types / alpha tokens | ~0.28–0.35 |
| Hapax legomena ratio | words appearing once / total types | ~0.45–0.55 |
| Average sentence length | tokens / sentences | ~12–18 words |
| Top content word (no stopwords) | most_common(1) | varies (likely a proper noun or thematic word) |
| Zipf's Law R² (log-log fit) | linear regression on log(rank) vs log(freq) | > 0.90 |

## Rubric

### Exercise 1: Tokenize and Profile (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Correct computation** | All statistics computed accurately; code handles punctuation tokens, edge cases | Statistics computed with minor errors or omissions | Major errors in token/type counting or TTR |
| **Comparison to reference** | Meaningful comparison to Gutenberg text with specific observations about what differs and why | Comparison present but observations are generic | No comparison or superficial side-by-side only |
| **Interpretation** | Insightful discussion of what TTR and sentence length reveal (and don't reveal) about Joycean prose vs. the reference | Some interpretation present | Statistics reported without interpretation |

### Exercise 2: Concordance as Close Reading (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Concordance generation** | Concordances for all 5 keywords generated and displayed correctly | Most keywords covered | Fewer than 3 keywords or technical errors |
| **Pattern identification** | Identifies non-obvious distributional patterns (e.g., *sea* and *mother* co-occurring in proximity) | Identifies at least one pattern | No patterns identified |
| **Argumentative paragraph** | Well-constructed argument connecting two words based on concordance evidence, with specific line citations | Argument present but vague or unsupported | No argument or purely impressionistic |

### Exercise 3: Frequency and Stopwords (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Frequency plots** | Both plots (with/without stopwords) are clear, well-labeled, and readable | Plots present but poorly formatted | Missing plots or major errors |
| **Stopword effect analysis** | Specific discussion of what emerges after stopword removal, with attention to proper nouns and thematic vocabulary | General observation that content words emerge | No analysis of the difference |
| **Surprise factor** | Identifies at least one genuinely surprising frequency finding and reflects on what it means for "aboutness" | Notes something interesting | No engagement with the results |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Zipf's Law plot with log-log regression and R² value | +3 |
| Discussion of lexical richness beyond TTR (Yule's K, MATTR) | +3 |
| Subword tokenization experiment (BPE on "Chrysostomos") | +2 |
| Free indirect discourse identification attempt | +2 |

## Reference Implementation

See [`solutions/week01_telemachus.py`](solutions/week01_telemachus.py)
