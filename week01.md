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
