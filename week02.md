# Week 2: Nestor
### *"A pier, a disappointed bridge" — History as nightmare, pedagogy as catechism, language as the medium of power.*

Nestor is the schoolroom chapter. Stephen teaches a listless history class, endures a hockey match, and suffers through a meeting with Mr. Deasy — a pompous, anti-Semitic headmaster who dispenses clichés as wisdom and wants Stephen to deliver his letter about foot-and-mouth disease to the newspapers. The episode's art is *history*; its technique is *catechism* (personal). Language here is transactional, pedagogical, authoritative — or trying to be. Deasy speaks in received phrases and prefabricated opinions. Stephen's interior life pushes against the episode's rigid surfaces.

**NLTK Focus:** Part-of-speech tagging and basic morphological analysis (`nltk.pos_tag`, `nltk.corpus.wordnet` for lemmatization, tagged corpus exploration)

**Pairing Rationale:**
POS tagging is grammatical taxonomy — the assignment of each word to its syntactic role. It is, in a sense, the schoolmaster's art: labeling, categorizing, imposing order on the unruly stream of language. Nestor is a chapter about people who think they know what things are and what they mean. Deasy is certain about history, about money, about Jews, about the role of women. POS tagging shares this confident taxonomic impulse — and, like Deasy, it is sometimes wrong in instructive ways. The episode's relatively conventional prose makes it a clean testbed for taggers, while its thematic concern with education and authority gives students a reason to think about what it means to *label* language computationally.

**Core Exercises:**

1. **Tag and tabulate.** POS-tag the full text of Nestor using NLTK's default tagger (`pos_tag` with the Penn Treebank tagset). Generate frequency counts for each tag. What is the ratio of nouns to verbs? Of adjectives to adverbs? Compare these ratios to the Brown Corpus (use `nltk.corpus.brown.tagged_words()`). Nestor is a chapter full of assertions and proclamations — does the POS distribution reflect this?

2. **Deasy vs. Stephen.** Separate Deasy's dialogue from Stephen's interior monologue (you'll need to do some manual or semi-manual segmentation — this is part of the exercise). POS-tag each subcorpus separately. Compare their distributions. Hypothesize: does Deasy use more nouns (the language of things and categories)? Does Stephen use more verbs, more abstract or modified constructions? Test your hypothesis quantitatively.

3. **Lemmatization and the weight of history.** Use `WordNetLemmatizer` to lemmatize all words in the episode. Identify the top 20 lemmas that appear more frequently in Nestor than in Telemachus (normalized by episode length). Do these lemmas capture the thematic shift — from sea and tower to school, money, history? Reflect on what lemmatization gains and loses: when Joyce writes *riddles* and *riddled*, should those map to the same base form?

**Diving Deeper:**

- NLTK's default POS tagger is a perceptron model trained on the Penn Treebank. Investigate how it was trained. What happens when you apply it to non-standard English? Tag a passage of Hiberno-English dialogue from Ulysses and assess the errors. Would a tagger trained on Irish English corpora perform better? (See the ICE-Ireland corpus.)
- The Penn Treebank tagset has 36 tags. The Universal Dependencies tagset has 17. Try mapping between them using `nltk.tag.mapping`. What distinctions are lost? What is gained? This is a question about the granularity of linguistic categories — relevant to Deasy's habit of collapsing distinctions.
- spaCy's POS tagger uses a neural model and tends to outperform NLTK's on modern text. Run the same analysis in spaCy and compare accuracy. Where do they disagree, and who is right?
- Explore NLTK's tagged corpora (`brown`, `treebank`, `conll2000`). These are the training data behind the tools we use. What biases do they encode? What kinds of English are over- or under-represented? (This connects to broader questions about whose language gets to be the "standard" — not unlike Deasy's assumptions about whose history matters.)
- Connection to Week 17 (Ithaca): that episode is structured as a literal catechism (Q&A). The pedagogical labeling impulse of POS tagging will find its ultimate Joycean expression there.
