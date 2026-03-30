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

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Apply POS tagging** to literary text using NLTK's perceptron tagger and the Penn Treebank tagset, and interpret the resulting tag distributions.
2. **Compare POS distributions** across texts and corpora quantitatively, using ratios (noun/verb, adjective/adverb) as stylistic indicators.
3. **Segment text** by speaker or voice (dialogue vs. narration) and use POS profiles to characterize distinct registers within a single episode.
4. **Lemmatize text** using WordNet-aware lemmatization and evaluate what lemmatization preserves and destroys in literary language.
5. **Identify distinctive vocabulary** between two texts using normalized frequency comparison of lemmatized forms.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Nestor) |
|---|---|---|
| Total tagged tokens | `len(pos_tag(word_tokenize(text)))` | ~7,000–8,500 |
| Noun/Verb ratio | NN*/VB* tag counts | ~1.2–1.6 |
| Adjective/Adverb ratio | JJ*/RB* tag counts | ~0.8–1.4 |
| POS distribution divergence from Brown | per-tag percentage difference | NOUN typically +2–5% vs. Brown |
| Dialogue/Interior token split | heuristic segmentation | ~30–40% dialogue |
| Unique lemmas | `len(set(lemmatized))` | ~2,000–2,800 |
| Distinctive lemmas (Nestor vs. Telemachus) | top-20 by normalized freq diff | expect *school*, *money*, *history*-adjacent |

## Rubric

### Exercise 1: Tag and Tabulate (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **POS tagging accuracy** | Correctly applied tagger; frequency table for top tags is accurate and well-formatted | Minor errors in counting or tag grouping | Major errors or missing tag groups |
| **Brown Corpus comparison** | Quantitative comparison with specific percentage differences noted; discusses what the deltas mean for Nestor's prose | Comparison present, observations generic | No comparison to external corpus |
| **Interpretation of ratios** | Connects noun/verb ratio to the episode's assertive, declarative character; discusses what high noun ratio means for a "schoolroom chapter" | Some interpretation | Ratios reported without analysis |

### Exercise 2: Deasy vs. Stephen (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Segmentation method** | Clear, documented heuristic for splitting dialogue from interior; acknowledges limitations and edge cases | Working segmentation with some documentation | No segmentation or broken method |
| **Quantitative comparison** | POS profiles for both voices computed and compared; specific differences highlighted with counts | Comparison present but thin | Profiles computed but not compared |
| **Hypothesis testing** | Explicit hypothesis stated, tested against data, and evaluated honestly (whether confirmed or refuted) | Hypothesis present but loosely tested | No hypothesis or untested assertion |

### Exercise 3: Lemmatization (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Correct lemmatization** | WordNet-aware lemmatization with POS mapping; top-20 distinctive lemmas identified | Lemmatization works but POS not used | Lemmatization errors or not attempted |
| **Thematic interpretation** | Distinctive lemmas connected to episode themes (school, money, history); reflection on what the thematic shift looks like computationally | Some thematic connection made | Lemmas listed without interpretation |
| **Lemmatization loss** | Specific examples of meaningful distinctions collapsed by lemmatization (e.g., *riddles/riddled*); reflection on when lemmatization helps vs. hinders | At least one example discussed | No discussion of limitations |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Tagger error analysis on Hiberno-English passages | +3 |
| Penn Treebank ↔ Universal Dependencies mapping comparison | +2 |
| spaCy vs. NLTK tagger comparison with accuracy assessment | +3 |
| Exploration of tagged corpus biases (Brown, Treebank) | +2 |

## Reference Implementation

See [`solutions/week02_nestor.py`](solutions/week02_nestor.py)
