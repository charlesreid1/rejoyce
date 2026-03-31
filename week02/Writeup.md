# Week 2 Writeup: Nestor -- POS Tagging & Morphological Analysis

## Overview

This week's script (`week02_nestor.py`) applies NLTK's part-of-speech tagger, voice segmentation heuristics, and WordNet-aware lemmatization to Episode 2 of *Ulysses* ("Nestor"). The three exercises map onto the episode's thematic concerns: taxonomic labeling (the schoolmaster's art), the contrast between Deasy's authoritative speech and Stephen's interior life, and the weight of distinctive vocabulary as the novel shifts from tower and sea to school, money, and history.

---

## Exercise 1: Tag and Tabulate

### What the code does

The function `tag_and_tabulate()` tokenizes the full text of Nestor with `word_tokenize()`, then POS-tags every token using `nltk.pos_tag()` (NLTK's pretrained averaged perceptron tagger, which assigns Penn Treebank tags). It counts tag frequencies with a `Counter`, groups tags into broad categories (all NN* tags as nouns, all VB* as verbs, JJ* as adjectives, RB* as adverbs), and computes two key ratios: noun/verb and adjective/adverb.

The function `compare_to_brown()` then maps the Penn Treebank tags to the Universal Dependencies tagset via a hand-built dictionary and compares the percentage distribution to the Brown Corpus (loaded via `nltk.corpus.brown.tagged_words(tagset='universal')`).

### Interpreting the output

**Top 15 POS tags in Nestor:**

| Tag | Count | What it represents |
|-----|------:|---------------------|
| NN | 831 | Singular common nouns |
| . | 505 | Sentence-final punctuation |
| IN | 491 | Prepositions / subordinating conjunctions |
| DT | 463 | Determiners |
| NNP | 404 | Proper nouns (singular) |
| , | 339 | Commas |
| PRP | 308 | Personal pronouns |
| VBD | 300 | Past tense verbs |
| JJ | 298 | Adjectives |
| NNS | 244 | Plural nouns |
| RB | 200 | Adverbs |
| VB | 166 | Base form verbs |
| PRP$ | 157 | Possessive pronouns |
| CC | 155 | Coordinating conjunctions |
| VBP | 103 | Non-3rd person singular present verbs |

The **noun/verb ratio is 1.827** and the **adjective/adverb ratio is 1.529**. The exercise sheet predicted a noun/verb ratio in the range 1.2--1.6 and an adj/adv ratio of 0.8--1.4. Both actual values land above the predicted ranges, suggesting Nestor is even more noun-heavy and adjective-heavy than expected. This makes thematic sense: the episode is dominated by assertions, declarations, and categorical statements -- Deasy naming things, Stephen cataloguing his impressions of the schoolroom. The high noun count reflects a prose style oriented toward objects, categories, and facts rather than actions and processes.

The dominance of VBD (past tense verbs, 300 occurrences) over VB (base form, 166) and VBP (present non-3rd, 103) is notable. Nestor is a chapter about history -- "a nightmare from which I am trying to awake" -- and its grammar leans past tense, narrating what has already happened.

**Brown Corpus comparison:**

The universal-tag comparison reveals:

- **NOUN: +2.98% over Brown.** Nestor is more noun-dense than standard edited American English, consistent with the declarative, taxonomic style of the episode.
- **PRON: +4.74% over Brown.** A striking surplus of pronouns. This reflects the dialogue-heavy structure -- Deasy saying "I," "you," "they," and Stephen's interior "he," "his," constantly deictic, constantly pointing.
- **X: +20.28% over Brown.** This is the largest discrepancy and represents a mapping problem (see TODO). Tags that do not map to any Universal category in the hand-built dictionary fall into "X." This includes punctuation tags like `.` and `,`, the CD (cardinal number) tag, possessives, particles (RP, TO), and others. The Brown Corpus uses its own universal mapping that correctly classifies these, while the script's manual mapping omits many Penn Treebank tags. This inflates "X" and deflates other categories (note ADP at -3.62%, DET at -3.46%, PRT at -2.57%).
- **ADJ: -1.59%, ADV: -1.17%.** Slightly fewer modifiers than Brown, though the raw adj/adv ratio within Nestor is high. The deficit is relative and may be partly an artifact of the mapping issue.

---

## Exercise 2: Deasy vs. Stephen

### What the code does

The function `split_deasy_stephen()` uses a simple heuristic: lines beginning with an em-dash (`---` or `--`) are classified as dialogue; all other non-empty lines become "interior/narration." This captures Joyce's typographical convention (he uses em-dashes instead of quotation marks to introduce speech), but it conflates all speakers' dialogue, not just Deasy's -- Stephen's spoken words, the boys' shouts, and any other character's speech all land in the "Dialogue" bin. The function `compare_voices()` then POS-tags each subcorpus separately and computes the same ratios.

### Interpreting the output

**Token counts:**
- Dialogue: 1,527 tokens
- Interior/Narration: 4,022 tokens

This gives dialogue roughly 27.5% of the text, slightly below the exercise sheet's predicted 30--40%. The split is reasonable for Nestor, where much of the episode is narrated action and Stephen's interior thought, with dialogue concentrated in the Deasy interview.

**POS distributions and ratios:**

| Metric | Dialogue | Interior/Narration |
|--------|:--------:|:------------------:|
| Noun/Verb ratio | 1.359 | 2.127 |
| Adj/Adv ratio | 1.339 | 1.579 |
| Noun count | 382 | 1,121 |
| Verb count | 281 | 527 |

The exercise hypothesized that Deasy (dialogue) would use more nouns -- "the language of things and categories" -- while Stephen (interior) would use more verbs and modified constructions. The data **refutes this hypothesis**, or at least complicates it:

- The **interior/narration** text has a much higher noun/verb ratio (2.127 vs. 1.359). Narration and interior monologue are more noun-dense than dialogue.
- **Dialogue** is more verb-dense in relative terms. Spoken language naturally involves more verb usage: imperatives, questions, declaratives with active verbs ("I told you," "you will see," "come here").

This is actually a well-known finding in corpus linguistics: speech tends to be more verb-heavy than written or narrative prose. The narration packs in nouns because it describes scenes, objects, and settings; dialogue moves through actions and states. So Deasy's speech is not more noun-heavy -- it is more verb-driven, as spoken registers typically are. The exercise's hypothesis was a useful starting point that the data overturns in an instructive way.

The adjective/adverb ratio is higher in the interior text (1.579 vs. 1.339), indicating slightly more adjectival modification in the descriptive, narrated passages. This also tracks with the distinction between action-oriented dialogue and descriptive narration.

A notable feature of the dialogue POS distribution: NNP (proper nouns) is the single most frequent tag at 193 occurrences, outranking even NN (165). This reflects the heavy use of names and forms of address in the dialogue -- "Mr Deasy," "Sargent," "Stephen" -- the social apparatus of the schoolroom and the headmaster's office.

---

## Exercise 3: Lemmatization and the Weight of History

### What the code does

The function `lemmatize_and_compare()` lemmatizes both Nestor and Telemachus using `WordNetLemmatizer`. Crucially, it uses POS-aware lemmatization: each token's Penn Treebank tag is mapped to a WordNet POS category (noun, verb, adjective, adverb) via `get_wordnet_pos()`, and this is passed to the lemmatizer. This matters because "left" lemmatizes to "leave" only if the tagger identifies it as a verb; as a noun or adjective, it would remain "left." Only alphabetic tokens are included, and a minimum frequency threshold of 3 is applied.

The distinctive lemma score is computed as the difference in normalized frequencies: `(count_in_nestor / total_nestor) - (count_in_telemachus / total_telemachus)`. The top 20 lemmas by this measure are reported.

The function `lemmatization_loss_examples()` then finds all cases where multiple surface forms collapse to a single lemma, illustrating what lemmatization destroys.

### Interpreting the output

**Top 20 distinctive lemmas (Nestor vs. Telemachus):**

The most distinctive lemmas are: *mr*, *be*, *deasy*, *sir*, *of*, *their*, *the*, *know*, *a*, *they*, *what*, *will*, *this*, *back*, *no*, *do*, *but*, *never*, *sargent*, *just*.

Thematic observations:

- **mr** and **sir**: The language of formal address, hierarchy, and the schoolroom. Nestor is saturated with the social scaffolding of student-teacher, employee-employer relationships. These lemmas capture the episode's institutional register.
- **deasy** and **sargent**: Character names unique to this episode. Their presence as top distinctive lemmas is expected but not very informative thematically -- they are proper names that simply do not appear in Telemachus.
- **know**, **will**, **never**: Epistemologically loaded words. Deasy claims to know things; Stephen grapples with what can be known about history. "Will" captures Deasy's assertive future orientation ("you will see"). "Never" resonates with the episode's sense of historical foreclosure.
- **their**, **they**: Third-person plural pronouns. In Nestor, Deasy talks about groups -- the English, the Jews, women -- in categorical terms. This pronoun surplus reflects his habit of making sweeping claims about collectives.
- **what**: The interrogative mode. The schoolroom runs on questions and answers (the catechism technique). "What" marks the pedagogical and inquisitive register.

The exercise sheet predicted lemmas adjacent to *school*, *money*, and *history*. The actual top 20 are dominated by function words and character names rather than content words. This is a limitation of the normalized frequency difference method when applied without stopword filtering: function words like *the*, *a*, *of*, *but* overwhelm content words because even small percentage differences in very frequent words produce large absolute differences. The thematically revealing lemmas (*know*, *will*, *never*) are present but buried among function words.

**Lemmatization collapses:**

The output lists cases where multiple surface forms reduce to one lemma. Notable examples:

- **know** <- *knew, known, knows*: All three temporal perspectives on knowing collapse into one form. In Nestor, the distinction between what was known, what is known, and what is knowable matters enormously to the episode's meditation on history.
- **leave** <- *leaves, left*: Here, lemmatization potentially confuses two etymologically distinct meanings. "Leaves" (noun, foliage) and "left" (verb, departed) may not share meaning even if they share a lemma. The tagger must get the POS right for the lemmatizer to handle this correctly.
- **star** <- *stared, stars*: This is a clear error. "Stared" (past tense of "stare") has been lemmatized to "star" rather than "stare," because the tagger may have misidentified the POS, or the lemmatizer's lookup defaulted incorrectly. This is a meaningful bug -- "stare" and "star" are different words.
- **sin** <- *sinned, sins*: Lemmatization correctly groups these, but the theological resonance of the inflected forms (the past tense "sinned" carrying weight of completed action, historical guilt) is lost.
- **bear** <- *born, borne*: These collapse into one lemma, but "born" (brought into existence) and "borne" (carried, endured) occupy different semantic spaces in Joyce's text.
- **swallow** <- *swallowed, swallowing*: Correctly grouped, capturing a physical action repeated in the episode.

The exercise prompt specifically asks about *riddles* and *riddled* -- whether they should map to the same base form. These do not appear in the output, likely because they do not both occur in Nestor with sufficient frequency, or because the POS tagger assigns them different categories (noun vs. adjective) that route them to different lemmas.

---

## Summary

The three exercises reveal Nestor as a noun-heavy, past-tense-dominated, formally addressed episode. The Brown Corpus comparison confirms its noun surplus, though the universal tag mapping has significant gaps that inflate the "X" category. The voice comparison overturns the intuitive hypothesis that Deasy's speech would be more nominal -- in fact, dialogue is more verb-dense, consistent with general patterns of spoken vs. written language. The lemmatization exercise surfaces distinctive vocabulary that captures the episode's institutional and epistemological register, though function words dominate the top 20 in the absence of stopword filtering. The lemmatization collapse analysis reveals both the power and peril of reducing inflected forms: meaningful temporal and semantic distinctions are lost alongside genuinely redundant variation.
