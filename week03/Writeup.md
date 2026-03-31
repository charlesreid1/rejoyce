# Week 3 Writeup: Proteus -- Stemming & Language Identification

## Overview

This week's script (`week03_proteus.py`) applies three NLTK stemming algorithms to the Proteus episode of *Ulysses*, builds a stopword-based language detector for Joyce's multilingual passages, and performs derivational morphology analysis on twenty unusual words from the text. The output illuminates both the capabilities and the sharp limitations of rule-based NLP when confronted with Joyce's deliberately shape-shifting prose.

---

## Exercise 1: The Stemmer's Struggle

### What the code does

The function `stemmers_struggle()` applies three NLTK stemmers -- `PorterStemmer`, `LancasterStemmer`, and `SnowballStemmer('english')` -- to every unique alphabetic token of three or more characters in the episode. For each word-stem pair, it computes `nltk.metrics.distance.edit_distance` (the Levenshtein distance between the original word and its stem). The results are sorted by descending edit distance, and the top 10 most aggressive reductions are printed per stemmer. Finally, the code computes a disagreement rate: the percentage of unique words where the three stemmers produce at least two distinct stems.

### Interpreting the output

**Porter and Snowball** produce identical top-10 lists. This is expected: the Snowball English stemmer is a direct reimplementation of Porter's algorithm in the Snowball framework, so on standard English input they agree almost perfectly. Their most aggressive reductions reach edit distance 6 (e.g., "witnesses" -> "wit", "imitations" -> "imit") and distance 5 for several others ("delectation" -> "delect", "hospitality" -> "hospit", "carefully" -> "care"). The massive Joycean coinage "contransmagnificandjewbangtantiality" is stemmed to "contransmagnificandjewbangtanti" (distance 5) -- the stemmer only manages to nibble the Latin suffix "-ality" off the end of a 38-character monster.

**Lancaster** is markedly more aggressive. It achieves an edit distance of 9 on "contransmagnificandjewbangtantiality" (stripping it down to "contransmagnificandjewbangt") and produces several striking over-reductions: "existence" -> "ex" (distance 7), "socialiste" -> "soc" (distance 7), "signatures" -> "sign" (distance 6), "squealing" -> "squ" (distance 6), "carefully" -> "car" (distance 6). Lancaster applies more aggressive iterative rules and frequently reduces words to fragments that no longer carry recognizable meaning. "Squealing" -> "squ" and "existence" -> "ex" are essentially destructive reductions where the stem has lost its semantic connection to the original word.

**Disagreement rate: 36.6%** (813 out of 2222 unique words). This falls squarely within the exercise's expected range of 25-40% and reflects the substantial algorithmic differences between Lancaster and the Porter/Snowball pair. On standard English text, disagreement rates are typically lower; Joyce's vocabulary of foreign words, compounds, and neologisms pushes the rate toward the upper end.

### Which stemmer handles Joyce best?

Porter and Snowball are clearly more conservative and produce fewer absurd results. Lancaster's aggressive iterative stripping is designed to collapse many surface forms to a common root, but on Joyce's unusual vocabulary this aggressiveness becomes destructive. The "best failures" -- Lancaster reducing "existence" to "ex" and "squealing" to "squ" -- illustrate that more aggressive stemming is not necessarily better stemming, especially on literary text full of unusual morphology.

### Connection to the episode

Proteus is the chapter of metamorphosis, where words and perceptions refuse to hold a fixed shape. Stemming assumes there is a stable root beneath inflected forms. Joyce's vocabulary -- compounds like "peacocktwittering" and "abstrusiosities," the epic neologism "contransmagnificandjewbangtantiality" -- is designed to resist exactly this kind of reduction to essences. The stemmers' struggle with these words enacts the episode's philosophical argument: identity is not so easily pinned down.

---

## Exercise 2: Multilingual Detection

### What the code does

The function `detect_languages()` builds a sliding-window (sentence-level) language detector. For each sentence, it tokenizes the text, lowercases all alphabetic tokens, and computes the proportion of tokens that appear in each of five stopword lists: English (from `nltk.corpus.stopwords`), French, German, Italian, and a hand-curated Latin list defined in the `LATIN_STOPWORDS` set. The language with the highest stopword overlap score wins; if the top non-English score is below 0.1, the sentence defaults to English.

A second function, `non_english_token_proportion()`, takes a different approach: it builds a set of all English words from WordNet's synsets and counts how many tokens in the episode are absent from that set.

### Interpreting the output

**Language detection summary:**
- 683 of 709 sentences (96.3%) are classified as English
- 16 sentences (2.3%) as French
- 6 sentences (0.8%) as Italian
- 3 sentences (0.4%) as Latin
- 1 sentence (0.1%) as German

The exercise predicted 3-8% non-English sentences; the output yields 3.7%, right at the low end of that range.

**Where the detector succeeds:** The French detections are largely accurate. The script correctly identifies the extended French conversation Stephen recalls from his time in Paris: "Qui vous a mis dans cette fichue position?", "C'est le pigeon, Joseph", "Moi, je suis socialiste", "Je ne crois pas en l'existence de Dieu", "Mon pere, oui". These are complete French sentences with high stopword density, making them easy targets for the heuristic.

The Italian detections include "O si, certo!" (correctly Italian) and "Pico della Mirandola like" (a mixed sentence -- the name is Italian, but "like" is English; the detector picks Italian because "della" appears in the Italian stopword list).

**Where the detector fails or is ambiguous:**
- "De boys up in de hayloft" is classified as Latin because "de" appears in the Latin stopword list. This is actually dialectal English (or Hiberno-English), where "de" is a phonetic spelling of "the." The detector has no way to distinguish homographs across languages.
- "Descende, calve, ut ne nimium decalveris" is classified as French, but this is actually Latin. The words "ne" and "ut" appear in both French and Latin stopword lists, and "de" prefixed forms are common in French. The small Latin stopword list loses to the larger French list.
- "Hat, tie, overcoat, nose" is classified as German, which is clearly wrong -- this is plain English. The word "hat" happens to appear in the German stopword list (it is the German third-person singular of "haben"). This is a classic false positive from stopword-based detection.
- The single-word or very short sentences throughout the episode are particularly vulnerable to misclassification because a single stopword match can swing the entire score.

**Non-English token proportion: 35.1%** (2022 of 5760 alpha tokens). This is far above the exercise's expected range of 5-15%. The sample of "non-English" tokens reveals the problem immediately: the list includes common English words like "the", "that", "and", "was", "them". These are function words that do not appear as lemmas in WordNet because WordNet is a lexical database of content words (nouns, verbs, adjectives, adverbs). Stopwords and function words are systematically absent from WordNet. The heuristic of using WordNet as an English dictionary is therefore badly flawed -- it massively overestimates the non-English proportion by misclassifying the most frequent English words as non-English.

### Connection to the episode

The detector's failures are thematically fitting. Proteus is Joyce's chapter about the instability of perception and the slipperiness of language. Stephen's macaronic sentences -- switching between English, French, Latin, and Italian within a single thought -- are designed to defeat exactly this kind of classification. The shared vocabulary between Romance languages ("de," "ne," "non") creates a zone of ambiguity that the detector cannot resolve, just as Stephen's consciousness refuses to settle into a single linguistic register.

---

## Exercise 3: Derivational Morphology and Neologism

### What the code does

The function `morphological_analysis()` takes a list of 20 unusual words from Proteus and looks each one up in WordNet. For words found in WordNet, it prints the number of synsets and the first definition. For words not in WordNet, the helper function `hypothesize_parse()` attempts a morphological decomposition: it tries splitting the word at every position and checking whether each half appears in WordNet, then checks for known prefixes. If no decomposition works, the word is labeled "Sui generis / foreign borrowing."

### Interpreting the output

**WordNet coverage: 9 out of 20 words (45%).** This falls within the exercise's expected range of 40-60%.

**Words found in WordNet (9):**
- *ineluctable* -- "impossible to avoid or evade" (1 synset). A learned English word from Latin *ineluctabilis*.
- *maestro* -- "an artist of consummate skill" (1 synset). An Italian borrowing fully absorbed into English.
- *dogsbody* -- "a worker who has to do all the unpleasant or boring work" (1 synset). British/Irish slang.
- *childbed* -- "concluding state of pregnancy" (1 synset). An archaic English compound.
- *deathbed* -- "the last few hours before death" (2 synsets). Standard English compound.
- *omphalos* -- "a scar where the umbilical cord was attached" (1 synset). Greek borrowing for navel/center.
- *augur* -- 3 synsets. Latin-origin word meaning a Roman religious official or to predict.
- *protean* -- "taking on different forms" (1 synset). From the Greek god Proteus -- the episode's namesake.
- *metempsychosis* -- "after death the soul begins a new cycle of existence" (1 synset). Greek philosophical term that recurs throughout *Ulysses* (Molly's famous garbling of it in Calypso).

**Words not in WordNet (11):**

Successfully decomposed compounds (both halves in WordNet):
- *snotgreen* -> "snot" + "green" -- a Joycean compound adjective (from the opening of Telemachus, recurring here).
- *scrotumtightening* -> "scrotum" + "tightening" -- another Joycean compound adjective describing the sea.
- *seaspawn* -> "sea" + "spawn" -- compound noun.
- *wavespeech* -> "wave" + "speech" -- compound noun, personifying the sea.
- *bridebed* -> "bride" + "bed" -- compound noun, parallel to the known words "childbed" and "deathbed."

These five compounds follow productive English compounding rules (noun+noun or noun+adjective). They are not in the dictionary but are immediately interpretable -- Joyce exploits the open-ended productivity of English compounding.

Partially decomposed (one half in WordNet):
- *nacheinander* -> "nac" + "heinander" -- This is actually a German word meaning "one after another." The decomposition is spurious; "nac" appears in WordNet only by accident (it may match an obscure entry). The word is a straight German borrowing.
- *nebeneinander* -> "neb" + "eneinander" -- Also German, meaning "next to one another." Same problem: "neb" (a beak) is a real English word but irrelevant here.
- *diaphane* -> "dia" + "phane" -- From Greek *diaphanes* (transparent). The split into "dia" (a variant of "diya"?) + "phane" is not meaningful; this is a Greek/French borrowing.
- *adiaphane* -> "adiaph" + "ane" -- Joyce's coinage: the negation of "diaphane" (the opaque). The split is again spurious.
- *contransmagnificandjewbangtantiality* -> "con" + "transmagnificandjewbangtantiality" -- The decomposer finds "con" in WordNet but cannot parse the remainder. This is Joyce's famous theological burlesque, packing "contra," "trans," "magnificent," "Jew," "bang," "substantial," and "-ity" into a single impossible word. A full decomposition would require recursive compound splitting.

Labeled as sui generis:
- *thalatta* -- The Greek exclamation meaning "the sea! the sea!" (from Xenophon's *Anabasis*). Correctly identified as a foreign borrowing that resists English morphological analysis.

### Connection to the episode

The morphological analysis reveals three categories of Joyce's unusual vocabulary: (1) productive English compounds that any speaker could coin and understand (snotgreen, seaspawn, wavespeech); (2) foreign borrowings from the languages of Stephen's education -- German philosophical terms, Greek classical references, Italian musical terms; and (3) sui generis coinages like "contransmagnificandjewbangtantiality" that pack multiple languages and morphemes into a single portmanteau. WordNet covers the established borrowings and the standard English words but cannot handle the compounds or coinages. This is precisely the limit of dictionary-based NLP: it can only recognize words that have been previously catalogued. Joyce's lexical creativity -- the Protean shape-shifting of language -- operates in the spaces between dictionaries.

---

## Summary of Metrics

| Metric | Expected Range | Observed | Notes |
|---|---|---|---|
| Stemmer disagreement rate | 25-40% | 36.6% | Within range; driven by Lancaster divergence |
| Max edit distance (Lancaster) | > 8 | 9 | On "contransmagnificandjewbangtantiality" |
| Non-English sentence proportion | 3-8% | 3.7% | At low end of range |
| Non-English token proportion | 5-15% | 35.1% | Far above range; WordNet heuristic is flawed |
| WordNet coverage of target words | 40-60% | 45% | Within range |
| Successful compound decomposition | 30-50% | ~45% (5/11 non-WN words) | Within range |
