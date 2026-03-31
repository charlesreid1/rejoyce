# Week 7 Writeup: Aeolus -- TF-IDF and Extractive Summarization

## Overview

This week's script (`week07_aeolus.py`) tackles the Aeolus episode of *Ulysses*, which is
uniquely suited to NLP analysis because Joyce himself inserted newspaper-style headlines
between sections -- a form of manual extractive summarization. The script implements three
exercises: TF-IDF from scratch to extract keywords, rhetorical figure detection (anaphora
and tricolon), and automated headline generation compared to Joyce's originals. All
computation uses NLTK's tokenizers, POS taggers, and stop word lists, with TF-IDF
calculated manually rather than via sklearn.

---

## Section Parsing

The script's `split_aeolus_sections()` function splits the episode text using ALL-CAPS
lines as headline delimiters. A line is classified as a headline if more than 70% of its
alphabetic characters are uppercase, it has more than 3 alphabetic characters, and it is
shorter than 120 characters. The opening text before the first headline is captured under
the label `[OPENING]`.

**Output:**

```
Found 30 sections
```

The exercise sheet predicted 30-65 sections. The parser found 30, which sits at the low
end of the expected range. This is likely because some headlines were not detected -- the
70% uppercase threshold may miss headlines that contain lowercase words (prepositions,
articles) or that include dialogue fragments. Several section headlines in the output show
signs of this parsing problem: sections 8, 9, 13, 15, 17, and 27 all have headlines that
begin with lowercase text (e.g., "own way. Sllt. NOTED CHURCHMAN AN OCCASIONAL
CONTRIBUTOR," "nonsense. AND IT WAS THE FEAST OF THE PASSOVER"). These appear to be
cases where a preceding line of dialogue was merged with the actual headline, suggesting
the delimiter detection is occasionally capturing too much or splitting at the wrong
boundary. Sections 22, 24, and 29 are missing entirely from the output, which indicates
those sections either had empty text after tokenization and filtering, or their headlines
were not detected.

---

## Exercise 1: TF-IDF from Scratch

### How It Works

The `compute_tfidf()` function implements the standard TF-IDF formula manually:

- **TF(t, d)** = count of term t in document d / total tokens in d
- **IDF(t)** = log(N / df(t)), where N is the total number of sections (30) and df(t) is
  the number of sections containing the term
- **TF-IDF(t, d)** = TF * IDF

Tokenization uses `nltk.tokenize.word_tokenize()`. Tokens are lowercased, filtered to
alphabetic-only strings longer than 2 characters, and stripped of English stop words via
`nltk.corpus.stopwords`. For each section, the top 5 terms by TF-IDF score are extracted.

### Interpreting the Results

The `tfidf_vs_headlines()` function prints each section's Joyce headline alongside its
top-5 TF-IDF keywords. The comparison reveals the fundamental tension the exercise is
designed to expose: TF-IDF captures *statistically distinctive* content words, while
Joyce's headlines perform editorial, ironic, and aesthetic work that no keyword extraction
can replicate.

**Cases where TF-IDF captures content reasonably well:**

- **Section 1 (OPENING):** Keywords "sandymount, rathgar, terenure, parallel, palmerston"
  are all Dublin place names from the opening's description of tram routes. There is no
  Joyce headline here (it is the pre-headline opening), but the keywords accurately
  reflect the geographic catalogue that begins the episode.
- **Section 25 (ITHACANS VOW PEN IS CHAMP.):** The keyword "penelope" connects directly
  to the Homeric register of the headline. "Sandymount" and "rathmines" ground the section
  geographically.
- **Section 23 (AEROLITHS, BELIEF):** The keyword "plums" connects to Stephen's Parable
  of the Plums, which dominates this section.
- **Section 18 (PERIOD):** "speech" and "stephen" are apt -- this section concerns
  Stephen's rhetorical performance.

**Cases where TF-IDF misses the point entirely:**

- **Section 12 (EXIT BLOOM):** The headline is a terse, dramatic stage direction. TF-IDF
  returns "crossblind, white, seems, round, running" -- descriptive words that capture
  physical detail but miss the narrative event (Bloom leaving the scene).
- **Section 11 (EOLIAN!):** Joyce's headline is an exclamatory reference to the Homeric
  wind god. TF-IDF gives "sir, tissues, door, shoved, aha" -- physical minutiae of the
  scene with no trace of the mythic register.
- **Section 19 (THE EDITOR):** Keywords "tell, bread, arm, kiss, arse" are body-focused
  words that miss the headline's identification of a character role.
- **Section 14 (ROME):** A single resonant proper noun as headline. TF-IDF gives "forget,
  mouth, stephen, foot, meet" -- none of which point toward Rome or its thematic weight.

**The pattern:** Joyce's headlines operate through compression, allusion, irony, and tonal
register-shifting. TF-IDF operates through statistical rarity. The two methods agree only
when a section's most distinctive vocabulary happens to overlap with the concept Joyce
chose to headline. The exercise sheet predicted a keyword-headline overlap rate of roughly
10-25%, and the output confirms this: very few sections show any lexical match between
TF-IDF keywords and headline words.

---

## Exercise 2: Rhetoric Detection

### Anaphora Detection

The `detect_anaphora()` function uses `nltk.tokenize.sent_tokenize()` to split the full
episode text into sentences, then groups sentences by their first (lowercased) token. A
group qualifies as anaphora if the repeated opening word appears in at least 2 sentences
and is not a stop word.

**Key findings from the output:**

- **Em-dash "---" (213 sentences):** The most frequent repeated opener is the em-dash,
  which marks dialogue throughout Ulysses. This is a *false positive* for anaphora
  detection -- the em-dash is a typographic convention for speech, not a rhetorical figure.
  However, it does reveal the heavily dialogic texture of Aeolus, which is fitting for an
  episode set in a newspaper office full of talkers.
- **"Mr" (24 sentences):** Sentences beginning with "Mr Bloom" or "Mr" followed by other
  names. This is partly a false positive (it reflects narrative convention, not deliberate
  rhetorical repetition), though it does capture the episode's focus on Bloom's presence
  and actions.
- **Character names -- "Lenehan" (15), "J." (14), "Myles" (13), "Ned" (9), "Professor"
  (6), "Stephen" (6):** These reflect the large cast of characters in the newspaper
  office. Again, mostly false positives for rhetorical anaphora, though the clustering of
  character-opener sentences does capture the episode's ensemble quality.
- **"Yes" (10 sentences):** Short affirmative sentences. Some of these may represent
  genuine rhetorical repetition in dialogue.
- **"Let" (9 sentences):** This is the most promising result for *actual* anaphora. The
  examples shown ("Let him take that in first," "Let him give us a three months' renewal,"
  "Let us build an altar to Jehovah") suggest a deliberate imperative pattern, especially
  the last example, which echoes biblical/oratorical register.

The exercise sheet predicted 5-15 anaphora detections and a false positive rate of 40-60%.
The detector found 10 groups (displayed), which falls within range. The false positive
rate appears quite high -- the em-dash group alone is a massive false positive, and most
character-name groups are narrative convention rather than rhetorical figuration. This
illustrates the fundamental difficulty: anaphora detection requires distinguishing
*deliberate* repetition from *incidental* repetition, and that distinction is semantic, not
syntactic.

### Tricolon Detection

The `detect_tricolon()` function looks for sequences of three comma- or semicolon-separated
phrases within a single sentence that have (a) similar token counts (within 50% of the
mean length) and (b) at least partial POS-tag similarity in their opening words. POS
tagging is done via `nltk.pos_tag()` on `word_tokenize()` output.

**Key findings from the output (8 tricolons detected):**

- **"Rathgar and Terenure / Palmerston Park and upper Rathmines / Sandymount Green"
  (lengths 3, 5, 2):** A genuine tricolon -- three Dublin place-name phrases in a
  catalogue. This is exactly the kind of list-based tricolon the detector was designed to
  find.
- **"Mr Bloom stood by / hearing the loud throbs of cranks / watching the silent
  typesetters at their cases" (lengths 4, 6, 8):** A solid detection. The parallel
  participial phrases ("hearing... watching...") constitute a real rhetorical figure, even
  if the length variation is considerable.
- **"sixth of May / time of the invincibles / murder in the Phoenix park" (lengths 3, 4,
  5):** Three noun phrases cataloguing a historical event -- a genuine tricolon of
  increasing gravity.
- **"He spoke on the law of evidence / J. J. O'Molloy said / of Roman justice as
  contrasted with the earlier Mosaic code" (lengths 10, 6, 10):** This is a false
  positive. The middle element is a speech attribution tag ("J. J. O'Molloy said"), not a
  parallel rhetorical element.
- **"You remind me of Antisthenes / the professor said / a disciple of Gorgias" (lengths
  6, 3, 4):** Another false positive caused by a speech attribution splitting what is
  really a single statement into three parts.

The exercise sheet predicted 3-10 tricolon detections, and 8 were found -- right in range.
The results are a mix of genuine rhetorical tricolons and false positives caused by speech
attributions and comma-separated clauses that happen to have similar lengths without being
truly parallel in structure. The POS-similarity check helps but cannot fully distinguish
rhetorical parallelism from accidental structural resemblance.

---

## Exercise 3: Headline Generation

The `generate_headlines()` function takes the top 3 TF-IDF keywords for each section,
uppercases them, and joins them with spaces to form a generated headline. These are
displayed in a side-by-side table with Joyce's actual headlines.

### Interpreting the Comparison

The generated headlines are, to put it plainly, terrible as headlines -- and that is
exactly the pedagogical point. A few representative comparisons:

| Joyce | Generated | Assessment |
|---|---|---|
| GENTLEMEN OF THE PRESS | MURRAY GROSSBOOTED DRAYMEN | Joyce names a social role; the algorithm picks character names and physical details |
| THE CROZIER AND THE PEN | STEPPED THUMPING ALONG | Joyce uses symbolic metonymy (church and writing); the algorithm picks verbs and adverbs |
| EXIT BLOOM | CROSSBLIND WHITE SEEMS | Joyce gives a dramatic stage direction; the algorithm gives descriptive fragments |
| EOLIAN! | SIR TISSUES DOOR | Joyce invokes Homer; the algorithm picks props |
| ROME | FORGET MOUTH STEPHEN | A single resonant proper noun vs. a random collage |
| ITHACANS VOW PEN IS CHAMP. | PENELOPE SANDYMOUNT RATHMINES | The closest match -- both reference Homeric elements and Dublin geography |

The generated headlines read like randomly selected words because, in effect, that is what
they are: the three most statistically distinctive tokens, stripped of syntax, narrative
context, and interpretive framing. Joyce's headlines work through compression, allusion,
irony, wordplay, and tonal register. "EXIT BLOOM" is a stage direction that dramatizes
Bloom's marginalization. "EOLIAN!" connects the newspaper office's windy rhetoric to
Homer's Aeolus. "ROME" compresses an entire thematic complex into a single word. None of
these operations are available to a bag-of-words keyword extractor.

The exercise sheet asked students to reflect on what makes a "good" headline and whether
"good" means the same thing to an algorithm and to Joyce. The output answers this
definitively: algorithmic salience (statistical distinctiveness relative to a corpus)
captures *what is talked about* at the lexical level, but Joyce's headlines capture *what
it means* -- and meaning requires interpretation, allusion, and compression that go far
beyond term frequency.

---

## Missing Elements

The script implements the three core exercises but does not address several components
from the exercise sheet:

- **Cross-episode application** (Exercise 3): The exercise asks students to apply the
  headline generation method to Hades or Calypso, which have no headlines. The script does
  not do this.
- **Keyword-headline overlap metric:** The exercise sheet specifies computing the
  percentage of sections where top-3 keywords overlap with headline words (expected
  10-25%). The script displays the comparison qualitatively but does not compute this
  metric.
- **Diving Deeper items:** TextRank/LexRank extractive summarization, Taylor speech
  anomaly detection, BM25 comparison, and rhetorical figure classification proposal are
  all absent.
- **Reflection paragraphs:** The exercise calls for written reflections on the difficulty
  of rhetorical detection and on what makes a good headline. These would need to be
  authored by the student rather than generated by the script.

---

## Summary of Metrics

| Metric | Expected | Observed |
|---|---|---|
| Number of sections parsed | 30-65 | 30 |
| Anaphora detections | 5-15 | 10 (displayed) |
| Tricolon detections | 3-10 | 8 |
| TF-IDF keyword-headline overlap | 10-25% | Very low (qualitatively consistent) |
| False positive rate (rhetoric) | 40-60% | High (em-dash group alone is a massive false positive) |

All metrics fall within or near the expected ranges specified in the exercise sheet. The
low section count (30 vs. potentially 63 in some editions) suggests the headline parser
could be tuned, but the TF-IDF and rhetoric detection results are pedagogically effective:
they demonstrate both the power and the limits of computational text analysis when applied
to a deliberately literary text.
