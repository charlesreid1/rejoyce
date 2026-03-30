# Week 7: Aeolus
### *"SUFFICIENT FOR THE DAY..." — Rhetoric, headlines, and the machinery of public language.*

Bloom enters the offices of the *Freeman's Journal* to place an ad; Stephen arrives to deliver Deasy's foot-and-mouth letter. Around them swirls the wind of rhetoric: editor Myles Crawford holds forth, professor MacHugh recites John F. Taylor's famous speech on the Irish language, and everyone talks in performative, quotable, oratorical prose. The episode's technique is *enthymemic* (the truncated syllogism of persuasion); its art is *rhetoric*; its organ is the *lungs*. Most conspicuously, Joyce interpolates newspaper-style **headlines** between sections — typographic intrusions that were added in revision and that range from stentorian ("IN THE HEART OF THE HIBERNIAN METROPOLIS") to parodic ("SOPPY WETDAY SITTING IN ALL THE PUBS") to cryptic. These headlines compress, distort, editorialize. They are Joyce's own exercise in automatic summarization.

**NLTK Focus:** TF-IDF, keyword extraction, and extractive summarization heuristics (`nltk.text`, `nltk.FreqDist`, TF-IDF via `sklearn` or manual computation, collocation measures as salience proxies)

**Pairing Rationale:**
A newspaper is a machine for deciding what matters. Headlines compress articles into a few words; editors select which stories run and how prominently. TF-IDF performs an analogous operation: it identifies the words that are most *distinctive* to a document relative to a corpus — the words that would make the best headlines. Joyce gave Aeolus its own headlines, which means students have a ground truth to work against: can a TF-IDF-based keyword extractor produce anything like the summaries Joyce wrote? The answer is instructively mixed. Joyce's headlines range from genuinely informative to deliberately misleading to pure sonic play — and only the first category is the kind that automated methods can approximate. The gap between computable salience and Joycean headlines teaches students what summarization algorithms actually capture and what remains stubbornly human.

**Core Exercises:**

1. **TF-IDF from scratch.** Divide Aeolus into sections using Joyce's headlines as delimiters. Treat each section as a separate "document" and the full set of sections as the corpus. Compute TF-IDF scores for every term in every section (implement the formula manually rather than using a library — understanding the math matters). For each section, extract the top 5 keywords by TF-IDF score. Compare these computationally extracted keywords to Joyce's actual headlines. Where does TF-IDF capture the section's content? Where does Joyce's headline do something that TF-IDF cannot — irony, wordplay, tonal commentary?

2. **Rhetoric detection.** The episode is dense with named rhetorical figures: anaphora, chiasmus, asyndeton, tricolon. Using NLTK's tokenizers and POS taggers, write pattern-detection functions for at least two figures. For **anaphora**: detect sequences of sentences or clauses that begin with the same word(s). For **tricolon**: detect sequences of three parallel phrases (you might approximate this as three consecutive phrases of similar POS-tag structure and similar length). Run these detectors on the episode. Do they find real rhetorical figures, or do they produce false positives? What makes rhetorical structure hard to detect computationally?

3. **Headline generation.** Using your TF-IDF keywords, attempt to generate a headline for each section of Aeolus. You can use a simple template (e.g., select the top 2–3 keywords and arrange them), or a more creative heuristic. Compare your generated headlines to Joyce's originals. Now apply the same method to sections of Hades or Calypso (which have no headlines). Do the generated "headlines" usefully characterize those sections? Write a paragraph reflecting on what makes a *good* headline — and whether "good" means the same thing to an algorithm and to Joyce.

**Diving Deeper:**

- Modern extractive summarization (TextRank, LexRank) uses graph-based methods to identify the most "central" sentences in a document. Apply TextRank (via `sumy` or `gensim.summarization`) to each section of Aeolus and compare the extracted sentences to the headlines. TextRank selects for representativeness; Joyce's headlines select for... what, exactly?
- The rhetorical figures in Aeolus have been catalogued by critics (notably Don Gifford's *Ulysses Annotated*). Stuart Gilbert identified over 90 distinct figures in the episode. This is one of the few literary classification tasks with an expert-annotated ground truth. Could you frame rhetorical figure detection as a supervised classification problem? What features would you need?
- John F. Taylor's speech on the Irish language, quoted in the episode, is a real historical text that Joyce repurposed. This raises questions about quotation, allusion, and textual reuse that connect to plagiarism detection and text reuse algorithms (shingling, MinHash). Can you detect the Taylor speech as a statistically anomalous passage within the episode's text?
- TF-IDF is the workhorse of classical information retrieval. Explore its successors: BM25, which adjusts for document length; dense retrieval with BERT embeddings. How do these methods change which terms are considered "important" in a literary text?
- Connection to Week 12 (Cyclops): that episode is narrated by an anonymous barfly and interrupted by parodic interpolations (gigantist pastiches of legal, scientific, and journalistic prose). Aeolus's headlines and Cyclops's interpolations are structural cousins — both force the reader to negotiate between competing textual registers. The classification tools from this week will be useful for distinguishing those registers.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Implement TF-IDF from scratch** (without sklearn), understanding the term frequency, inverse document frequency, and their product.
2. **Segment structured text** using typographic or formatting cues (headlines as section delimiters).
3. **Extract and evaluate keywords** computationally, and compare them to human-authored summaries (Joyce's headlines).
4. **Detect rhetorical figures** (anaphora, tricolon) using POS-based pattern matching, and assess precision/recall of the detectors.
5. **Articulate the gap** between computable salience and human editorial judgment in summarization.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Aeolus) |
|---|---|---|
| Number of headline-delimited sections | parsing ALL-CAPS lines | ~30–65 sections |
| Top TF-IDF keyword match rate | % of sections where top-3 keywords overlap with headline words | ~10–25% (low — that's the point) |
| Anaphora detections | sentence sequences sharing opening words | ~5–15 |
| Tricolon detections | triple phrases of similar POS structure and length | ~3–10 |
| False positive rate (rhetoric) | manually assessed precision of detectors | ~40–60% |

## Rubric

### Exercise 1: TF-IDF from Scratch (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Manual implementation** | TF-IDF formula implemented correctly from scratch (not via sklearn); math explained | Implementation works but formula not discussed | Used library or formula incorrect |
| **Section parsing** | Headline-delimited sections correctly identified; edge cases handled | Most sections parsed correctly | Parsing broken or missing |
| **Keyword-headline comparison** | For each section, top-5 TF-IDF keywords compared to Joyce's headline with specific commentary on matches and mismatches | Comparison for some sections | No comparison |

### Exercise 2: Rhetoric Detection (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Anaphora detector** | Working detector; results manually verified; precision assessed | Detector works; some verification | Detector broken or untested |
| **Tricolon detector** | POS-based parallel structure detection implemented; results shown | Attempted but weak results | Not attempted |
| **Difficulty reflection** | Explains why rhetorical structure is computationally hard (ambiguity, nesting, figures that span sentences) | Brief reflection | No reflection |

### Exercise 3: Headline Generation (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Generated headlines** | Template-based or heuristic headlines generated for all sections; compared side-by-side with Joyce | Headlines for some sections | Not attempted |
| **Cross-episode application** | Same method applied to Hades or Calypso; generated "headlines" discussed | Applied to one other episode | No cross-episode test |
| **Reflection on "good" headlines** | Paragraph on what makes a good headline and whether algorithmic vs. Joycean headlines optimize for different things | Brief reflection | No reflection |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| TextRank extractive summarization comparison | +3 |
| Rhetorical figure supervised classification proposal | +2 |
| Taylor speech anomaly detection (text reuse algorithms) | +3 |
| BM25 comparison to TF-IDF | +2 |

## Reference Implementation

See [`solutions/week07_aeolus.py`](solutions/week07_aeolus.py)
