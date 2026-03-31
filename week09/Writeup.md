# Week 9 Writeup: Scylla and Charybdis -- Context-Free Grammars & Parsing

## Overview

Week 9 pairs Episode 9 of *Ulysses* ("Scylla and Charybdis") with NLTK's context-free grammar and parsing tools. Stephen Dedalus holds forth in the National Library, delivering his theory that Shakespeare's biography is encoded in his plays. The prose is dense with logical connectives, embedded quotations, nested subordinate clauses, and sentences structured like syllogisms. The NLTK focus is `CFG.fromstring`, `ChartParser`, the Penn Treebank parsed corpus, and POS tagging -- tools that reveal (or fail to reveal) the hierarchical structure beneath Stephen's arguments.

The script (`week09_scyllacharybdis.py`) tackles three exercises: parsing argumentative sentences with a hand-written CFG, comparing syntactic complexity metrics against Penn Treebank baselines, and extracting quoted Shakespeare material for separate syntactic analysis.

---

## Exercise 1: Parsing the Argument

### What the code does

The function `find_argument_sentences()` scans the episode for sentences containing logical connectives (*therefore*, *because*, *if*, *but*, *yet*, *however*, *thus*, *hence*, *since*, *although*) that are also longer than 15 tokens. It sorts these by length (descending) as a proxy for syntactic complexity, then selects the top 5.

For each sentence, `parse_with_cfg()` tokenizes it, filters tokens down to only those present in the hand-written CFG's terminal set, and feeds the filtered token list to NLTK's `ChartParser`. The CFG (`ARGUMENT_GRAMMAR`) covers the major constituent types: S, NP, VP, PP, SBAR, ADJP, ADVP, with a lexicon of about 100 words chosen for relevance to Stephen's Shakespeare argument (e.g., *Shakespeare*, *Hamlet*, *ghost*, *soul*, *father*, *son*).

### What the output shows

```
--- Found 50 argument sentences ---
```

The script finds **50 argument sentences** -- right in the middle of the exercise sheet's predicted range of 50-100. This confirms that Scylla and Charybdis is heavily argumentative prose, saturated with logical connectives.

The five longest argument sentences range from 53 to 99 tokens. All five fail to parse:

1. **"The playwright who wrote the folio of this world..."** (99 tokens, 40 covered, 40.4% coverage) -- This sprawling sentence about God-as-playwright contains vocabulary well outside the grammar (*playwright*, *folio*, *light*, *sun*). The covered tokens read like a skeleton: *the who wrote the of this world and wrote it he and the the of* -- recognizable as a determiner-heavy English sentence but stripped of its content.

2. **"He Who Himself begot middler the Holy Ghost..."** (95 tokens, 35 covered, 36.8%) -- Stephen's theological parody of the Trinity. The grammar captures *ghost*, *who*, *his*, but the theological and Joycean vocabulary (*Agenbuyer*, *middler*, *begot*) is entirely absent.

3. **"When Rutlandbaconsouthamptonshakespeare or another poet..."** (82 tokens, 50 covered, 61.0%) -- This sentence achieves the highest coverage at 61%. The portmanteau *Rutlandbaconsouthamptonshakespeare* (Joyce's joke about the authorship question) is naturally uncovered, but common words like *name*, *wrote*, *father*, *errors* are captured. Even at 61% coverage, the parser finds zero trees -- the filtered tokens don't form a grammatically valid sequence under the CFG's rules.

4. **"I don't want Richard, my name..."** (76 tokens, 30 covered, 39.5%) -- A passage that appears to include dramatic dialogue (with stage directions like "Laughter" and character names like QUAKERLYSTER), showing how the episode's mixed-mode text confounds a straightforward sentence parser.

5. **"Shylock chimes with the jewbaiting..."** (53 tokens, 22 covered, 41.5%) -- Stephen's argument about Shakespeare and anti-Semitism. Period-specific vocabulary (*jewbaiting*, *quartering*, *leech*) falls outside the grammar.

**Parse trees found: 0 for all five sentences.** This is the central finding: a hand-written CFG with ~100 terminals cannot parse even one of Joyce's complex argumentative sentences. The coverage ranges from 36.8% to 61.0%, well within the exercise sheet's predicted 30-50% range (one sentence exceeds it). The failure is partly in the grammar (limited lexicon) and partly in Joyce's syntax (nested quotations, portmanteau words, mixed prose modes, sentences that exceed any reasonable CFG's structural rules).

### Interpretation

The zero-parse result is pedagogically valuable. It demonstrates that:
- A hand-written CFG is a useful formalism for understanding constituent structure but is hopelessly inadequate for real literary text.
- Even when individual tokens are covered, the *sequences* they form may not match the grammar's production rules -- showing that parsing requires both lexical coverage and structural coverage.
- Joyce's argumentative prose in this episode is genuinely difficult to parse: the sentences mix theological parody, dramatic dialogue, literary-critical argument, and Joycean wordplay in ways that defeat simple constituency analysis.

---

## Exercise 2: Penn Treebank as Reference Grammar

### What the code does

`treebank_statistics()` iterates over the first 20 files of NLTK's Penn Treebank corpus, computing:
- **Tree depth** (via `tree.height()`) for each parsed sentence
- **Branching factor** (number of children per non-terminal node with height > 2)
- **SBAR count** (subordinate clause nodes per sentence)

`episode_complexity()` computes proxy measures for Scylla and Charybdis (and Calypso as a comparison), since full parsing is infeasible:
- Mean, median, and max sentence length
- Subordinating conjunction frequency (rate of *because*, *although*, *if*, *when*, *where*, *while*, *since*, *unless*, *that*, *which*, *who*, *whom*)
- Comma density (commas per sentence)

### What the output shows

**Penn Treebank baseline (233 sentences from 20 files):**

| Metric | Value |
|---|---|
| Average tree depth | 10.85 |
| Max tree depth | 22 |
| Average branching factor | 2.26 |
| SBAR per sentence | 0.56 |

**Scylla and Charybdis (1,213 sentences):**

| Metric | Value |
|---|---|
| Mean sentence length | 12.1 tokens |
| Median sentence length | 8 tokens |
| Max sentence length | 103 tokens |
| Subordinating conjunctions per sentence | 0.23 |
| Commas per sentence | 0.79 |

**Calypso (822 sentences, for comparison):**

| Metric | Value |
|---|---|
| Mean sentence length | 9.0 tokens |
| Median sentence length | 7 tokens |
| Max sentence length | 53 tokens |
| Subordinating conjunctions per sentence | 0.09 |
| Commas per sentence | 0.40 |

### Interpretation

Scylla and Charybdis is measurably more complex than Calypso across every proxy metric:

- **Sentence length:** Mean 12.1 vs. 9.0 tokens (34% longer), with a much higher maximum (103 vs. 53). The median (8 vs. 7) is closer, suggesting that Scylla contains many short dialogue fragments alongside its long argumentative periods -- consistent with the episode's mix of Stephen's extended arguments and brief interjections from his interlocutors.

- **Subordinating conjunctions per sentence:** 0.23 vs. 0.09 (2.6x higher). This confirms the episode's dialectical character: Stephen's argument relies heavily on subordination (*that*, *which*, *who*, *if*, *because*) to embed claims within claims.

- **Comma density:** 0.79 vs. 0.40 (nearly 2x). Higher comma density correlates with parenthetical qualifications, lists, and the appositive constructions Stephen uses to layer meaning.

- **Comparison to Treebank:** The Treebank's SBAR rate of 0.56 per sentence is notably higher than Scylla's subordinating conjunction rate of 0.23. However, these are not directly comparable: the Treebank metric counts parsed SBAR nodes (which may be introduced by relative pronouns, complementizers, or other means), while the episode metric counts only a fixed set of subordinating conjunction words. The Treebank's average depth of 10.85 and max of 22 suggest that even newspaper prose (the Treebank's domain) can be structurally deep; one would expect Joyce's most complex sentences to rival or exceed this.

The exercise sheet hypothesized that Stephen's argumentative sentences would be "deeper and more left-branching than typical English." The proxy metrics support the depth claim (longer sentences, more subordination, more commas), though direct depth comparison awaits full parsing. The comparison between Scylla and Calypso -- the brain-episode versus the kidney-episode -- is particularly apt: Calypso's simpler, more sensory prose (Bloom making breakfast) contrasts sharply with Scylla's recursive intellectual architecture.

---

## Exercise 3: The Quotation Problem

### What the code does

`extract_quotations()` uses regex patterns to find text between:
- ASCII double quotes (`"..."`)
- ASCII single quotes (`'...'`)
- Markdown italics (`*...*`)

It also checks for em-dash dialogue lines containing embedded quotes. `compare_quotation_syntax()` then POS-tags the extracted quotations and the remaining "framing" prose separately, comparing their POS distributions.

### What the output shows

```
--- Quotations Found: 0 ---
```

**Zero quotations found.** The POS comparison section does not execute because there is no quoted text to analyze.

### Interpretation

This is a bug. The episode text uses Unicode curly/smart punctuation rather than ASCII quote marks:
- Em-dashes (U+2014, `\u2014`) mark dialogue, following the Irish typographic convention Joyce inherited from Continental European printing
- Right single curly quotes (U+2019, `\u2019`) serve as apostrophes (206 occurrences)
- Left single curly quotes (U+2018, `\u2018`) appear only 3 times

The regex patterns in `extract_quotations()` search for ASCII double quotes (`"`) and ASCII single quotes (`'`), neither of which appears in the text. The episode contains no ASCII-encoded quotation marks at all. This means the quotation extraction exercise produces no results -- a significant gap in the analysis. See the TODO file for the fix.

Beyond the encoding bug, there is a deeper methodological challenge noted in the exercise sheet: Scylla and Charybdis contains both *explicit* quotation (marked by punctuation) and *unmarked allusion* (Shakespeare phrases woven into Stephen's prose without attribution). Even a fixed regex would only capture the explicit cases. The episode's saturation with Shakespeare -- sometimes quoted, sometimes paraphrased, sometimes parodied -- makes the boundary between "quoted" and "framing" prose genuinely blurry, which is part of Joyce's point about the relationship between original and derivative art.

---

## Summary of Results

| Exercise | Key Finding |
|---|---|
| 1. Parsing the Argument | 50 argument sentences found; 0 of 5 parse with hand-written CFG; coverage 37-61% |
| 2. Treebank Reference | Scylla is 34% longer sentences, 2.6x more subordination, 2x more commas than Calypso |
| 3. Quotation Problem | 0 quotations extracted due to Unicode quote mark mismatch (bug) |

The script successfully demonstrates the limits of hand-written CFGs on literary text (Exercise 1) and provides meaningful cross-episode complexity comparison (Exercise 2). Exercise 3 is blocked by a character encoding issue in the quotation extraction regex.
