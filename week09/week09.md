# Week 9: Scylla and Charybdis
### *"A man of genius makes no mistakes. His errors are volitional and are the portals of discovery" — The Library, the dialectic, and the deep structure of argument.*

Stephen holds court in the National Library, presenting his theory of Shakespeare's biography: that the bard's life is encoded in his plays, that Hamlet is Shakespeare's dead son, that Ann Hathaway's infidelity is the wound at the center of the work. The episode is a formal debate — Stephen argues against the Platonist mystics (AE, Eglinton) who believe art transcends its maker, and against the Aristotelians who deny the biographical reading. The technique is *dialectic*; the art is *literature*; the organ is the *brain*. The prose is dense with embedded quotations, logical connectives ("therefore," "if... then," "but"), parenthetical qualifications, and nested subordinate clauses. Sentences have the structure of syllogisms. Arguments have the structure of trees.

**NLTK Focus:** Context-free grammars and syntactic parsing (`nltk.parse`, `RecursiveDescentParser`, `ChartParser`, `CFG.fromstring`, the Penn Treebank parsed corpus, dependency structure)

**Pairing Rationale:**
A dialectical argument has hierarchical structure: premises support conclusions, objections embed counter-premises, qualifications modify claims. This is exactly what a syntactic parse tree represents — the hierarchical organization of a sentence into nested constituents. Scylla and Charybdis is the most syntactically ambitious episode so far: Stephen's Shakespeare theory is delivered in long, architecturally complex sentences whose surface linearity conceals deep recursive structure. (Consider: "He was himself a lord of language and had made himself a coistrel gentleman and he had written *Romeo and Juliet* with the survey of all Italy and a tale of a honeymoon and he had sent...") Parsing these sentences reveals the skeleton of Stephen's argument — and the places where the skeleton breaks. The episode's dialectical method (thesis vs. antithesis, Plato vs. Aristotle, art vs. life) also maps onto the formal duality between constituency parsing and dependency parsing: two ways of revealing a sentence's deep structure that sometimes agree and sometimes don't.

**Core Exercises:**

1. **Parsing the argument.** Select 5 syntactically complex sentences from Scylla and Charybdis — prioritize sentences where Stephen is making an argument (look for logical connectives: *therefore*, *because*, *if*, *but*, *yet*). Use NLTK's `ChartParser` with a hand-written CFG that covers the major constituent types (S, NP, VP, PP, SBAR, ADJP, ADVP). Draw the parse trees (using `tree.draw()` or `tree.pretty_print()`). Where does your grammar succeed? Where does it fail — and is the failure in your grammar or in Joyce's syntax? Identify at least one sentence that seems genuinely ambiguous (admitting more than one valid parse tree). What are the competing readings?

2. **Penn Treebank as reference grammar.** Using NLTK's parsed Treebank corpus (`nltk.corpus.treebank`), extract statistics on sentence depth (maximum tree depth), branching factor (average number of children per non-terminal), and the distribution of clause types (SBAR, relative clauses, parentheticals). Now POS-tag and parse 20 sentences from Scylla and Charybdis using a PCFG trained on the Treebank (`nltk.parse.pchart`, `nltk.grammar.induce_pcfg`). Compare the structural statistics. Does Joyce's dialectical prose differ measurably from Treebank English in depth, branching, or subordination? (Hypothesis: Stephen's argumentative sentences are deeper and more left-branching than typical English.)

3. **The quotation problem.** Scylla and Charybdis is saturated with quotations from Shakespeare — sometimes attributed, sometimes not, sometimes modified. Extract all quoted material from the episode (you'll need to handle both explicit quote marks and unmarked allusion — start with the explicit cases). Parse these quoted fragments separately from Stephen's framing prose. Compare their syntactic profiles. Does early modern English Shakespeare have a measurably different syntactic structure from Joyce's early modern Irish English? Where does the Treebank-trained parser struggle most with Shakespeare's syntax, and what does this tell you about the training data assumptions of statistical parsers?

**Diving Deeper:**

- NLTK's chart parser is a pedagogical tool. For serious parsing of literary text, explore the Stanford Parser (accessible via NLTK's `StanfordParser` interface) or spaCy's dependency parser. Run the same sentences through multiple parsers and compare their outputs. Where parsers disagree, the disagreement often points to genuine syntactic ambiguity — which in Joyce is frequently *deliberate*.
- Dependency parsing offers an alternative to constituency parsing: instead of nested brackets, it represents sentence structure as directed links between words (head → dependent). Use spaCy to generate dependency parses for Stephen's argument sentences. Visualize them with `displacy`. Does the dependency representation make the argument's logical structure more or less visible than the constituency tree?
- The Chomsky hierarchy classifies formal grammars by generative power: regular → context-free → context-sensitive → recursively enumerable. English is generally agreed to be mildly context-sensitive (see Shieber, 1985, on Swiss German cross-serial dependencies). Is there anything in Joyce's syntax — the nested quotations, the self-referential constructions — that pushes beyond context-free? This is a hard question, but a productive one.
- Stephen's Shakespeare theory is itself a *parsing* of Shakespeare's texts — an attempt to recover the biographical deep structure beneath the literary surface. The episode thus thematizes the interpretive act that computational parsing formalizes. There is a rich critical literature on this: see Kenner (1980) on Joyce's art of mechanical reproduction and Attridge (1988) on Joyce and the limits of formalism.
- Connection to Week 17 (Ithaca): that episode's catechism structure (question → answer → question) is a radically different formal organization of information. Where Scylla's dialectic nests arguments recursively, Ithaca's catechism proceeds linearly. Comparing the parse structures of both episodes will reveal two competing models of how knowledge can be organized in prose.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Write context-free grammars** covering major English constituent types and use NLTK's ChartParser to produce parse trees.
2. **Evaluate grammar coverage** — understanding what percentage of a literary text's vocabulary and structure a hand-written CFG can handle.
3. **Extract structural statistics** (tree depth, branching factor, subordinate clause frequency) from the Penn Treebank and use them as baselines for literary comparison.
4. **Identify syntactic ambiguity** in complex literary sentences and produce competing parse trees.
5. **Separate quoted material** from framing prose and compare their syntactic profiles.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Scylla & Charybdis) |
|---|---|---|
| Argument sentences found | sentences with logical connectives > 15 tokens | ~50–100 |
| CFG terminal coverage | % of tokens in grammar's terminal set | ~30–50% (hand-written grammar is limited) |
| Parse trees per sentence | ambiguous parses for selected sentences | 0–5+ per sentence |
| Mean sentence length | tokens per sentence | ~18–25 (longer than most episodes) |
| Subordinating conjunctions per sentence | that/which/who/because/if/etc. per sentence | ~1.2–2.0 |
| Comma density | commas per sentence | ~2–4 |
| Quotations extracted | quoted passages > 3 words | ~20–50 |

## Rubric

### Exercise 1: Parsing the Argument (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **CFG design** | Grammar covers S, NP, VP, PP, SBAR, ADJP, ADVP with reasonable lexicon; documented and justified | Basic grammar with S, NP, VP | Trivial or broken grammar |
| **Parse trees** | 5 complex sentences selected; trees drawn or pretty-printed; successes and failures discussed | 3+ sentences parsed | Fewer than 3 or no trees shown |
| **Ambiguity identification** | At least 1 genuinely ambiguous sentence found with competing parses; interpretations discussed | Ambiguity noted but not analyzed | No ambiguity discussion |

### Exercise 2: Treebank Reference (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Treebank statistics** | Depth, branching, SBAR rate extracted from Treebank sample | Some statistics extracted | Treebank not used |
| **Episode comparison** | Complexity proxies computed for Scylla and a simpler episode; differences quantified | Some comparison | No comparison |
| **Interpretation** | Connects higher depth/subordination to the episode's dialectical, argumentative character | Brief interpretation | No interpretation |

### Exercise 3: The Quotation Problem (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Quotation extraction** | Explicit quotes extracted via regex; count reported; samples shown | Some quotes extracted | Extraction broken |
| **Syntactic comparison** | POS distributions compared between quoted and framing prose; differences discussed | Basic comparison | No comparison |
| **Parser struggle analysis** | Identifies where Treebank-trained parser fails on Shakespeare syntax and explains training data assumptions | Some failure noted | No analysis |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Stanford/spaCy dependency parser comparison | +3 |
| Dependency vs. constituency visualization of the same sentences | +3 |
| Discussion of Chomsky hierarchy and Joyce's syntax | +2 |
| Kenner/Attridge critical connection | +2 |

## Reference Implementation

See [`week09_scyllacharybdis.py`](week09_scyllacharybdis.py)
