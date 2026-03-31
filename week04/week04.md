# Week 4: Calypso
### *"Mr Leopold Bloom ate with relish the inner organs of beasts and fowls" — The world according to a man who notices everything.*

We leave Stephen's brooding intellect and enter the mind of Leopold Bloom — ad canvasser, cuckold, amateur scientist, lover of kidneys. The shift is seismic. Where Stephen thinks in abstractions and literary allusion, Bloom thinks in particulars: the cost of things, the names of streets, what the cat wants, how his wife's body looks under the blankets. The episode's art is *economics*; its organ is the *kidney*; its technique is *narrative (mature)*. Calypso is the novel's great reorientation toward the concrete, the specific, the named. Bloom's consciousness is an inventory of Dublin.

**NLTK Focus:** Chunking and Named Entity Recognition (`nltk.chunk`, `ne_chunk`, `RegexpParser`, noun phrase extraction)

**Pairing Rationale:**
Named Entity Recognition is the technology of noticing what Bloom notices: proper nouns, places, organizations, quantities — the specific, nameable furniture of the world. Where POS tagging (Week 2) assigned grammatical categories, NER asks a more Bloomian question: *what is this a name of?* The episode's art, economics, is fundamentally about tracking named entities and their relationships (who owns what, who owes whom, what costs how much). Bloom's mind is an NER engine running at full tilt — tagging every person, street, brand, and price he encounters. This week also introduces chunking more broadly: the extraction of meaningful multi-word phrases from tagged text, which mirrors the way Bloom's thoughts organize the world into concrete noun phrases rather than Stephen's abstract verb-heavy meditations.

**Core Exercises:**

1. **NER as characterization.** Run NLTK's `ne_chunk` on both Calypso and Proteus (Week 3). Extract and classify all named entities by type (PERSON, GPE, ORGANIZATION, etc.). Compare the two episodes quantitatively: How many named entities per 1,000 tokens does each contain? What types dominate in each? The hypothesis: Bloom's chapter is entity-dense and skews toward places and things; Stephen's is entity-sparse and skews toward persons (mostly historical or literary). Test it. What does this reveal about two ways of being conscious?

2. **Noun phrase chunking.** Write a chunking grammar using `nltk.RegexpParser` that captures noun phrases (e.g., `NP: {<DT>?<JJ>*<NN.*>+}`). Apply it to Calypso. Extract all noun phrases and rank them by frequency. Do the top NPs capture the episode's domestic texture — the kidney, the cat, the bed, the letter? Now modify your grammar to capture prepositional phrases (`PP: {<IN><NP>}`). What spatial and relational patterns emerge? Bloom's Dublin is a world of things *in* places.

3. **Entity co-occurrence and narrative structure.** For each paragraph in Calypso, record which named entities appear. Build a simple co-occurrence matrix: which entities tend to appear in the same paragraphs? Visualize this as a network (use `networkx` if desired). Does the structure of Bloom's associations — Molly linked to the bed and the letter, Dlugacz linked to the porkbutcher and the kidney — emerge from the data? Can you reconstruct the episode's movement through space by tracking place-name sequences?

**Diving Deeper:**

- NLTK's `ne_chunk` uses a maximum entropy classifier trained on the ACE corpus. It is, frankly, not very good on literary text. Compare its output to spaCy's NER (`en_core_web_sm` or `en_core_web_trf`). Where does each fail on Joyce? Common failure modes: Irish place names, archaic terms, Bloom's interior abbreviations.
- The Information Extraction pipeline (tokenize → POS tag → chunk → NER) is a classic cascade architecture. Each stage's errors propagate forward. Trace a specific error through the pipeline: find a case where a POS tagging mistake causes a chunking failure that causes an NER miss.
- Bloom's economic consciousness connects to the emerging field of computational literary economics — extracting prices, transactions, and financial relationships from fiction. See Bode (2018) on quantitative approaches to realist fiction's engagement with material culture.
- Modern NER systems can extract far more fine-grained entity types (food, disease, chemical, event). The OntoNotes and WNUT datasets push well beyond PERSON/GPE/ORG. What entity types would a Joyce-specific NER system need?
- Connection to Week 10 (Wandering Rocks): that episode tracks dozens of Dubliners moving through the city simultaneously. The NER and co-occurrence techniques from this week become essential infrastructure for mapping that episode's crowded geography.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Run and evaluate** NLTK's named entity recognition pipeline on literary text, and categorize entities by type (PERSON, GPE, ORGANIZATION, etc.).
2. **Compare NER density** across episodes as a proxy for different modes of consciousness (Bloom's concrete particularity vs. Stephen's abstraction).
3. **Write chunking grammars** using `RegexpParser` to extract noun phrases and prepositional phrases, and interpret the resulting phrase inventories thematically.
4. **Build co-occurrence matrices** from entity extraction and reason about narrative structure through entity relationships.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Calypso) |
|---|---|---|
| Named entities found | total NER extractions | ~80–150 |
| Entities per 1,000 tokens | entity_count / token_count * 1000 | ~6–12 (Calypso) vs. ~3–6 (Proteus) |
| Dominant entity type | most frequent NE label | PERSON or GPE |
| Unique noun phrases | distinct NP chunks extracted | ~400–800 |
| Top NP frequency | most common noun phrase count | varies (expect domestic vocabulary) |
| Co-occurring entity pairs | pairs sharing a paragraph | ~20–60 unique pairs |
| Most connected entity | entity appearing in most paragraphs | likely "Bloom" or "Molly" |

## Rubric

### Exercise 1: NER as Characterization (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **NER extraction** | Correct extraction from both episodes; entity type counts tabulated; densities computed per 1,000 tokens | Extraction works on one episode; comparison attempted | Extraction errors or only one episode processed |
| **Quantitative comparison** | Specific density numbers compared; entity type distributions contrasted with clear percentages | General comparison noted | No quantitative comparison |
| **Interpretive argument** | Connects entity density to Bloom's concrete vs. Stephen's abstract consciousness; supports with specific examples of what NER captures/misses | Some interpretation present | Metrics without interpretation |

### Exercise 2: Noun Phrase Chunking (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Grammar design** | Working NP and PP grammars; grammar rules documented and justified | NP grammar works; PP not attempted | Grammar broken or trivial |
| **NP frequency analysis** | Top NPs ranked and discussed; domestic texture identified (kidney, cat, bed, letter) | Top NPs listed; some thematic connection | NPs extracted but not analyzed |
| **PP spatial patterns** | Prepositional phrases analyzed for spatial/relational structure; "Bloom's Dublin is a world of things *in* places" explored computationally | PPs extracted; basic analysis | PPs not attempted |

### Exercise 3: Entity Co-occurrence (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Co-occurrence matrix** | Paragraph-level co-occurrence computed; top pairs identified; entity trajectory plotted | Matrix computed; some pairs noted | Matrix not computed or broken |
| **Network interpretation** | Co-occurrence patterns connected to narrative structure (Molly-bed-letter, Dlugacz-kidney) | Some pattern observed | No narrative connection |
| **Visualization** | Entity trajectory or network graph produced; clear and readable | Basic visualization | No visualization |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| spaCy NER comparison with error analysis | +3 |
| Pipeline error propagation traced (POS → chunk → NER) | +3 |
| networkx graph visualization of entity co-occurrence | +2 |
| Joyce-specific NER entity type proposal | +2 |

## Reference Implementation

See [`week04_calypso.py`](week04_calypso.py)
