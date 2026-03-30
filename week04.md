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
