# Week 17: Ithaca
### *"What did Bloom see on the range?" — The catechism of everything, knowledge extraction, and the skeleton of fact.*

Bloom and Stephen arrive at 7 Eccles Street. Bloom makes cocoa. They talk. Stephen leaves. Bloom goes to bed. These ordinary events are narrated in the form of an **impersonal catechism** — a relentless sequence of questions and answers that anatomizes every detail of Bloom's world with the exhaustive precision of a scientific encyclopedia. What is the temperature of the water? What route did the water travel from Roundwood reservoir? What did Bloom see on the dresser? (An inventory follows.) What were his thoughts on the stars? (A disquisition on astronomy follows.) What were the contents of the second drawer? (A list that runs for pages.) The technique is *catechism (impersonal)*; the art is *science*; the organ is the *skeleton* — the hard, articulated, mineral structure beneath the flesh. Where Eumaeus was vague and hedging, Ithaca is mercilessly specific. Where Eumaeus felt, Ithaca enumerates.

**NLTK Focus:** Information extraction, relation extraction, and knowledge graph construction (`nltk.chunk`, `nltk.sem`, relation patterns, structured output parsing; `networkx` and `rdflib` for knowledge representation)

**Pairing Rationale:**
Ithaca is the episode that *wants* to be a database. Its Q&A format is a proto-API: structured queries yielding structured responses. Its contents — inventories, budgets, measurements, genealogies, astronomical calculations — are the raw material of knowledge graphs. Where a conventional novelist would write "Bloom made cocoa," Joyce writes a question-and-answer pair that specifies the water source, the heating method, the ingredients, the cream brand, and the spoon's material. This is information extraction's fantasy: text so structured that the entities and relations practically parse themselves. Students will build a knowledge graph from Ithaca's Q&A pairs — extracting entities, properties, and relationships, then representing them as triples (subject, predicate, object) and visualizing the result. The graph is Bloom's world reduced to its skeleton, which is exactly what the episode intends.

**Core Exercises:**

1. **Parse the catechism.** Ithaca's Q&A format is consistent enough to parse semi-automatically. Write a parser that segments the episode into question-answer pairs (detect questions by their interrogative syntax and the structural pattern of the text). How many Q&A pairs are there? (Scholars count approximately 309.) Classify the questions by type: *what* (entity/inventory), *why* (causal), *how* (procedural), *did* (yes/no), *what were* (list). Visualize the distribution. Which question type dominates? The distribution tells you what kind of knowledge the episode is interested in — and it's overwhelmingly *what*: the episode wants to name and catalog, not explain.

2. **Triple extraction.** For each Q&A pair, attempt to extract one or more knowledge triples in the form (subject, predicate, object). Start with the simplest cases: inventory questions ("What did the first drawer contain?") yield triples like (*first_drawer*, *contains*, *writing_materials*), (*first_drawer*, *contains*, *sealing_wax*). Procedural questions yield triples like (*water*, *traveled_from*, *Roundwood_reservoir*). Relational questions yield triples like (*Bloom*, *made_cocoa_for*, *Stephen*). You will not be able to extract triples from every pair — some questions and answers are too syntactically complex or too abstract. That's fine. Extract what you can (aim for 100+ triples) and build a knowledge graph using `networkx` (nodes = entities, edges = predicates). Visualize it. What does the skeleton of Bloom's world look like? Is it centered on Bloom, on the house, on the objects?

3. **The question Ithaca doesn't ask.** Ithaca's catechism is not random — Joyce chose which questions to ask and which to leave unasked. Analyze the distribution of question topics: how many questions concern physical objects vs. human relationships vs. abstract concepts vs. Bloom's inner life? Build a **topic treemap** (`squarify` or `plotly.express.treemap`) showing the proportional weight of each topic domain. Now compare this distribution to the proportional weight of the same domains in an earlier Bloom episode (Calypso or Lestrygonians), measured by the proportion of sentences devoted to each. The comparison reveals what the catechism *selects for* — the skeletal structure that the impersonal voice considers worth cataloging — and what it suppresses. (Hypothesis: Ithaca dramatically over-represents objects and under-represents emotions. The skeleton has no flesh.)

**Diving Deeper:**

- The Ithaca episode has been called a precursor to the Semantic Web — structured data embedded in natural language. RDF (Resource Description Framework) is the W3C standard for representing knowledge as triples. Convert your extracted triples to RDF using `rdflib` and write SPARQL queries against your Bloom knowledge base. Can you answer questions about Bloom's world ("What objects are in the kitchen?", "Who has Bloom interacted with today?") using formal queries against structured data extracted from a novel? The exercise is half serious, half absurd — which is appropriate for an episode that catalogs the stars and the contents of a drawer with equal solemnity.
- Modern information extraction pipelines (OpenIE, Stanford IE) can extract triples at scale. Run an off-the-shelf system on Ithaca and compare its output to your hand-crafted extractions. Where does the automated system excel (simple possessive and locative relations) and where does it fail (Joyce's long, nested answer-paragraphs that embed multiple relations in subordinate clauses)?
- Ithaca's Q&A pairs are, formally, a dialogue between an unnamed questioner and an unnamed answerer. This makes the episode a test case for question-answering systems. Feed Ithaca's questions to a modern QA model (e.g., a BERT-based model via HuggingFace `transformers`) with the episode text as context. Can the model locate the correct answer span? Where it fails, analyze whether the failure is due to the question's complexity, the answer's length, or the model's inability to handle Joyce's syntax.
- The lists in Ithaca (contents of drawers, items in Bloom's budget, books on his shelf) are themselves data structures — ordered sequences with implicit schemas. Can you infer the schema? (The drawer inventory is implicitly categorized: writing implements, then correspondence, then keepsakes.) Cluster the listed items using word embeddings and see whether the clusters match the implicit categories. This is schema induction from literary text.
- Connection to Week 2 (Nestor): that episode's technique was also catechism, but *personal* — Deasy's pedagogical bluster. Compare the Nestor and Ithaca catechisms structurally: question types, answer lengths, information density per Q&A pair. Ithaca's impersonal catechism should be measurably denser and more informative. The two episodes bracket the novel's relationship with the interrogative form — from the human teacher who knows less than he thinks to the inhuman voice that knows everything and feels nothing.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Parse structured literary text** (Q&A catechism format) into question-answer pairs using regex and structural heuristics.
2. **Classify questions by type** (what, why, how, yes/no, etc.) and analyze what the distribution reveals about the episode's epistemological priorities.
3. **Extract knowledge triples** (subject, predicate, object) from semi-structured text using POS patterns and heuristic rules.
4. **Analyze topic distribution** to identify what the catechism selects for (objects, science) and what it suppresses (emotions, inner life).

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Ithaca) |
|---|---|---|
| Q&A pairs parsed | catechism segmentation | ~250–350 (scholars count ~309) |
| Dominant question type | classify_question distribution | "what" overwhelmingly dominant |
| Mean answer length | tokens per answer | ~50–150 words |
| Max answer length | longest answer | several hundred words |
| Knowledge triples extracted | (subject, predicate, object) tuples | aim for 100+ |
| Most common subject | entity appearing most as triple subject | likely "bloom" or "water" |
| Physical object % of questions | topic classification | substantially higher than Calypso |

## Rubric

### Exercise 1: Parse the Catechism (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Q&A parsing** | All Q&A pairs extracted; count close to scholarly ~309; format correctly identified | Most pairs extracted | Parsing broken or incomplete |
| **Question classification** | 5+ question types; distribution visualized; "what" dominance discussed | 3+ types classified | Fewer than 3 types |
| **Answer statistics** | Length distribution (mean, median, max, min) reported and discussed | Basic stats | No statistics |

### Exercise 2: Triple Extraction (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Extraction method** | POS-based and pattern-based triple extraction; inventory lists handled; 100+ triples | 50+ triples; basic patterns | Fewer than 50 or broken |
| **Knowledge graph** | Triples visualized as networkx graph; central nodes identified | Some graph structure | No graph |
| **Graph interpretation** | Discusses whether the graph centers on Bloom, the house, or the objects; connects to "skeleton" metaphor | Brief interpretation | No interpretation |

### Exercise 3: The Question Ithaca Doesn't Ask (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Topic classification** | 6+ topic domains; proportional weight visualized as treemap or bar chart | 4+ domains | Fewer than 4 |
| **Cross-episode comparison** | Topic distribution compared to Calypso or another Bloom episode | Some comparison | No comparison |
| **Selection/suppression analysis** | Identifies what catechism over-represents (objects) and under-represents (emotions); connects to "skeleton has no flesh" | Brief discussion | No analysis |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| RDF conversion with SPARQL queries on Bloom's world | +3 |
| BERT QA model applied to Ithaca questions | +3 |
| Schema induction on list items via word embeddings | +2 |
| Nestor vs. Ithaca catechism structural comparison | +2 |

## Reference Implementation

See [`solutions/week17_ithaca.py`](solutions/week17_ithaca.py)
