# Week 10: Wandering Rocks
### *"The lacquey by the door of Dillon's auctionrooms shook his handbell" — Nineteen fragments, one city, and the problem of simultaneous narration.*

Wandering Rocks is the hinge of the novel — structurally and ontologically. It consists of 19 short sections, each following a different Dubliner through the city at roughly the same time on the afternoon of June 16, 1904. Father Conmee walks through the northside. Blazes Boylan buys a gift for Molly. A one-legged sailor begs. A crumpled throwaway floats down the Liffey. The episode has no single protagonist and no single perspective; its organ is *none* (it sits outside the body schema, as if the novel briefly became a machine rather than a person). Its technique is *labyrinth*; its art is *mechanics*. Most distinctively, Joyce inserts brief **interpolations** — fragments from one section that intrude into another, marking simultaneous events happening elsewhere in the city. The effect is cinematic: cross-cutting, parallel editing, Dublin as a system of synchronized clocks.

**NLTK Focus:** Text similarity, document clustering, and cross-segment entity tracking (`nltk.cluster`, cosine similarity, vector space models, `nltk.metrics.distance`)

**Pairing Rationale:**
Wandering Rocks presents 19 mini-documents and asks the reader to reconstruct the hidden connections between them — shared characters who appear in multiple sections, interpolations that link simultaneous events, geographic paths that intersect. This is fundamentally a problem of *cross-document analysis*: measuring similarity between text segments, clustering related passages, and tracking entities across boundaries. The episode's labyrinthine technique is what happens when you fragment a narrative into pieces and challenge the reader (or the algorithm) to reassemble the map. Vector space representations — turning each section into a bag-of-words vector and measuring cosine distances — provide a computational model of the question "which of these 19 fragments belong together?" The interpolations, meanwhile, are a devilish test case: they are textually *foreign* to their host sections (a sentence about the viceregal cavalcade appears in a section about a bookstall), which means they should be detectable as anomalies — if your similarity metrics are good enough.

**Core Exercises:**

1. **Section similarity matrix.** Represent each of the 19 sections as a TF-IDF vector (building on Week 7). Compute pairwise cosine similarity for all 19×19 pairs. Visualize this as a heatmap. Which sections are most similar to each other? Do the clusters correspond to shared characters, shared locations, or shared themes? Identify the top 5 most similar section pairs and explain each: is the similarity driven by a shared character (Bloom appears in multiple sections), a shared location (several sections cross the same streets), or something else?

2. **Interpolation detection.** The interpolations are sentences that belong, narratively, to a different section than the one they appear in. They are, in effect, *misplaced* sentences. For each section, compute the average TF-IDF vector, then score every sentence by its cosine similarity to the section centroid. Flag sentences with abnormally low similarity scores. Do your lowest-scoring sentences correspond to Joyce's interpolations? What is the precision and recall of this anomaly detector? Where does it fail, and why? (Some interpolations are very short and share common words with their host; some sections are internally heterogeneous enough that legitimate sentences score low.)

3. **Entity tracking across the labyrinth.** Extract named entities (using the NER tools from Week 4) from each section. Build a section × entity occurrence matrix. Which entities appear in the most sections? (The viceregal cavalcade and the Liffey throwaway are designed to thread through the episode as tracking devices.) Construct a bipartite graph of sections and shared entities (using `networkx`). Does the graph's structure reveal Joyce's architectural plan — the sections that are "supposed" to connect? Overlay the geographic trajectory of 2–3 entities across sections and sketch a rough map of their paths through Dublin.

**Diving Deeper:**

- Topic modeling (LDA via `gensim`) offers an alternative to TF-IDF clustering. Fit a topic model to the 19 sections and examine the learned topics. Do the topics correspond to characters, locations, or thematic domains? Experiment with different numbers of topics (k=3, 5, 8, 12). At what granularity does the model best capture the episode's structure?
- The interpolations create a kind of *hypertext* — links between sections that the reader must follow. Compare Joyce's technique to the structure of hyperlinked documents. Could you compute a PageRank-style centrality score for each section, using entity co-occurrence as the link structure? Which section is the most "central" to the episode's network?
- Franco Moretti's *Graphs, Maps, Trees* (2005) and *Atlas of the European Novel* (1998) pioneered computational-geographic approaches to fiction. Wandering Rocks is the ultimate test case for literary cartography. If you have access to a historical map of 1904 Dublin, can you geolocate the sections and compute actual walking distances? Correlate geographic distance with textual similarity. Do nearby sections share more vocabulary?
- Text reuse detection (shingling, MinHash, locality-sensitive hashing) is the industrial-strength version of the interpolation detection task. These methods are used for plagiarism detection and news article de-duplication. Try implementing a MinHash-based near-duplicate detector — do the interpolations show up as near-duplicates of their "source" sections?
- Connection to Week 15 (Circe): that episode also features textual intrusions — stage directions, hallucinatory apparitions, the return of characters from earlier episodes. But where Wandering Rocks' interpolations are mechanical and geographic, Circe's are psychological and surreal. The anomaly detection framework from this week will need radical adaptation.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Represent text segments as TF-IDF vectors** and compute pairwise cosine similarity to build a document similarity matrix.
2. **Visualize similarity** as a heatmap and interpret clusters in terms of shared characters, locations, or themes.
3. **Detect textual anomalies** (interpolations) by scoring sentences against their section centroid and flagging outliers.
4. **Track named entities** across document segments and build section × entity matrices that reveal the episode's architectural plan.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Wandering Rocks) |
|---|---|---|
| Sections parsed | structural segmentation of episode | ~15–25 (ideally 19) |
| Similarity matrix dimensions | n × n pairwise cosine | 19 × 19 |
| Top-5 pair similarity | highest cosine between non-identical sections | ~0.1–0.4 |
| Anomalous sentences flagged | sentences with cosine < 0.1 to centroid | ~10–30 |
| Interpolation detection precision | manually verified anomalies / flagged | ~20–50% |
| Multi-section entities | entities appearing in 2+ sections | ~10–25 |
| Most connected section pair | pair with most shared entities | varies |

## Rubric

### Exercise 1: Section Similarity Matrix (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **TF-IDF computation** | Correct TF-IDF vectors for all sections; cosine similarity matrix computed | Vectors computed; some errors | Vectors broken or missing |
| **Heatmap visualization** | Clear heatmap with labeled axes; clusters visible and discussed | Heatmap produced | No visualization |
| **Pair analysis** | Top-5 similar pairs identified with explanation (shared character, location, theme) and shared keyword evidence | Some pairs noted | No pair analysis |

### Exercise 2: Interpolation Detection (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Anomaly scoring** | Per-sentence cosine to section centroid computed; threshold-based flagging | Some scoring done | Not attempted |
| **Precision/recall assessment** | Flagged sentences manually checked against known interpolations; P/R estimated | Some manual verification | No verification |
| **Failure analysis** | Explains why detection fails (short interpolations, internally heterogeneous sections) | Brief failure notes | No analysis |

### Exercise 3: Entity Tracking (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Entity extraction** | NER run on each section; section × entity matrix built | Entities extracted; basic tracking | Extraction broken |
| **Cross-section analysis** | Multi-section entities identified; bipartite graph or connected pairs shown | Some cross-section tracking | No cross-section analysis |
| **Architectural interpretation** | Entity trajectories connected to Joyce's design (cavalcade threading through, throwaway floating) | Some interpretation | No interpretation |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| LDA topic modeling of the 19 sections | +3 |
| PageRank centrality of sections via entity co-occurrence | +3 |
| Geographic correlation (textual similarity vs. walking distance) | +2 |
| MinHash near-duplicate detection for interpolations | +2 |

## Reference Implementation

See [`week10_wanderingrocks.py`](week10_wanderingrocks.py)
