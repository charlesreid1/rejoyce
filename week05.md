# Week 5: Lotus Eaters
### *"This is my body" — Narcotics, sacraments, and the drift of words into other words.*

Bloom wanders through mid-morning Dublin in a pleasant fog: he collects a clandestine letter from Martha Clifford (his pen-pal flirtation), watches a church service, orders a lotion at the chemist, and contemplates a bath. Everything in this episode is about substitution and altered states. The Eucharist transforms bread into body. Martha's letter substitutes for real contact (and contains a beautiful malapropism: "I do not like that other world" for *word*). Bloom plans to bathe — to dissolve himself in warm water. The episode's art is *botany/chemistry*; its technique is *narcissism*; its organ is the *genitals* (desire sublimated, deferred, dissolved). Language itself goes soft at the edges — words echo, double, blur into near-synonyms.

**NLTK Focus:** WordNet and lexical semantics (`nltk.corpus.wordnet`, synsets, hypernymy, hyponymy, meronymy, semantic similarity, `wup_similarity`, `path_similarity`)

**Pairing Rationale:**
WordNet is a network of substitutions — it tells you that every word is connected to other words through synonymy, hypernymy, and meronymy, that meaning is never fixed to a single form but drifts through a web of related terms. This is the Lotus Eaters' epistemology exactly. The Eucharist is the ultimate semantic operation: *bread* IS *body* (synonymy? hypernymy? metaphor?). Martha writes *world* for *word* — a one-character substitution that opens a semantic chasm. Bloom thinks about how a "languid floating flower" describes both a lotus and a body in a bath. WordNet gives us a formal vocabulary for these relationships: how close is *flower* to *drug*? How many hypernym steps from *bread* to *body*? The narcotic quality of the episode is the narcotic quality of browsing WordNet itself — you follow one link, then another, and soon you've drifted very far from where you started.

**Core Exercises:**

1. **Semantic fields of narcosis.** Identify 10–15 key content words from Lotus Eaters that carry the episode's thematic weight (e.g., *flower*, *languid*, *body*, *pin*, *altar*, *bath*, *drug*, *floating*, *dissolve*, *communion*). For each, retrieve all synsets from WordNet. Map the hypernym trees for each synset up to the root. Where do the trees converge? Do thematically related words share common ancestors? Build a simple visualization showing the hypernym paths for your word set — the structure should reveal the episode's deep semantic topology.

2. **Martha's malapropism and semantic distance.** Martha writes "I do not like that other world" (meaning *word*). Using WordNet's similarity measures (`path_similarity`, `wup_similarity`, `lin_similarity`), compute the semantic distance between *world* and *word*. Now compare this to other near-homophone pairs in the episode. Is there a relationship between phonological closeness and semantic distance? (The answer will be no, and that's the point — the pun lives in the gap between sound-similarity and meaning-similarity.) Extend this: find 5 other word pairs in the episode that are phonologically close but semantically distant, and vice versa.

3. **Substitution chains.** Starting from the word *body*, build a chain of substitutions through WordNet: replace *body* with its closest synonym, then replace that synonym with *its* closest synonym, and so on for 10 steps. Where do you end up? Repeat starting from *bread*, *flower*, *drug*, and *water*. Do any of the chains converge? This exercise makes visible the way meaning drifts through lexical space — a computational enactment of the episode's logic of narcotic substitution and transubstantiation.

**Diving Deeper:**

- WordNet's similarity measures are based on taxonomic distance (path length, lowest common subsumer). But *body/bread* feels related through cultural and sacramental association, not taxonomy. Explore distributional similarity as an alternative: using word2vec or GloVe embeddings (via `gensim`), compute cosine similarities for the same word pairs. Where do distributional and taxonomic measures agree? Where do they diverge? Which captures Joycean association better?
- Martha's *world/word* substitution is a type of speech error (malapropism). Computational models of speech errors draw on both phonological and semantic similarity. See Dell (1986) and Vitevitch (2002) on the architecture of the mental lexicon. Can you build a simple model that predicts likely malapropisms using NLTK and a pronunciation dictionary (`nltk.corpus.cmudict`)?
- WordNet is an expert-curated resource encoding one model of how English organizes meaning. It is not the only model. FrameNet organizes words by semantic frames; VerbNet by syntactic-semantic verb classes. How would a FrameNet analysis of Lotus Eaters differ from a WordNet analysis? (NLTK provides access to FrameNet: `nltk.corpus.framenet`.)
- The episode's sacramental logic (bread = body, wine = blood) anticipates contemporary debates about metaphor in NLP. Is metaphor a semantic relationship that WordNet can represent? Or is it fundamentally different from synonymy and hypernymy? See Shutova (2010) on computational approaches to metaphor.
- Connection to Week 9 (Scylla and Charybdis): that episode's literary-critical debate about Shakespeare involves the relationship between names and essences, between the word and the thing. The lexical-semantic tools from this week provide the formal machinery to revisit those questions.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Navigate WordNet** programmatically — retrieving synsets, hypernym paths, and lowest common subsumers for arbitrary words.
2. **Compute and interpret semantic similarity** using multiple measures (path_similarity, wup_similarity) and understand what each captures.
3. **Map semantic fields** by tracing hypernym convergence among thematically related words, revealing the deep taxonomic structure beneath surface vocabulary.
4. **Distinguish phonological from semantic similarity** and articulate why puns and malapropisms exploit the gap between them.
5. **Build substitution chains** through WordNet's synonym network and reflect on how meaning drifts through lexical space.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Lotus Eaters) |
|---|---|---|
| Synsets per thematic word | `len(wn.synsets(word))` averaged over 15 words | ~5–15 synsets |
| Hypernym depth (avg) | average path length to root | ~6–10 steps |
| WuP similarity (world/word) | `wn.synsets('world')[0].wup_similarity(wn.synsets('word')[0])` | very low (~0.1–0.3) |
| Substitution chain length before dead-end | steps until no new synonym found | ~4–10 |
| Chain convergence | number of start-word pairs whose chains share a word | 0–3 pairs |
| Average synset count per content word | total synsets / content words with synsets | ~4–8 |

## Rubric

### Exercise 1: Semantic Fields of Narcosis (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Synset retrieval** | 10–15 words investigated; synsets, definitions, hypernym paths all reported | 8+ words; some paths traced | Fewer than 8 words or paths not traced |
| **Convergence analysis** | Lowest common subsumers identified for thematic pairs; convergence points interpreted (e.g., *body* and *bread* sharing an ancestor) | Some LCS computed | No convergence analysis |
| **Visualization** | Hypernym tree visualization produced showing path structure | Textual representation of paths | No structural representation |

### Exercise 2: Martha's Malapropism (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Similarity computation** | path_similarity and wup_similarity for world/word plus 5+ additional pairs; phonological distance via CMUdict | At least 3 pairs computed | Only world/word or computation errors |
| **Sound vs. meaning analysis** | Clear demonstration that phonological closeness does not predict semantic closeness; specific examples | General observation made | No comparison of the two dimensions |
| **Reflection** | Connects the finding to Joyce's use of puns, homophony, and the episode's theme of substitution | Brief reflection | No reflection |

### Exercise 3: Substitution Chains (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Chain construction** | 5+ starting words; chains of 10 steps; dead-ends noted | 3+ chains constructed | Fewer than 3 or chains broken |
| **Convergence check** | All chain pairs checked for shared words; results reported | Some convergence checked | No convergence analysis |
| **Thematic reflection** | Connects semantic drift to the episode's logic of narcotic substitution and transubstantiation | Brief connection made | No thematic connection |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| word2vec/GloVe distributional similarity comparison | +3 |
| Malapropism predictor using CMUdict + semantic distance | +3 |
| FrameNet analysis comparison to WordNet | +2 |
| Metaphor and the limits of taxonomic semantics | +2 |

## Reference Implementation

See [`solutions/week05_lotuseaters.py`](solutions/week05_lotuseaters.py)
