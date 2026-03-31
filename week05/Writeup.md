# Week 5 Writeup: Lotus Eaters -- WordNet & Lexical Semantics

## Overview

This week pairs NLTK's WordNet interface with Episode 5 of *Ulysses*, "Lotus Eaters," an episode saturated with substitution: the Eucharist (bread becomes body), Martha Clifford's malapropism ("world" for "word"), and Bloom's drift through a narcotic mid-morning of baths, flowers, and chemist shops. The exercises use WordNet's synset hierarchy, similarity measures, and synonym networks to formalize the episode's logic of semantic drift.

The script (`week05_lotuseaters.py`) loads the episode text from `txt/05lotuseaters.txt` and runs three core exercises plus a bonus metric. It relies on `nltk.corpus.wordnet` for synsets and similarity, `nltk.corpus.cmudict` for phonological comparison, and `nltk.edit_distance` for phoneme-level distance.

---

## Exercise 1: Semantic Fields of Narcosis

### What the code does

The function `semantic_fields()` takes 15 thematic words hand-picked from the episode -- *flower, languid, body, pin, altar, bath, drug, floating, dissolve, communion, bread, blood, lotus, water, sacrament* -- and for each word:

1. Retrieves all synsets via `wn.synsets(word)`.
2. Takes the first (most frequent) synset.
3. Calls `ss.hypernym_paths()` to get every path from that synset up to the root entity.
4. Selects the shortest path and reports its depth and final four ancestors.

It then computes the Lowest Common Subsumer (LCS) for every pair of words using `ss1.lowest_common_hypernyms(ss2)` and reports Wu-Palmer similarity (`wup_similarity`) for pairs where the score exceeds 0.3.

### Interpreting the output

**Hypernym depths.** The 15 words range from depth 1 (*languid*, classified as the satellite adjective `dreamy.s.02`, which has no noun hypernym tree) to depth 11 (*flower*, *floating*, *lotus*). The botanical words (*flower*, *lotus*) sit deep in the taxonomy under `vascular_plant`, while the sacramental and ritual words (*communion*, *sacrament*) cluster under `activity`. The corporeal/material words (*body*, *bread*, *blood*, *water*, *drug*) group under `physical_entity`, `substance`, or `matter` at middling depths (5--7).

**Convergence points.** The LCS analysis reveals the episode's deep semantic topology through several clusters:

- **The physical-object cluster.** *flower*, *body*, *pin*, *altar*, *bath*, and *lotus* all share `whole.n.02` as a common ancestor, meaning WordNet groups them as "wholes" -- complete physical entities. This is fitting: the episode keeps juxtaposing whole objects (a pin, an altar, a bath, a flower) as if they were interchangeable vessels.

- **The substance cluster.** *drug*, *bread*, *blood*, and *water* converge at `substance.n.07` or `matter.n.03`. This is the Eucharistic group: the things that transform into one another during communion. That WordNet places drug, bread, blood, and water under a shared "substance" ancestor formally mirrors the episode's transubstantiation logic. The *drug/bread* pair has a notably high WuP similarity of 0.571 -- they are taxonomically close, both being substances that enter the body.

- **The activity cluster.** *floating*, *communion*, and *sacrament* converge at `activity.n.01`, with *communion/sacrament* scoring 0.667 (the highest WuP similarity among all pairs). *floating* and *communion* share the same ancestor at 0.600 -- an accidental but thematically resonant connection, since the episode links the floating sensation of the bath to the drift of the communion ritual.

- **The botanical pair.** *flower* and *lotus* converge at `vascular_plant.n.01` with a high WuP similarity of 0.727, the highest in the dataset. This is unsurprising taxonomically but meaningful for the episode: the lotus is the episode's governing flower, and their tight semantic bond reflects how Joyce collapses the generic ("a languid floating flower") into the specific (the lotus of the title).

- **body/bread.** This is the sacramental pair par excellence. Their LCS is `physical_entity.n.01` with a WuP similarity of only 0.308. WordNet's taxonomy puts them far apart -- body is a `natural_object`, bread is `food/baked_goods`. The low score captures the fact that transubstantiation is not a taxonomic relationship; it is metaphorical, sacramental, cultural. WordNet cannot represent "bread IS body" because that equivalence operates outside hyponymy. This is a key finding: the gap between taxonomic distance and felt thematic closeness is itself a datum about the limits of WordNet.

### Average synset count per thematic word

The exercise prompt asked for approximately 5--15 synsets per thematic word. While the script reports synset counts implicitly (stored in `hypernym_data[word]['num_synsets']`), these are not printed. The bonus section shows the episode-wide average is 9.36 synsets per content word, which falls in the expected range and indicates that Lotus Eaters' vocabulary is semantically rich -- most words carry many possible meanings, reinforcing the episode's theme of polysemy and substitution.

---

## Exercise 2: Martha's Malapropism and Semantic Distance

### What the code does

The function `marthas_malapropism()` computes three measures for 10 near-homophone pairs drawn from the episode's thematic concerns:

1. **Path similarity** (`ss1[0].path_similarity(ss2[0])`): based on the shortest path between two synsets in the hypernym tree. Ranges from 0 to 1; higher means closer.
2. **Wu-Palmer similarity** (`ss1[0].wup_similarity(ss2[0])`): based on depth of the two synsets and their lowest common subsumer. Also 0 to 1.
3. **Phonological distance**: loads the CMU Pronouncing Dictionary via `cmudict.dict()`, retrieves phoneme sequences for each word, and computes edit distance (`nltk.edit_distance`) between them. Lower means more similar in sound.

### Interpreting the output

The results table:

| Pair | Path Sim | WuP Sim | Phon Dist |
|------|----------|---------|-----------|
| world/word | 0.091 | 0.167 | 1 |
| flower/flour | 0.067 | 0.222 | 0 |
| altar/alter | 0.083 | 0.154 | 0 |
| body/bawdy | 0.077 | 0.143 | 1 |
| sole/soul | 0.071 | 0.235 | 0 |
| sun/son | 0.091 | 0.444 | 0 |
| holy/wholly | 0.111 | 0.200 | 0 |
| rite/right | 0.077 | 0.333 | 0 |
| bread/bred | 0.091 | 0.167 | 0 |
| wine/whine | 0.062 | 0.211 | 0 |

**The central finding:** Phonological distance is near-zero for almost every pair (most are perfect homophones with phon_dist = 0; *world/word* and *body/bawdy* differ by just one phoneme). Yet semantic similarity is uniformly low -- all path similarities are below 0.12 and all WuP similarities are below 0.45. Sound-alike words do not mean alike. This is the exercise's intended punchline: puns and malapropisms exploit the gap between phonological closeness and semantic distance.

**Martha's world/word specifically:** Path similarity of 0.091 and WuP similarity of 0.167, with a phonological distance of only 1. The words sound nearly identical but are semantically remote -- *world* (a physical or conceptual domain) and *word* (a unit of language) share no taxonomic neighborhood. Martha's substitution is devastating precisely because so small a phonological slip opens so large a semantic chasm. "I do not like that other world" transforms a complaint about vocabulary into an existential declaration.

**sun/son** has the highest WuP similarity at 0.444, making it the pair where phonological and semantic closeness are most aligned. This is interesting because the sun/son pun is itself ancient and theologically loaded (the "Son" who is also the "Sun" of righteousness), suggesting that some homophones have accrued cultural semantic proximity that taxonomic measures partially capture.

**flower/flour** has a phonological distance of 0 (perfect homophones) but very low semantic similarity (0.067 path, 0.222 WuP) -- a plant vs. ground grain. These words are etymologically related (flour is the "flower" of the wheat) but WordNet's synchronic taxonomy does not encode that history.

---

## Exercise 3: Substitution Chains

### What the code does

The function `substitution_chain(start_word, steps=10)` builds a chain by:

1. Looking up all synsets for the current word.
2. Iterating through each synset's lemmas to find the first lemma name that is neither the current word nor any previously visited word.
3. Adding that new word to the chain and repeating.
4. Stopping at 10 steps or when no unvisited synonym can be found (a "dead end").

It then runs this for five starting words -- *body, bread, flower, drug, water* -- and checks all pairs of chains for convergence (shared words).

### Interpreting the output

The chains:

- **body** -> organic structure -> [dead end] (2 steps)
- **bread** -> breadstuff -> staff of life -> [dead end] (3 steps)
- **flower** -> bloom -> blooming -> blossom -> prime -> prime quantity -> [dead end] (6 steps)
- **drug** -> dose -> dosage -> [dead end] (3 steps)
- **water** -> H2O -> [dead end] (2 steps)

Most chains die quickly -- after 2 to 3 steps. The *flower* chain is the longest at 6 steps, drifting beautifully from *flower* through *bloom* and *blossom* (botanical synonyms) into *prime* (because "bloom" can mean "prime of life") and then *prime quantity* (a mathematical term). This is semantic drift made visible: a flower becomes a number through a chain of synonym substitutions. It enacts in miniature the Lotus Eaters' narcotic logic, where meaning slides sideways through polysemy.

The *bread* chain reaches "staff of life" -- a poetic/biblical epithet -- before hitting a dead end. This is thematically apt: bread, the Eucharistic substance, arrives at a phrase connoting sustenance and the sacred before the chain breaks.

**Convergence.** The convergence check reports that every pair of chains "converges" at `[dead end]`, which is a bug rather than a real finding (see TODO). No actual word overlap exists between any two chains. The chains remain isolated, each drifting into its own semantic cul-de-sac. This is itself meaningful: WordNet's synonym networks are sparse enough that 10 steps of substitution from thematically related starting points do not produce convergence. The "narcotic substitution" of the episode -- where body, bread, water, and flower all seem to blur into one another -- is not captured by WordNet's synonym links. It operates at a level of association (cultural, metaphorical, imagistic) that exceeds what a lexical taxonomy can represent.

**Chain length vs. expectations.** The exercise prompt anticipated chains of 4--10 steps before dead-ending. Most chains here are shorter (2--3 steps), with only *flower* reaching 6. This suggests the synonym-traversal strategy (first unvisited lemma of the first synset) is quite conservative. A strategy that explored more synsets per step or used similarity-ranked candidates might produce longer chains.

---

## Bonus: Average Synset Count

### What the code does

The function `avg_synset_count(text)` tokenizes the full episode text, filters for alphabetic tokens longer than 3 characters, and computes the average number of WordNet synsets per content word.

### Interpreting the output

- **Average synsets per content word: 9.36**
- 3,115 words have at least one synset out of 3,705 content words (84%)

An average of 9.36 synsets per word is above the exercise prompt's expected range of 4--8, indicating that Lotus Eaters uses highly polysemous vocabulary. This aligns with the episode's thematic core: words in this chapter carry many meanings, drift between senses, and resist fixity. The high polysemy count is a quantitative signature of the episode's "narcotic" quality -- every word potentially means several things at once, and the reader floats among possible interpretations just as Bloom floats through his mid-morning errands.

The 84% coverage rate (words found in WordNet) is reasonable; the 16% not found likely includes proper nouns (Dublin place names, character names), archaic or dialectal terms, and very short words filtered out by the length threshold.

---

## Summary of Methods Used

| NLTK Function | Purpose in This Exercise |
|---|---|
| `wn.synsets(word)` | Retrieve all synsets for a word |
| `ss.hypernym_paths()` | Trace path from synset to root |
| `ss.lowest_common_hypernyms(ss2)` | Find where two words' hypernym trees converge |
| `ss.wup_similarity(ss2)` | Wu-Palmer similarity (depth-based) |
| `ss.path_similarity(ss2)` | Path-based similarity (shortest path length) |
| `cmudict.dict()` | Load CMU Pronouncing Dictionary for phoneme sequences |
| `nltk.edit_distance(p1, p2)` | Compute edit distance between phoneme sequences |
| `word_tokenize(text)` | Tokenize episode text for bonus analysis |
