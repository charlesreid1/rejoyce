# Rejoyce

An 18-week course using Python's Natural Language Toolkit (NLTK) to analyze
James Joyce's *Ulysses*. Each of the novel's 18 episodes is paired with a
different NLP technique, chosen to reflect the episode's literary style,
Homeric parallels, and place in Joyce's own schemas. The course moves from
foundational techniques (tokenization, frequency analysis) toward more
sophisticated methods (classification, language modeling, topic modeling),
mirroring the arc of *Ulysses* itself as it grows progressively more
experimental.

The `txt/` directory contains the full text of *Ulysses* (via Project
Gutenberg), split by episode. Each week has its own directory (`week01/`
through `week18/`), containing the exercise sheet and Python solution.

## Weekly exercises

- **Week 1 - Telemachus: Tokenization & corpus exploration.** Tokenize and profile text statistics, build concordances for thematic words, and analyze frequency distributions with and without stopwords.
- **Week 2 - Nestor: POS tagging & morphological analysis.** POS-tag the text and compare distributions, segment dialogue vs. interior monologue by voice, and lemmatize to identify distinctive vocabulary.
- **Week 3 - Proteus: Stemming & language identification.** Compare three stemmers on Joyce's vocabulary, detect multilingual passages with stopword heuristics, and analyze the derivational morphology of neologisms.
- **Week 4 - Calypso: Named entity recognition & chunking.** Run NER on Bloom vs. Stephen episodes and compare entity density, write chunking grammars for noun and prepositional phrases, and build entity co-occurrence matrices.
- **Week 5 - Lotus Eaters: WordNet & lexical semantics.** Trace semantic fields through hypernym trees, compute semantic distance for malapropisms vs. homophones, and build substitution chains through synonym networks.
- **Week 6 - Hades: Sentiment analysis & affective lexicons.** Plot VADER sentiment trajectory across the funeral, compare Bloom's interior register to the narrator's, and examine SentiWordNet's context-free scoring limitations.
- **Week 7 - Aeolus: TF-IDF & extractive summarization.** Implement TF-IDF from scratch to extract keywords per section, compare results to Joyce's own headlines, and detect rhetorical figures like anaphora and tricolon.
- **Week 8 - Lestrygonians: N-gram models & language generation.** Train bigram and trigram models and generate text, compute perplexity across episodes, and extract bigram associations to trace Bloom's associative logic.
- **Week 9 - Scylla and Charybdis: Context-free grammars & parsing.** Write CFGs and parse complex argumentative sentences, compare syntactic depth to Treebank baselines, and parse quoted Shakespeare separately from Joyce's framing prose.
- **Week 10 - Wandering Rocks: Text similarity & document clustering.** Build a TF-IDF similarity matrix for the episode's 19 sections, detect interpolations as anomalies, and track named entities across sections to reveal narrative architecture.
- **Week 11 - Sirens: Phonetic analysis & sequence detection.** Match the overture's 63 fragments to their body passages, compute alliteration and assonance density, and track recurring motifs with edit distance trajectories.
- **Week 12 - Cyclops: Text classification & genre detection.** Annotate barfly narration vs. interpolations and train a Naive Bayes classifier, profile the barfly's distinctive voice, and quantify genre parody as feature amplification.
- **Week 13 - Nausicaa: Stylometry & authorship attribution.** Compute stylometric profiles for both halves of the episode, implement Burrows' Delta to measure distance from other texts, and build a cliche detector for prefabricated language.
- **Week 14 - Oxen of the Sun: Diachronic corpus analysis & historical style.** Build period-specific stylistic profiles from historical corpora, segment the episode into style periods and compare to real baselines, and train a period classifier to date Joyce's imitations.
- **Week 15 - Circe: Entity extraction & network visualization.** Parse the dramatic format to extract 100+ speakers, build interaction graphs and compute centrality metrics, and construct a cumulative entity network across all prior episodes.
- **Week 16 - Eumaeus: Corpus-wide metrics & data visualization.** Compile a master dataset of 25+ metrics across all episodes, build a multi-panel dashboard with heatmaps and radar charts, and audit prior analyses for errors.
- **Week 17 - Ithaca: Information extraction & knowledge graphs.** Parse the catechism Q&A format into 300+ pairs and classify question types, extract knowledge triples from semi-structured answers, and analyze topic distribution.
- **Week 18 - Penelope: Text segmentation & topic modeling.** Apply TextTiling to Molly's unsegmented monologue, run LDA to detect latent topics and visualize as a stacked area chart, and recompute Week 1 metrics to close the arc.
