# Week 8 Writeup: Lestrygonians -- N-gram Models and Language Generation

## Overview

This week pairs the Lestrygonians episode (Episode 8 of *Ulysses*) with n-gram language models and probabilistic text generation. The central analogy is between peristalsis -- the involuntary muscular wave that moves food through the digestive tract -- and the Markov property that drives n-gram models: each word is conditioned only on the immediately preceding word(s), just as each of Bloom's lunchtime thoughts is triggered by the one before it. The script uses NLTK's `nltk.lm` module (MLE and Laplace estimators), `nltk.util.bigrams` and `nltk.util.trigrams`, and standard tokenization via `nltk.tokenize.word_tokenize` and `sent_tokenize`.

---

## Exercise 1: Train and Generate

**What the exercise asks:** Build bigram and trigram language models for Lestrygonians and Proteus using `nltk.lm.MLE`, generate sentences from each, and compare the linguistic fingerprints of machine-generated Bloom versus machine-generated Stephen.

**How the code works:** The function `train_ngram_model` tokenizes the episode text into lowercase word-level sentences, feeds them through `padded_everygram_pipeline` (which adds start/end padding and generates all n-grams up to the specified order), and fits an `MLE` model. The function `generate_sentences` then calls `model.generate()` with a random seed, strips out padding tokens, and joins the resulting words.

### Lestrygonians (bigram) output

The bigram model produces recognizably Bloomian fragments that capture the episode's local texture -- short, clipped observations strung together by loose association:

- *"bloom walked, silver over her mouth full... heights, off some day"* -- This captures Bloom's peripatetic mode: walking, noticing, drifting. The phrase "her mouth full" evokes the episode's obsession with eating and mouths.
- *"her fat on. thinking... cycleshop"* -- The abrupt pivot from a bodily observation to "thinking" to "cycleshop" mimics Bloom's tendency to observe something physical and then jump to an unrelated commercial detail.
- *"filleted lemon sole, slush of your hare"* -- Pure Bloomian food language, the kind of menu-scanning that dominates the Burton restaurant scene.

The bigram model captures Bloom's characteristic juxtapositions -- food, commerce, women, Dublin geography -- but the sentences lack any sustained grammatical coherence. This is the Markov limitation: with only one word of context, the model cannot maintain clause structure.

### Lestrygonians (trigram) output

The trigram model produces more grammatical but much shorter output:

- *"mr bloom walked towards dawson street, mr bloom asked."* -- A perfectly coherent Bloomian sentence. The trigram context is enough to lock in the common phrase "Mr Bloom walked towards" and complete it with a plausible Dublin street name.
- *"chargesheets crammed with cases get their percentage manufacturing crime."* -- This reads like one of Bloom's compressed social observations, a sardonic thought about the legal system.

The trigram model's sentences are shorter because with more context, the model more quickly reaches end-of-sentence tokens. It trades the bigram model's wild associative sprawl for local grammatical correctness.

### Proteus (bigram) output

Stephen's generated text is immediately distinguishable from Bloom's:

- *"occam thought through the suck his feet are consubstantial father"* -- The vocabulary alone marks this as Stephen: "Occam," "consubstantial," the philosophical register that is entirely absent from Bloom's episodes.
- *"je suis socialiste... et erant valde bona"* -- The multilingual fragments (French, Latin) are a Stephen signature. Bloom thinks in English about practical things; Stephen thinks in multiple languages about abstractions.
- *"tripudium... fourworde"* -- Archaisms and learned vocabulary that reflect Stephen's literary self-consciousness.

### Proteus (trigram) output

- *"his shadow lay over the dead"* -- Somber, literary, imagistic. Compare with Bloom's trigram output about walking towards Dawson Street. Stephen's model generates poetic fragments; Bloom's generates pedestrian (literally) ones.
- *"moon, his feet, curling, unfurling many crests, every ninth, breaking, plashing"* -- A sea-description that captures Proteus's Sandymount Strand setting, with the rhythmic, incantatory quality of Stephen's prose.

### Character comparison

The models successfully distinguish the two characters' linguistic fingerprints. Bloom's generated text is concrete, commercial, food-oriented, and syntactically clipped. Stephen's is abstract, multilingual, philosophical, and rhythmically elaborate. This confirms that even a simple statistical model can detect the deep stylistic differences Joyce built into his two protagonists' interior monologues. What neither model can capture is the *purpose* behind these word chains -- Bloom's sentences have no destination, no emotional arc, because the Markov property prevents any long-range planning.

---

## Exercise 2: Perplexity as Style Measure

**What the exercise asks:** Train a Laplace-smoothed bigram model on Lestrygonians and compute its perplexity on four test texts: Lestrygonians itself, Calypso (another Bloom episode), Proteus (a Stephen episode), and a reference prose text. Rank by perplexity and interpret.

**How the code works:** The `compute_perplexity` function trains a `Laplace` bigram model (which adds one to every count to avoid zero probabilities), then evaluates it on test text by generating padded bigrams from the test sentences and calling `model.perplexity()`. Laplace smoothing is essential here because literary text has an enormous vocabulary and many word combinations that appear in one episode but not another.

### Results

| Test Text | Perplexity |
|---|---|
| Lestrygonians (self) | 390.93 |
| Calypso (Bloom) | 582.01 |
| Proteus (Stephen) | 728.61 |
| Emma (Austen) | 1288.26 |

### Interpretation

The ranking is exactly what we would predict, and it confirms the exercise's hypothesis:

1. **Lestrygonians (390.93)** -- The model is least surprised by its own training data. This is the baseline. A perplexity of 391 is still quite high in absolute terms, which reflects the richness and unpredictability of Joyce's prose even within a single episode. (For comparison, a well-trained bigram model on ordinary English newspaper text might achieve perplexities in the 200-300 range.)

2. **Calypso (582.01)** -- The second-lowest perplexity confirms that Bloom's language in Calypso resembles his language in Lestrygonians. Both episodes are Bloom's interior monologue; they share vocabulary (food, Dublin, Molly, practical concerns) and syntactic patterns (short sentences, incomplete thoughts, concrete nouns). The model correctly identifies another Bloom episode as the closest stylistic match.

3. **Proteus (728.61)** -- Stephen's prose is significantly more surprising to a Bloom-trained model than Bloom's own prose in a different episode. The gap between Calypso (582) and Proteus (729) quantifies the stylistic distance between the two characters. Stephen's philosophical vocabulary, multilingual phrases, and longer syntactic structures are all bigram patterns that the Lestrygonians model rarely or never encountered.

4. **Emma (Austen) (1288.26)** -- Jane Austen's prose is the most surprising of all, nearly doubling Proteus's perplexity. This is not because Austen writes badly -- it is because her language belongs to an entirely different century, genre, and register. The formal syntax of Regency-era narration, the vocabulary of drawing rooms and propriety, and the long balanced clauses are all maximally foreign to a model trained on Bloom's fragmented, early-20th-century Dublin consciousness.

The key insight: perplexity functions as a **stylistic distance measure**. The model's "surprise" is a proxy for linguistic dissimilarity. The ranking -- same episode < same character < different character < different author/century -- maps cleanly onto our intuitive understanding of stylistic proximity in *Ulysses*.

A subtlety worth noting: Stephen's prose is closer to Bloom's than Austen's is, even though Stephen and Bloom are very different thinkers. This is because they still share the same novel -- the same Dublin setting, the same historical moment, many of the same character names and place names. The shared context of *Ulysses* makes even its most contrasting voices more alike than either is to an entirely different text.

---

## Exercise 3: Associative Chains

**What the exercise asks:** Extract bigrams ranked by conditional probability P(w2|w1), identify the top 20 strongest associations, then extract cross-sentence boundary bigrams that capture Bloom's inter-sentence associative logic.

**How the code works:** The `associative_chains` function computes bigram and unigram frequency distributions from lowercased alphabetic tokens, calculates conditional probabilities as count(w1,w2)/count(w1) with a minimum frequency threshold of 3, and sorts by probability. For cross-sentence bigrams, it uses `sent_tokenize` to find sentence boundaries and pairs the last word of each sentence with the first word of the next.

### Top 20 Bigram Associations

The top associations fall into several clear categories:

**Tokenization artifacts (contractions):** "didn't," "wouldn't," "isn't," "doesn't," "couldn't," "don't" all appear as bigrams like `didn -> t` with probability 1.0 because the tokenizer splits contractions into two tokens. These are mechanically perfect associations but linguistically uninteresting -- they tell us Bloom uses contractions frequently (which itself is a marker of informal, spoken-register interior monologue, unlike the more formal prose of the narration sections).

**Character names:** `davy -> byrne` (1.0, 18 occurrences), `paddy -> leonard` (1.0, 8 occurrences), `bantam -> lyons` (1.0, 4 occurrences), `nosey -> flynn` (0.96, 24 occurrences), `blind -> stripling` (1.0, 3 occurrences). These are the episode's dramatis personae: Davy Byrne's pub is where Bloom eventually eats lunch, and Paddy Leonard, Bantam Lyons, and Nosey Flynn are the drinkers he encounters there. The blind stripling is a figure Bloom observes with characteristic empathy. These name-bigrams confirm that the episode is dominated by these particular social encounters.

**The protagonist:** `mr -> bloom` (0.81, 38 occurrences) is the single most frequent association in the episode. The 0.81 probability means that when "mr" appears, it is followed by "bloom" 81% of the time -- confirming Bloom's centrality.

**Collocations:** `lombard -> street` (1.0), `number -> one` (1.0), `washed -> in` (1.0), `ought -> to` (1.0), `used -> to` (0.9), `want -> to` (0.78), `telling -> me` (0.8). These are fixed phrases and collocations that appear in Bloom's speech and thought patterns.

**Thematic note:** The exercise asks whether the top associations capture the episode's thematic preoccupations with food, body, Molly, and memory. The answer is mixed. The character names and place names do capture the social architecture of the episode (Davy Byrne's pub, the various Dubliners Bloom encounters). However, the highest-probability bigrams are dominated by names and contractions rather than by thematic vocabulary like "food," "eat," "Molly," or "memory." This is because thematic words tend to appear in varied contexts (Bloom thinks about food in many different constructions), which lowers their conditional probability for any single following word. The most thematically revealing associations would appear further down the ranked list, where words like "eat," "taste," or "Molly" might show moderate-probability associations with multiple following words.

### Cross-Sentence Associative Links

The most frequent cross-sentence transitions are:

- `said -> he` (10 times), `said -> i` (7 times) -- These are dialogue tags. Many sentences in the episode end with "said" and the next sentence begins with the speaker's pronoun. These are structurally unremarkable but reveal how much of the episode consists of reported speech (Bloom is in a social setting at Davy Byrne's).

- `coming -> is` (3 times), `that -> yes` (3 times) -- The `that -> yes` transition captures a distinctive Bloomian tic: Bloom frequently ends a thought and then affirms it to himself ("Yes"). This self-confirming habit is one of his most recognizable verbal signatures.

- `out -> molly` (2 times) -- Molly surfaces at sentence boundaries, which is thematically significant. Bloom's thoughts of Molly are intrusive -- they break into his other trains of thought rather than being logically connected to them.

- `so -> davy` (3 times) -- Bloom's thoughts return repeatedly to the pub as a destination, often after a digressive "so" that signals a return to practical concerns.

### Sample Cross-Sentence Transitions

The ten sampled transitions demonstrate several types of associative logic:

1. **`[lamb] -> [his]`**: *"Blood of the Lamb. His slow feet walked him riverward..."* -- A religious phrase (Blood of the Lamb) triggers a shift to Bloom's own body in motion. The association is **thematic/sensory**: "lamb" (sacrifice, blood, also meat) connects to Bloom's physical self walking through Dublin. The word "blood" bridges the sacred and the bodily.

2. **`[kippur] -> [crossbuns]`**: *"Yom Kippur. Crossbuns."* -- This is a brilliant **thematic** association: one religious food tradition (the Jewish fast of Yom Kippur) triggers another (the Christian tradition of hot cross buns). Bloom, who is both Jewish and Catholic by background, naturally bridges the two traditions through their shared connection to ritual and food.

3. **`[one] -> [timeball]`**: *"After one. Timeball on the ballastoffice is down."* -- A **temporal** association: "one" (the time, after one o'clock) triggers a thought about the timeball, a physical time-keeping device. Bloom's mind moves from abstract time to concrete time-telling apparatus.

4. **`[up] -> [sister]`**: *"Molly tasting it, her veil up. Sister?"* -- A **sensory/memory** chain: the image of Molly with her veil up triggers a sudden associative jump to "Sister?" -- possibly a nun, possibly a relative. The connection seems to run through the veil (nuns wear veils; Molly's veil is up).

5. **`[yes] -> [mrs]`**: *"O yes! Mrs Miriam Dandrade..."* -- An **idiosyncratic/personal** association: Bloom's affirmative "yes" unlocks a specific memory of a specific person. The exclamation mark suggests the memory arrived with a jolt of recognition.

6. **`[hornies] -> [that]`**: *"hornies. That horsepoliceman the day Joe Chamberlain..."* -- A **phonetic/sensory** association: "hornies" (slang) triggers "horsepoliceman," linked by the shared "hor-" sound. The phonetic similarity pulls a specific memory (Joe Chamberlain's visit) into the stream.

7. **`[front] -> [trams]`**: *"Trinity's surly front. Trams passed one another..."* -- A **spatial/visual** association: Bloom sees Trinity College's facade and his eye follows the movement in front of it -- trams passing. This is pure visual stream of consciousness, the eye moving across a scene.

8. **`[eh] -> [showing]`**: *"How time flies, eh? Showing long red pantaloons..."* -- An **idiosyncratic** jump: the conversational filler "eh?" clears the mental slate, and Bloom's attention lands on a visual detail (red pantaloons). The association is not logical but perceptual -- Bloom's eye wanders when his conversation lapses.

9. **`[it] -> [women]`**: *"Then she mightn't like it. Women won't pick up pins."* -- A **thematic** association: thinking about a specific woman's preferences ("she mightn't like it") triggers a generalization about women as a class. This is a characteristic Bloom move: from the particular to the general, from Molly to Women.

10. **`[heady] -> [yes]`**: *"Too heady. Yes, it is."* -- Another instance of Bloom's self-affirming "Yes." He makes a judgment ("too heady"), pauses at the sentence boundary, and then confirms it to himself. The cross-sentence "yes" is a rhythmic signature of Bloom's thinking.

These transitions collectively demonstrate that Bloom's associative logic operates through multiple channels simultaneously -- sensory perception, thematic resonance, phonetic similarity, spatial contiguity, and pure personal idiosyncrasy. The n-gram model captures the local texture of these transitions but cannot explain *why* they occur; the explanation requires the kind of close reading that the model's statistical output enables but cannot itself perform.

---

## Summary

The three exercises build a cumulative picture of what n-gram models can and cannot capture about literary style:

- **Exercise 1** shows that even simple bigram/trigram models produce text that is recognizably "Bloomian" or "Stephenian" in vocabulary and local texture, but globally incoherent -- confirming that stream of consciousness derives its character from local word-to-word transitions (which n-grams capture) layered over long-range thematic structure (which they do not).

- **Exercise 2** demonstrates that perplexity functions as a quantitative measure of stylistic distance, correctly ranking texts from most-similar (same episode) to least-similar (different author and century), with another Bloom episode falling between same-episode and a Stephen episode.

- **Exercise 3** reveals the specific associative machinery of Bloom's mind: character names and fixed phrases dominate the highest-probability bigrams, while cross-sentence transitions expose the multiple channels (sensory, thematic, phonetic, idiosyncratic) through which one thought triggers the next -- the peristaltic logic of consciousness that gives the episode its name and structure.
