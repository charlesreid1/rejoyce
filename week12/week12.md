# Week 12: Cyclops
### *"I was just passing the time of day with old Troy of the D.M.P. at the corner of Arbour hill there and be damned but a bloody sweep came along..." — One eye, many voices, and the art of monstrous inflation.*

There is no episode in Ulysses — and possibly no chapter in all of English literature — quite like Cyclops. The primary narration is delivered by an unnamed Dublin barfly: opinionated, profane, gossipy, casually anti-Semitic, hilarious, unreliable. He recounts a scene in Barney Kiernan's pub where Bloom confronts "the citizen" (a bombastic Irish nationalist modeled on Michael Cusack) and various Dublin types drink, argue, and hold forth on politics, hanging, love, and the Irish nation. So far, so realist. But the episode is periodically invaded by **gigantist interpolations** — enormous parodic set-pieces that inflate whatever was just mentioned into the register of some other genre entirely. A casual mention of a legal proceeding triggers a page-long parody of courtroom reportage. A list of wedding gifts becomes a catalog of mythological treasures. A description of the citizen morphs into a passage of mock-heroic saga. A discussion of capital punishment spawns a grotesquely detailed newspaper account of an execution. Each interpolation is a pitch-perfect pastiche of a specific genre — legal, journalistic, scientific, biblical, epic, sentimental — swollen to absurd proportions.

The episode's technique is *gigantism*; its art is *politics*; its organ is the *eye* (singular — the Cyclops sees with one eye, and the episode is about the distortions of monocular vision: nationalism, bigotry, the inability to see another perspective). The interpolations are the textual manifestation of gigantism: language inflated until it becomes grotesque, genre conventions amplified until they expose their own absurdity.

**NLTK Focus:** Text classification, feature engineering, and genre/register detection (`nltk.classify`, `NaiveBayesClassifier`, `DecisionTreeClassifier`, feature extraction, evaluation metrics)

**Pairing Rationale:**
Cyclops hands you a supervised classification problem on a silver platter. The episode contains two radically distinct text types — the barfly's vernacular narration and the gigantist interpolations — and the interpolations themselves subdivide into a dozen recognizable genres (legal, scientific, epic, journalistic, romantic, biblical, etc.). Each genre has characteristic features: legal prose has latinate vocabulary, passive constructions, and nested subordinate clauses; journalistic prose has short sentences, named sources, and the inverted pyramid; epic prose has formulaic epithets, catalogues, and dactylic rhythm. The classifier's job is to learn these features from data and use them to sort text into categories — which is exactly what a reader does, instinctively, when they hit an interpolation and recognize it as parody. The beauty of the pairing is that Joyce has done the work of a hostile adversarial data scientist: he has taken genre features and exaggerated them to the point of caricature, which means the features are *more* separable than in real-world genre classification, but the underlying question — what makes legal prose *legal*? what makes epic *epic*? — is the same. And the one-eyed Cyclops is the classifier itself: a system that sees each text through a single feature vector, reducing the rich, ironic, multi-layered thing that Joyce actually wrote to a label.

**Core Exercises:**

1. **Annotate and classify.** Manually segment Cyclops into its component text types. At minimum, distinguish: (a) the barfly's primary narration, (b) the gigantist interpolations. If you're ambitious — and you should be, this is Cyclops — further classify the interpolations by genre: legal, journalistic, epic/mythological, scientific/encyclopedic, biblical/liturgical, sentimental/romantic, and any others you identify. (There is no canonical count; scholars disagree on the exact number. Your annotation is a critical act.) Now extract features from each segment. Start with simple bag-of-words features, then add: average sentence length, type-token ratio, proportion of latinate vocabulary (approximate using word length as a proxy or check against an etymological word list), proportion of passive voice (approximate via POS patterns: `<VB.*> <VBN>`), and proportion of named entities. Train an NLTK `NaiveBayesClassifier` using 70% of your annotated segments as training data and 30% as test data. Report accuracy, and more importantly, examine the `show_most_informative_features()` output. Which features does the classifier rely on to distinguish genres? Are they the features you'd have chosen?

2. **The barfly's fingerprint.** The unnamed narrator has one of the most distinctive voices in the novel — colloquial Dublin English, heavy with slang, hedging, direct address, and performative vulgarity. Extract features that characterize his register specifically: frequency of first-person pronouns, frequency of discourse markers (*says I*, *says he*, *begob*, *be damned*, *bloody*), average sentence length, proportion of dialogue verbs, use of the historical present tense. Build a profile. Now take this feature profile and scan it against other episodes: can you find passages in Aeolus, Hades, or Wandering Rocks that statistically resemble the barfly's voice? (There shouldn't be many — his register is uniquely informal for the novel.) Use your classifier to generate a "barfly probability score" for every paragraph in Ulysses so far. Where does it spike outside of Cyclops?

3. **Gigantism as feature amplification.** Here is the exercise that gets at the heart of what Joyce is doing. Take the feature vectors for the interpolations and compare them to *real* examples of their source genres. For the legal parody: find an actual legal document (NLTK's `reuters` corpus or a legal corpus) and extract the same features. For the journalistic parody: use a news corpus. For each genre, compute the ratio: how much more "legal" is Joyce's legal parody than real legal prose? How much more "journalistic"? Quantify the exaggeration. Joyce's gigantism should manifest as feature values that are *systematically more extreme* than the real-world genre they parody — longer sentences than real legal prose, more epithets than real epic, more passive voice than real scientific writing. This is parody as statistical caricature: take every characteristic feature and crank it past 11. Can you measure the dial position?

**Diving Deeper:**

- The Naive Bayes classifier makes a strong independence assumption: it treats each feature as independent of all others. This is linguistically absurd (the presence of *whereas* in a sentence is not independent of sentence length or latinate vocabulary — they all correlate because they all signal formal register). But NB often works anyway, a phenomenon known as the "Naive Bayes paradox." Explore this: compare NB's performance to a `DecisionTreeClassifier` or a `MaxentClassifier` (logistic regression) that can model feature interactions. Does the more sophisticated model actually perform better on Cyclops, or is NB's simplicity sufficient when the genres are as exaggerated as Joyce makes them?
- Cyclops is one of the most politically charged episodes of the novel — it stages a confrontation between Bloom's cosmopolitan humanism and the citizen's exclusionary nationalism. Modern NLP has its own politics-of-classification problem: text classifiers trained on biased data reproduce and amplify those biases. The connection is not metaphorical. The citizen's one-eyed worldview ("a nation is the same people living in the same place") is a classification heuristic — and a bad one. Explore work on fairness in NLP: Blodgett et al. (2020), Bender et al. (2021). What happens when a sentiment classifier trained on standard American English encounters Hiberno-English dialect features? Does it misclassify? This is the computational version of the citizen's bigotry.
- Genre classification at scale is a well-studied problem. See Kessler et al. (1997) and Sharoff et al. (2010) for foundational work on automatic genre identification. The key insight from this literature: genre is not a fixed taxonomy but a fuzzy, overlapping, historically contingent system of expectations — precisely the sort of thing Joyce understood and exploited. Does a classifier trained on 19th-century genre distinctions apply to 20th-century parody of those genres?
- The interpolations in Cyclops have been compared to Bakhtin's concept of *heteroglossia* — the coexistence of multiple social languages within a single text. Bakhtin argued that the novel as a form is defined by this multi-voicedness. Your classifier is, in a sense, a Bakhtin machine: it detects and labels the distinct social languages in the text. See Bakhtin's "Discourse in the Novel" (1935) and Morson & Emerson's *Mikhail Bakhtin: Creation of a Prosaics* (1990).
- Connection to Week 7 (Aeolus): that episode's headlines and rhetorical registers were a gentler version of Cyclops' multi-register structure. But where Aeolus maintains a clear hierarchy (the narration is primary, the headlines are paratextual), Cyclops gives the interpolations equal textual weight — sometimes more. The classification framework allows a precise comparison: are the register shifts in Cyclops more extreme (higher inter-class distance) than in Aeolus? Your feature vectors can answer this.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Manually annotate** text segments by genre/register and understand annotation as a critical act.
2. **Engineer features** for text classification: bag-of-words, sentence length, TTR, POS proportions, passive voice rate, discourse markers.
3. **Train and evaluate** a Naive Bayes classifier using NLTK, including accuracy metrics, train/test splitting, and most-informative-feature inspection.
4. **Profile a distinctive voice** (the barfly narrator) and scan for it across other texts.
5. **Quantify parody as feature amplification** — measuring how Joyce exaggerates genre characteristics beyond their real-world baselines.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Cyclops) |
|---|---|---|
| Barfly segments | heuristic segmentation count | ~60–120 |
| Interpolation segments | formal/parodic segments | ~10–30 |
| NB accuracy (barfly vs. interp) | 70/30 train/test split | ~0.70–0.90 |
| Most informative feature | NB top feature | likely avg_sent_len or discourse_markers |
| Feature amplification ratio | interpolation feature / baseline feature | > 1.5x for key features |
| Barfly probability outside Cyclops | discourse_marker similarity | near 0 for most episodes |

## Rubric

### Exercise 1: Annotate and Classify (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Annotation** | Text segmented into barfly + interpolation types; interpolations further sub-classified by genre (legal, epic, journalistic, etc.) | Binary segmentation (barfly/interpolation) | No segmentation or broken |
| **Feature extraction** | 6+ features including sentence length, TTR, word length, POS proportions, passive voice, discourse markers | 4+ features | Fewer than 4 |
| **Classification** | NB trained; accuracy reported; most-informative features shown and discussed; features connected to genre characteristics | Classifier runs; accuracy reported | Classifier broken |

### Exercise 2: The Barfly's Fingerprint (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Profile construction** | All key features computed for the barfly voice; distinctive markers identified | Some profile features | Incomplete profile |
| **Cross-episode scan** | 5+ episodes scanned for barfly-like passages; "barfly probability" computed | 3+ episodes scanned | Fewer than 3 |
| **Uniqueness assessment** | Demonstrates that the barfly's register is uniquely informal for the novel; discusses what makes it distinctive | Some uniqueness noted | No assessment |

### Exercise 3: Gigantism as Feature Amplification (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Baseline comparison** | Interpolation features compared to real-genre baselines or normal-prose baselines; ratios computed | Some comparison | No comparison |
| **Amplification quantified** | Feature ratios show systematic exaggeration; "dial position" discussed | Ratios computed | No quantification |
| **Interpretation** | Connects quantitative amplification to Joyce's satiric method; discusses parody as statistical caricature | Brief interpretation | No interpretation |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| DecisionTree or MaxEnt comparison to NB | +3 |
| Fairness-in-NLP connection (dialect bias, Blodgett et al.) | +3 |
| Genre classification at scale (Kessler et al. framework) | +2 |
| Bakhtin/heteroglossia connection | +2 |

## Reference Implementation

See [`week12_cyclops.py`](week12_cyclops.py)
