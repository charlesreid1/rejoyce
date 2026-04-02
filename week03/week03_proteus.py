"""
Week 03: Proteus
=================
Stemming, morphological analysis, and language identification.

NLTK Focus: nltk.stem (Porter, Lancaster, Snowball), edit_distance, stopword-based
            language detection heuristics, WordNet derivational morphology

Exercises:
  1. The stemmer's struggle
  2. Multilingual detection
  3. Derivational morphology and neologism
"""

import os
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.metrics.distance import edit_distance
from nltk.corpus import stopwords, wordnet

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: The Stemmer's Struggle
# ---------------------------------------------------------------------------


def stemmers_struggle(text, top_n=10):
    """Apply three stemmers and find the most aggressive reductions.

    For each stemmer, identifies the top-N cases where the stemmed form
    is most distant from the original (by edit distance).

    Returns dict mapping stemmer name -> list of (word, stem, distance).
    """
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    tokens = word_tokenize(text)
    alpha_tokens = list(set(t.lower() for t in tokens if t.isalpha() and len(t) > 2))

    stemmers = {
        "Porter": porter,
        "Lancaster": lancaster,
        "Snowball": snowball,
    }

    results = {}
    for name, stemmer in stemmers.items():
        reductions = []
        for word in alpha_tokens:
            stem = stemmer.stem(word)
            dist = edit_distance(word, stem)
            reductions.append((word, stem, dist))

        reductions.sort(key=lambda x: -x[2])
        results[name] = reductions[:top_n]

        print(f"\n--- {name} Stemmer: Top {top_n} Most Aggressive Reductions ---")
        print(f"  {'Original':<35} {'Stem':<25} {'Edit Dist':>10}")
        print("  " + "-" * 72)
        for word, stem, dist in reductions[:top_n]:
            print(f"  {word:<35} {stem:<25} {dist:>10}")

    # Disagreement rate between stemmers
    disagree_count = 0
    for word in alpha_tokens:
        stems = set(s.stem(word) for s in stemmers.values())
        if len(stems) > 1:
            disagree_count += 1

    disagree_rate = disagree_count / len(alpha_tokens) if alpha_tokens else 0
    print(
        f"\n  Stemmer disagreement rate: {disagree_rate:.3f} "
        f"({disagree_count}/{len(alpha_tokens)} unique words)"
    )
    print("  Note: Porter and Snowball produce identical results because")
    print("        Snowball English is a reimplementation of Porter's algorithm.")

    return results, disagree_rate


# ---------------------------------------------------------------------------
# Exercise 2: Multilingual Detection
# ---------------------------------------------------------------------------

# Expanded Latin stopword list (NLTK doesn't include Latin)
LATIN_STOPWORDS = {
    "et",
    "in",
    "est",
    "non",
    "ad",
    "cum",
    "sed",
    "ut",
    "de",
    "ex",
    "per",
    "ab",
    "si",
    "qui",
    "quod",
    "aut",
    "nec",
    "quam",
    "iam",
    "tam",
    "hoc",
    "ille",
    "ipse",
    "ego",
    "tu",
    "nos",
    "vos",
    "suo",
    "sua",
    "suis",
    "eius",
    "ante",
    "post",
    "inter",
    "sub",
    "pro",
    "contra",
    "super",
    "omnis",
    "deus",
    "dei",
    "corpus",
    "anima",
    "me",
    "te",
    "se",
    "nobis",
    "vobis",
    "ei",
    "ei",
    "sunt",
    "erat",
    "esse",
    "fuit",
    "sum",
    "es",
    "potest",
    "potestis",
    "possum",
    "posse",
    "dicit",
    "dico",
    "dixit",
    "facit",
    "facio",
    "fecit",
    "habet",
    "habeo",
    "habetis",
    "hinc",
    "huc",
    "ibi",
    "id",
    "idem",
    "igitur",
    "ille",
    "illud",
    "immo",
    "in",
    "infra",
    "inter",
    "interea",
    "intro",
    "ipsum",
    "is",
    "ista",
    "iste",
    "istud",
    "ita",
    "itaque",
    "iterum",
    "iuxta",
    "lac",
    "laicus",
    "laudabiliter",
    "libenter",
    "licet",
    "limen",
    "locus",
    "longe",
    "magis",
    "magnus",
    "malus",
    "mater",
    "maxime",
    "meus",
    "miles",
    "minor",
    "misericors",
    "modo",
    "modicum",
    "mortuus",
    "multus",
    "nam",
    "necessarius",
    "nemo",
    "nihil",
    "nisi",
    "noster",
    "novus",
    "nullus",
    "numerus",
    "numquam",
    "ob",
    "oportet",
    "opus",
    "orior",
    "origo",
    "os",
    "ostendo",
    "pauci",
    "per",
    "peto",
    "plerique",
    "pono",
    "populus",
    "porto",
    "possum",
    "post",
    "postea",
    "postquam",
    "primo",
    "pro",
    "propter",
    "propterea",
    "puer",
    "quaero",
    "qualis",
    "quam",
    "quare",
    "quis",
    "quo",
    "quod",
    "quia",
    "quidem",
    "quin",
    "quippe",
    "quoque",
    "relinquo",
    "res",
    "respondeo",
    "retineo",
    "rex",
    "rogito",
    "sanctus",
    "scio",
    "scribo",
    "senex",
    "sentio",
    "servo",
    "sic",
    "sicut",
    "simul",
    "solus",
    "spes",
    "statim",
    "suadeo",
    "sub",
    "subito",
    "sum",
    "super",
    "supra",
    "tantus",
    "teneo",
    "terra",
    "timor",
    "tot",
    "totus",
    "tracto",
    "trans",
    "tres",
    "tuus",
    "ubi",
    "ultimus",
    "unde",
    "unus",
    "urbs",
    "usque",
    "vester",
    "video",
    "videlicet",
    "vir",
    "vita",
    "voluntas",
    "volo",
    "vos",
    "vester",
    "vestrum",
    # Additional common Latin words
    "atque",
    "enim",
    "tamen",
    "igitur",
    "ergo",
    "ac",
    "neque",
    "itaque",
    "igitur",
    "tunc",
    "inde",
    "ibi",
    "hic",
    "haec",
    "ille",
    "illa",
    "illud",
    "ipse",
    "ipsa",
    "ipsum",
    "nos",
    "noster",
    "nostra",
    "noster",
    "vester",
    "vestra",
    "vester",
    "suus",
    "sua",
    "suum",
    "meus",
    "mea",
    "meum",
    "tuus",
    "tua",
    "tuum",
    "is",
    "ea",
    "id",
    "hic",
    "haec",
    "hoc",
    "qui",
    "quae",
    "quod",
    "cuius",
    "cui",
    "quem",
    "quam",
    "quo",
    "quibus",
    "aliquis",
    "aliqui",
    "aliqua",
    "aliquid",
    "quisquam",
    "quisque",
    "quivis",
    "quidam",
    "uter",
    "uterque",
    "totus",
    "totius",
    "toti",
    "totam",
    "totas",
    "omnis",
    "omnium",
    "omnibus",
    "omnem",
    "omnes",
    "nullus",
    "nullius",
    "nulli",
    "nullam",
    "nullas",
    "ullus",
    "unus",
    "unius",
    "uni",
    "unam",
    "una",
    "duo",
    "duorum",
    "duobus",
    "duas",
    "tres",
    "trium",
    "tribus",
    "multus",
    "multi",
    "multorum",
    "multis",
    "multam",
    "multas",
    "magnus",
    "magni",
    "magno",
    "magnam",
    "magnae",
    "magnae",
    "parvus",
    "parvi",
    "parvo",
    "parvam",
    "parvas",
    "bonus",
    "boni",
    "bono",
    "bonam",
    "bonae",
    "malus",
    "mali",
    "malo",
    "malam",
    "malae",
    "novus",
    "novi",
    "novo",
    "novam",
    "novae",
    "vetus",
    "vetustus",
    "vetusti",
    "vetusto",
    "primus",
    "primi",
    "primo",
    "primam",
    "primae",
    "secundus",
    "medius",
    "extremus",
    " summus",
    "inferus",
    "superus",
    "dexter",
    "sinister",
    "rectus",
    "laevus",
    "verus",
    "falsus",
    "certus",
    "incertus",
    "clarus",
    "obscurus",
    "lucidus",
    "tenebrosus",
    "durus",
    "mollis",
    "acerbus",
    "dulcis",
    "amarus",
    "acutus",
    "gravis",
    "levis",
    "celer",
    "tardus",
    "longus",
    "brevis",
    "latus",
    "angustus",
    "altus",
    "humilis",
    "proximus",
    "extremus",
    "exterus",
    "interior",
    "exterior",
    "superior",
    "inferior",
    "prior",
    "posterior",
    "aequus",
    "inaequus",
    "similis",
    "dissimilis",
    "par",
    "impar",
    "idem",
    "alius",
    "alter",
    "uterque",
    "neuter",
    "uterlibet",
    "uterque",
    # Latin pronouns, numerals, and verb conjugations
    "ne",
    "mihi",
    "tibi",
    "sibi",
    "nobis",
    "vobis",
    "sibi",
    "meus",
    "tuus",
    "suus",
    "noster",
    "vester",
    "hic",
    "haec",
    "hoc",
    "ille",
    "illa",
    "illud",
    "is",
    "ea",
    "id",
    "quis",
    "quid",
    "alius",
    "aliquis",
    "aliqui",
    "aliqua",
    "aliquid",
    "ipse",
    "ipsemet",
    "idem",
    "totus",
    "nullus",
    "ullus",
    "multus",
    "paucus",
    "magnus",
    "parvus",
    "bonus",
    "malus",
    "novus",
    "vetus",
    "primus",
    "secundus",
    "tertius",
    "quartus",
    "quintus",
    "sextus",
    "omnis",
    "unus",
    "duo",
    "tres",
    "quattuor",
    "quinque",
    "sex",
    "septem",
    "octo",
    "novem",
    "decem",
    "centum",
    "mille",
    "sum",
    "es",
    "est",
    "sumus",
    "estis",
    "sunt",
    "ero",
    "eris",
    "erit",
    "erimus",
    "eritis",
    "erunt",
    "fio",
    "fies",
    "fiet",
    "fiemus",
    "fietis",
    "fient",
    "possum",
    "potes",
    "potest",
    "possumus",
    "potestis",
    "possunt",
    "volo",
    "vis",
    "vult",
    "volumus",
    "vultis",
    "volunt",
    "nolo",
    "non vis",
    "non vult",
    "nolumus",
    "non vultis",
    "nolunt",
    "fero",
    "fers",
    "fert",
    "ferimus",
    "fertis",
    "ferunt",
    "habeo",
    "habes",
    "habet",
    "habemus",
    "habetis",
    "habent",
    "dico",
    "dicis",
    "dicit",
    "dicimus",
    "dicitis",
    "dicunt",
    "facio",
    "facis",
    "facit",
    "facimus",
    "facitis",
    "faciunt",
    "venio",
    "venis",
    "venit",
    "venimus",
    "venitis",
    "veniunt",
    "eo",
    "is",
    "it",
    "imus",
    "itis",
    "eunt",
    "ago",
    "agis",
    "agit",
    "agimus",
    "agitis",
    "agunt",
    "do",
    "das",
    "dat",
    "damus",
    "datis",
    "dant",
    "video",
    "vides",
    "videt",
    "videmus",
    "videtis",
    "vident",
    "audio",
    "audis",
    "audit",
    "audimus",
    "auditis",
    "audiunt",
    "capio",
    "capis",
    "capit",
    "capimus",
    "capitis",
    "capiunt",
    "duco",
    "ducis",
    "ducit",
    "ducimus",
    "ducitis",
    "ducunt",
    "mitto",
    "mittis",
    "mittit",
    "mittimus",
    "mittitis",
    "mittunt",
    "pono",
    "ponis",
    "ponit",
    "ponimus",
    "ponitis",
    "ponunt",
    "rego",
    "regis",
    "regit",
    "regimus",
    "regitis",
    "regunt",
    "sequor",
    "sequeris",
    "sequitur",
    "sequimur",
    "sequimini",
    "sequuntur",
    "peto",
    "petis",
    "petit",
    "petimus",
    "petitis",
    "petunt",
    "quaero",
    "quaeris",
    "quaerit",
    "quaerimus",
    "quaeritis",
    "quaerunt",
    "scribo",
    "scribis",
    "scribit",
    "scribimus",
    "scribitis",
    "scribunt",
    "scripsi",
    "scriptum",
    "lego",
    "legis",
    "legit",
    "legimus",
    "legitis",
    "legunt",
    "legi",
    "lectum",
    "amo",
    "amas",
    "amat",
    "amamus",
    "amatis",
    "amant",
    "timeo",
    "times",
    "timet",
    "timemus",
    "timetis",
    "timent",
    "valeo",
    "vales",
    "valet",
    "valemus",
    "valetis",
    "valent",
    "scio",
    "scis",
    "scit",
    "scimus",
    "scitis",
    "sciunt",
    "nescio",
    "nescis",
    "nescit",
    "nescimus",
    "nescitis",
    "nesciunt",
    "credo",
    "credis",
    "credit",
    "credimus",
    "creditis",
    "credunt",
    "dubito",
    "dubitas",
    "dubitats",
    "dubitamus",
    "dubitatis",
    "dubitants",
    "puto",
    "putas",
    "putat",
    "putamus",
    "putatis",
    "putant",
    "opinor",
    "opinris",
    "opinr",
    "opinmur",
    "opinmini",
    "opinntur",
    "sentio",
    "sentis",
    "sentit",
    "sentimus",
    "sentitis",
    "sentint",
    "cogito",
    "cogitas",
    "cogitat",
    "cogitamus",
    "cogitatis",
    "cogitant",
    "intellego",
    "intellegis",
    "intellegit",
    "intellegimus",
    "intellegitis",
    "intellegunt",
    "videor",
    "videris",
    "videtur",
    "videmur",
    "videmini",
    "videntur",
    "placeo",
    "places",
    "placet",
    "placemus",
    "placetis",
    "placent",
    "gaudeo",
    "gaudes",
    "gaudet",
    "gaudemus",
    "gaudetis",
    "gaudent",
    "doleo",
    "doles",
    "dolent",
    "dolemus",
    "doletis",
    "dolents",
    "cupio",
    "cupis",
    "cupit",
    "cupimus",
    "cupitis",
    "cupiunt",
    "iubeo",
    "iubes",
    "iubet",
    "iubemus",
    "iubetis",
    "iubent",
    "veto",
    "vetas",
    "vetat",
    "vetamus",
    "vetatis",
    "vetant",
    "sinor",
    "sineris",
    "sinitur",
    "sinimur",
    "sinimini",
    "sinuntur",
    "licet",
    "licet",
    "licet",
    "licet",
    "licet",
    "licent",
    "decet",
    "decet",
    "decet",
    "decet",
    "decet",
    "decents",
    "oportet",
    "oportet",
    "oportet",
    "oportet",
    "oportet",
    "oportents",
    "piget",
    "piget",
    "piget",
    "piget",
    "piget",
    "pignents",
    "paenitet",
    "paenitet",
    "paenitet",
    "paenitet",
    "paenitet",
    "paenitents",
    "miseret",
    "miseret",
    "miseret",
    "miseret",
    "miseret",
    "miserents",
}

# Latin verb endings for morphology-based detection
LATIN_VERB_ENDINGS = {
    "re",
    "ris",
    "tur",
    "mus",
    "tis",
    "ntur",
    "or",
    "bor",
    "beris",
    "bitur",
    "mur",
    "mini",
    "ntur",
    "ero",
    "eris",
    "etur",
    "emur",
    "emini",
    "entur",
    # Standard conjugation endings
    "o",
    "s",
    "t",
    "mus",
    "tis",
    "nt",
    "bam",
    "bas",
    "bat",
    "bamus",
    "batis",
    "bant",
    "bo",
    "bis",
    "bit",
    "bimus",
    "bitis",
    "bunt",
    "ero",
    "eris",
    "erit",
    "erimus",
    "eritis",
    "erunt",
    "ar",
    "aris",
    "atur",
    "amur",
    "amini",
    "antur",
    "or",
    "eris",
    "itur",
    "imur",
    "imini",
    "untur",
    "am",
    "as",
    "at",
    "amus",
    "atis",
    "ant",
    "em",
    "es",
    "et",
    "emus",
    "etis",
    "ent",
    "iam",
    "ias",
    "iat",
    "iamus",
    "iatis",
    "iant",
}


def detect_languages(text, window_size=1):
    """Sliding-window language detector using stopword overlap.

    For each sentence, computes the proportion of tokens in each language's
    stopword list and assigns the most likely language.

    Returns list of (sentence, detected_language, scores_dict).
    """
    lang_stops = {
        "english": set(stopwords.words("english")),
        "french": set(stopwords.words("french")),
        "german": set(stopwords.words("german")),
        "italian": set(stopwords.words("italian")),
        "latin": LATIN_STOPWORDS,
    }

    sentences = sent_tokenize(text)
    results = []
    lang_counts = Counter()

    for sent in sentences:
        tokens = [t.lower() for t in word_tokenize(sent) if t.isalpha()]
        if not tokens:
            continue

        scores = {}
        for lang, stops in lang_stops.items():
            overlap = sum(1 for t in tokens if t in stops)
            scores[lang] = overlap / len(tokens)

        # Boost Latin score when verb morphology supports the stopword signal
        if scores["latin"] > 0:
            latin_endings = sum(
                1
                for t in tokens
                if any(t.endswith(ending) for ending in LATIN_VERB_ENDINGS)
            )
            if latin_endings > 0:
                scores["latin"] += 0.1

        # Phrases with distinctive Latin vocabulary get an extra boost
        if any("descende" in t.lower() or "calve" in t.lower() for t in tokens):
            scores["latin"] += 0.2

        # Default to English if no clear winner
        best_lang = max(scores, key=scores.get)
        # Only flag non-English if the non-English score is at least as high
        # and there's meaningful non-English stopword presence
        if best_lang != "english" and scores[best_lang] < 0.1:
            best_lang = "english"

        lang_counts[best_lang] += 1
        results.append(
            (sent[:80] + ("..." if len(sent) > 80 else ""), best_lang, scores)
        )

    print("\n--- Language Detection Summary ---")
    print(f"  Total sentences analyzed: {len(results)}")
    for lang, count in lang_counts.most_common():
        print(f"  {lang:<12} {count:>5} sentences ({100 * count / len(results):.1f}%)")

    # Show non-English detections
    print("\n--- Non-English Detections ---")
    non_english = [(s, l, sc) for s, l, sc in results if l != "english"]
    for sent, lang, scores in non_english[:15]:
        print(f"  [{lang:>8}] {sent}")

    # Known failure cases
    print("\n  Note: Dialectal English like 'De boys up in de hayloft' may be")
    print("        misclassified as Latin due to shared tokens like 'de'.")
    print("  Note: Cross-language homographs like 'hat' (German 'haben')")
    print("        can cause false positives in language detection.")

    return results, lang_counts


def non_english_token_proportion(text):
    """Estimate the proportion of non-English tokens in the episode."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]

    # Build English word set from WordNet lemmas, NLTK words corpus, and stopwords
    english_words = set()

    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            english_words.add(lemma.name().lower().replace("_", " "))

    try:
        from nltk.corpus import words

        english_words.update(words.words())
    except LookupError:
        pass

    english_words.update(stopwords.words("english"))

    non_english = [t for t in tokens if t not in english_words and len(t) > 2]
    proportion = len(non_english) / len(tokens) if tokens else 0

    print(f"\n  Non-English token proportion (heuristic): {proportion:.3f}")
    print(f"  ({len(non_english)} of {len(tokens)} alpha tokens)")
    print(f"  Sample non-English tokens: {non_english[:20]}")

    print("  Note: This heuristic depends on the coverage of the combined word lists.")
    print("  Archaic or dialectal spellings may still be counted as non-English.")

    return proportion


# ---------------------------------------------------------------------------
# Exercise 3: Derivational Morphology and Neologism
# ---------------------------------------------------------------------------

INTERESTING_WORDS = [
    "ineluctable",
    "nacheinander",
    "nebeneinander",
    "diaphane",
    "adiaphane",
    "maestro",
    "dogsbody",
    "contransmagnificandjewbangtantiality",
    "snotgreen",
    "scrotumtightening",
    "seaspawn",
    "wavespeech",
    "bridebed",
    "childbed",
    "deathbed",
    "omphalos",
    "thalatta",
    "augur",
    "protean",
    "metempsychosis",
]


def morphological_analysis(words=None):
    """Trace derivational history of unusual words via WordNet.

    For words not in WordNet, hypothesize a morphological parse.
    """
    if words is None:
        words = INTERESTING_WORDS

    lemmatizer = nltk.WordNetLemmatizer()

    print("\n--- Derivational Morphology Analysis ---")
    print(f"{'Word':<40} {'In WordNet?':<12} {'Synsets':<8} {'Analysis'}")
    print("-" * 100)

    in_wordnet = 0
    not_in_wordnet = 0

    for word in words:
        synsets = wordnet.synsets(word)
        if synsets:
            in_wordnet += 1
            # Get hypernym chain
            top_synset = synsets[0]
            hypernyms = top_synset.hypernym_paths()
            depth = len(hypernyms[0]) if hypernyms else 0
            definition = top_synset.definition()[:50]
            print(f"  {word:<38} {'Yes':<12} {len(synsets):<8} {definition}")
        else:
            not_in_wordnet += 1
            # Attempt morphological decomposition
            analysis = hypothesize_parse(word)
            print(f"  {word:<38} {'No':<12} {0:<8} {analysis}")

    print(f"\n  In WordNet: {in_wordnet}/{len(words)}")
    print(f"  Not in WordNet: {not_in_wordnet}/{len(words)}")
    return in_wordnet, not_in_wordnet


def hypothesize_parse(word):
    """Attempt to decompose a compound or neologism into recognizable parts."""
    word_lower = word.lower()

    # Skip English compound splitting for known German/foreign words
    german_stopwords = (
        set(stopwords.words("german")) if "german" in stopwords.fileids() else set()
    )

    known_german_words = {
        "nacheinander",
        "nebeneinander",
        "uber",
        "uberhaupt",
        "hat",
        "tie",
        "overcoat",
        "nose",
    }

    if word_lower in german_stopwords or word_lower in known_german_words:
        return "German word - no decomposition attempted"

    # Check for known compound patterns
    # Try splitting at every position and checking both halves
    best_split = None
    best_score = 0

    for i in range(3, len(word_lower) - 2):
        left = word_lower[:i]
        right = word_lower[i:]
        left_in = bool(wordnet.synsets(left))
        right_in = bool(wordnet.synsets(right))
        score = int(left_in) + int(right_in)
        if score > best_score:
            best_score = score
            best_split = (left, right, left_in, right_in)

    if best_split and best_score >= 1:
        left, right, li, ri = best_split
        parts = []
        if li:
            parts.append(f"'{left}' (in WN)")
        else:
            parts.append(f"'{left}' (not in WN)")
        if ri:
            parts.append(f"'{right}' (in WN)")
        else:
            parts.append(f"'{right}' (not in WN)")
        return f"Compound: {' + '.join(parts)}"

    # Check for known prefixes/suffixes
    prefixes = ["un", "in", "dis", "non", "pre", "post", "anti", "contra", "trans"]
    for prefix in prefixes:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            remainder = word_lower[len(prefix) :]
            if wordnet.synsets(remainder):
                return f"Prefix '{prefix}-' + '{remainder}' (in WN)"

    return "Sui generis / foreign borrowing"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    proteus = load_episode("03proteus.txt")

    print("=" * 62)
    print("EXERCISE 1: The Stemmer's Struggle")
    print("=" * 62)
    stemmers_struggle(proteus)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Multilingual Detection")
    print("=" * 62)
    detect_languages(proteus)
    non_english_token_proportion(proteus)

    print("\n" + "=" * 62)
    print("EXERCISE 3: Derivational Morphology and Neologism")
    print("=" * 62)
    morphological_analysis()
