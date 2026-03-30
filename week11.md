# Week 11: Sirens
### *"Bronze by gold heard the hoofirons, steelyringing" — Language as music, prose as fugue, and the sound of meaning dissolving.*

Sirens is Joyce's fugue. It opens with an "overture" — 63 fragments of text, torn from their contexts, presented as pure sound: "Imperthnthn thnthnthn." "Chips, picking chips off rocky thumbnail, chips." "A sail! A veil awave upon the waves." These fragments will recur throughout the episode, developed, varied, and recombined as Bloom sits in the Ormond Hotel bar listening to Simon Dedalus sing and thinking about Molly and Blazes Boylan. The episode's technique is *fuga per canonem*; its art is *music*; its organ is the *ear*. Language subordinates meaning to sound. Words are stretched, compressed, onomatopoetically distorted. Sentences scan like musical phrases. Joyce said he wrote the episode using the formal structure of a fugue — subject, answer, countersubject, episode, stretto — and whether or not the musical analogy is exact (scholars disagree), the acoustic texture is unmistakable. This is prose that asks to be heard, not just read.

**NLTK Focus:** Phonetic analysis, sound patterning, and sequence repetition detection (`nltk.corpus.cmudict`, phonetic feature extraction, alliteration/assonance detection, string matching for motif recurrence)

**Pairing Rationale:**
If language is made of sounds before it is made of meanings, then the CMU Pronouncing Dictionary is the score beneath the text. Sirens forces the question that phoneticians and computational linguists both grapple with: what is the relationship between the sound of a word and its function? Joyce's fugal technique treats words as musical motifs — recurring sound patterns that are introduced, varied, and recombined according to quasi-musical logic. Detecting these patterns computationally requires exactly the tools of phonetic analysis: converting orthographic text to phonemic representation, measuring phonetic similarity, identifying repeated subsequences. The episode also foregrounds the phenomena that poetry has always known about — alliteration, assonance, consonance, internal rhyme — and asks whether prose can be *organized* by these principles rather than merely decorated by them. NLTK's phonetic resources let us test this empirically: is the sound patterning in Sirens statistically denser than in other episodes, or does it just *feel* that way?

**Core Exercises:**

1. **The overture decoded.** The episode's opening 63 fragments are torn from passages that appear later in the chapter. Treat the overture as a set of queries and the body of the episode as the search corpus. For each overture fragment, find its "source" passage in the body using string matching (start with exact substring matching, then move to fuzzy matching with `nltk.metrics.distance.edit_distance` for fragments that Joyce altered). What percentage of fragments can you match automatically? For unmatched fragments, examine whether the alteration is phonetic (sound preserved, spelling changed), semantic (meaning preserved, words changed), or purely musical (sound pattern preserved, content abandoned).

2. **Phonetic density analysis.** Using `cmudict`, convert the text of Sirens to a phonemic representation (skip words not in the dictionary — but note which words are missing; they're often Joyce's onomatopoeia, which is precisely the most phonetically interesting material). Compute phonetic patterning density for each paragraph: count instances of alliteration (repeated onset consonants in stressed syllables within a window), assonance (repeated vowel nuclei), and consonance (repeated coda consonants). Compare these densities to the same measures computed on Lestrygonians and Calypso. Is Sirens measurably more sound-patterned? Visualize the density across the episode — do the peaks correspond to the passages critics have identified as the fugal "stretto" (where themes pile up)?

3. **Motif tracking.** Identify 5–8 recurring verbal motifs from the overture (e.g., "bronze by gold," "jingle jaunty," "tap tap tap," "Blmstup"). Track every occurrence and variation of each motif through the episode. For each motif, record its exact form at each appearance and compute the edit distance from the "canonical" overture form. Plot the trajectory: do motifs become more or less distorted as the episode progresses? Do any motifs converge (appearing closer and closer together), mimicking the musical stretto? Build a "motif score" — a timeline of the episode showing when each motif sounds, like a simplified musical score.

**Diving Deeper:**

- The question of whether Sirens is "really" a fugue has generated a century of critical debate. Zack Bowen's *Bloom's Old Sweet Song* and Sebastian Walsh's more recent work attempt to map the episode onto strict fugal form. Your motif-tracking data gives you empirical evidence: does the recurrence pattern of verbal motifs match the formal structure of a fugue (exposition → development → recapitulation), or is the analogy impressionistic? This is a genuine open question.
- Phonaesthesia — the hypothesis that certain sounds carry inherent meaning (e.g., that /gl/ words tend to relate to light: *glow*, *gleam*, *glitter*, *glisten*) — is a contested but persistent idea in linguistics. See Bergen (2004) and Blasi et al. (2016). Sirens is a test case: does Joyce exploit phonaesthetic clusters? Are his sound patterns arbitrary or semantically motivated?
- The `pronouncing` Python library (built on CMU) provides a cleaner API for rhyme detection, syllable counting, and stress pattern extraction. Use it to compute the metrical pattern (stressed/unstressed syllables) of Sirens' prose. Are there passages that scan as regular verse? Joyce was a trained tenor — how much of his musical knowledge is encoded in the rhythmic structure?
- Audio analysis tools (Praat, `librosa`) can analyze actual recordings of Sirens read aloud. If you can find or produce a recording, compare the acoustic features (pitch contour, rhythm, spectral properties) to your text-based phonetic analysis. Where does the written text's sound patterning translate to audible musicality, and where is it a purely visual/orthographic phenomenon?
- Connection to Week 8 (Lestrygonians): that episode's sequential, associative logic was modeled with n-gram language models. Sirens' fugal repetition structure is *not* Markov — motifs recur across hundreds of words, violating the memorylessness assumption. This is a concrete example of long-range dependency in literary structure, and motivates the move from n-gram models to more powerful sequence models.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Convert text to phonemic representation** using the CMU Pronouncing Dictionary and extract onset consonants, vowel nuclei, and codas.
2. **Detect alliteration, assonance, and consonance** computationally and measure their density relative to other episodes.
3. **Match textual fragments** across a document using exact and fuzzy string matching (edit distance).
4. **Track recurring motifs** through an episode and visualize their distribution as a timeline/score.

## Metrics & Assessment Targets

| Metric | What to Compute | Expected Range (Sirens) |
|---|---|---|
| Overture fragment count | short lines at episode start | ~50–70 |
| Overture match rate | % of fragments matched to body passages | ~40–70% |
| Alliteration density | adjacent-pair onset matches / total pairs | higher than Calypso/Lestrygonians |
| Assonance density | 3-word window shared vowel nuclei / windows | measurably elevated |
| CMU dict coverage | % of words found in cmudict | ~75–85% (onomatopoeia missing) |
| Motifs tracked | recurring verbal patterns from overture | 5–8 motifs |
| Motif convergence | do motifs cluster later in episode (stretto)? | expect some clustering |

## Rubric

### Exercise 1: The Overture Decoded (30 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Fragment extraction** | Overture correctly parsed; ~63 fragments identified | Most fragments found | Parsing broken |
| **Matching** | Exact + fuzzy matching with edit distance; match rate reported; unmatched fragments categorized (phonetic alteration, semantic shift, purely musical) | Some matching done | Only exact matching or broken |
| **Analysis** | Discusses what kinds of alterations Joyce makes (sound-preserving, meaning-preserving, pure-sound) | Brief categorization | No analysis |

### Exercise 2: Phonetic Density (35 points)

| Criterion | Excellent (12) | Satisfactory (8) | Needs Work (4) |
|---|---|---|---|
| **Phonemic conversion** | CMU dict used correctly; missing words noted (and noted as the most interesting—onomatopoeia) | CMU dict used; some conversion | Conversion broken |
| **Density computation** | Alliteration, assonance, consonance all computed per paragraph window; cross-episode comparison | 2 of 3 measures computed | Only 1 or broken |
| **Visualization** | Density plotted across episode; peaks correlated to fugal structure (stretto) | Plot produced | No visualization |

### Exercise 3: Motif Tracking (25 points)

| Criterion | Excellent (10) | Satisfactory (7) | Needs Work (4) |
|---|---|---|---|
| **Motif identification** | 5–8 motifs from overture tracked; every occurrence and variation recorded | 3+ motifs tracked | Fewer than 3 |
| **Edit distance trajectory** | Distance from canonical form plotted per motif; distortion trends discussed | Some distance data | No distance tracking |
| **Motif score visualization** | Timeline showing when each motif sounds, like a musical score | Basic timeline | No visualization |

### Diving Deeper (10 points, bonus)

| Criterion | Points |
|---|---|
| Fugal structure analysis (exposition/development/recapitulation mapping) | +3 |
| Phonaesthetic cluster detection (gl- words, etc.) | +2 |
| Metrical pattern extraction using stress marks | +3 |
| Audio recording analysis comparison | +2 |

## Reference Implementation

See [`solutions/week11_sirens.py`](solutions/week11_sirens.py)
