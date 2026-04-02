# Week 1 Streamlit Dashboard Plan

## Context

The `rejoyce` project has 18 weeks of NLTK exercises analyzing Ulysses, but each week is a static Python script producing print output and saved PNGs. We want an interactive Streamlit dashboard for Week 1 (Telemachus — tokenization, concordance, frequency) that lets students explore beyond the fixed parameters of the exercises, building intuition for what these NLP tools reveal about literary text.

**Data available:** 18 chapter text files in `txt/` (e.g., `01telemachus.txt` ... `18penelope.txt`) plus `ulysses.txt`. Jane Austen's *Emma* via `nltk.corpus.gutenberg`.

**Existing code to reuse:** `week01/week01_telemachus.py` contains `load_episode()`, `tokenize_and_profile()`, `concordance_analysis()`, `frequency_analysis()`, `zipf_plot()`.

---

## File Structure

```
dashboard/
├── app.py              # Entry point: streamlit run dashboard/app.py
├── pages/
│   └── week01.py       # Week 1 page (future weeks get their own files here)
└── shared.py           # Shared helpers: episode loading, episode map, caching wrappers
```

Run with: `streamlit run dashboard/app.py`

**New dependency:** `streamlit` (add to `requirements.txt`). Use Streamlit's built-in charting (`st.bar_chart`) plus `matplotlib` for Zipf plot (already a dependency). Also add `scipy` (already used in `zipf_plot` but not in requirements.txt).

**Importing from week directories:** Each `weekNN/` folder already has self-contained functions (e.g., `week01_telemachus.py` exports `load_episode`, `tokenize_and_profile`, etc.). The dashboard imports these directly:

```python
# In dashboard/pages/week01.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from week01.week01_telemachus import load_episode, tokenize_and_profile
```

Alternatively, add an `__init__.py` to each `weekNN/` folder to make them proper packages. The `load_episode()` function uses `__file__`-relative paths, which will resolve correctly since the module lives in `week01/`.

`dashboard/shared.py` holds things used across all week pages: the episode name map, the `load_episode` wrapper with `@st.cache_data`, and any common sidebar logic.

---

## Page Layout: Three Sections

The page has a shared **episode selector sidebar** (dropdown of all 18 chapters, default: Telemachus) that feeds into all three exercise sections. Each section is wrapped in an `st.header` / `st.subheader`.

### Sidebar (shared controls)

- **Episode dropdown:** All 18 chapters, labeled like "01 — Telemachus", "02 — Nestor", etc. Default: 01.
- **Compare to:** Second dropdown (same 18 chapters + "Emma (Austen)" from Gutenberg). Default: Emma. Used by Exercise 1 and optionally Exercise 3.
- Brief intro text explaining Week 1's focus.

---

### Section 1: Corpus Profile & Comparison

**What it does:** Runs `tokenize_and_profile()` on the selected episode and the comparison text, displays results side-by-side.

**Widgets & visualizations:**

1. **Metrics row** — `st.columns(4)` showing the primary episode's key stats as `st.metric` cards with delta values vs. the comparison text:
   - Total tokens
   - Type-token ratio  
   - Avg sentence length
   - Hapax ratio

2. **Full comparison table** — `st.dataframe` with all 8 metrics for both texts, side by side. Columns: Metric | Selected Episode | Comparison Text | Difference.

3. **Radar chart** — Normalized (0-1 scaled) overlay of both texts across 5 axes: TTR, hapax ratio, avg sentence length (inverted so shorter = more Joycean), total types, total sentences. Uses `matplotlib` polar plot. This gives an instant visual "fingerprint" of each text's profile.

4. **"Profile all 18 episodes" expander** — An `st.expander` containing a precomputed table of all 18 episodes' profiles (with a "Compute all" button since it's slow). Highlights the selected episode row. Lets students see how lexical richness evolves across the novel.

**Interactive expansion beyond the original exercise:**
- The original only compares Telemachus vs. Emma. Now students can compare *any* chapter to *any* other chapter (or to Emma), immediately seeing how Joyce's style shifts across episodes.

---

### Section 2: Concordance Explorer

**What it does:** Interactive KWIC (keyword-in-context) display with thematic exploration tools.

**Widgets & visualizations:**

1. **Keyword input** — `st.text_input` with default value "mother" and a row of quick-select `st.button`s for the 5 canonical keywords: mother, sea, key, tower, God. Clicking a button populates the input.

2. **Context width slider** — `st.slider` from 40 to 120 characters (default 80), controlling how much surrounding text is shown.

3. **Concordance display** — Rendered as a styled table/dataframe where:
   - The keyword is **bolded** in each line
   - Left context is right-aligned, right context is left-aligned (classic KWIC format)
   - Shows occurrence count as `st.metric` above the table

4. **Keyword co-occurrence heatmap** — Below the concordance, an `st.pyplot` heatmap showing how often pairs of the 5 canonical keywords appear within the same concordance window (e.g., within 50 words of each other). This directly supports the exercise's goal of arguing for thematic connections between words. Color intensity = co-occurrence strength.

5. **"Dispersion plot" visualization** — A horizontal dot-strip chart showing *where* in the episode each occurrence of the keyword falls (by token position, normalized 0-100%). Students can overlay multiple keywords to see clustering. For example, "mother" and "sea" cluster together in certain passages. Uses `matplotlib` with colored markers per keyword (toggled via `st.multiselect`).

**Interactive expansion beyond the original exercise:**
- Original exercise fixes 5 keywords and one chapter. Now students can search *any* word in *any* chapter, adjust context width, and visually see co-occurrence and dispersion patterns that would take manual reading to discover.

---

### Section 3: Frequency & Zipf's Law

**What it does:** Interactive frequency distributions and Zipf's law visualization.

**Widgets & visualizations:**

1. **Controls row** — `st.columns` with:
   - `st.toggle` for stopword removal (default: off, so students see the "before" first)
   - `st.slider` for top-N words to display (10-100, default 30)
   - `st.text_input` for custom stopwords to add/remove (comma-separated)

2. **Frequency bar chart** — `st.bar_chart` (or `st.pyplot` for better labeling) showing top-N words. Color-coded: stopwords in gray, content words in coral. When toggle flips, the chart updates live — students see stopwords vanish and content words rise.

3. **Word frequency lookup** — `st.text_input` where students can type any word and instantly see its rank, raw count, and percentage of total tokens. Displayed as `st.metric` cards. Encourages curiosity: "how often does 'snotgreen' appear?"

4. **Zipf's law plot** — `st.pyplot` log-log scatter plot with:
   - Blue dots for actual rank-frequency data
   - Red dashed line for ideal Zipf (C/r)
   - R² value displayed prominently as `st.metric`
   - The word frequency lookup word highlighted as a labeled orange dot on the plot (so students can see where their word falls on the Zipf curve)
   - Optional: overlay a second episode's Zipf curve (from sidebar comparison selection) in a different color

5. **Comparison overlay toggle** — When enabled, overlays the comparison text's frequency bars (semi-transparent) behind the primary text's bars, making differences immediately visible.

**Interactive expansion beyond the original exercise:**
- Original generates two static bar charts and one Zipf plot. Now students can dynamically toggle stopwords, adjust the view, look up individual words, and see where words fall on the Zipf curve — connecting the abstract "law" to concrete vocabulary.

---

## Implementation Approach

### Code structure

**`dashboard/app.py`** — Entry point. Sets page config, renders a landing/nav page. Streamlit's multi-page app convention auto-discovers files in `dashboard/pages/`.

**`dashboard/shared.py`** — Shared across all week pages:
```
EPISODE_MAP: {"01telemachus.txt": "01 — Telemachus", ...}
DATA_DIR: path to txt/
cached_load_episode()    # @st.cache_data wrapper around week01's load_episode
cached_profile()         # @st.cache_data wrapper around tokenize_and_profile
episode_sidebar()        # common sidebar: episode picker + comparison picker
```

**`dashboard/pages/week01.py`** — Week 1 page:
```
imports from week01.week01_telemachus (tokenize_and_profile, concordance fns, etc.)
imports from dashboard.shared (cached loaders, sidebar)
section_1_profile()
section_2_concordance()
section_3_frequency()
```

### Key implementation details

- **Import from week directories:** `week01/week01_telemachus.py` already has well-factored functions. Import them directly into the dashboard page. `load_episode()` uses `os.path.dirname(__file__)` relative paths, so it resolves to `txt/` correctly as long as it's imported (not copied). Add `__init__.py` files to `week01/` (and future week dirs) to make imports clean.
- **`@st.cache_data`** wrappers in `shared.py` around the imported functions to avoid recomputation on every widget interaction.
- **Episode map:** Hardcode the 18 filenames with display names in `shared.py`.
- **Matplotlib figures** passed via `st.pyplot(fig)` — create figures without `plt.show()` or `plt.savefig()`.
- **Future weeks:** Each week adds a new `dashboard/pages/weekNN.py` that imports from its `weekNN/` directory. The pattern is established by week01.

### Files to create/modify

| File | Change |
|------|--------|
| `dashboard/app.py` | **New** — Streamlit entry point |
| `dashboard/shared.py` | **New** — shared episode map, caching, sidebar |
| `dashboard/pages/week01.py` | **New** — Week 1 page |
| `week01/__init__.py` | **New** — empty, makes week01 importable as a package |
| `requirements.txt` | Add `streamlit`, `scipy` |

---

## Verification

1. `pip install streamlit scipy` (if not already installed)
2. `cd /Users/creid/charlesreid1/rejoyce && streamlit run dashboard/app.py`
3. Navigate to the Week 1 page in the sidebar nav
4. Test each section:
   - Change episode dropdown → metrics update
   - Change comparison text → delta values and radar chart update
   - Type a custom keyword in concordance → KWIC table appears
   - Click canonical keyword buttons → concordance updates
   - Toggle stopwords → frequency chart redraws
   - Look up a word → see its position on Zipf curve
   - Adjust all sliders → responsive updates
