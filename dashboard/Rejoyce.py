"""
Rejoyce — Interactive NLTK Explorations of Ulysses
====================================================
Streamlit dashboard entry point. Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Rejoyce — NLTK × Ulysses",
    page_icon="📖",
    layout="wide",
)

st.title("Rejoyce — Interactive NLTK Explorations of Ulysses")

st.markdown("""
This dashboard accompanies the 18-week NLTK exercise series on James Joyce's *Ulysses*.
Each week's exercises become interactive: adjust parameters, compare chapters, and
explore patterns that static scripts can't reveal.

**Select a week from the sidebar** to begin.

---

### Available Weeks

- **Week 1 — Telemachus:** Tokenization, concordance, and frequency analysis
- **Week 2 — Nestor:** POS tagging, voice analysis (dialogue vs. narration), and lemmatization
- **Week 3 — Proteus:** Stemming, multilingual detection, and morphological analysis
- **Week 4 — Calypso:** Named entity recognition, noun phrase chunking, and entity co-occurrence
""")
