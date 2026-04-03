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

- **[Week 1 — Telemachus](/Week_1):** Tokenization, concordance, and frequency analysis
- **[Week 2 — Nestor](/Week_2):** POS tagging, voice analysis (dialogue vs. narration), and lemmatization
- **[Week 3 — Proteus](/Week_3):** Stemming, multilingual detection, and morphological analysis
- **[Week 4 — Calypso](/Week_4):** Named entity recognition, noun phrase chunking, and entity co-occurrence
- **[Week 5 — Lotus Eaters](/Week_5):** WordNet semantic similarity, malapropisms, and substitution chains
- **[Week 6 — Hades](/Week_6):** Sentiment analysis, narrative voice registers, and affective lexicons
- **[Week 7 — Aeolus](/Week_7):** TF-IDF, keyword extraction, rhetoric detection, and headline generation
""")
