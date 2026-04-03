"""Shared helpers for the Streamlit dashboard."""

import os
import sys
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from week01.week01_telemachus import load_episode as _load_episode, tokenize_and_profile as _tokenize_and_profile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")

EPISODE_MAP = {
    "01telemachus.txt": "01 — Telemachus",
    "02nestor.txt": "02 — Nestor",
    "03proteus.txt": "03 — Proteus",
    "04calypso.txt": "04 — Calypso",
    "05lotuseaters.txt": "05 — Lotus Eaters",
    "06hades.txt": "06 — Hades",
    "07aeolus.txt": "07 — Aeolus",
    "08lestrygonians.txt": "08 — Lestrygonians",
    "09scyllacharybdis.txt": "09 — Scylla & Charybdis",
    "10wanderingrocks.txt": "10 — Wandering Rocks",
    "11sirens.txt": "11 — Sirens",
    "12cyclops.txt": "12 — Cyclops",
    "13nausicaa.txt": "13 — Nausicaa",
    "14oxenofthesun.txt": "14 — Oxen of the Sun",
    "15circe.txt": "15 — Circe",
    "16eumaeus.txt": "16 — Eumaeus",
    "17ithaca.txt": "17 — Ithaca",
    "18penelope.txt": "18 — Penelope",
}

EPISODE_FILES = list(EPISODE_MAP.keys())
EPISODE_LABELS = list(EPISODE_MAP.values())


@st.cache_data
def cached_load_episode(filename):
    """Load episode text with caching."""
    return _load_episode(filename)


@st.cache_data
def cached_profile(text, label="Episode"):
    """Compute profile with caching."""
    return _tokenize_and_profile(text, label=label)


def episode_sidebar(default_index=0, caption="", description=""):
    """Render the shared sidebar and return (episode_filename, episode_label)."""
    with st.sidebar:
        st.header("Episode Selection")
        if caption:
            st.caption(caption)

        episode_label = st.selectbox(
            "Episode",
            EPISODE_LABELS,
            index=default_index,
        )
        episode_file = EPISODE_FILES[EPISODE_LABELS.index(episode_label)]

        if description:
            st.divider()
            st.markdown(description)

    return episode_file, episode_label
