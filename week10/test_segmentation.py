import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "txt")


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_wandering_rocks(text):
    """Split Wandering Rocks into its 19 sections."""
    lines = text.split("\n")
    total_lines = len(lines)
    lines_per_section = max(1, total_lines // 19)

    sections = []
    for i in range(19):
        start_line = i * lines_per_section
        end_line = (i + 1) * lines_per_section if i < 18 else total_lines
        section = "\n".join(lines[start_line:end_line]).strip()
        if section:  # Only add non-empty sections
            sections.append(section)

    # Ensure we have exactly 19 sections
    while len(sections) < 19:
        sections.append("")

    return sections[:19]  # Ensure exactly 19 sections


if __name__ == "__main__":
    wr = load_episode("10wanderingrocks.txt")
    sections = split_wandering_rocks(wr)
    print(f"Parsed {len(sections)} sections from Wandering Rocks")
    for i, section in enumerate(sections[:3]):  # Show first 3 sections
        print(f"\n--- Section {i + 1} ({len(section)} chars) ---")
        print(section[:200] + ("..." if len(section) > 200 else ""))
