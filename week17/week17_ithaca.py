"""
Week 17: Ithaca
================
Information extraction, relation extraction, and knowledge graph construction.

NLTK Focus: nltk.chunk, nltk.sem, relation patterns, structured output parsing;
            networkx for knowledge representation

Exercises:
  1. Parse the catechism
  2. Triple extraction
  3. The question Ithaca doesn't ask
"""

import os
import re
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Q&A Parsing
# ---------------------------------------------------------------------------

def parse_catechism(text):
    """Parse Ithaca's Q&A format into question-answer pairs.

    The format alternates between questions (interrogative sentences)
    and answers (declarative, often long).
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    qa_pairs = []
    current_question = None
    current_answer_parts = []

    for para in paragraphs:
        # Detect question: ends with '?' or starts with interrogative word
        is_question = (para.strip().endswith('?') or
                       re.match(r'^(What|Why|How|Did|Was|Were|Had|Where|When|Which|Who)\b',
                                para.strip()))

        if is_question:
            # Save previous Q&A pair
            if current_question is not None:
                answer = ' '.join(current_answer_parts)
                qa_pairs.append((current_question, answer))
            current_question = para.strip()
            current_answer_parts = []
        else:
            current_answer_parts.append(para)

    # Last pair
    if current_question is not None and current_answer_parts:
        qa_pairs.append((current_question, ' '.join(current_answer_parts)))

    return qa_pairs


def classify_question(question):
    """Classify a question by its interrogative type."""
    q = question.lower().strip()
    if q.startswith('what'):
        if 'what were' in q or 'what did' in q:
            return 'what (list/inventory)'
        return 'what (entity)'
    elif q.startswith('why'):
        return 'why (causal)'
    elif q.startswith('how'):
        return 'how (procedural)'
    elif q.startswith(('did', 'was', 'were', 'had')):
        return 'yes/no'
    elif q.startswith('where'):
        return 'where (locative)'
    elif q.startswith('when'):
        return 'when (temporal)'
    elif q.startswith('which'):
        return 'which (selection)'
    elif q.startswith('who'):
        return 'who (person)'
    return 'other'


# ---------------------------------------------------------------------------
# Exercise 1: Parse the Catechism
# ---------------------------------------------------------------------------

def exercise_parse_catechism():
    """Parse and classify all Q&A pairs in Ithaca."""
    ithaca = load_episode('17ithaca.txt')
    qa_pairs = parse_catechism(ithaca)

    print(f"--- Catechism Parsed ---")
    print(f"  Total Q&A pairs: {len(qa_pairs)}")

    # Classify questions
    q_types = Counter()
    for question, answer in qa_pairs:
        qtype = classify_question(question)
        q_types[qtype] += 1

    print(f"\n--- Question Type Distribution ---")
    for qtype, count in q_types.most_common():
        pct = 100 * count / len(qa_pairs)
        print(f"  {qtype:<25} {count:>5} ({pct:>5.1f}%)")

    # Answer length statistics
    answer_lengths = [len(word_tokenize(a)) for _, a in qa_pairs]
    print(f"\n--- Answer Length Statistics ---")
    print(f"  Mean answer length:   {sum(answer_lengths)/len(answer_lengths):.1f} words")
    print(f"  Median answer length: {sorted(answer_lengths)[len(answer_lengths)//2]} words")
    print(f"  Max answer length:    {max(answer_lengths)} words")
    print(f"  Min answer length:    {min(answer_lengths)} words")

    # Show sample Q&A pairs
    print(f"\n--- Sample Q&A Pairs ---")
    for q, a in qa_pairs[:8]:
        print(f"\n  Q: {q[:100]}")
        print(f"  A: {a[:120]}...")

    # Plot question type distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    types = [t for t, _ in q_types.most_common()]
    counts = [c for _, c in q_types.most_common()]
    ax.barh(types, counts, color='steelblue')
    ax.set_xlabel('Count')
    ax.set_title('Ithaca: Question Type Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'week17_questions.png'), dpi=150)
    plt.close()

    return qa_pairs, q_types


# ---------------------------------------------------------------------------
# Exercise 2: Triple Extraction
# ---------------------------------------------------------------------------

def extract_triples(qa_pairs, max_pairs=None):
    """Extract knowledge triples (subject, predicate, object) from Q&A pairs.

    Uses simple heuristic patterns:
    - "X contained Y" → (X, contains, Y)
    - "X was Y" → (X, is, Y)
    - "X of Y" → (Y, has, X)
    """
    triples = []

    pairs_to_process = qa_pairs[:max_pairs] if max_pairs else qa_pairs

    for question, answer in pairs_to_process:
        answer_sents = sent_tokenize(answer)
        for sent in answer_sents:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)

            # Pattern: NP VBD/VBZ NP
            # Find subject-verb-object patterns
            for i in range(len(tagged)):
                word, tag = tagged[i]

                # Pattern: "X contained Y" / "X was Y"
                if tag in ('VBD', 'VBZ', 'VBN') and i > 0 and i < len(tagged) - 1:
                    # Simple: take preceding noun as subject, following as object
                    subject = None
                    obj = None

                    # Look back for subject noun
                    for j in range(i-1, max(i-5, -1), -1):
                        if tagged[j][1].startswith('NN'):
                            subject = tagged[j][0]
                            break

                    # Look forward for object noun
                    for j in range(i+1, min(i+5, len(tagged))):
                        if tagged[j][1].startswith('NN'):
                            obj = tagged[j][0]
                            break

                    if subject and obj and subject != obj:
                        predicate = word.lower()
                        triples.append((subject.lower(), predicate, obj.lower()))

            # Pattern: list items (detect "X, Y, Z" lists as contained-in relations)
            if ',' in sent and len(tokens) > 5:
                # Check if this looks like an inventory
                items = [t.strip() for t in sent.split(',') if t.strip()]
                if len(items) >= 3:
                    # Try to extract the container from the question
                    q_tokens = word_tokenize(question.lower())
                    container = None
                    for qt in reversed(q_tokens):
                        if qt.isalpha() and len(qt) > 3 and qt not in ('what', 'were', 'the', 'did'):
                            container = qt
                            break
                    if container:
                        for item in items[:5]:  # Limit
                            item_words = [w for w in word_tokenize(item)
                                          if w.isalpha() and len(w) > 2]
                            if item_words:
                                triples.append((container, 'contains', item_words[0].lower()))

    # Deduplicate
    unique_triples = list(set(triples))
    return unique_triples


def exercise_triple_extraction():
    """Extract and display knowledge triples."""
    ithaca = load_episode('17ithaca.txt')
    qa_pairs = parse_catechism(ithaca)
    triples = extract_triples(qa_pairs)

    print(f"--- Knowledge Triples ---")
    print(f"  Total triples extracted: {len(triples)}")

    # Show sample triples
    print(f"\n--- Sample Triples ---")
    for s, p, o in triples[:30]:
        print(f"  ({s}, {p}, {o})")

    # Most common subjects and predicates
    subjects = Counter(s for s, _, _ in triples)
    predicates = Counter(p for _, p, _ in triples)
    objects = Counter(o for _, _, o in triples)

    print(f"\n--- Most Common Subjects ---")
    for s, c in subjects.most_common(10):
        print(f"  {s:<25} {c:>5}")

    print(f"\n--- Most Common Predicates ---")
    for p, c in predicates.most_common(10):
        print(f"  {p:<25} {c:>5}")

    print(f"\n--- Most Common Objects ---")
    for o, c in objects.most_common(10):
        print(f"  {o:<25} {c:>5}")

    return triples


# ---------------------------------------------------------------------------
# Exercise 3: The Question Ithaca Doesn't Ask
# ---------------------------------------------------------------------------

def topic_distribution(qa_pairs):
    """Analyze the topic distribution of Ithaca's questions."""
    topic_keywords = {
        'physical_objects': {'drawer', 'table', 'chair', 'cup', 'key', 'door',
                            'shelf', 'bed', 'lamp', 'range', 'kitchen', 'book',
                            'furniture', 'contents', 'articles'},
        'human_relations': {'bloom', 'stephen', 'molly', 'father', 'mother',
                           'son', 'wife', 'husband', 'friend', 'companion'},
        'abstract_concepts': {'reason', 'cause', 'purpose', 'meaning', 'thought',
                             'memory', 'belief', 'feeling', 'emotion', 'desire'},
        'science_math': {'water', 'temperature', 'light', 'weight', 'distance',
                        'calculation', 'measurement', 'star', 'astronomical',
                        'mathematical', 'scientific'},
        'economics': {'money', 'cost', 'budget', 'expenditure', 'income',
                     'savings', 'financial', 'price', 'pound'},
        'geography': {'street', 'road', 'route', 'dublin', 'eccles', 'city',
                     'house', 'garden', 'direction', 'church'},
    }

    topic_counts = Counter()
    question_topics = []

    for question, answer in qa_pairs:
        q_lower = question.lower()
        a_lower = answer.lower()[:200]  # First 200 chars of answer
        combined = q_lower + ' ' + a_lower

        tokens = set(word_tokenize(combined))
        matched_topic = 'other'
        max_overlap = 0

        for topic, keywords in topic_keywords.items():
            overlap = len(tokens & keywords)
            if overlap > max_overlap:
                max_overlap = overlap
                matched_topic = topic

        topic_counts[matched_topic] += 1
        question_topics.append((question, matched_topic))

    print(f"\n--- Question Topic Distribution ---")
    total = len(qa_pairs)
    for topic, count in topic_counts.most_common():
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {topic:<20} {count:>5} ({pct:>5.1f}%) {bar}")

    # Compare to Calypso
    calypso = load_episode('04calypso.txt')
    cal_sents = sent_tokenize(calypso)
    cal_topics = Counter()
    for sent in cal_sents:
        tokens = set(word_tokenize(sent.lower()))
        matched = 'other'
        max_o = 0
        for topic, keywords in topic_keywords.items():
            overlap = len(tokens & keywords)
            if overlap > max_o:
                max_o = overlap
                matched = topic
        cal_topics[matched] += 1

    cal_total = len(cal_sents)
    print(f"\n--- Comparison: Ithaca vs. Calypso Topic Distribution ---")
    print(f"  {'Topic':<20} {'Ithaca %':>10} {'Calypso %':>12}")
    print("  " + "-" * 44)
    all_topics = sorted(set(list(topic_counts.keys()) + list(cal_topics.keys())))
    for topic in all_topics:
        ip = 100 * topic_counts.get(topic, 0) / total
        cp = 100 * cal_topics.get(topic, 0) / cal_total
        print(f"  {topic:<20} {ip:>9.1f}% {cp:>11.1f}%")

    return topic_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 62)
    print("EXERCISE 1: Parse the Catechism")
    print("=" * 62)
    qa_pairs, q_types = exercise_parse_catechism()

    print("\n" + "=" * 62)
    print("EXERCISE 2: Triple Extraction")
    print("=" * 62)
    exercise_triple_extraction()

    print("\n" + "=" * 62)
    print("EXERCISE 3: The Question Ithaca Doesn't Ask")
    print("=" * 62)
    topic_distribution(qa_pairs)
