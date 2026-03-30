"""
Week 09: Scylla and Charybdis
===============================
Context-free grammars and syntactic parsing.

NLTK Focus: nltk.parse, RecursiveDescentParser, ChartParser, CFG.fromstring,
            Penn Treebank parsed corpus, dependency structure

Exercises:
  1. Parsing the argument
  2. Penn Treebank as reference grammar
  3. The quotation problem
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.parse.chart import ChartParser
from nltk.grammar import CFG, induce_pcfg
from nltk.corpus import treebank
from nltk.tree import Tree
import matplotlib.pyplot as plt

for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                 'averaged_perceptron_tagger_eng', 'treebank']:
    nltk.download(resource, quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'txt')


def load_episode(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Exercise 1: Parsing the Argument
# ---------------------------------------------------------------------------

# A hand-written CFG covering major English constituent types
ARGUMENT_GRAMMAR = CFG.fromstring("""
    S -> NP VP | S CC S | SBAR S | S SBAR
    SBAR -> IN S | WDT S | WP S
    NP -> DT NN | DT JJ NN | NNP | PRP | DT NN NN | NP PP | NP CC NP
    NP -> DT NNS | DT JJ NNS | NNP NNP | PRP$ NN | NNP NNP NNP
    VP -> VBD NP | VBD | VBZ NP | VBD VP | MD VP | VB NP | VB
    VP -> VBD ADJP | VBZ ADJP | VBD PP | VBZ PP | VP CC VP
    VP -> VBD NP PP | VB NP PP | VBN NP | VBN PP
    PP -> IN NP
    ADJP -> JJ | RB JJ | JJ CC JJ

    CC -> 'and' | 'but' | 'or' | 'yet' | 'nor'
    DT -> 'the' | 'a' | 'an' | 'his' | 'her' | 'this' | 'that' | 'no'
    IN -> 'in' | 'of' | 'to' | 'for' | 'with' | 'from' | 'by' | 'on' | 'at' | 'as' | 'through' | 'into' | 'upon'
    IN -> 'because' | 'if' | 'that' | 'when' | 'where' | 'after' | 'before'
    WDT -> 'which' | 'that' | 'who'
    WP -> 'who' | 'whom' | 'what'
    MD -> 'will' | 'would' | 'could' | 'should' | 'may' | 'might' | 'must' | 'can'
    PRP -> 'he' | 'she' | 'it' | 'they' | 'we' | 'I' | 'him' | 'them'
    PRP$ -> 'his' | 'her' | 'their' | 'its' | 'my' | 'our'

    NN -> 'man' | 'genius' | 'art' | 'life' | 'father' | 'son' | 'play'
    NN -> 'world' | 'work' | 'name' | 'soul' | 'body' | 'mind' | 'word'
    NN -> 'woman' | 'ghost' | 'king' | 'love' | 'death' | 'truth'
    NNS -> 'errors' | 'plays' | 'words' | 'portals' | 'mistakes'
    NNP -> 'Shakespeare' | 'Hamlet' | 'Stephen' | 'Aristotle' | 'Plato'
    NNP -> 'Ann' | 'Hathaway' | 'Joyce' | 'God' | 'Dublin'

    VBD -> 'was' | 'made' | 'had' | 'wrote' | 'said' | 'believed'
    VBD -> 'knew' | 'lived' | 'died' | 'loved' | 'found' | 'proved'
    VBZ -> 'is' | 'has' | 'makes' | 'proves' | 'says' | 'means'
    VB -> 'be' | 'make' | 'have' | 'prove' | 'say' | 'know' | 'write'
    VBN -> 'written' | 'made' | 'born' | 'proved' | 'called' | 'known'
    VBG -> 'making' | 'writing' | 'proving' | 'being'

    JJ -> 'great' | 'volitional' | 'little' | 'old' | 'own' | 'dead'
    JJ -> 'true' | 'perceptive' | 'spiritual' | 'new'
    RB -> 'not' | 'never' | 'therefore' | 'always' | 'only' | 'merely'
""")


def find_argument_sentences(text, connectives=None):
    """Find syntactically complex sentences with logical connectives."""
    if connectives is None:
        connectives = {'therefore', 'because', 'if', 'but', 'yet',
                       'however', 'thus', 'hence', 'since', 'although'}

    sentences = sent_tokenize(text)
    argument_sents = []

    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        if any(c in tokens for c in connectives):
            # Prefer longer, more complex sentences
            if len(tokens) > 15:
                argument_sents.append(sent)

    # Sort by length (complexity proxy)
    argument_sents.sort(key=lambda s: -len(word_tokenize(s)))
    return argument_sents


def parse_with_cfg(sentence, grammar=None):
    """Attempt to parse a sentence with the hand-written CFG.

    Returns list of parse trees (may be empty if grammar doesn't cover the sentence).
    """
    if grammar is None:
        grammar = ARGUMENT_GRAMMAR

    tokens = word_tokenize(sentence.lower())
    # Only keep tokens that are in the grammar's terminal set
    terminals = set()
    for prod in grammar.productions():
        if prod.is_lexical():
            terminals.add(prod.rhs()[0])

    filtered = [t for t in tokens if t in terminals]

    if len(filtered) < 3:
        return [], tokens, filtered

    parser = ChartParser(grammar)
    try:
        trees = list(parser.parse(filtered))
    except ValueError:
        trees = []

    return trees, tokens, filtered


def parsing_exercise(text):
    """Run the parsing exercise on Scylla and Charybdis."""
    argument_sents = find_argument_sentences(text)

    print(f"--- Found {len(argument_sents)} argument sentences ---")
    print(f"  (showing top 5 by length)\n")

    results = []
    for sent in argument_sents[:5]:
        print(f"  Original: {sent[:120]}...")
        trees, tokens, filtered = parse_with_cfg(sent)
        print(f"  Tokens: {len(tokens)}, Grammar-covered: {len(filtered)}")
        print(f"  Parse trees found: {len(trees)}")
        if trees:
            print(f"  First parse:")
            trees[0].pretty_print()
        elif filtered:
            print(f"  Grammar coverage: {len(filtered)/len(tokens)*100:.1f}%")
            print(f"  Covered tokens: {' '.join(filtered[:15])}")
        print()
        results.append((sent, trees, tokens, filtered))

    return results


# ---------------------------------------------------------------------------
# Exercise 2: Penn Treebank as Reference Grammar
# ---------------------------------------------------------------------------

def treebank_statistics():
    """Extract structural statistics from the Penn Treebank for comparison."""
    print("--- Penn Treebank Structural Statistics ---")

    depths = []
    branching_factors = []
    sbar_count = 0
    total_sents = 0

    for fileid in treebank.fileids()[:20]:  # Sample
        for tree in treebank.parsed_sents(fileid):
            total_sents += 1
            depths.append(tree.height())

            # Branching factor
            for subtree in tree.subtrees():
                if subtree.height() > 2:  # Non-terminal, non-preterminal
                    branching_factors.append(len(subtree))

            # Count SBAR (subordinate clauses)
            for subtree in tree.subtrees():
                if subtree.label() == 'SBAR':
                    sbar_count += 1

    avg_depth = sum(depths) / len(depths) if depths else 0
    avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
    sbar_per_sent = sbar_count / total_sents if total_sents else 0

    stats = {
        'total_sentences': total_sents,
        'avg_depth': avg_depth,
        'max_depth': max(depths) if depths else 0,
        'avg_branching': avg_branching,
        'sbar_per_sentence': sbar_per_sent,
    }

    print(f"  Sentences analyzed: {total_sents}")
    print(f"  Average tree depth: {avg_depth:.2f}")
    print(f"  Max tree depth: {max(depths) if depths else 0}")
    print(f"  Average branching factor: {avg_branching:.2f}")
    print(f"  SBAR (subordinate clauses) per sentence: {sbar_per_sent:.2f}")

    return stats


def episode_complexity(text, label="Scylla and Charybdis"):
    """Estimate syntactic complexity of the episode.

    Since we can't parse the full episode with a hand-written CFG,
    we use proxy measures: sentence length, subordinating conjunction
    frequency, comma density.
    """
    sentences = sent_tokenize(text)
    tokens_per_sent = [len(word_tokenize(s)) for s in sentences]

    # Subordinating conjunctions as complexity proxy
    sub_conjs = {'because', 'although', 'if', 'when', 'where', 'while',
                 'since', 'unless', 'that', 'which', 'who', 'whom'}
    all_tokens = word_tokenize(text.lower())
    sub_conj_count = sum(1 for t in all_tokens if t in sub_conjs)
    sub_conj_rate = sub_conj_count / len(sentences) if sentences else 0

    # Comma density
    comma_count = text.count(',')
    comma_per_sent = comma_count / len(sentences) if sentences else 0

    print(f"\n--- Syntactic Complexity Proxies: {label} ---")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Mean sentence length: {sum(tokens_per_sent)/len(tokens_per_sent):.1f} tokens")
    print(f"  Median sentence length: {sorted(tokens_per_sent)[len(tokens_per_sent)//2]} tokens")
    print(f"  Max sentence length: {max(tokens_per_sent)} tokens")
    print(f"  Subordinating conjunctions per sentence: {sub_conj_rate:.2f}")
    print(f"  Commas per sentence: {comma_per_sent:.2f}")

    return {
        'mean_sent_len': sum(tokens_per_sent) / len(tokens_per_sent),
        'max_sent_len': max(tokens_per_sent),
        'sub_conj_per_sent': sub_conj_rate,
        'comma_per_sent': comma_per_sent,
    }


# ---------------------------------------------------------------------------
# Exercise 3: The Quotation Problem
# ---------------------------------------------------------------------------

def extract_quotations(text):
    """Extract explicitly quoted material from the episode.

    Looks for text between quotation marks or in italicized passages.
    """
    # Find text between various quotation marks
    patterns = [
        r'"([^"]+)"',       # Double quotes
        r"'([^']+)'",       # Single quotes (careful with apostrophes)
        r'\*([^*]+)\*',     # Italics (markdown)
    ]

    quotations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match.split()) > 3:  # At least 4 words
                quotations.append(match)

    # Also look for em-dash dialogue patterns that might contain quoted speech
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('—') and len(stripped) > 20:
            # Check for embedded quotes
            inner_quotes = re.findall(r'"([^"]+)"', stripped)
            quotations.extend(q for q in inner_quotes if len(q.split()) > 3)

    # Deduplicate
    seen = set()
    unique = []
    for q in quotations:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


def compare_quotation_syntax(text):
    """Compare POS distributions of quoted vs. framing prose."""
    quotations = extract_quotations(text)

    print(f"\n--- Quotations Found: {len(quotations)} ---")
    for q in quotations[:10]:
        print(f"  \"{q[:80]}...\"" if len(q) > 80 else f"  \"{q}\"")

    # POS-tag quotations vs. rest of text
    quote_text = ' '.join(quotations)
    # Remove quotations from text for framing prose
    frame_text = text
    for q in quotations:
        frame_text = frame_text.replace(q, '')

    if quote_text.strip():
        quote_tags = Counter(tag for _, tag in pos_tag(word_tokenize(quote_text)))
        frame_tags = Counter(tag for _, tag in pos_tag(word_tokenize(frame_text)))

        qt = sum(quote_tags.values())
        ft = sum(frame_tags.values())

        print(f"\n--- POS Comparison: Quoted vs. Framing Prose ---")
        print(f"{'Tag':<8} {'Quoted %':>10} {'Frame %':>10} {'Diff':>8}")
        print("-" * 38)
        all_tags = sorted(set(list(quote_tags.keys()) + list(frame_tags.keys())))
        for tag in all_tags:
            qp = 100 * quote_tags.get(tag, 0) / qt if qt else 0
            fp = 100 * frame_tags.get(tag, 0) / ft if ft else 0
            if qp > 1 or fp > 1:
                print(f"  {tag:<6} {qp:>9.2f}% {fp:>9.2f}% {qp-fp:>+7.2f}%")

    return quotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    scylla = load_episode('09scyllacharybdis.txt')

    print("=" * 62)
    print("EXERCISE 1: Parsing the Argument")
    print("=" * 62)
    parsing_exercise(scylla)

    print("\n" + "=" * 62)
    print("EXERCISE 2: Treebank Reference & Episode Complexity")
    print("=" * 62)
    treebank_statistics()
    episode_complexity(scylla)
    # Compare with a simpler episode
    calypso = load_episode('04calypso.txt')
    episode_complexity(calypso, label="Calypso (for comparison)")

    print("\n" + "=" * 62)
    print("EXERCISE 3: The Quotation Problem")
    print("=" * 62)
    compare_quotation_syntax(scylla)
