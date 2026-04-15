"""
preprocessing.py
────────────────
Converts raw text into clean, searchable tokens.
Every document and every query passes through this file.

DATA STRUCTURES USED:
  set   → stopwords        O(1) membership check per token
  dict  → stem cache       O(1) lookup after first stem computation
  list  → token output     ordered sequence of processed tokens
"""

import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords

nltk.download("stopwords", quiet=True)

# ── Stopwords ────────────────────────────────────────────────────────
# DATA STRUCTURE: set
# Why set and not list?
#   list check → O(n)  — scans every element until match
#   set check  → O(1)  — direct hash lookup, instant
# With 179 stopwords and thousands of tokens per document,
# this difference adds up fast.
# ─────────────────────────────────────────────────────────────────────
STOPWORDS = set(nltk_stopwords.words("english"))

# ── Stem cache ───────────────────────────────────────────────────────
# DATA STRUCTURE: dict  {word: stemmed_word}
# PorterStemmer.stem() is slow — it applies ~60 suffix rules per call.
# Caching results means each unique word is stemmed only once.
# Every repeat call is an O(1) dict lookup instead.
# Example: "learning", "learned", "learns" → all cached as "learn"
# ─────────────────────────────────────────────────────────────────────
_stem_cache: dict[str, str] = {}
_stemmer = PorterStemmer()


def stem(word: str) -> str:
    """Return stem of word, computing only on first call."""
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]


def get_bigrams(tokens: list[str]) -> list[str]:
    """
    Sliding window of size 2 over token list.

    DATA STRUCTURE: list of str
    Joins adjacent pairs with '_' so "neural network" becomes
    "neural_network" — a single indexable token.
    This lets users search for exact phrases and get better results.

    Example:
      tokens  = ["neural", "network", "train"]
      bigrams = ["neural_network", "network_train"]
    """
    return [
        f"{tokens[i]}_{tokens[i + 1]}"
        for i in range(len(tokens) - 1)
    ]


def preprocess(text: str, use_bigrams: bool = False) -> list[str]:
    """
    Full text cleaning pipeline. Returns list of tokens.

    Steps:
      1. Lowercase           → "Neural" and "neural" become same token
      2. Strip punctuation   → remove . , ! ? etc
      3. Tokenise            → split on whitespace into word list
      4. Remove stopwords    → drop "the", "is", "a" etc  (set O(1))
      5. Stem each token     → "learning" → "learn"        (dict cache)
      6. Add bigrams         → "neural_network" as one token (optional)

    Args:
      text        : raw input string
      use_bigrams : if True, append bigrams to the token list

    Returns:
      list[str] of clean tokens ready for indexing or querying
    """
    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — remove all punctuation characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3 — split into word list
    tokens = text.split()

    # Step 4 — remove stopwords using set (O(1) per token)
    tokens = [w for w in tokens if w not in STOPWORDS]

    # Step 5 — stem using cached stemmer (O(1) after first call)
    tokens = [stem(w) for w in tokens]

    # Step 6 — optionally append bigrams
    if use_bigrams and len(tokens) >= 2:
        tokens = tokens + get_bigrams(tokens)

    return tokens

