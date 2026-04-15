"""
search.py
─────────
Terminal interface for the ML search engine.
Loads the saved index and handles all query types.

QUERY TYPES SUPPORTED:
  Normal  : neural network
  Phrase  : "gradient descent"      (wrap in quotes)
  Boolean : transformer AND attention
            neural OR deep
            learning NOT supervised

Run:
  python search.py
"""

from indexer import load_index, build_index, save_index
from ranking import rank_documents, phrase_search, boolean_search
import json
import os

DATA_PATH = os.path.join("data", "all_documents.json")


# ─────────────────────────────────────────────────────────────────────
# STARTUP — load or rebuild index
# ─────────────────────────────────────────────────────────────────────

def initialise() -> tuple:
    """
    Load index from disk. If not found, try to build from
    saved raw documents. If those are missing too, tell user
    to run run_scrapers.py first.

    Returns:
      pos_index, doc_lengths, meta_store, total_docs
    """
    pos_index, tf_index, doc_lengths, meta_store, total_docs = load_index()

    if pos_index is not None:
        return pos_index, doc_lengths, meta_store, total_docs

    # Index missing — try to rebuild from saved raw documents
    if os.path.exists(DATA_PATH):
        print("  Index not found — rebuilding from saved documents...")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            all_docs = json.load(f)
        pos_index, tf_index, doc_lengths, meta_store, total_docs = build_index(all_docs)
        save_index(pos_index, tf_index, doc_lengths, meta_store, total_docs)
        return pos_index, doc_lengths, meta_store, total_docs

    # Nothing found — user must run the pipeline first
    print("\n  No index found. Please run first:")
    print("    python run_scrapers.py\n")
    exit(1)


# ─────────────────────────────────────────────────────────────────────
# QUERY ROUTER
# ─────────────────────────────────────────────────────────────────────

def detect_query_type(query: str) -> str:
    """
    Detect which query mode to use based on syntax.

    Rules:
      Starts and ends with "  → phrase search
      Contains AND / OR / NOT  → boolean search
      Anything else            → normal BM25 ranked search
    """
    stripped = query.strip()
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2:
        return "phrase"
    tokens = stripped.upper().split()
    if any(op in tokens for op in ("AND", "OR", "NOT")):
        return "boolean"
    return "normal"


def run_query(
    query      : str,
    pos_index  : dict,
    doc_lengths: dict,
    meta_store : dict,
    total_docs : int,
) -> list[tuple[str, float]]:
    """
    Route query to the right handler and return ranked results.

    DATA STRUCTURE returned:
      list[ (doc_id, score) ]  — sorted by score descending
      For phrase/boolean, score = BM25 score of matching docs
      For normal, score = BM25 score directly
    """
    query_type = detect_query_type(query)

    if query_type == "phrase":
        # Strip surrounding quotes then phrase search
        clean = query.strip().strip('"')
        matched_ids = phrase_search(clean, pos_index)

        if not matched_ids:
            return []

        # DS: set for O(1) membership check during filter
        matched_set = set(matched_ids)

        # Rank the matched docs by BM25 score
        all_ranked = rank_documents(clean, pos_index, doc_lengths, total_docs)
        return [(doc_id, score) for doc_id, score in all_ranked
                if doc_id in matched_set]

    elif query_type == "boolean":
        # Boolean returns a set of matching doc_ids
        matched_set = boolean_search(query, pos_index)

        if not matched_set:
            return []

        # Rank matched docs by BM25 using the raw query words
        all_ranked = rank_documents(query, pos_index, doc_lengths, total_docs)
        results = [(doc_id, score) for doc_id, score in all_ranked
                   if doc_id in matched_set]

        # If BM25 had no overlap with boolean set, return unscored
        if not results:
            return [(doc_id, 0.0) for doc_id in list(matched_set)[:10]]

        return results

    else:
        # Normal BM25 ranked search
        return rank_documents(query, pos_index, doc_lengths, total_docs)


# ─────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────

def display_results(
    results   : list[tuple[str, float]],
    meta_store: dict,
    query     : str,
):
    """
    Print ranked results in a clean terminal layout.

    For each result shows:
      Rank, title, source type badge, BM25 score, URL, snippet
    """
    if not results:
        print("\n  No results found.\n")
        return

    # DS: dict lookup O(1) per result for metadata
    print(f"\n  {len(results)} result(s) for: '{query}'")
    print("  " + "─" * 56)

    for rank, (doc_id, score) in enumerate(results, start=1):
        meta = meta_store.get(doc_id, {})

        title   = meta.get("title",   "Untitled")
        url     = meta.get("url",     "")
        snippet = meta.get("snippet", "")
        source  = meta.get("source",  "")
        dtype   = meta.get("type",    "article")

        # Source badge
        badge = {
            "wikipedia": "[WIKI]  ",
            "arxiv"    : "[PAPER] ",
            "tds"      : "[BLOG]  ",
            "youtube"  : "[VIDEO] ",
        }.get(source, "[DOC]   ")

        # Trim snippet to 160 chars
        snippet_display = snippet[:160].replace("\n", " ")
        if len(snippet) > 160:
            snippet_display += "..."

        print(f"\n  Rank {rank}  {badge}  Score: {round(score, 4)}")
        print(f"  Title   : {title}")
        print(f"  URL     : {url if url else 'N/A'}")
        print(f"  Preview : {snippet_display}")
        print("  " + "─" * 56)

    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("        ML / AI SEARCH ENGINE")
    print("=" * 58)
    print("  Query modes:")
    print('    Normal  →  neural network')
    print('    Phrase  →  "gradient descent"')
    print('    Boolean →  transformer AND attention')
    print('              neural OR deep')
    print('              learning NOT supervised')
    print("  Type 'exit' to quit.")
    print("=" * 58 + "\n")

    print("  Loading index...")
    pos_index, doc_lengths, meta_store, total_docs = initialise()
    print(f"  Ready. {total_docs} ML documents indexed.\n")

    while True:
        try:
            query = input("  Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!\n")
            break

        if not query:
            continue

        if query.lower() == "exit":
            print("\n  Goodbye!\n")
            break

        query_type = detect_query_type(query)
        print(f"  Mode: {query_type.upper()}")

        results = run_query(query, pos_index, doc_lengths, meta_store, total_docs)
        display_results(results, meta_store, query)


if __name__ == "__main__":
    main()