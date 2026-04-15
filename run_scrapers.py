"""
run_scrapers.py
───────────────
Master script — runs all scrapers, combines results,
builds the search index, and saves everything to disk.

Run this once to populate your search engine:
  python run_scrapers.py

After this you never need to run it again unless you want
to refresh the data. The index is saved to disk and loaded
instantly on every subsequent run.

DATA STRUCTURE FLOW:
  Each scraper returns  list[dict]
  We combine with  +    list[dict] + list[dict] + ...
  Pass one flat         list[dict]  to build_index()
  Index saved as        dict → JSON on disk

WHY A FLAT list[dict]?
  Simple to combine   : wiki_docs + arxiv_docs = one list, O(n)
  Simple to iterate   : one loop in build_index() handles all types
  Simple to extend    : add a new scraper → just append its list
"""

import json
import os
import time

from scraper.wikipedia_scraper import scrape as scrape_wikipedia
from scraper.arxiv_scraper      import scrape as scrape_arxiv
from scraper.tds_scraper        import scrape as scrape_tds
from scraper.youtube_scraper    import scrape as scrape_youtube
from indexer                    import build_index, save_index

DATA_DIR       = "data"
ALL_DOCS_PATH  = os.path.join(DATA_DIR, "all_documents.json")


def run():
    start_time = time.time()

    print("=" * 55)
    print("   ML SEARCH ENGINE — DATA PIPELINE")
    print("=" * 55)

    # ── Step 1: Run all scrapers ──────────────────────────────────────
    # Each returns list[dict] with the same schema.
    # We collect them separately first so we can report counts.
    # DS: list[dict] per scraper → combined into one flat list
    # ─────────────────────────────────────────────────────────────────

    print("\n[1/4] Scraping Wikipedia...")
    wiki_docs = scrape_wikipedia(max_articles=40)

    print("\n[2/4] Scraping ArXiv...")
    arxiv_docs = scrape_arxiv(max_per_query=10)

    print("\n[3/4] Scraping Towards Data Science...")
    tds_docs = scrape_tds(max_posts=30)

    print("\n[4/4] Scraping YouTube...")
    yt_docs = scrape_youtube(max_per_query=5)

    # ── Step 2: Combine all documents ────────────────────────────────
    # DS: list concatenation — O(n) total
    # All scrapers return the same dict schema so we can safely
    # combine them into one flat list with simple + operator.
    # ─────────────────────────────────────────────────────────────────
    all_docs = wiki_docs + arxiv_docs + tds_docs + yt_docs

    print("\n" + "─" * 55)
    print(f"  Wikipedia articles : {len(wiki_docs)}")
    print(f"  ArXiv papers       : {len(arxiv_docs)}")
    print(f"  TDS blog posts     : {len(tds_docs)}")
    print(f"  YouTube videos     : {len(yt_docs)}")
    print(f"  Total documents    : {len(all_docs)}")
    print("─" * 55)

    if len(all_docs) == 0:
        print("\nNo documents collected. Check your internet connection.")
        return

    # ── Step 3: Save raw documents to data/ folder ───────────────────
    # Saves all_documents.json so you can inspect what was scraped
    # without rebuilding from scratch.
    # ─────────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ALL_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    print(f"\n  Raw documents saved → {ALL_DOCS_PATH}")

    # ── Step 4: Build index ───────────────────────────────────────────
    # Pass the flat list[dict] to build_index().
    # It handles all document types uniformly — article, paper, video.
    # ─────────────────────────────────────────────────────────────────
    print("\n  Building search index...")
    pos_index, tf_index, doc_lengths, meta_store, total_docs = build_index(all_docs)

    # ── Step 5: Save index to disk ────────────────────────────────────
    # After this step, app.py and search.py load from disk instantly.
    # No need to re-scrape or re-index on every run.
    # ─────────────────────────────────────────────────────────────────
    print("\n  Saving index to disk...")
    save_index(pos_index, tf_index, doc_lengths, meta_store, total_docs)

    elapsed = round(time.time() - start_time, 1)

    print("\n" + "=" * 55)
    print(f"  Done in {elapsed}s")
    print(f"  {total_docs} documents indexed and ready to search.")
    print(f"  Run  python app.py  to start the web interface.")
    print(f"  Run  python search.py  for the terminal interface.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run()