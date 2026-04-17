# ML Search Engine
### Domain-Specific Search Engine for Machine Learning & AI


A fully functional search engine built from scratch in Python — no pre-built search library used. Collects real ML/AI content from Wikipedia, ArXiv, Towards Data Science, and YouTube; builds a searchable inverted index using custom data structures; ranks results with the BM25 algorithm; and presents them through a Google-style Flask web interface.


## Performance

| Metric | Value |
|--------|-------|
| Query response time | **< 50 ms** |
| Documents indexed | **150+ real ML documents** |
| Index load on startup | **< 1 second** |
| Hash table lookup | **O(1)** |
| Top-k memory | **O(10) constant** |

---

## Quick Start

```bash
git clone https://github.com/your-username/ml-search-engine
cd ml-search-engine
pip install -r requirements.txt
python scraper.py   # builds index (~2 min, run once)
python app.py       # starts Flask at localhost:5000
```

For the terminal interface:
```bash
python search.py
```

---

## Query Modes

| Mode | Syntax | How it works |
|------|--------|-------------|
| **Normal** | `neural network` | BM25-ranked results with TF saturation and length normalisation |
| **Phrase** | `"gradient descent"` | Exact consecutive-token match using positional index |
| **Boolean** | `CNN AND image NOT video` | AND / OR / NOT on posting sets — O(min(\|A\|,\|B\|)) |

---

## Data Sources

| Source | Method | Content type |
|--------|--------|-------------|
| Wikipedia | REST API | ML/AI articles |
| ArXiv | Atom XML | Research papers |
| Towards Data Science | RSS feed | ML blogs |
| YouTube | Data API v3 | Tutorial videos |

---

## Data Structures — Every Choice Justified

Each data structure replaces a slow naive approach and produces a measurable Big-O improvement:

### HashTableChaining — `indexer.py`
Implements the inverted index. Hash function: `h = (h × 31 + ord(char)) % capacity`. Collisions resolved by prepending to a linked list (O(1) insert). Token positions stored as `list[int]` per word per doc to enable phrase search.

- **Before:** O(D × W) — scan all documents for every query
- **After:** O(1) — hash lookup directly to matching posting list

> Chaining chosen over Linear Probing because key count is unknown before scraping, load factor can safely exceed 1.0, and values are large nested dicts. Both implementations are included in `indexer.py`.

### MinHeap (size 10) — `ranking.py`
Hand-built min-heap for top-k selection. New score is compared to the root (weakest of top-10) in O(1). If weaker → rejected immediately. If stronger → replaces root, `sift_down` restores heap in O(log 10). Memory stays constant at O(10) regardless of how many documents match.

- **Before:** O(k log k) + O(k) memory
- **After:** O(k log 10) + O(10) constant memory

### Set — `preprocessing.py`
Python `set` for stopword lookup — replaces a linear list scan on 179 stopwords.

- **Before:** O(n) list scan per token
- **After:** O(1) hash lookup

### Dict cache — `preprocessing.py`
Memoises the Porter stemmer. A word is stemmed once; all subsequent occurrences retrieve the result from cache.

- **Before:** 60 stemming rules × N calls per repeated word
- **After:** O(1) after first call

### `list[int]` positions — `indexer.py`
Stores per-document token positions to enable exact phrase verification: set intersection finds candidate docs, positional check confirms consecutiveness.

- **Before:** Phrase search impossible
- **After:** O(p) positional check where p = phrase length

### Set operations — `ranking.py`
Boolean AND / OR / NOT implemented via Python set intersection, union, and difference on posting sets.

- **Before:** O(\|A\| × \|B\|) nested loops
- **After:** O(min(\|A\|, \|B\|))

---

## BM25 Ranking Algorithm

BM25 fixes two weaknesses of plain TF-IDF:
1. **Term frequency saturation** — repeated occurrences add diminishing returns, preventing keyword stuffing
2. **Document length normalisation** — longer documents are penalised unless the word appears proportionally more

```
score  = IDF × TF_BM25

IDF    = log( (N − df + 0.5) / (df + 0.5) + 1 )

TF     = (count × (K1 + 1)) / (count + K1 × (1 − B + B × dl/avgdl))
```

**Parameters:** K1 = 1.5, B = 0.75

---

## Pipeline

```
Scrape → Preprocess → Index → Persist → Query → Rank → Display
```

1. **Scrape** — 4 scrapers fetch data as `list[dict]`
2. **Preprocess** — lowercase → strip punctuation → tokenise → remove stopwords (set) → stem (dict cache) → bigrams
3. **Index** — tokens inserted into `HashTableChaining` with positions stored as `list[int]`
4. **Persist** — index saved to `index.json`, metadata to `meta.json` (loads in < 1 sec on next run)
5. **Query** — query cleaned the same way, routed to normal / phrase / boolean handler
6. **Rank** — BM25 scores accumulated per doc; MinHeap of size 10 selects top-k
7. **Display** — metadata attached via O(1) lookup; Flask renders appropriate card type

---

## Project Structure

```
ml-search-engine/
├── app.py              # Flask web server — localhost:5000
├── search.py           # Terminal search interface
├── indexer.py          # HashTableChaining + HashTableProbing implementations
├── ranking.py          # BM25 scorer + hand-built MinHeap top-k
├── preprocessing.py    # Tokeniser, stopword set, stem dict cache
├── scrapers/
│   ├── wikipedia.py    # Wikipedia REST API scraper
│   ├── arxiv.py        # ArXiv Atom XML scraper
│   ├── tds.py          # Towards Data Science RSS scraper
│   └── youtube.py      # YouTube Data API v3 scraper
├── index.json          # Persisted inverted index (generated)
├── meta.json           # Document metadata (generated)
├── templates/
│   └── index.html      # Google-style UI with 4 result card types
└── requirements.txt
```

---

## Requirements

### Hardware
| Component | Minimum |
|-----------|---------|
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| RAM | 4 GB (index ≈ 100 MB); 8 GB recommended |
| Storage | 50 MB for code + index; 200 MB with full dataset |
| Internet | Required once for scraping only |

### Software
```
Python 3.9+
flask
requests
beautifulsoup4
feedparser
nltk
```

Install all dependencies:
```bash
pip install -r requirements.txt
```



## Industry Relevance

The techniques implemented here — inverted indexes, positional indexes, BM25 ranking, phrase search, boolean queries — are the same techniques used by **Elasticsearch**, **Apache Solr**, **Apache Lucene**, and **Google**.

**Data structure patterns applicable across industry:**
- `HashTableChaining` → database indexing, DNS caches, compiler symbol tables, network routing
- `MinHeap` for top-k → recommendation engines, log monitoring, streaming analytics, task schedulers
- Dict memoisation → dynamic programming, API response caching, function result caching
- Set operations → database query optimisation, access control, recommendation filtering
