"""
app.py
──────
Flask web interface for the ML Search Engine.
Google-style UI with 4 result card types:
  article, paper, video, blog post

Run:
  python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, request, render_template_string
from indexer import load_index, build_index, save_index
from ranking import rank_documents, phrase_search, boolean_search
import json, os

app = Flask(__name__)

DATA_PATH = os.path.join("data", "all_documents.json")

# ── Load index at startup ─────────────────────────────────────────────
pos_index, tf_index, doc_lengths, meta_store, total_docs = load_index()
if pos_index is None:
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            all_docs = json.load(f)
        pos_index, tf_index, doc_lengths, meta_store, total_docs = build_index(all_docs)
        save_index(pos_index, tf_index, doc_lengths, meta_store, total_docs)
    else:
        print("No index found. Run: python run_scrapers.py")
        exit(1)

# ─────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{% if query %}{{ query }} — ML Search{% else %}ML Search Engine{% endif %}</title>
<style>
  /* ── Reset & base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --blue:    #1a73e8;
    --blue-h:  #1557b0;
    --green:   #0d652d;
    --gray1:   #202124;
    --gray2:   #4d5156;
    --gray3:   #70757a;
    --gray4:   #e8eaed;
    --gray5:   #f8f9fa;
    --border:  #dfe1e5;
    --red:     #c5221f;
    --purple:  #681da8;
    --orange:  #e37400;
    --shadow:  0 1px 6px rgba(32,33,36,.28);
    --radius:  24px;
  }
  body {
    font-family: arial, sans-serif;
    font-size: 14px;
    color: var(--gray1);
    background: #fff;
    min-height: 100vh;
  }

  /* ── HOME PAGE (no query) ── */
  .home {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 20px;
  }
  .home-logo {
    font-size: 68px;
    font-weight: 700;
    letter-spacing: -2px;
    margin-bottom: 28px;
    line-height: 1;
  }
  .home-logo span:nth-child(1) { color: #4285f4; }
  .home-logo span:nth-child(2) { color: #ea4335; }
  .home-logo span:nth-child(3) { color: #fbbc05; }
  .home-logo span:nth-child(4) { color: #4285f4; }
  .home-logo span:nth-child(5) { color: #34a853; }
  .home-logo span:nth-child(6) { color: #ea4335; }
  .home-logo span:nth-child(7) { color: #4285f4; }
  .home-logo span:nth-child(8) { color: #fbbc05; }
  .home-logo span:nth-child(9) { color: #34a853; }
  .home-subtitle {
    font-size: 16px;
    color: var(--gray3);
    margin-bottom: 32px;
    letter-spacing: 0.5px;
  }
  .home-search-wrap {
    width: 100%;
    max-width: 584px;
  }

  /* ── SEARCH BAR ── */
  .search-form { width: 100%; }
  .search-bar {
    display: flex;
    align-items: center;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 16px 10px 20px;
    background: #fff;
    transition: box-shadow .2s, border-color .2s;
    gap: 8px;
  }
  .search-bar:hover,
  .search-bar:focus-within {
    box-shadow: var(--shadow);
    border-color: transparent;
  }
  .search-icon {
    flex-shrink: 0;
    width: 20px; height: 20px;
    fill: var(--gray3);
  }
  .search-input {
    flex: 1;
    border: none;
    outline: none;
    font-size: 16px;
    color: var(--gray1);
    background: transparent;
    min-width: 0;
  }
  .search-input::placeholder { color: var(--gray3); }
  .search-btn {
    flex-shrink: 0;
    background: var(--blue);
    color: #fff;
    border: none;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background .15s;
    white-space: nowrap;
  }
  .search-btn:hover { background: var(--blue-h); }

  /* ── RESULTS PAGE HEADER ── */
  .results-header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px 0;
    display: flex;
    align-items: center;
    gap: 24px;
    position: sticky;
    top: 0;
    background: #fff;
    z-index: 10;
  }
  .header-logo {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -1px;
    text-decoration: none;
    flex-shrink: 0;
  }
  .header-logo span:nth-child(1) { color: #4285f4; }
  .header-logo span:nth-child(2) { color: #ea4335; }
  .header-logo span:nth-child(3) { color: #fbbc05; }
  .header-logo span:nth-child(4) { color: #4285f4; }
  .header-logo span:nth-child(5) { color: #34a853; }
  .header-logo span:nth-child(6) { color: #ea4335; }
  .header-logo .s2:nth-child(7)  { color: #4285f4; }
  .header-search-wrap {
    flex: 1;
    max-width: 600px;
    padding-bottom: 10px;
  }
  .header-search-wrap .search-bar { padding: 7px 12px 7px 16px; }
  .header-search-wrap .search-input { font-size: 15px; }

  /* ── RESULTS BODY ── */
  .results-body {
    max-width: 720px;
    margin: 0 auto;
    padding: 20px 24px 60px;
  }
  .results-meta {
    font-size: 13px;
    color: var(--gray3);
    margin-bottom: 20px;
  }
  .results-meta strong { color: var(--gray2); }

  /* ── FILTER TABS ── */
  .filter-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .tab {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid transparent;
    background: var(--gray5);
    color: var(--gray2);
    text-decoration: none;
    transition: all .15s;
  }
  .tab:hover { background: var(--gray4); }
  .tab.active {
    background: #e8f0fe;
    color: var(--blue);
    border-color: #c5d4fb;
  }
  .tab-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
  }

  /* ── QUERY TYPE BADGE ── */
  .query-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }
  .badge-normal  { background: #e8f0fe; color: #1a73e8; }
  .badge-phrase  { background: #fce8e6; color: #c5221f; }
  .badge-boolean { background: #e6f4ea; color: #0d652d; }

  /* ── RESULT CARDS ── */
  .result-card {
    margin-bottom: 28px;
    padding-bottom: 28px;
    border-bottom: 1px solid var(--gray4);
  }
  .result-card:last-child { border-bottom: none; }

  /* Article / Blog card */
  .card-source-line {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  .card-favicon {
    width: 16px; height: 16px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
  }
  .card-favicon-fallback {
    width: 16px; height: 16px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 9px; font-weight: 700; color: #fff;
    flex-shrink: 0;
  }
  .card-source-name { font-size: 12px; color: var(--gray2); }
  .card-source-url  { font-size: 12px; color: var(--gray3); }
  .card-title {
    font-size: 18px;
    font-weight: 400;
    color: var(--blue);
    text-decoration: none;
    line-height: 1.33;
    display: block;
    margin-bottom: 6px;
  }
  .card-title:hover { text-decoration: underline; }
  .card-snippet {
    font-size: 14px;
    color: var(--gray2);
    line-height: 1.58;
  }
  .card-snippet em {
    font-style: normal;
    font-weight: 700;
    color: var(--gray1);
  }
  .card-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .card-tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 500;
  }
  .tag-wiki    { background: #e8f0fe; color: #185abc; }
  .tag-arxiv   { background: #fce8e6; color: #b31412; }
  .tag-tds     { background: #e6f4ea; color: #137333; }
  .tag-youtube { background: #fef7e0; color: #b06000; }
  .score-tag   { background: var(--gray5); color: var(--gray3); }

  /* Paper card extras */
  .card-authors {
    font-size: 12px;
    color: var(--green);
    margin-bottom: 4px;
  }

  /* Image card */
  .card-with-image {
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }
  .card-image {
    width: 120px;
    height: 80px;
    object-fit: cover;
    border-radius: 8px;
    flex-shrink: 0;
    border: 1px solid var(--gray4);
  }
  .card-image-content { flex: 1; min-width: 0; }

  /* Video card */
  .video-thumb-wrap {
    position: relative;
    width: 200px;
    flex-shrink: 0;
  }
  .video-thumb {
    width: 200px;
    height: 112px;
    object-fit: cover;
    border-radius: 8px;
    display: block;
    border: 1px solid var(--gray4);
  }
  .video-play-btn {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 40px; height: 40px;
    background: rgba(0,0,0,0.65);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }
  .video-play-btn svg { fill: #fff; width: 16px; height: 16px; }
  .video-card-wrap {
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }
  .video-content { flex: 1; min-width: 0; }
  .video-channel {
    font-size: 12px;
    color: var(--gray3);
    margin-top: 4px;
  }

  /* ── NO RESULTS ── */
  .no-results {
    text-align: center;
    padding: 60px 20px;
    color: var(--gray2);
  }
  .no-results h2 { font-size: 20px; margin-bottom: 10px; font-weight: 400; }
  .no-results p  { font-size: 14px; color: var(--gray3); line-height: 1.6; }

  /* ── TIPS BOX ── */
  .tips-box {
    background: var(--gray5);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 32px;
    font-size: 13px;
    color: var(--gray2);
    line-height: 1.7;
  }
  .tips-box strong { color: var(--gray1); }
  .tips-box code {
    background: var(--gray4);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--purple);
  }

  /* ── HOME TIPS ── */
  .home-tips {
    margin-top: 24px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    max-width: 584px;
  }
  .home-tip-pill {
    background: var(--gray5);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: var(--gray2);
    cursor: pointer;
    transition: background .15s;
    text-decoration: none;
  }
  .home-tip-pill:hover { background: var(--gray4); }

  /* ── STATS BAR ── */
  .stats-bar {
    background: var(--gray5);
    border-bottom: 1px solid var(--border);
    padding: 8px 24px;
    font-size: 12px;
    color: var(--gray3);
    display: flex;
    gap: 20px;
  }
  .stats-bar span { color: var(--gray2); font-weight: 500; }

  @media (max-width: 600px) {
    .results-header { flex-wrap: wrap; gap: 10px; }
    .card-with-image, .video-card-wrap { flex-direction: column; }
    .video-thumb-wrap, .video-thumb { width: 100%; }
    .card-image { width: 100%; height: 160px; }
  }
</style>
</head>
<body>

{% if not query %}
<!-- ══════════════ HOME PAGE ══════════════ -->
<div class="home">
  <div class="home-logo">
    <span>M</span><span>L</span><span>S</span><span>e</span><span>a</span><span>r</span><span>c</span><span>h</span><span>+</span>
  </div>
  <div class="home-subtitle">Machine Learning & AI Search Engine</div>
  <div class="home-search-wrap">
    <form class="search-form" method="get" action="/">
      <div class="search-bar">
        <svg class="search-icon" viewBox="0 0 24 24">
          <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
        <input class="search-input" type="text" name="q"
               placeholder="Search ML topics, papers, videos..."
               autofocus autocomplete="off">
        <button class="search-btn" type="submit">Search</button>
      </div>
    </form>
    <div class="home-tips">
      <a class="home-tip-pill" href="/?q=neural+network">neural network</a>
      <a class="home-tip-pill" href='/?q=%22gradient+descent%22'>"gradient descent"</a>
      <a class="home-tip-pill" href="/?q=transformer+AND+attention">transformer AND attention</a>
      <a class="home-tip-pill" href="/?q=deep+learning">deep learning</a>
      <a class="home-tip-pill" href="/?q=reinforcement+learning">reinforcement learning</a>
    </div>
  </div>
</div>

{% else %}
<!-- ══════════════ RESULTS PAGE ══════════════ -->

<!-- Sticky header -->
<div class="results-header">
  <a class="header-logo" href="/">
    <span>M</span><span>L</span><span>S</span><span>+</span>
  </a>
  <div class="header-search-wrap">
    <form class="search-form" method="get" action="/">
      <div class="search-bar">
        <svg class="search-icon" viewBox="0 0 24 24">
          <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
        <input class="search-input" type="text" name="q"
               value="{{ query }}" autocomplete="off">
        <button class="search-btn" type="submit">Search</button>
      </div>
    </form>
  </div>
</div>

<!-- Stats bar -->
<div class="stats-bar">
  <div>Total indexed: <span>{{ total_docs }} documents</span></div>
  <div>Results: <span>{{ results|length }}</span></div>
  <div>Mode: <span>{{ query_type|upper }}</span></div>
</div>

<!-- Results body -->
<div class="results-body">

  <!-- Query type badge -->
  <div class="query-badge badge-{{ query_type }}">
    {% if query_type == 'phrase' %}Phrase search
    {% elif query_type == 'boolean' %}Boolean search
    {% else %}Ranked search
    {% endif %}
  </div>

  <!-- Filter tabs -->
  {% if results %}
  <div class="filter-tabs">
    <a class="tab active" href="/?q={{ query|urlencode }}">
      All&nbsp;<span style="color:var(--gray3);font-weight:400">({{ results|length }})</span>
    </a>
    {% if counts.article > 0 %}
    <a class="tab" href="/?q={{ query|urlencode }}&type=article">
      <span class="tab-dot" style="background:#4285f4"></span>Articles ({{ counts.article }})
    </a>
    {% endif %}
    {% if counts.paper > 0 %}
    <a class="tab" href="/?q={{ query|urlencode }}&type=paper">
      <span class="tab-dot" style="background:#ea4335"></span>Papers ({{ counts.paper }})
    </a>
    {% endif %}
    {% if counts.video > 0 %}
    <a class="tab" href="/?q={{ query|urlencode }}&type=video">
      <span class="tab-dot" style="background:#fbbc05"></span>Videos ({{ counts.video }})
    </a>
    {% endif %}
  </div>
  {% endif %}

  <!-- Result cards -->
  {% if results %}
    {% for doc_id, score, meta in results %}

    <div class="result-card">

      {% if meta.type == 'video' and meta.image %}
      <!-- ── VIDEO CARD ── -->
      <div class="video-card-wrap">
        <div class="video-thumb-wrap">
          <a href="{{ meta.url }}" target="_blank" rel="noopener">
            <img class="video-thumb" src="{{ meta.image }}"
                 alt="{{ meta.title }}" loading="lazy"
                 onerror="this.style.display='none'">
            <div class="video-play-btn">
              <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </div>
          </a>
        </div>
        <div class="video-content">
          <div class="card-source-line">
            <div class="card-favicon-fallback" style="background:#ff0000">YT</div>
            <span class="card-source-name">YouTube</span>
            <span class="card-source-url">youtube.com</span>
          </div>
          <a class="card-title" href="{{ meta.url }}" target="_blank" rel="noopener">
            {{ meta.title }}
          </a>
          <div class="card-snippet">{{ meta.snippet[:200] }}...</div>
          <div class="card-meta">
            <span class="card-tag tag-youtube">VIDEO</span>
            <span class="card-tag score-tag">Score: {{ "%.3f"|format(score) }}</span>
          </div>
        </div>
      </div>

      {% elif meta.image and meta.type == 'article' and meta.source == 'wikipedia' %}
      <!-- ── WIKI ARTICLE WITH IMAGE ── -->
      <div class="card-with-image">
        <img class="card-image" src="{{ meta.image }}"
             alt="{{ meta.title }}" loading="lazy"
             onerror="this.style.display='none'">
        <div class="card-image-content">
          <div class="card-source-line">
            <div class="card-favicon-fallback" style="background:#3366cc">W</div>
            <span class="card-source-name">Wikipedia</span>
            <span class="card-source-url">en.wikipedia.org</span>
          </div>
          <a class="card-title" href="{{ meta.url }}" target="_blank" rel="noopener">
            {{ meta.title }}
          </a>
          <div class="card-snippet">{{ meta.snippet[:220] }}...</div>
          <div class="card-meta">
            <span class="card-tag tag-wiki">WIKI</span>
            <span class="card-tag score-tag">Score: {{ "%.3f"|format(score) }}</span>
          </div>
        </div>
      </div>

      {% elif meta.type == 'paper' %}
      <!-- ── ARXIV PAPER CARD ── -->
      <div class="card-source-line">
        <div class="card-favicon-fallback" style="background:#b31412">Ax</div>
        <span class="card-source-name">ArXiv</span>
        <span class="card-source-url">arxiv.org</span>
      </div>
      <a class="card-title" href="{{ meta.url }}" target="_blank" rel="noopener">
        {{ meta.title }}
      </a>
      <div class="card-snippet">{{ meta.snippet[:240] }}...</div>
      <div class="card-meta">
        <span class="card-tag tag-arxiv">PAPER</span>
        <span class="card-tag score-tag">Score: {{ "%.3f"|format(score) }}</span>
      </div>

      {% else %}
      <!-- ── GENERIC ARTICLE / BLOG CARD ── -->
      {% set src_label = {'tds': 'Towards Data Science', 'wikipedia': 'Wikipedia'}.get(meta.source, meta.source|title) %}
      {% set src_url   = {'tds': 'towardsdatascience.com', 'wikipedia': 'en.wikipedia.org'}.get(meta.source, '') %}
      {% set src_color = {'tds': '#00ab6c', 'wikipedia': '#3366cc'}.get(meta.source, '#555') %}
      {% set src_short = {'tds': 'TDS', 'wikipedia': 'W'}.get(meta.source, meta.source[:2]|upper) %}
      <div class="card-source-line">
        <div class="card-favicon-fallback" style="background:{{ src_color }}">{{ src_short }}</div>
        <span class="card-source-name">{{ src_label }}</span>
        {% if src_url %}<span class="card-source-url">{{ src_url }}</span>{% endif %}
      </div>
      <a class="card-title" href="{{ meta.url }}" target="_blank" rel="noopener">
        {{ meta.title }}
      </a>
      <div class="card-snippet">{{ meta.snippet[:240] }}...</div>
      <div class="card-meta">
        {% if meta.source == 'tds' %}
          <span class="card-tag tag-tds">BLOG</span>
        {% else %}
          <span class="card-tag tag-wiki">ARTICLE</span>
        {% endif %}
        <span class="card-tag score-tag">Score: {{ "%.3f"|format(score) }}</span>
      </div>
      {% endif %}

    </div>
    {% endfor %}

    <!-- Tips box -->
    <div class="tips-box">
      <strong>Search tips</strong><br>
      Phrase search: <code>"neural network"</code> &nbsp;|&nbsp;
      Boolean: <code>transformer AND attention</code> &nbsp;|&nbsp;
      Exclude: <code>learning NOT supervised</code>
    </div>

  {% else %}
  <!-- No results -->
  <div class="no-results">
    <h2>No results found for "{{ query }}"</h2>
    <p>Try different keywords, or use broader terms.<br>
       Example: try <strong>neural</strong> instead of <strong>neural networks architecture</strong></p>
    <div class="tips-box" style="text-align:left;margin-top:24px">
      <strong>Search tips</strong><br>
      Phrase: <code>"gradient descent"</code> &nbsp;|&nbsp;
      Boolean: <code>deep AND learning</code> &nbsp;|&nbsp;
      Exclude: <code>neural NOT recurrent</code>
    </div>
  </div>
  {% endif %}

</div>
{% endif %}

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────
# QUERY LOGIC
# ─────────────────────────────────────────────────────────────────────

def detect_query_type(query: str) -> str:
    s = query.strip()
    if s.startswith('"') and s.endswith('"') and len(s) > 2:
        return "phrase"
    if any(op in s.upper().split() for op in ("AND", "OR", "NOT")):
        return "boolean"
    return "normal"


def run_query(query: str, filter_type: str = None) -> tuple:
    query_type = detect_query_type(query)

    if query_type == "phrase":
        clean       = query.strip().strip('"')
        matched_ids = set(phrase_search(clean, pos_index))
        all_ranked  = rank_documents(clean, pos_index, doc_lengths, total_docs)
        results     = [(d, s) for d, s in all_ranked if d in matched_ids]

    elif query_type == "boolean":
        matched_ids = boolean_search(query, pos_index)
        all_ranked  = rank_documents(query, pos_index, doc_lengths, total_docs)
        results     = [(d, s) for d, s in all_ranked if d in matched_ids]
        if not results:
            results = [(d, 0.0) for d in list(matched_ids)[:10]]

    else:
        results = rank_documents(query, pos_index, doc_lengths, total_docs)

    # Attach metadata
    enriched = []
    for doc_id, score in results:
        meta = meta_store.get(doc_id, {})
        if filter_type and meta.get("type") != filter_type:
            continue
        enriched.append((doc_id, score, meta))

    return enriched, query_type


# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    query       = request.args.get("q", "").strip()
    filter_type = request.args.get("type", None)

    if not query:
        return render_template_string(
            TEMPLATE, query="", results=[], query_type="normal",
            total_docs=total_docs, counts={}
        )

    results, query_type = run_query(query, filter_type)

    # Count results by type for filter tabs
    counts = {"article": 0, "paper": 0, "video": 0}
    all_results, _ = run_query(query)   # unfiltered for tab counts
    for _, _, meta in all_results:
        t = meta.get("type", "article")
        if t in counts:
            counts[t] += 1

    return render_template_string(
        TEMPLATE,
        query      = query,
        results    = results,
        query_type = query_type,
        total_docs = total_docs,
        counts     = counts,
    )


if __name__ == "__main__":
    print("\n  ML Search Engine running at http://127.0.0.1:5000\n")
    app.run(debug=True)