"""
ranking.py
──────────
Scores and ranks documents using BM25 + a hand-built Min-Heap
Priority Queue.

WHY WE REPLACED sorted() WITH A HEAP:
  Before : scores dict → sorted() → top 10
           sorted() loads ALL matching docs into memory and sorts
           everything even if we only want top 10.
           Time: O(k log k)  Space: O(k)  — k = all matching docs

  After  : scores dict → Min-Heap of size k → top 10 extracted
           Heap keeps only 10 elements in memory at all times.
           Every new score either gets rejected immediately (O(log 10))
           or replaces the smallest in the heap (O(log 10)).
           Time: O(k log 10) ≈ O(k)   Space: O(10) = O(1)

  The difference matters when thousands of docs match a query.
  We never store or compare more than 10 elements at once.

SYLLABUS MAPPING:
  Module 5 — Binary Heap and Their Applications
  Lab 5    — Demonstrate Heap Sort Algorithm
  CO2      — Design and implement appropriate data structure
  CO3      — Analyze and differentiate searching/sorting algorithms
"""

import math
from preprocessing import preprocess


# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — MIN-HEAP (Priority Queue) — built from scratch
# ═════════════════════════════════════════════════════════════════════

class MinHeap:
    """
    A Min-Heap based Priority Queue.
    Stores (score, doc_id) pairs.
    The ROOT always holds the SMALLEST score.

    WHY MIN-HEAP FOR TOP-K?
    ────────────────────────
    Intuition: imagine you want to keep the 10 best players
    from a tryout of 1000 people. You maintain a waiting list
    of exactly 10. When a new player arrives:
      - If they are worse than the worst on your list → reject them
      - If they are better → remove the worst, add the new player

    The Min-Heap makes "find the worst on the list" instant — O(1).
    Replacing the worst takes O(log k) time.
    Result: we never store more than k elements.

    INTERNAL STRUCTURE — array-based binary tree:
    ─────────────────────────────────────────────
    The heap is stored as a flat Python list.
    A node at index i has:
      Left child  → index  2*i + 1
      Right child → index  2*i + 2
      Parent      → index  (i-1) // 2

    Example — heap with 5 elements stored as list:
      Index:  0     1     2     3     4
      Value: [1.2, 3.4, 2.1, 5.6, 4.0]

      Tree view:
              1.2  (root = smallest)
             /   \\
           3.4   2.1
           / \\
         5.6  4.0

    HEAP PROPERTY: every parent ≤ both its children.
    This guarantees root = minimum always.
    """

    def __init__(self, max_size: int):
        """
        Args:
          max_size : maximum number of elements to keep (our k = top_n)
        """
        self._data: list[tuple[float, str]] = []  # list of (score, doc_id)
        self._max_size = max_size

    # ── Helpers ──────────────────────────────────────────────────────

    def _parent(self, i: int) -> int:
        return (i - 1) // 2

    def _left(self, i: int) -> int:
        return 2 * i + 1

    def _right(self, i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def size(self) -> int:
        return len(self._data)

    def is_full(self) -> bool:
        return len(self._data) == self._max_size

    def peek_min(self) -> float:
        """
        Return the SMALLEST score in the heap without removing it.
        Since root is always the minimum, this is O(1).
        We use this to quickly decide: is a new score good enough
        to enter the top-k?
        """
        if not self._data:
            return float('-inf')
        return self._data[0][0]   # root = index 0 = minimum score

    # ── Core operations ──────────────────────────────────────────────

    def _sift_up(self, i: int) -> None:
        """
        Move element at index i UP the tree until heap property holds.

        Called after INSERTION — new element added at the bottom
        may be smaller than its parent, violating the heap property.

        Process:
          Compare with parent → if smaller, swap → repeat
          Stop when: parent ≤ current  OR  we reach the root

        Time: O(log n) — tree height is log(n)

        Example:
          Insert 0.5 into heap [1.2, 3.4, 2.1]:
          Added at index 3 → [1.2, 3.4, 2.1, 0.5]
          0.5 < parent(3.4) at index 1 → swap
          → [1.2, 0.5, 2.1, 3.4]
          0.5 < parent(1.2) at index 0 → swap
          → [0.5, 1.2, 2.1, 3.4]
          Reached root → done
        """
        while i > 0:
            p = self._parent(i)
            if self._data[i][0] < self._data[p][0]:
                self._swap(i, p)
                i = p
            else:
                break   # heap property satisfied — stop

    def _sift_down(self, i: int) -> None:
        """
        Move element at index i DOWN the tree until heap property holds.

        Called after DELETION of root — we move the last element
        to the root and then sift it down into its correct position.

        Process:
          Find the smallest child → if smaller than current, swap
          → repeat from the new position
          Stop when: both children ≥ current  OR  no children left

        Time: O(log n) — tree height is log(n)

        Example:
          Root deleted from [1.2, 3.4, 2.1, 5.6, 4.0]
          Move last (4.0) to root → [4.0, 3.4, 2.1, 5.6]
          Children of 4.0: left=3.4, right=2.1 → smallest=2.1 → swap
          → [2.1, 3.4, 4.0, 5.6]
          Children of 4.0 at index 2: none → done
        """
        n = len(self._data)
        while True:
            smallest = i
            l = self._left(i)
            r = self._right(i)

            # Check if left child exists and is smaller
            if l < n and self._data[l][0] < self._data[smallest][0]:
                smallest = l

            # Check if right child exists and is even smaller
            if r < n and self._data[r][0] < self._data[smallest][0]:
                smallest = r

            if smallest == i:
                break   # current is already smallest — heap property holds

            self._swap(i, smallest)
            i = smallest   # continue sifting down from new position

    def push(self, score: float, doc_id: str) -> None:
        """
        Insert (score, doc_id) into the heap.

        Strategy:
          If heap not full → add and sift up
          If heap IS full → only add if score > current minimum
            Because: we want TOP scores, not bottom scores
            If new score ≤ minimum of current top-k → it won't
            make the top-k → reject immediately, O(1) check

        Time: O(log k) where k = max_size (our top_n limit)
        """
        if not self.is_full():
            # Heap has room — add at end and sift up
            self._data.append((score, doc_id))
            self._sift_up(len(self._data) - 1)
        elif score > self._data[0][0]:
            # New score beats the current worst in our top-k
            # Replace the root (worst) with the new element
            self._data[0] = (score, doc_id)
            self._sift_down(0)   # restore heap property
        # else: score ≤ minimum in heap → cannot be in top-k → ignore

    def pop_min(self) -> tuple[float, str]:
        """
        Remove and return the element with the SMALLEST score.

        Process:
          1. Save root (minimum)
          2. Move last element to root
          3. Remove last position
          4. Sift down the new root

        Time: O(log k)

        We use this when extracting results — popping all k elements
        gives us results in ascending order, so we reverse at the end
        to get descending (highest score first).
        """
        if not self._data:
            raise IndexError("pop from empty heap")

        # Swap root with last element
        self._swap(0, len(self._data) - 1)
        min_item = self._data.pop()   # remove last (original root)
        if self._data:
            self._sift_down(0)        # restore heap property from new root

        return min_item

    def extract_all_sorted_desc(self) -> list[tuple[str, float]]:
        """
        Extract all elements sorted by score DESCENDING (best first).

        Process:
          Pop minimum repeatedly → get ascending order
          Reverse → descending order

        Time: O(k log k) where k = number of elements in heap
        Space: O(k)

        This is essentially Heap Sort on our k results.
        """
        result = []
        while self._data:
            score, doc_id = self.pop_min()
            result.append((doc_id, score))

        result.reverse()   # ascending → descending
        return result


# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — BM25 SCORING
# ═════════════════════════════════════════════════════════════════════

# BM25 constants
K1 = 1.5   # term frequency saturation
B  = 0.75  # document length normalisation


def _avg_doc_length(doc_lengths: dict) -> float:
    """Average token count across all indexed documents."""
    if not doc_lengths:
        return 1.0
    return sum(doc_lengths.values()) / len(doc_lengths)


def rank_documents(
    query       : str,
    pos_index   : dict,
    doc_lengths : dict,
    total_docs  : int,
    top_n       : int = 10,
) -> list[tuple[str, float]]:
    """
    Score all matching documents with BM25 and return top_n results
    using a Min-Heap Priority Queue instead of full sort.

    FULL ALGORITHM WALKTHROUGH:
    ───────────────────────────
    Step 1  Preprocess query → list of stemmed tokens
    Step 2  For each query token:
              a. Look up posting list in index → O(1)
              b. Compute BM25 IDF for this token
              c. For each doc in posting list:
                   Compute BM25 TF
                   Add IDF × TF to scores[doc]   → O(1) dict write
    Step 3  For each (doc, score) in scores dict:
              Push into MinHeap of size top_n
              Heap automatically keeps only the best top_n  → O(log top_n)
    Step 4  Extract all from heap in descending order → O(top_n log top_n)

    TIME COMPLEXITY:
      Step 2: O(Q × df)  — Q = query terms, df = matching docs per term
      Step 3: O(k × log top_n)  — k = total unique matching docs
      Step 4: O(top_n × log top_n)  — tiny, top_n is usually 10

    SPACE COMPLEXITY:
      scores dict: O(k)  — all matching docs
      heap:        O(top_n)  — only 10 elements ever in heap

    BM25 FORMULA RECAP:
      IDF  = log( (N - df + 0.5) / (df + 0.5) + 1 )
      TF   = (count × (K1+1)) / (count + K1 × (1 - B + B × dl/avgdl))
      score += IDF × TF

    Args:
      query       : raw query string from user
      pos_index   : dict{word → dict{doc → list[int]}}
      doc_lengths : dict{doc → int}
      total_docs  : int
      top_n       : how many results to return

    Returns:
      list of (doc_id, score) sorted by score descending
    """

    # ── Step 1: Preprocess query ──────────────────────────────────────
    query_terms = preprocess(query, use_bigrams=False)
    if not query_terms:
        return []

    avg_dl = _avg_doc_length(doc_lengths)

    # ── Step 2: Compute BM25 scores ───────────────────────────────────
    # DS: dict { doc_id → float }
    # Accumulates score contributions from each query term.
    scores: dict[str, float] = {}

    for term in query_terms:
        if term not in pos_index:
            continue

        postings = pos_index[term]      # dict { doc_id → list[positions] }
        df       = len(postings)        # how many docs contain this term

        # BM25 IDF — terms in fewer docs get higher IDF (more informative)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

        for doc_id, positions in postings.items():
            count = len(positions)                      # raw term freq
            dl    = doc_lengths.get(doc_id, avg_dl)    # this doc's length

            # BM25 TF — saturates at high counts, penalises long docs
            tf_bm25 = (count * (K1 + 1)) / (
                count + K1 * (1 - B + B * (dl / avg_dl))
            )

            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_bm25

    if not scores:
        return []

    # ── Step 3: Push scores into Min-Heap ────────────────────────────
    # DS: MinHeap of size top_n
    #
    # This is the KEY improvement over sorted():
    #
    # OLD approach:
    #   all_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #   return all_results[:top_n]
    #   → sorts ALL k matching docs even though we want only top_n
    #
    # NEW approach:
    #   heap maintains top_n best scores seen so far
    #   any doc that can't beat the current worst top_n is rejected
    #   immediately without being stored
    #   → heap never grows beyond top_n = 10 elements
    #
    # WHY MIN-HEAP AND NOT MAX-HEAP?
    #   Max-heap would give us the maximum at the root.
    #   But we need to know the MINIMUM of our current top-k
    #   to decide if a new score is good enough to replace it.
    #   Min-heap puts the weakest of our top-k at the root → O(1) check.
    # ─────────────────────────────────────────────────────────────────
    heap = MinHeap(max_size=top_n)

    for doc_id, score in scores.items():
        heap.push(score, doc_id)
        # push() internally:
        #   if heap not full      → add it  (O(log top_n))
        #   if score > heap root  → replace root, sift down  (O(log top_n))
        #   if score ≤ heap root  → reject, do nothing  (O(1))

    # ── Step 4: Extract results in descending order ───────────────────
    # extract_all_sorted_desc() pops min repeatedly → ascending
    # then reverses → descending (best first)
    # Time: O(top_n × log top_n)  — always tiny since top_n = 10
    return heap.extract_all_sorted_desc()


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — PHRASE SEARCH (unchanged — uses positional index)
# ═════════════════════════════════════════════════════════════════════

def phrase_search(phrase: str, pos_index: dict) -> list[str]:
    """
    Find documents containing all words of a phrase consecutively.

    Two-stage algorithm:
      Stage 1: Set intersection → candidate docs with ALL words  O(min|A|,|B|)
      Stage 2: Positional check → verify words are adjacent      O(p)

    DS used: set (intersection), list[int] (position checking)
    """
    terms = preprocess(phrase, use_bigrams=False)
    if not terms:
        return []

    # Stage 1 — set intersection across all terms
    posting_sets = [set(pos_index.get(t, {}).keys()) for t in terms]
    candidate_docs = posting_sets[0]
    for s in posting_sets[1:]:
        candidate_docs = candidate_docs & s

    if not candidate_docs:
        return []

    # Stage 2 — positional consecutiveness check
    matched = []
    for doc_id in candidate_docs:
        pos_lists = [pos_index[t][doc_id] for t in terms]
        found = False
        for start_pos in pos_lists[0]:
            if all(
                (start_pos + offset) in set(pos_lists[offset])
                for offset in range(1, len(terms))
            ):
                found = True
                break
        if found:
            matched.append(doc_id)

    return matched


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — BOOLEAN SEARCH (unchanged — uses set operations)
# ═════════════════════════════════════════════════════════════════════

def _get_doc_set(term: str, pos_index: dict) -> set:
    """Preprocess a single term and return its set of matching doc_ids."""
    stems = preprocess(term, use_bigrams=False)
    if not stems:
        return set()
    return set(pos_index.get(stems[0], {}).keys())


def boolean_search(query: str, pos_index: dict) -> set:
    """
    AND / OR / NOT boolean query using set operations.

    DS: set operations
      AND → intersection  A & B   O(min|A|,|B|)
      OR  → union         A | B   O(|A|+|B|)
      NOT → difference    A - B   O(|A|)
    """
    tokens = query.split()
    result = None
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token.upper() == "AND":
            i += 1
            continue

        elif token.upper() == "OR":
            i += 1
            if i >= len(tokens):
                break
            docs   = _get_doc_set(tokens[i], pos_index)
            result = result | docs if result is not None else docs

        elif token.upper() == "NOT":
            i += 1
            if i >= len(tokens):
                break
            docs   = _get_doc_set(tokens[i], pos_index)
            result = result - docs if result is not None else set()

        else:
            docs   = _get_doc_set(token, pos_index)
            result = result & docs if result is not None else docs

        i += 1

    return result or set()


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — HEAP DEMO (run this file directly to see the heap work)
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MIN-HEAP PRIORITY QUEUE — LIVE DEMONSTRATION")
    print("="*55)

    print("\n  Building heap of size 3 (top_n = 3)...")
    heap = MinHeap(max_size=3)

    scores_to_insert = [
        (2.5, "wiki_001"),
        (8.1, "arxiv_003"),
        (1.2, "tds_007"),
        (9.4, "wiki_012"),
        (5.6, "arxiv_009"),
        (3.3, "tds_002"),
        (7.7, "yt_001"),
    ]

    for score, doc in scores_to_insert:
        before = list(heap._data)
        heap.push(score, doc)
        after  = list(heap._data)
        action = "added " if len(after) > len(before) else (
                 "replaced min" if after != before else "rejected")
        print(f"  push({score}, {doc:12s})  → {action:15s}  heap={[round(x[0],1) for x in heap._data]}")

    print(f"\n  Heap root (weakest of top-3): {heap.peek_min()}")
    print("\n  Extracting results (best first):")
    results = heap.extract_all_sorted_desc()
    for rank, (doc, score) in enumerate(results, 1):
        print(f"    Rank {rank}: {doc:12s}  score={round(score, 1)}")

    print("\n  Heap correctly kept only the 3 highest-scored documents.")
    print("="*55 + "\n")