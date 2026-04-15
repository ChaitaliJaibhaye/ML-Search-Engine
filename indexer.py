"""
indexer.py
──────────
Builds, saves, and loads the search index using hand-built Hash Tables.

WHAT CHANGED FROM THE PREVIOUS VERSION:
  Before: used Python's built-in dict for the inverted index.
          dict is a black box — you cannot explain its internals.

  After:  Two hash tables built from scratch:
            1. HashTableChaining  — Separate Chaining (linked list per slot)
            2. HashTableProbing   — Linear Probing   (open addressing)
          The search engine uses HashTableChaining for the final index.
          Both are demonstrated and compared at the bottom of this file.

SYLLABUS MAPPING:
  Module 9 — Hash Table: overview, theory, implementation,
             separate chaining, linear probing, hash table context
  Lab 10   — Simple Implementation of a Hash Table
  CO2      — Design and implement appropriate data structure
  CO3      — Analyze and differentiate various searching algorithms

WHY HASH TABLE FOR THE INVERTED INDEX?
  The inverted index maps word → {doc → positions}.
  We need to look up a word millions of times per day.
  Hash table gives O(1) average lookup — nothing is faster.
  Arrays would need O(n) scan. BST gives O(log n). Hash = O(1).
"""

import os
import json
import math
from preprocessing import preprocess


# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — SEPARATE CHAINING HASH TABLE
# ═════════════════════════════════════════════════════════════════════

class Node:
    """
    A single node in a linked list chain.

    Each node stores one key-value pair.
    When two keys hash to the same slot (collision),
    they form a chain of nodes at that slot.

    DS: Singly linked list node
    """
    def __init__(self, key: str, value):
        self.key   = key
        self.value = value
        self.next  = None   # pointer to next node in the chain


class HashTableChaining:
    """
    Hash Table with Separate Chaining for collision resolution.

    STRUCTURE:
    ──────────
    An array of size `capacity`.
    Each slot holds either None or the HEAD of a linked list.
    When two keys hash to the same index, they form a chain.

    VISUAL EXAMPLE (capacity = 7):

    Slot 0: None
    Slot 1: ["neural" → {...}] → ["network" → {...}] → None
    Slot 2: ["learn"  → {...}] → None
    Slot 3: None
    Slot 4: ["deep"   → {...}] → None
    Slot 5: ["model"  → {...}] → ["train" → {...}] → None
    Slot 6: None

    Slot 1 has TWO words ("neural" and "network") that happened to
    hash to the same index. They are chained together as a linked list.

    WHY SEPARATE CHAINING?
    ───────────────────────
    The inverted index can have thousands of unique words.
    With chaining, collisions just add a node to the chain.
    The table never "fills up" — chains can grow as needed.
    Load factor (n/capacity) can exceed 1.0 — still works fine.
    Best for use cases where key count is hard to predict in advance
    — exactly our situation (we don't know how many unique ML words).

    TIME COMPLEXITY:
      Insert : O(1) average, O(n) worst (all keys in one chain)
      Search : O(1) average, O(n) worst
      Delete : O(1) average, O(n) worst
      Average assumes good hash function + low load factor.

    SPACE: O(n + capacity) — n nodes + capacity slot pointers
    """

    def __init__(self, capacity: int = 1024):
        """
        Args:
          capacity : number of slots in the array.
                     Larger = fewer collisions but more memory.
                     We use 1024 as default (power of 2 is common).
        """
        self._capacity = capacity
        # DS: array (Python list) of linked list heads
        # Each element is either None or a Node
        self._buckets: list[Node | None] = [None] * capacity
        self._size = 0   # number of key-value pairs stored

    # ── Hash function ─────────────────────────────────────────────────

    def _hash(self, key: str) -> int:
        """
        Convert a string key to a slot index.

        METHOD: Polynomial rolling hash
        ─────────────────────────────────
        Treat the string as a base-31 number.
        Each character contributes: char_value × 31^position

        Why 31?
          - Prime number → reduces clustering
          - Small enough to avoid overflow quickly
          - Used by Java's String.hashCode() — proven effective

        Formula:
          hash = (ord(c0) × 31^0 + ord(c1) × 31^1 + ...) % capacity

        Then modulo capacity to get a valid slot index [0, capacity).

        Example: hash("neural", capacity=7)
          n=110, e=101, u=117, r=114, a=97, l=108
          hash = (110×1 + 101×31 + 117×961 + 114×29791 + ...) % 7
          = some index in [0, 6]

        Time: O(len(key)) — proportional to word length
        """
        h = 0
        for char in key:
            h = (h * 31 + ord(char)) % self._capacity
        return h

    # ── Core operations ───────────────────────────────────────────────

    def put(self, key: str, value) -> None:
        """
        Insert or update key-value pair.

        Steps:
          1. Compute hash → get slot index  O(len(key))
          2. Walk the chain at that slot
          3. If key found → update value
          4. If key not found → prepend new node to chain  O(1)

        Prepending to the chain (not appending) keeps insert O(1)
        regardless of chain length — we don't need to walk to the end.

        Time: O(1) average — O(n) worst if all keys collide
        """
        index = self._hash(key)
        node  = self._buckets[index]

        # Walk the chain — check if key already exists
        while node is not None:
            if node.key == key:
                node.value = value   # update existing key
                return
            node = node.next

        # Key not found — prepend new node at the head of the chain
        # Prepend = O(1), append = O(chain_length)
        new_node              = Node(key, value)
        new_node.next         = self._buckets[index]   # point to old head
        self._buckets[index]  = new_node               # new head
        self._size           += 1

    def get(self, key: str):
        """
        Retrieve value for key. Returns None if not found.

        Steps:
          1. Compute hash → slot index  O(len(key))
          2. Walk the chain comparing keys
          3. Return value if found, None otherwise

        Time: O(1) average — O(chain_length) worst
        """
        index = self._hash(key)
        node  = self._buckets[index]

        while node is not None:
            if node.key == key:
                return node.value
            node = node.next

        return None   # key not found

    def contains(self, key: str) -> bool:
        """O(1) average membership check."""
        return self.get(key) is not None

    def keys(self):
        """
        Iterate over all keys in the hash table.
        Walks every bucket and every node in each chain.
        Time: O(capacity + n) — must check all slots even empty ones
        """
        for bucket in self._buckets:
            node = bucket
            while node is not None:
                yield node.key
                node = node.next

    def items(self):
        """Iterate over all (key, value) pairs."""
        for bucket in self._buckets:
            node = bucket
            while node is not None:
                yield node.key, node.value
                node = node.next

    def load_factor(self) -> float:
        """
        Load factor = number of entries / number of slots.
        Measures how full the table is.
        Chaining works well even at load factor > 1.
        Rule of thumb: keep below 2.0 for good average performance.
        """
        return self._size / self._capacity

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: str) -> bool:
        return self.contains(key)

    def __getitem__(self, key: str):
        """
        Support bracket notation: pos_index[word]
        Used in ranking.py when fetching posting lists.
        Raises KeyError if word not found — same behaviour as dict.
        """
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self):
        """
        Support: for word in pos_index
        and:     if word in pos_index  (via __contains__)
        Returns a generator of all keys in the table.
        """
        return self.keys()


# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — LINEAR PROBING HASH TABLE
# ═════════════════════════════════════════════════════════════════════

_DELETED = object()   # sentinel — marks a slot as "was deleted"


class HashTableProbing:
    """
    Hash Table with Linear Probing for collision resolution.

    STRUCTURE:
    ──────────
    A single flat array. Each slot holds a (key, value) pair or None.
    When a collision occurs, we probe forward one slot at a time
    until we find an empty slot.

    VISUAL EXAMPLE (capacity = 7):

    Index: 0       1         2        3      4       5       6
           None  "neural"  "learn"   None  "deep"  "model"  None
                 {...}     {...}           {...}    {...}

    If "network" also hashes to index 1:
      → index 1 is taken → try index 2 → taken → try index 3 → empty!
      → store "network" at index 3

    Index: 0       1         2        3         4       5       6
           None  "neural"  "learn"  "network"  "deep"  "model"  None

    WHY LINEAR PROBING?
    ───────────────────
    All data lives in one contiguous array → cache-friendly.
    Modern CPUs fetch memory in cache lines — nearby data loads fast.
    No linked list node overhead — each entry is just one array slot.

    PROBLEM — Clustering:
    ─────────────────────
    When many keys hash nearby, they form a "cluster" of consecutive
    filled slots. New insertions probe through the whole cluster.
    This is called PRIMARY CLUSTERING — the main weakness of linear probing.

    LOAD FACTOR LIMIT:
    ──────────────────
    Must keep load factor below 0.7 (70% full).
    Above 0.7, clustering gets severe and performance degrades fast.
    When load factor exceeds threshold → REHASH (double the array size).

    TIME COMPLEXITY:
      Insert : O(1) average with low load factor
      Search : O(1) average with low load factor
      Both degrade to O(n) when heavily loaded

    SPACE: O(capacity) — one flat array, no extra nodes
    """

    _LOAD_FACTOR_THRESHOLD = 0.7   # rehash when this full

    def __init__(self, capacity: int = 16):
        self._capacity = capacity
        # DS: two parallel arrays — keys and values
        self._keys:   list = [None] * capacity
        self._values: list = [None] * capacity
        self._size    = 0

    def _hash(self, key: str) -> int:
        """Same polynomial rolling hash as chaining version."""
        h = 0
        for char in key:
            h = (h * 31 + ord(char)) % self._capacity
        return h

    def _rehash(self) -> None:
        """
        Double the array size and reinsert all existing entries.

        WHY REHASH?
        ───────────
        As the table fills up, probing sequences get longer.
        Doubling the capacity halves the load factor instantly,
        restoring O(1) average performance.

        Process:
          1. Save old arrays
          2. Create new arrays of double the size
          3. Reinsert every existing key-value pair
             (keys must be re-hashed — new capacity changes slot positions)

        Time: O(n) — must reinsert all n entries
        This is amortised O(1) per insertion overall.
        """
        old_keys   = self._keys
        old_values = self._values

        self._capacity *= 2
        self._keys   = [None] * self._capacity
        self._values = [None] * self._capacity
        self._size   = 0

        for i in range(len(old_keys)):
            if old_keys[i] is not None and old_keys[i] is not _DELETED:
                self.put(old_keys[i], old_values[i])

    def put(self, key: str, value) -> None:
        """
        Insert or update key-value pair using linear probing.

        Steps:
          1. Check load factor → rehash if needed
          2. Compute hash → starting index
          3. Probe forward until empty slot or matching key found
          4. Insert or update

        PROBING SEQUENCE: index, index+1, index+2, ... (wraps around)

        Time: O(1) average (with rehashing keeping load factor low)
        """
        if self._size / self._capacity >= self._LOAD_FACTOR_THRESHOLD:
            self._rehash()

        index = self._hash(key)

        while True:
            slot_key = self._keys[index]

            if slot_key is None or slot_key is _DELETED:
                # Empty or previously deleted slot → insert here
                self._keys[index]   = key
                self._values[index] = value
                self._size += 1
                return

            if slot_key == key:
                # Key already exists → update value
                self._values[index] = value
                return

            # Slot taken by different key → probe next slot
            # Wrap around using modulo
            index = (index + 1) % self._capacity

    def get(self, key: str):
        """
        Retrieve value for key using linear probing.

        Must follow the same probe sequence used during insertion.
        Stop when: key found, or empty slot (key was never inserted).
        DELETED sentinel slots are skipped — key might be further ahead.

        Time: O(1) average
        """
        index = self._hash(key)

        while True:
            slot_key = self._keys[index]

            if slot_key is None:
                return None   # empty slot → key definitely not here

            if slot_key == key:
                return self._values[index]

            # _DELETED or different key → keep probing
            index = (index + 1) % self._capacity

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self):
        for i in range(self._capacity):
            if self._keys[i] is not None and self._keys[i] is not _DELETED:
                yield self._keys[i]

    def items(self):
        for i in range(self._capacity):
            if self._keys[i] is not None and self._keys[i] is not _DELETED:
                yield self._keys[i], self._values[i]

    def load_factor(self) -> float:
        return self._size / self._capacity

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: str) -> bool:
        return self.contains(key)


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — WHY WE CHOOSE CHAINING FOR THE INVERTED INDEX
# ═════════════════════════════════════════════════════════════════════
#
# SEPARATE CHAINING wins for our use case because:
#
# 1. UNPREDICTABLE KEY COUNT
#    We don't know how many unique ML words exist before scraping.
#    Chaining handles any number without degrading — chains just grow.
#    Probing MUST rehash when 70% full — risky with unknown key count.
#
# 2. LOAD FACTOR > 1 IS FINE
#    Chaining works correctly even with load factor of 2, 3, or more.
#    Each slot's chain simply holds more nodes.
#    Probing breaks down above 0.7 due to clustering.
#
# 3. VALUES ARE LARGE DICTS THEMSELVES
#    Each value is a dict{doc_id → list[positions]}.
#    Separate chaining stores each value in a Node — flexible size.
#    Linear probing stores values in a fixed-size flat array — wasteful.
#
# 4. NO CLUSTERING PROBLEM
#    Probing suffers from primary clustering — long probe sequences
#    when nearby slots are all taken.
#    Chaining has no clustering — each slot's chain is independent.
#
# WHEN WOULD PROBING WIN?
#    Small fixed-size tables with simple values (integers).
#    Cache-critical code where memory locality matters more than flexibility.
#    Our inverted index does NOT fit this profile.
#
# ═════════════════════════════════════════════════════════════════════


INDEX_PATH = "index.json"
META_PATH  = "meta.json"


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD INDEX USING HashTableChaining
# ═════════════════════════════════════════════════════════════════════

def build_index(documents: list[dict]) -> tuple:
    """
    Build the inverted index from a list of document dicts.
    Uses HashTableChaining as the core data structure.

    DATA FLOW:
    ──────────
    documents (list[dict])
        ↓  for each doc
    tokens = preprocess(doc["text"])
        ↓  for each (position, token)
    positional_ht.put(token, {})        ← HashTableChaining insert
    positional_ht.get(token)[doc_id]    ← HashTableChaining lookup
        .append(position)               ← list append

    FINAL STRUCTURE inside HashTableChaining:
      key   = stemmed word (str)
      value = plain dict { doc_id → list[int] }
              (inner dict stays as Python dict — one level of
               custom hash table is enough to demonstrate the concept)

    Args:
      documents : list of dicts with keys id/title/text/url/type/source/image

    Returns:
      positional_ht  HashTableChaining  { word → { doc_id → list[int] } }
      tf_index       dict               { word → { doc_id → float } }
      doc_lengths    dict               { doc_id → int }
      meta_store     dict               { doc_id → metadata dict }
      total_docs     int
    """

    # ── Core index — our hand-built hash table ────────────────────────
    # DS: HashTableChaining
    # key   = stemmed/processed word
    # value = dict { doc_id → list[int] of positions }
    positional_ht = HashTableChaining(capacity=2048)

    # ── Supporting structures (plain dicts) ───────────────────────────
    doc_lengths: dict[str, int]  = {}
    meta_store:  dict[str, dict] = {}

    total_docs = len(documents)
    print(f"  Building index from {total_docs} documents...")

    for doc in documents:
        doc_id = doc["id"]

        # Save metadata separately
        meta_store[doc_id] = {
            "title"  : doc.get("title",  ""),
            "url"    : doc.get("url",    ""),
            "type"   : doc.get("type",   "article"),
            "source" : doc.get("source", ""),
            "image"  : doc.get("image",  ""),
            "snippet": doc.get("text",   "")[:300],
        }

        tokens = preprocess(doc.get("text", ""), use_bigrams=True)
        doc_lengths[doc_id] = len(tokens)

        for position, word in enumerate(tokens):

            # ── HashTableChaining.get() — O(1) average ────────────────
            posting = positional_ht.get(word)

            if posting is None:
                # First time seeing this word → create its posting dict
                # HashTableChaining.put() — O(1) average
                positional_ht.put(word, {doc_id: [position]})
            else:
                # Word already in table → update its posting dict
                if doc_id not in posting:
                    posting[doc_id] = []
                posting[doc_id].append(position)
                # No need to put() again — we mutated the value in-place

    # ── Build TF index from the hash table ───────────────────────────
    # Iterate all entries: O(capacity + n)
    tf_index: dict[str, dict[str, float]] = {}

    for word, posting in positional_ht.items():
        tf_index[word] = {}
        for doc_id, positions in posting.items():
            raw_count            = len(positions)
            length               = doc_lengths.get(doc_id, 1)
            tf_index[word][doc_id] = raw_count / length   # normalised TF

    unique_terms = len(tf_index)
    lf           = positional_ht.load_factor()
    print(f"  Index built: {unique_terms} unique terms, {total_docs} docs.")
    print(f"  Hash table load factor: {lf:.3f}  "
          f"({positional_ht._capacity} slots, {len(positional_ht)} entries)")

    return positional_ht, tf_index, doc_lengths, meta_store, total_docs


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — SAVE AND LOAD (JSON serialisation)
# ═════════════════════════════════════════════════════════════════════

def save_index(positional_ht, tf_index, doc_lengths, meta_store, total_docs,
               path=INDEX_PATH):
    """
    Serialise all index structures to JSON files on disk.

    HashTableChaining is converted to a plain dict for JSON serialisation
    because JSON does not understand custom objects.
    On load, we rebuild the hash table from the plain dict.
    """
    # Convert HashTableChaining → plain dict for JSON
    pos_dict = {}
    for word, posting in positional_ht.items():
        pos_dict[word] = posting

    payload = {
        "pos"   : pos_dict,
        "tf"    : tf_index,
        "dl"    : doc_lengths,
        "total" : total_docs,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_store, f, ensure_ascii=False, indent=2)

    print(f"  Index saved → {path}")
    print(f"  Metadata saved → {META_PATH}")


def load_index(path=INDEX_PATH):
    """
    Load index from disk and reconstruct the HashTableChaining.

    Steps:
      1. Read JSON → plain dict
      2. Insert each entry into a new HashTableChaining
      3. Return reconstructed hash table alongside other structures
    """
    if not os.path.exists(path) or not os.path.exists(META_PATH):
        return None, None, None, None, None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_store = json.load(f)

    # Reconstruct HashTableChaining from saved plain dict
    pos_dict      = data["pos"]
    positional_ht = HashTableChaining(capacity=max(2048, len(pos_dict) * 2))

    for word, posting in pos_dict.items():
        positional_ht.put(word, posting)

    print(f"  Index loaded ({data['total']} docs, "
          f"{len(positional_ht)} terms in hash table).")

    return (
        positional_ht,
        data["tf"],
        data["dl"],
        meta_store,
        data["total"],
    )


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — DEMO (run this file directly to see both tables)
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HASH TABLE DEMONSTRATION — SEPARATE CHAINING vs LINEAR PROBING")
    print("="*60)

    sample_words = [
        ("neural",     {"wiki_001": [4, 17]}),
        ("network",    {"wiki_001": [5, 18], "arxiv_003": [2]}),
        ("learn",      {"wiki_001": [9], "tds_002": [3, 11]}),
        ("deep",       {"arxiv_003": [1, 7]}),
        ("gradient",   {"wiki_002": [6]}),
        ("descent",    {"wiki_002": [7]}),
        ("transform",  {"arxiv_005": [0, 4, 9]}),
        ("attention",  {"arxiv_005": [5]}),
    ]

    # ── Demo 1: Separate Chaining ─────────────────────────────────────
    print("\n  [1] SEPARATE CHAINING (capacity=7 for demo):")
    ht_chain = HashTableChaining(capacity=7)

    for word, posting in sample_words:
        ht_chain.put(word, posting)
        slot = ht_chain._hash(word)
        print(f"    put('{word:12s}')  →  slot {slot}  "
              f"  load={ht_chain.load_factor():.2f}")

    print(f"\n    Total entries : {len(ht_chain)}")
    print(f"    Load factor   : {ht_chain.load_factor():.2f}  "
          f"(> 1.0 is fine for chaining!)")
    print(f"\n    get('neural') → {ht_chain.get('neural')}")
    print(f"    get('xyz')    → {ht_chain.get('xyz')}")

    print("\n    Slot contents (showing chain lengths):")
    for i, bucket in enumerate(ht_chain._buckets):
        chain = []
        node  = bucket
        while node:
            chain.append(node.key)
            node = node.next
        if chain:
            print(f"      Slot {i}: {' → '.join(chain)}")

    # ── Demo 2: Linear Probing ────────────────────────────────────────
    print("\n  [2] LINEAR PROBING (capacity=16):")
    ht_probe = HashTableProbing(capacity=16)

    for word, posting in sample_words:
        ideal_slot  = ht_probe._hash(word)
        ht_probe.put(word, posting)
        # find where it actually landed
        actual_slot = None
        for i in range(ht_probe._capacity):
            if ht_probe._keys[i] == word:
                actual_slot = i
                break
        collision = " ← COLLISION PROBED" if actual_slot != ideal_slot else ""
        print(f"    put('{word:12s}')  →  ideal={ideal_slot:2d}  "
              f"actual={actual_slot:2d}{collision}")

    print(f"\n    Total entries : {len(ht_probe)}")
    print(f"    Load factor   : {ht_probe.load_factor():.2f}  "
          f"(must stay below 0.7)")

    # ── Comparison table ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    rows = [
        ("Collision handling", "Linked list per slot",   "Probe next slot"),
        ("Load factor limit",  "Can exceed 1.0",         "Must stay < 0.7"),
        ("Memory layout",      "Scattered (nodes)",      "Contiguous array"),
        ("Cache performance",  "Moderate",               "Better (locality)"),
        ("Clustering issue",   "None",                   "Primary clustering"),
        ("Best for",           "Unknown key count",      "Known fixed keys"),
        ("Used in our index",  "YES",                    "No"),
    ]
    print(f"  {'Property':<22} {'Separate Chaining':<28} {'Linear Probing'}")
    print("  " + "-"*58)
    for prop, chain_val, probe_val in rows:
        print(f"  {prop:<22} {chain_val:<28} {probe_val}")
    print("="*60 + "\n")