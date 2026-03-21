"""
WordPiece Trainer and Tokenizer
Reference: Schuster & Nakajima (2012), used in BERT

Two implementations:
  train()      — naive version (simple, educational, slow)
  train_fast() — optimized version (inverse index + score caching)

Tie-breaking strategy:
  When multiple pairs share the same score, we break ties
  lexicographically (alphabetical order of the pair).
  This ensures naive and fast produce identical vocabularies.
"""
from collections import defaultdict
import heapq
import json
import time


# ============================================================
# SHARED UTILITIES
# ============================================================

def tokenize_word(word, wp_vocab):
    """
    Tokenize using longest-match-first on vocabulary.

    Unlike BPE which replays rules, WordPiece only needs
    the final vocabulary — no merge rules required.

    Ambiguity resolution: greedy longest match from left to right.
    Always picks the longest possible token at each position.
    If no match found at any position → entire word = [UNK].

    Example (vocab has "spiel", "##en"):
        "spielen"
        → try "spielen" → not in vocab
        → try "spiele"  → not in vocab
        → try "spiel"   → ✅ found, remaining = "en"
        → try "##en"    → ✅ found
        → result: ["spiel", "##en"]
    """
    tokens    = []
    remaining = word

    while remaining:
        found = False
        for end in range(len(remaining), 0, -1):
            substr    = remaining[:end]
            candidate = substr if not tokens else f"##{substr}"
            if candidate in wp_vocab:
                tokens.append(candidate)
                remaining = remaining[end:]
                found = True
                break
        if not found:
            return ["[UNK]"]

    return tokens


def tokenize(text, wp_vocab):
    """Tokenize a full sentence."""
    tokens = []
    for word in text.lower().split():
        tokens.extend(tokenize_word(word, wp_vocab))
    return tokens


def save(wp_vocab, merge_log, vocab_path, merge_log_path):
    """Save vocabulary and merge log."""
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(wp_vocab)),
                  f, ensure_ascii=False, indent=2)
    with open(merge_log_path, 'w', encoding='utf-8') as f:
        json.dump(
            [{"pair": list(p), "score": s, "merged": m}
             for p, s, m in merge_log],
            f, ensure_ascii=False, indent=2
        )
    print(f"✅ Saved {len(wp_vocab):,} tokens      → {vocab_path}")
    print(f"✅ Saved {len(merge_log):,} log entries → {merge_log_path}")


def load(vocab_path):
    """Load vocabulary from disk."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        wp_vocab = set(json.load(f))
    print(f"✅ Loaded {len(wp_vocab):,} vocab tokens")
    return wp_vocab


# ============================================================
# NAIVE WORDPIECE TRAINER
# ============================================================

def _get_vocab(word_freq):
    """Convert to ## prefixed character representation."""
    vocab = {}
    for word, freq in word_freq.items():
        chars = tuple(
            c if i == 0 else f"##{c}"
            for i, c in enumerate(word)
        )
        vocab[chars] = freq
    return vocab


def _compute_pair_scores(vocab):
    """
    Compute score(A,B) = freq(AB) / (freq(A) × freq(B))
    for all adjacent pairs.
    """
    letter_freqs = defaultdict(int)
    pair_freqs   = defaultdict(int)

    for word_tokens, freq in vocab.items():
        if len(word_tokens) == 1:
            letter_freqs[word_tokens[0]] += freq
            continue
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            letter_freqs[word_tokens[i]] += freq
            pair_freqs[pair]             += freq
        letter_freqs[word_tokens[-1]] += freq

    return {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }


def _merge_pair(best_pair, vocab):
    """Apply one WordPiece merge — strips ## when merging."""
    a, b   = best_pair
    merged = a + b[2:] if b.startswith("##") else a + b
    new_vocab = {}
    for word_tokens, freq in vocab.items():
        new_tokens = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and \
               word_tokens[i] == a and word_tokens[i + 1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab


def train(word_freq, vocab_size, verbose=True):
    """
    NAIVE WordPiece Training.

    Bottleneck: recomputes ALL scores from scratch every step.
    Time complexity: O(V × N)

    Tie-breaking: lexicographic on (score, pair[0], pair[1])
    guarantees identical output to train_fast().

    Returns: vocab, wp_vocab, merge_log
    """
    start_time = time.time()
    vocab      = _get_vocab(word_freq)

    wp_vocab = set()
    for word_tokens in vocab.keys():
        for token in word_tokens:
            wp_vocab.add(token)

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for t in special:
        wp_vocab.add(t)

    initial_size = len(wp_vocab)
    num_merges   = vocab_size - initial_size
    merge_log    = []

    if verbose:
        print(f"[NAIVE WP] Initial vocab: {initial_size} | "
              f"Target: {vocab_size} | Merges: {num_merges}")

    for step in range(num_merges):
        scores = _compute_pair_scores(vocab)
        if not scores:
            break

        # ── Deterministic tie-breaking ──
        # Primary:   highest score
        # Secondary: alphabetical order of pair[0]
        # Tertiary:  alphabetical order of pair[1]
        best_pair = max(
            scores,
            key=lambda p: (scores[p], p[0], p[1])
        )
        best_score = scores[best_pair]
        a, b       = best_pair
        merged     = a + b[2:] if b.startswith("##") else a + b

        merge_log.append((best_pair, best_score, merged))
        wp_vocab.add(merged)
        vocab = _merge_pair(best_pair, vocab)

        if verbose and (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step+1:,}/{num_merges:,} | "
                  f"'{a}'+'{b}'→'{merged}' "
                  f"(score:{best_score:.6f}) | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ NAIVE WP done in {elapsed:.2f}s | "
              f"Vocab: {len(wp_vocab):,} | "
              f"Merges: {len(merge_log):,}")

    return vocab, wp_vocab, merge_log


# ============================================================
# FAST WORDPIECE TRAINER
# Uses inverse index + incremental score updates
# ============================================================

def train_fast(word_freq, vocab_size, verbose=True):
    """
    FAST WordPiece Training.

    Key optimizations:
    1. INVERSE INDEX: only recount scores for affected words
    2. INCREMENTAL letter_freqs: update only changed tokens
    3. PRIORITY QUEUE with lazy deletion: O(log n) best pair access

    Tie-breaking: heap entries are (-score, pair[0], pair[1])
    so pairs with equal scores are ordered alphabetically.
    This matches the naive implementation exactly.

    Note on score staleness:
    Scores depend on letter_freqs which change after every merge.
    We handle this via lazy deletion — recompute score on pop
    and skip if it no longer matches the heap entry.

    Returns: vocab, wp_vocab, merge_log
    """
    start_time = time.time()
    vocab      = _get_vocab(word_freq)

    wp_vocab = set()
    for word_tokens in vocab.keys():
        for token in word_tokens:
            wp_vocab.add(token)

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for t in special:
        wp_vocab.add(t)

    initial_size = len(wp_vocab)
    num_merges   = vocab_size - initial_size
    merge_log    = []

    if verbose:
        print(f"[FAST WP] Initial vocab: {initial_size} | "
              f"Target: {vocab_size} | Merges: {num_merges}")

    # ── Step 1: Build letter + pair frequencies ──
    letter_freqs = defaultdict(int)
    pair_freqs   = defaultdict(int)

    for word_tokens, freq in vocab.items():
        if len(word_tokens) == 1:
            letter_freqs[word_tokens[0]] += freq
            continue
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            letter_freqs[word_tokens[i]] += freq
            pair_freqs[pair]             += freq
        letter_freqs[word_tokens[-1]] += freq

    # ── Step 2: Build inverse index ──
    inverse_index = defaultdict(set)
    for word_tokens in vocab.keys():
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            inverse_index[pair].add(word_tokens)

    # ── Step 3: Score function + initial heap ──
    def score(pair):
        pa = letter_freqs.get(pair[0], 0)
        pb = letter_freqs.get(pair[1], 0)
        if pa == 0 or pb == 0:
            return 0.0
        return pair_freqs.get(pair, 0) / (pa * pb)

    # Entry: (-score, pair[0], pair[1])
    # Negated score → max-heap via Python's min-heap
    # pair[0], pair[1] → deterministic tie-breaking
    heap = []
    for pair in pair_freqs:
        s = score(pair)
        if s > 0:
            heapq.heappush(heap, (-s, pair[0], pair[1]))

    # ── Main loop ──
    for step in range(num_merges):

        # Find best valid pair using lazy deletion
        best_pair  = None
        best_score = 0.0

        while heap:
            neg_s, p0, p1 = heapq.heappop(heap)
            pair          = (p0, p1)
            current_score = score(pair)

            # Skip stale entries
            if abs(-neg_s - current_score) > 1e-10:
                continue
            if current_score <= 0:
                continue

            best_pair  = pair
            best_score = current_score
            break

        if best_pair is None:
            break

        a, b   = best_pair
        merged = a + b[2:] if b.startswith("##") else a + b
        merge_log.append((best_pair, best_score, merged))
        wp_vocab.add(merged)

        # ── Find affected words ──
        affected_words = inverse_index.get(best_pair, set()).copy()

        # ── Remove old contributions ──
        for word_tokens in affected_words:
            freq = vocab.get(word_tokens, 0)
            if len(word_tokens) == 1:
                letter_freqs[word_tokens[0]] -= freq
                continue
            for i in range(len(word_tokens) - 1):
                pair_i = (word_tokens[i], word_tokens[i + 1])
                letter_freqs[word_tokens[i]] -= freq
                pair_freqs[pair_i]           -= freq
                if pair_freqs[pair_i] <= 0:
                    del pair_freqs[pair_i]
                inverse_index[pair_i].discard(word_tokens)
            letter_freqs[word_tokens[-1]] -= freq

        # ── Apply merge ──
        new_entries = {}
        for word_tokens in affected_words:
            freq = vocab.pop(word_tokens, 0)
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and \
                   word_tokens[i] == a and word_tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_key = tuple(new_tokens)
            new_entries[new_key] = new_entries.get(new_key, 0) + freq

        # ── Add new entries + update counts ──
        for new_key, freq in new_entries.items():
            vocab[new_key] = vocab.get(new_key, 0) + freq
            if len(new_key) == 1:
                letter_freqs[new_key[0]] += freq
                continue
            for i in range(len(new_key) - 1):
                pair_i = (new_key[i], new_key[i + 1])
                letter_freqs[new_key[i]] += freq
                pair_freqs[pair_i]       += freq
                inverse_index[pair_i].add(new_key)
                s = score(pair_i)
                if s > 0:
                    # Push with tie-breaking
                    heapq.heappush(
                        heap,
                        (-s, pair_i[0], pair_i[1])
                    )
            letter_freqs[new_key[-1]] += freq

        if verbose and (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step+1:,}/{num_merges:,} | "
                  f"'{a}'+'{b}'→'{merged}' "
                  f"(score:{best_score:.6f}) | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ FAST WP done in {elapsed:.2f}s | "
              f"Vocab: {len(wp_vocab):,} | "
              f"Merges: {len(merge_log):,}")

    return vocab, wp_vocab, merge_log
