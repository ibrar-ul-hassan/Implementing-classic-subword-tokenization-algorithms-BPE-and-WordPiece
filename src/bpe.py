"""
BPE (Byte Pair Encoding) Trainer and Tokenizer
Reference: Sennrich et al. (2016)

Two implementations:
  train()      — naive version (simple, educational, slow)
  train_fast() — optimized version (priority queue + inverse index)
"""
from collections import defaultdict
import heapq
import json
import time


# ============================================================
# SHARED UTILITIES
# ============================================================

def get_vocab(word_freq):
    """
    Convert word frequency dict into character-split representation.
    Example: "hug": 10  →  ("h","u","g"): 10
    """
    return {tuple(word): freq for word, freq in word_freq.items()}


def tokenize_word(word, merge_rules):
    """
    Tokenize a single word by replaying merge rules in order.

    This is identical for both naive and fast BPE —
    the tokenizer only needs the final merge_rules list.

    Ambiguity resolution: rule ORDER resolves all ambiguity.
    Earlier rules always take priority.
    """
    tokens = list(word)
    for (a, b) in merge_rules:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and \
               tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def tokenize(text, merge_rules):
    """Tokenize a full sentence word by word."""
    tokens = []
    for word in text.lower().split():
        tokens.extend(tokenize_word(word, merge_rules))
    return tokens


def save(merge_rules, bpe_vocab, merge_rules_path, vocab_path):
    """Save merge rules and vocabulary to disk."""
    with open(merge_rules_path, 'w', encoding='utf-8') as f:
        json.dump([list(p) for p in merge_rules],
                  f, ensure_ascii=False, indent=2)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(bpe_vocab)),
                  f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(merge_rules):,} merge rules → {merge_rules_path}")
    print(f"✅ Saved {len(bpe_vocab):,} tokens     → {vocab_path}")


def load(merge_rules_path, vocab_path):
    """Load merge rules and vocabulary from disk."""
    with open(merge_rules_path, 'r', encoding='utf-8') as f:
        merge_rules = [tuple(p) for p in json.load(f)]
    with open(vocab_path, 'r', encoding='utf-8') as f:
        bpe_vocab = set(json.load(f))
    print(f"✅ Loaded {len(merge_rules):,} merge rules")
    print(f"✅ Loaded {len(bpe_vocab):,} vocab tokens")
    return merge_rules, bpe_vocab


# ============================================================
# NAIVE BPE TRAINER
# Simple to understand — rescans full vocab every step
# ============================================================

def _get_pair_counts(vocab):
    """Count frequency of every adjacent pair across all words."""
    pair_counts = defaultdict(int)
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pair_counts[(word_tokens[i], word_tokens[i + 1])] += freq
    return pair_counts


def _merge_pair(best_pair, vocab):
    """Apply one merge rule to the entire vocabulary."""
    a, b = best_pair
    merged = a + b
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
    NAIVE BPE Training.

    How it works:
    1. Split every word into characters
    2. Count all adjacent pairs across corpus
    3. Merge the most frequent pair
    4. Repeat from step 2

    Bottleneck: step 2 rescans the ENTIRE vocabulary every iteration.
    Time complexity: O(V × N) where V = vocab size, N = num merges

    Args:
        word_freq:  {word: frequency}
        vocab_size: target vocabulary size
        verbose:    print progress

    Returns:
        vocab, merge_rules, bpe_vocab
    """
    start_time = time.time()
    vocab = get_vocab(word_freq)

    # Build initial character vocabulary
    bpe_vocab = set()
    for word_tokens in vocab.keys():
        for token in word_tokens:
            bpe_vocab.add(token)

    initial_size = len(bpe_vocab)
    num_merges   = vocab_size - initial_size
    merge_rules  = []

    if verbose:
        print(f"[NAIVE BPE] Initial vocab: {initial_size} | "
              f"Target: {vocab_size} | Merges: {num_merges}")

    for step in range(num_merges):
        # ← THIS is the bottleneck: full rescan every step
        pair_counts = _get_pair_counts(vocab)

        if not pair_counts:
            break

        best_pair  = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        if best_count < 2:
            break

        new_token = best_pair[0] + best_pair[1]
        merge_rules.append(best_pair)
        bpe_vocab.add(new_token)
        vocab = _merge_pair(best_pair, vocab)

        if verbose and (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step+1:,}/{num_merges:,} | "
                  f"'{best_pair[0]}'+'{best_pair[1]}'→'{new_token}' "
                  f"(freq:{best_count:,}) | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ NAIVE BPE done in {elapsed:.2f}s | "
              f"Vocab: {len(bpe_vocab):,} | "
              f"Rules: {len(merge_rules):,}")

    return vocab, merge_rules, bpe_vocab


# ============================================================
# FAST BPE TRAINER
# Uses priority queue + inverse index for efficiency
# ============================================================

def _build_inverse_index(vocab):
    """
    Build inverse index: pair → set of words containing that pair.

    This is the key data structure for fast BPE.
    Instead of scanning ALL words to find where a pair appears,
    we can look it up directly in O(1).

    Example:
        vocab has words: ("h","u","g"), ("p","u","g"), ("b","u","g")
        inverse_index[("u","g")] = {
            ("h","u","g"), ("p","u","g"), ("b","u","g")
        }
    """
    inverse_index = defaultdict(set)
    for word_tokens in vocab.keys():
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            inverse_index[pair].add(word_tokens)
    return inverse_index


def _get_pair_counts_for_words(words, vocab):
    """
    Count pairs only for a specific set of words.
    Used to update counts after a merge — only recount affected words.
    """
    pair_counts = defaultdict(int)
    for word_tokens in words:
        freq = vocab.get(word_tokens, 0)
        for i in range(len(word_tokens) - 1):
            pair_counts[(word_tokens[i], word_tokens[i + 1])] += freq
    return pair_counts


def train_fast(word_freq, vocab_size, verbose=True):
    """
    FAST BPE Training using Priority Queue + Inverse Index.

    Key optimizations over naive:

    1. PRIORITY QUEUE (max-heap):
       Always get the best pair in O(log n)
       instead of scanning all pairs in O(n)

    2. INVERSE INDEX:
       pair → {words containing this pair}
       After a merge, only recount pairs in AFFECTED words
       instead of rescanning the entire vocabulary

    3. LAZY DELETION:
       Instead of removing stale entries from the heap
       (expensive), we mark them as invalid and skip them
       when we pop them off the heap.

    Time complexity: O(M × W × L)
    Where M = merges, W = avg words per pair, L = avg word length
    Much faster than naive O(M × V) in practice.

    Args:
        word_freq:  {word: frequency}
        vocab_size: target vocabulary size
        verbose:    print progress

    Returns:
        vocab, merge_rules, bpe_vocab
    """
    start_time = time.time()
    vocab = get_vocab(word_freq)

    # Build initial character vocabulary
    bpe_vocab = set()
    for word_tokens in vocab.keys():
        for token in word_tokens:
            bpe_vocab.add(token)

    initial_size = len(bpe_vocab)
    num_merges   = vocab_size - initial_size
    merge_rules  = []

    if verbose:
        print(f"[FAST BPE] Initial vocab: {initial_size} | "
              f"Target: {vocab_size} | Merges: {num_merges}")

    # ── Step 1: Count all pairs once ──
    pair_counts   = defaultdict(int)
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pair_counts[(word_tokens[i], word_tokens[i + 1])] += freq

    # ── Step 2: Build inverse index ──
    inverse_index = _build_inverse_index(vocab)

    # ── Step 3: Build max-heap ──
    # Python's heapq is a MIN-heap, so we negate counts
    # Entry format: (-count, pair)
    # We use a dict to track the "current valid count" for each pair
    # This enables lazy deletion of stale heap entries
    heap = []
    for pair, count in pair_counts.items():
        heapq.heappush(heap, (-count, pair))

    # ── Main training loop ──
    for step in range(num_merges):

        # Find the best valid pair using lazy deletion
        best_pair  = None
        best_count = 0

        while heap:
            neg_count, pair = heapq.heappop(heap)
            current_count   = pair_counts.get(pair, 0)

            # Lazy deletion: skip if this entry is stale
            # (the count in the heap no longer matches reality)
            if -neg_count != current_count:
                continue

            if current_count < 2:
                break

            best_pair  = pair
            best_count = current_count
            break

        if best_pair is None:
            break

        # Create new merged token
        new_token = best_pair[0] + best_pair[1]
        merge_rules.append(best_pair)
        bpe_vocab.add(new_token)

        # ── Step 4: Find only affected words ──
        affected_words = inverse_index.get(best_pair, set()).copy()

        # ── Step 5: Update counts for affected words only ──
        # Remove old pair counts for affected words
        for word_tokens in affected_words:
            old_freq = vocab.get(word_tokens, 0)
            for i in range(len(word_tokens) - 1):
                old_pair = (word_tokens[i], word_tokens[i + 1])
                pair_counts[old_pair] -= old_freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            # Also update inverse index
            for i in range(len(word_tokens) - 1):
                old_pair = (word_tokens[i], word_tokens[i + 1])
                inverse_index[old_pair].discard(word_tokens)

        # ── Step 6: Apply merge to affected words only ──
        a, b = best_pair
        new_vocab_entries = {}
        for word_tokens in affected_words:
            freq = vocab.pop(word_tokens, 0)
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and \
                   word_tokens[i] == a and word_tokens[i + 1] == b:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_key = tuple(new_tokens)
            # Accumulate in case multiple old forms map to same new form
            new_vocab_entries[new_key] = \
                new_vocab_entries.get(new_key, 0) + freq

        # ── Step 7: Add new entries, update counts + index ──
        for new_key, freq in new_vocab_entries.items():
            vocab[new_key] = vocab.get(new_key, 0) + freq
            for i in range(len(new_key) - 1):
                new_pair = (new_key[i], new_key[i + 1])
                pair_counts[new_pair] = \
                    pair_counts.get(new_pair, 0) + freq
                inverse_index[new_pair].add(new_key)
                # Push updated count to heap
                heapq.heappush(heap, (-pair_counts[new_pair], new_pair))

        if verbose and (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step+1:,}/{num_merges:,} | "
                  f"'{a}'+'{b}'→'{new_token}' "
                  f"(freq:{best_count:,}) | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ FAST BPE done in {elapsed:.2f}s | "
              f"Vocab: {len(bpe_vocab):,} | "
              f"Rules: {len(merge_rules):,}")

    return vocab, merge_rules, bpe_vocab
