"""
WordPiece Trainer and Tokenizer
Reference: Schuster & Nakajima (2012), used in BERT
"""
from collections import defaultdict
import json


def get_vocab(word_freq):
    """Convert to ## prefixed character representation."""
    vocab = {}
    for word, freq in word_freq.items():
        chars = tuple(
            c if i == 0 else f"##{c}"
            for i, c in enumerate(word)
        )
        vocab[chars] = freq
    return vocab


def compute_pair_scores(vocab):
    """
    Compute WordPiece score for every adjacent pair.
    score(A,B) = freq(AB) / (freq(A) * freq(B))
    """
    letter_freqs = defaultdict(int)
    pair_freqs   = defaultdict(int)

    for word_tokens, freq in vocab.items():
        if len(word_tokens) == 1:
            letter_freqs[word_tokens[0]] += freq
            continue
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i+1])
            letter_freqs[word_tokens[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[word_tokens[-1]] += freq

    return {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }


def merge_pair(best_pair, vocab):
    """Apply one WordPiece merge — strips ## when merging."""
    a, b = best_pair
    merged = a + b[2:] if b.startswith("##") else a + b
    new_vocab = {}
    for word_tokens, freq in vocab.items():
        new_tokens = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens)-1 and \
               word_tokens[i] == a and word_tokens[i+1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab


def train(word_freq, vocab_size, verbose=True):
    """
    Train WordPiece algorithm.
    Returns: (vocab, wp_vocab, merge_log)
    """
    vocab = get_vocab(word_freq)

    wp_vocab = set()
    for word_tokens in vocab.keys():
        for token in word_tokens:
            wp_vocab.add(token)

    special = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
    for t in special:
        wp_vocab.add(t)

    initial_size = len(wp_vocab)
    num_merges   = vocab_size - initial_size
    merge_log    = []

    if verbose:
        print(f"WordPiece Training | Initial vocab: {initial_size} | "
              f"Target: {vocab_size} | Merges: {num_merges}")

    for step in range(num_merges):
        scores = compute_pair_scores(vocab)
        if not scores:
            break

        best_pair  = max(scores, key=scores.get)
        best_score = scores[best_pair]
        a, b       = best_pair
        merged     = a + b[2:] if b.startswith("##") else a + b

        merge_log.append((best_pair, best_score, merged))
        wp_vocab.add(merged)
        vocab = merge_pair(best_pair, vocab)

        if verbose and (step+1) % 200 == 0:
            print(f"  Step {step+1:,}/{num_merges:,} | "
                  f"'{a}'+'{b}'→'{merged}' "
                  f"(score:{best_score:.6f})")

    if verbose:
        print(f"✅ WordPiece done | Vocab: {len(wp_vocab):,} | "
              f"Merges: {len(merge_log):,}")

    return vocab, wp_vocab, merge_log


def tokenize_word(word, wp_vocab):
    """Tokenize using longest-match-first on vocabulary."""
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
        json.dump(sorted(list(wp_vocab)), f, ensure_ascii=False, indent=2)
    with open(merge_log_path, 'w', encoding='utf-8') as f:
        json.dump(
            [{"pair": list(p), "score": s, "merged": m}
             for p, s, m in merge_log],
            f, ensure_ascii=False, indent=2
        )
    print(f"✅ Saved {len(wp_vocab):,} tokens   → {vocab_path}")
    print(f"✅ Saved {len(merge_log):,} log entries → {merge_log_path}")


def load(vocab_path):
    """Load vocabulary from disk."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        wp_vocab = set(json.load(f))
    print(f"✅ Loaded {len(wp_vocab):,} vocab tokens")
    return wp_vocab
