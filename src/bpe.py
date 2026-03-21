"""
BPE (Byte Pair Encoding) Trainer and Tokenizer
Reference: Sennrich et al. (2016)
"""
from collections import defaultdict
import json
import os


def get_vocab(word_freq):
    """Convert word frequency dict into character-split representation."""
    return {tuple(word): freq for word, freq in word_freq.items()}


def get_pair_counts(vocab):
    """Count frequency of every adjacent pair across all words."""
    pair_counts = defaultdict(int)
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pair_counts[(word_tokens[i], word_tokens[i+1])] += freq
    return pair_counts


def merge_pair(best_pair, vocab):
    """Apply one merge rule to the entire vocabulary."""
    a, b = best_pair
    merged = a + b
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
    Train BPE algorithm.
    Returns: (vocab, merge_rules, bpe_vocab)
    """
    vocab = get_vocab(word_freq)

    bpe_vocab = set()
    for word_tokens in vocab.keys():
        for token in word_tokens:
            bpe_vocab.add(token)

    initial_size = len(bpe_vocab)
    num_merges   = vocab_size - initial_size
    merge_rules  = []

    if verbose:
        print(f"BPE Training | Initial vocab: {initial_size} | "
              f"Target: {vocab_size} | Merges: {num_merges}")

    for step in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break

        best_pair  = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        if best_count < 2:
            break

        new_token = best_pair[0] + best_pair[1]
        merge_rules.append(best_pair)
        bpe_vocab.add(new_token)
        vocab = merge_pair(best_pair, vocab)

        if verbose and (step+1) % 200 == 0:
            print(f"  Step {step+1:,}/{num_merges:,} | "
                  f"'{best_pair[0]}'+'{best_pair[1]}'→'{new_token}' "
                  f"(freq:{best_count:,})")

    if verbose:
        print(f"✅ BPE done | Vocab: {len(bpe_vocab):,} | "
              f"Rules: {len(merge_rules):,}")

    return vocab, merge_rules, bpe_vocab


def tokenize_word(word, merge_rules):
    """Tokenize a single word using learned merge rules."""
    tokens = list(word)
    for (a, b) in merge_rules:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(a+b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def tokenize(text, merge_rules):
    """Tokenize a full sentence."""
    tokens = []
    for word in text.lower().split():
        tokens.extend(tokenize_word(word, merge_rules))
    return tokens


def save(merge_rules, bpe_vocab, merge_rules_path, vocab_path):
    """Save merge rules and vocabulary to disk."""
    with open(merge_rules_path, 'w', encoding='utf-8') as f:
        json.dump([list(p) for p in merge_rules], f,
                  ensure_ascii=False, indent=2)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(bpe_vocab)), f,
                  ensure_ascii=False, indent=2)
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
