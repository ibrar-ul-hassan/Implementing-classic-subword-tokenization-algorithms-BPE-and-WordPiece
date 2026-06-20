# Implementing Classic Subword Tokenization Algorithms: BPE and WordPiece

Building the engines of modern language models from scratch — two subword tokenizers, **Byte Pair Encoding (BPE)** and **WordPiece**, implemented in Python in both naive and efficient ("fast") forms, then compared scientifically on a German corpus.

**Team:** Ibrar · Dayo · Lama  
**Supervisor:** Marie Candito  
**Course:** NLP Project — Master 1, Université Paris Cité (2025–2026)

---

## What this project is

A neural network only understands numbers, so before any model can read text, that text must be cut into units (*tokens*) and each unit given a number. This project builds the two most widely used tokenizers that do this job — BPE (used by GPT, RoBERTa) and WordPiece (used by BERT) — **from scratch**, without using any ready-made tokenization library to do the actual work.

The project has three goals:

1. **Implement** both algorithms correctly, in a naive (slow, simple) form and an efficient (fast) form.
2. **Resolve tokenization ambiguity** with a principled strategy.
3. **Evaluate** the two algorithms scientifically against a morphological gold standard.

---

## Repository structure

```
src/
  config.py         # all paths and parameters in one place
  bpe.py            # BPE trainer (naive + fast) and tokenizer
  wordpiece.py      # WordPiece trainer (naive + fast) and tokenizer
00_setup.ipynb      # download + clean the corpus, build word-frequency dict
01_bpe.ipynb        # train BPE (naive + fast), verify they agree
02_wordpiece.ipynb  # train WordPiece (naive + fast), verify
03_experiments.ipynb# run all five experiments, save results + figures
data/
  experiment_results.json   # single source of truth for every reported number
  *.png                     # figures generated from the results
```

All algorithmic logic lives in `src/`. The notebooks only call those modules, so the algorithms stay readable and testable.

---

## User manual (how to run)

The project runs in **Google Colab** (the approved compute environment for the course).

### Requirements

- A Google account with access to Google Colab.
- A GitHub personal-access token stored in Colab **Secrets** under the name `GITHUB_1` (used to clone this repository and push results back).
- The only external Python dependency beyond the standard library is `datasets`, which `00_setup.ipynb` installs automatically. Figures use `matplotlib`, pre-installed in Colab.

### Steps

1. Open **`00_setup.ipynb`** and run all cells. This downloads and cleans a slice of German Wikipedia and builds the word-frequency dictionary. This is the slow step (a few minutes) and only needs to be done once per session.
2. Open **`01_bpe.ipynb`** and run all cells to train BPE (naive + fast) and verify the two implementations agree.
3. Open **`02_wordpiece.ipynb`** and run all cells to train WordPiece and run the same verification.
4. Open **`03_experiments.ipynb`** and run all cells to reproduce all five experiments, regenerate the figures, and save every result to `data/experiment_results.json`.

Each notebook starts with a single setup cell that clones or updates the repo and imports the modules. **If a Colab session restarts, just re-run that first cell and everything else works again.**

### Using the algorithms directly

Both modules expose a small, symmetric API:

```python
import bpe, wordpiece

# --- BPE ---
vocab, merge_rules, bpe_vocab = bpe.train(word_freq, vocab_size=10000)
vocab, merge_rules, bpe_vocab = bpe.train_fast(word_freq, vocab_size=10000)
tokens = bpe.tokenize_word("unbreakability", merge_rules)
tokens = bpe.tokenize("the cats are running", merge_rules)

# --- WordPiece ---
vocab, wp_vocab, merge_log = wordpiece.train(word_freq, vocab_size=10000)
vocab, wp_vocab, merge_log = wordpiece.train_fast(
    word_freq, vocab_size=10000, min_pair_freq=0   # optional frequency threshold
)
tokens = wordpiece.tokenize_word("spielen", wp_vocab)   # -> ['spiel', '##en']
tokens = wordpiece.tokenize("der hund läuft", wp_vocab)
```

Each function documents its arguments and return values in its docstring (the "online help").

---

## How the two algorithms differ

Both grow a vocabulary by merging pairs one at a time, but they choose *which* pair to merge differently:

- **BPE** merges the **most frequent** adjacent pair. Simple, robust, and it records an ordered list of merge rules that makes tokenization perfectly reproducible.
- **WordPiece** merges the pair with the highest **score = freq(AB) / (freq(A) × freq(B))**. This rewards *informative* pairs (pieces that appear together more than expected) rather than merely frequent ones. It marks word-continuation pieces with a `##` prefix.

**Ambiguity resolution.** BPE replays its merge rules in learned order, so the result is deterministic. WordPiece tokenizes by **greedy longest-match** on its vocabulary. When two pairs tie on count or score during training, both implementations break the tie **lexicographically**, so naive and fast make the same choices.

---

## The fast implementations

A naive trainer re-reads the whole corpus on every merge. The fast versions avoid this with:

- an **inverse index** (pair → words containing it), so only affected words are revisited;
- **incremental counts** (subtract the old contribution, add the new) instead of recounting;
- a **priority queue (heap)** with lazy deletion for O(log n) access to the best pair.

WordPiece's fast version is genuinely harder than BPE's, because its score has a **global denominator** that can change due to a merge elsewhere — which is exactly where one of our bugs lived (see below).

---

## Results summary

All numbers below are computed by `03_experiments.ipynb` and saved to `data/experiment_results.json`. Trained on the top-N most frequent words of German Wikipedia.

### 1. Training speed (naive vs fast)

| Corpus | BPE naive | BPE fast | WP naive | WP fast |
|---|---|---|---|---|
| 5,000 words | 23.8 s | 1.1 s | 51.5 s | 0.23 s |
| 10,000 words | 54.2 s | 1.4 s | 102.9 s | 0.38 s |
| 20,000 words | 109.9 s | 3.3 s | 235.5 s | 0.76 s |

The key result is the **shape**, not a single ratio: naive training time grows with the corpus while fast training time stays nearly constant. (Speed-up ratios vary with cloud-hardware timing noise, roughly 21–40× for BPE and 220–310× for WordPiece in our runs.)

### 2. WordPiece frequency threshold (`min_pair_freq`)

On real data the score formula inflates the score of very rare pairs, producing junk merges (e.g. `##wj`). A minimum pair-frequency threshold (a heuristic noted in the HuggingFace docs) cleans this up — but it is a **trade-off**, not a free win:

| Threshold | Naive/fast overlap | Vocabulary reached |
|---|---|---|
| 0 (off) | 58% | 1100 / 1100 |
| 50,000 | 88% | 1100 / 1100 |
| 200,000 | 94% | 771 / 1100 (saturated) |
| 500,000 | 95% | 357 / 1100 (saturated) |

A fixed threshold eventually filters out *every* remaining pair, so training halts early and the vocabulary **saturates** below the requested size. The threshold improves merge *quality* at the cost of merge *quantity*.

### 3. Vocabulary size vs tokenization length (threshold off)

| Vocab | BPE tokens/sentence | WordPiece tokens/sentence |
|---|---|---|
| 1,000 | 10.6 | 33.4 |
| 5,000 | 7.6 | 27.4 |
| 10,000 | 7.0 | 25.2 |

BPE is consistently more compact than un-thresholded WordPiece on German, because WordPiece's informative-pair criterion builds long pieces more slowly than BPE's frequency merging.

### 4. Attention cost

Self-attention builds an **n × n** matrix for a sequence of n tokens, so cost grows with n². At vocab 1,000, WordPiece's fragmented tokenization implies an attention matrix **~10× larger** than BPE's on the same text. Tokenizer choice is an efficiency lever, not just preprocessing.

### 5. Morphological evaluation (the core finding)

How well does each algorithm's split align with real German morphemes? Presented as a three-way comparison rather than a single flattering number:

| Configuration | Vocabulary | Exact match | Morpheme recall |
|---|---|---|---|
| **BPE** | 10,000 | 4 / 20 (20%) | **0.91** |
| WordPiece (no threshold) | 10,000 | 0 / 20 (0%) | 0.27 |
| WordPiece (threshold 200k) | 969 (saturated) | 3 / 20 (15%) | 0.54 |

**BPE's frequency criterion aligns with German morphology almost for free** (0.91 recall). WordPiece needs the threshold to approach it (0.27 → 0.54) and only by accepting a saturated vocabulary.

We also **cross-checked this against spaCy** (the automatic tool the brief suggested). Since spaCy gives lemmas, not morpheme splits, we used it for lemma-root consistency and a spaCy-expanded recall metric. It independently agreed: BPE kept the lemma root in 2/3 inflected families vs WordPiece's 0/3, and scored 0.62 recall vs WordPiece's 0.14. Two independent methods reaching the same conclusion is strong evidence the finding is real.

An **inflection-family test** (the `trink-` paradigm: *trinke, trinkst, trinkt, getrunken*) is unflattering to **both** algorithms (BPE 0.54, WordPiece 0.25, neither with an exact match): neither cleanly isolates the shared root. This is an honest limit of subword tokenization — these methods are statistical, not linguistic.

---

## Problems we found and fixed (transparency note)

We document the bugs we hit, because understanding them is part of understanding the algorithms. These were diagnosed in debugging sessions assisted by **Claude (Anthropic)** and cross-checked against the **HuggingFace** documentation; we attribute every source of a solution, as our supervisor asked.

- **Stale-heap bug in fast WordPiece.** Fast WordPiece initially disagreed with naive WordPiece, which we had wrongly called a "finding." It was a bug: stale heap entries (whose score changed because of a merge elsewhere) were *discarded* instead of being re-inserted with their corrected score, so a pair could be lost forever. Fixing it (re-push instead of discard) raised naive/fast agreement from ~19% to ~55%. BPE never had this problem because its decision has no global denominator.
- **Missing training cells.** The committed notebooks were missing the cells that actually call the training functions, so a fresh checkout failed. Rebuilt so every notebook runs end to end and every speed number is computed live.
- **Threshold saturation.** The `min_pair_freq` threshold we added to clean up WordPiece's merges turned out to saturate the vocabulary at a fixed value — a real trade-off we now report as a result rather than hide.

Our WordPiece scoring was verified against the canonical worked example in the HuggingFace documentation (it reproduces the documented first merge exactly).

---

## References

- Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. https://aclanthology.org/P16-1162
- Schuster, M., Nakajima, K. (2012). *Japanese and Korean Voice Search.* ICASSP 2012.
- Mielke, S. J., et al. (2021). *Between Words and Characters.* https://arxiv.org/abs/2112.10508
- HuggingFace LLM Course — [BPE (ch. 6.5)](https://huggingface.co/learn/llm-course/en/chapter6/5) and [WordPiece (ch. 6.6)](https://huggingface.co/learn/llm-course/en/chapter6/6)
- Corpus: Wikimedia German Wikipedia (`20231101.de`) via the HuggingFace `datasets` library.
