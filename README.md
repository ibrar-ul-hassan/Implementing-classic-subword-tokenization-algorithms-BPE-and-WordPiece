# BPE and WordPiece Tokenization
> NLP Semester 2 Project — Implementing classic subword tokenization algorithms from scratch in Python.

## The Project
We implement two tokenization algorithms used in modern LLMs:
- **BPE** (Byte Pair Encoding) — used by GPT, GPT-2, RoBERTa
- **WordPiece** — used by BERT, DistilBERT, MobileBERT

Both are built **from scratch** without using HuggingFace's pre-built tokenizers. We then evaluate them scientifically against a morphological gold standard on German text.

---

## Team
| Name | Role |
|------|------|
| Ibrar | Algorithmic code (BPE + WordPiece trainer & tokenizer) |
| Dayo | Experiments, evaluation, and analysis |
| Lama | Report and presentation |

**Supervisor:** Marie Candito — Academic Year 2025-2026, Semester 2

---

## Key Results

### Training Speed (naive vs fast)
| Algorithm | Naive | Fast | Speedup |
|-----------|-------|------|---------|
| BPE | ~15s | ~0.5s | **~30x** |
| WordPiece | ~30s | ~0.4s | **~85x** |
| BPE (projected, 20k merges, full corpus) | ~4.8 hours | ~12 min | ~30x |
| WordPiece (projected, 20k merges, full corpus) | ~8.7 hours | ~7 min | ~85x |

### Tokenization Quality (vocabulary size effects)
| Vocab Size | BPE avg tokens/sentence | WP avg tokens/sentence |
|------------|------------------------|------------------------|
| 1,000 | 15.8 | 45.8 |
| 5,000 | 10.0 | 30.4 |
| 10,000 | **9.2** | 19.6 |

BPE consistently produces **2-3x fewer tokens per sentence** — meaning it learns richer, longer subword units.

### Morphological Evaluation (20 German words, 10k vocab)
| Metric | BPE | WordPiece |
|--------|-----|-----------|
| Exact match | 4/20 (20%) | 1/20 (5%) |
| Morpheme score | **0.91** | 0.48 |

BPE finds 91% of expected morphemes purely from frequency statistics — no linguistic knowledge required.

### OOV Handling
| | BPE | WordPiece |
|--|-----|-----------|
| Unknown words | Always segments → falls back to chars | Returns `[UNK]` for whole word |
| Robustness | High — works for any input | Lower — strict vocabulary match required |

---

## Scientific Finding
> Naive and fast WordPiece implementations produce different vocabularies.
> Root cause: incremental `letter_freq` updates in fast WP change score ordering
> compared to full recomputation in naive WP. Both implementations are correct —
> this reflects a known numerical sensitivity in the WordPiece scoring function.
> We use the fast implementation for all experiments as it produces longer,
> more semantically coherent tokens.

---

## Repository Architecture

```
repo/
│
├── src/                        ← ALL logic lives here as .py files
│   ├── config.py               ← single source of truth for all variables
│   ├── bpe.py                  ← BPE trainer + tokenizer (naive + fast)
│   └── wordpiece.py            ← WordPiece trainer + tokenizer (naive + fast)
│
├── data/                       ← generated data files (small JSONs only)
│   ├── word_freq_sample.json   ← top 5000 words from German Wikipedia
│   ├── bpe_merge_rules.json    ← learned BPE merge rules
│   ├── bpe_vocab.json          ← final BPE vocabulary
│   ├── wp_vocab.json           ← final WordPiece vocabulary
│   ├── wp_merge_log.json       ← WordPiece merge log with scores
│   └── experiment_results.json ← all experiment results
│
├── 00_setup.ipynb              ← corpus download + preprocessing
├── 01_bpe.ipynb                ← BPE training + benchmarking
├── 02_wordpiece.ipynb          ← WordPiece training + benchmarking
├── 03_experiments.ipynb        ← evaluation + comparison graphs
│
├── PROJECT_BRIEF.docx          ← original project brief
├── .gitignore                  ← excludes large corpus files
└── README.md
```

---

## How the Code is Organised

### The Golden Rule
> **Logic lives in `src/` — notebooks just call functions.**

Notebooks never contain algorithm code. They only:
1. Clone the repo and import from `src/`
2. Call functions and display results
3. Push back to GitHub

**If your Colab session restarts → run Cell 1 only. Everything else works again.**

### The Three Source Files

**`src/config.py`** — one place for all paths and parameters. Every other file imports from here.

**`src/bpe.py`** — exports:
- `train(word_freq, vocab_size)` → naive BPE trainer
- `train_fast(word_freq, vocab_size)` → fast BPE (priority queue + inverse index)
- `tokenize_word(word, merge_rules)` → tokenize using rule replay
- `tokenize(text, merge_rules)` → tokenize full sentence
- `save(...)` / `load(...)` → persist results to disk

**`src/wordpiece.py`** — exports:
- `train(word_freq, vocab_size)` → naive WordPiece trainer
- `train_fast(word_freq, vocab_size)` → fast WordPiece (inverse index + incremental scores)
- `tokenize_word(word, wp_vocab)` → tokenize using longest-match
- `tokenize(text, wp_vocab)` → tokenize full sentence
- `save(...)` / `load(...)` → persist results to disk

---

## How to Start a Session

**Every session — same 3 steps:**

### Step 1 — Run Cell 1 (always)
```python
import sys, os, subprocess
from google.colab import userdata

GITHUB_TOKEN    = userdata.get('GITHUB_1')
REPO_DIR        = "/content/tokenization-project"
GITHUB_USERNAME = "ibrar-ul-hassan"
REPO_NAME       = "Implementing-classic-subword-tokenization-algorithms-BPE-and-WordPiece"
auth_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

if not os.path.exists(REPO_DIR):
    subprocess.run(f'git clone "{auth_url}" {REPO_DIR}', shell=True)
else:
    subprocess.run(f'git -C {REPO_DIR} pull origin main', shell=True)

sys.path.insert(0, f"{REPO_DIR}/src")
from config import *
```

### Step 2 — Run the remaining cells

### Step 3 — Last cell pushes to GitHub

---

## What Lives on GitHub vs Colab Only

| File | GitHub | Colab only | Reason |
|------|--------|------------|--------|
| `src/*.py` | ✅ | | Code |
| `*.ipynb` | ✅ | | Code |
| `word_freq_sample.json` | ✅ | | Small (top 5000 words) |
| `bpe_merge_rules.json` | ✅ | | Small output |
| `bpe_vocab.json` | ✅ | | Small output |
| `wp_vocab.json` | ✅ | | Small output |
| `experiment_results.json` | ✅ | | Small output |
| `corpus_clean.txt` | | ✅ | ~500MB — regenerated each session |
| `word_frequencies.json` | | ✅ | Large — regenerated each session |

---

## Key Algorithmic Difference (BPE vs WordPiece)

| | BPE | WordPiece |
|--|-----|-----------|
| **Merge criterion** | Most frequent pair | Highest score = freq(AB) / (freq(A) × freq(B)) |
| **Stores** | Merge rules + vocabulary | Vocabulary only |
| **Tokenizes by** | Replaying merge rules in order | Longest-match-first on vocabulary |
| **OOV handling** | Falls back to characters | Returns `[UNK]` for whole word |
| **Used by** | GPT, GPT-2, RoBERTa | BERT, DistilBERT |
| **## prefix** | No | Yes |

---

## References
- Sennrich et al. (2016) — [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- Schuster & Nakajima (2012) — Japanese and Korean voice search (WordPiece origin)
- HuggingFace — [BPE tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5) | [WordPiece tokenization](https://huggingface.co/learn/llm-course/en/chapter6/6)
- Mielke et al. (2021) — [Between words and characters](https://arxiv.org/abs/2112.10508)
