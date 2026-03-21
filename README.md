# BPE and WordPiece Tokenization
> NLP Semester 2 Project — Implementing classic subword tokenization 
> algorithms from scratch in Python.

## The Project
We implement two tokenization algorithms used in modern LLMs:
- **BPE** (Byte Pair Encoding) — used by GPT, GPT-2, RoBERTa
- **WordPiece** — used by BERT, DistilBERT

Both are built from scratch without using HuggingFace's pre-built 
tokenizers. We then evaluate them scientifically against a 
morphological gold standard on German text.

---

## Team
| Name | Role |
|------|------|
| Ibrar | Algorithmic code (BPE + WordPiece trainer & tokenizer) |
| Dayo | Experiments, evaluation, and analysis |
| Lama | Report and presentation |

---

## Repository Architecture
```
repo/
│
├── src/                        ← ALL logic lives here as .py files
│   ├── config.py               ← single source of truth for all variables
│   ├── bpe.py                  ← BPE trainer + tokenizer
│   └── wordpiece.py            ← WordPiece trainer + tokenizer
│
├── data/                       ← generated data files (small JSONs only)
│   ├── word_freq_sample.json   ← top 5000 words from German Wikipedia
│   ├── bpe_merge_rules.json    ← learned BPE merge rules
│   ├── bpe_vocab.json          ← final BPE vocabulary
│   ├── wp_vocab.json           ← final WordPiece vocabulary
│   └── wp_merge_log.json       ← WordPiece merge log with scores
│
├── 00_setup.ipynb              ← corpus download + preprocessing
├── 01_bpe.ipynb                ← run BPE training + tokenization
├── 02_wordpiece.ipynb          ← run WordPiece training + tokenization
├── 03_experiments.ipynb        ← evaluation + comparison
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

This means if your Colab session restarts, **you only ever 
need to re-run Cell 1** and everything works again.

### The Three Source Files

**`src/config.py`** — one place for all paths and parameters.
Every other file imports from here. Never hardcode a path anywhere else.

**`src/bpe.py`** — contains:
- `train(word_freq, vocab_size)` → trains BPE, returns merge rules
- `tokenize_word(word, merge_rules)` → tokenizes using rule replay
- `tokenize(text, merge_rules)` → tokenizes full sentence
- `save(...)` / `load(...)` → persist results to disk

**`src/wordpiece.py`** — contains:
- `train(word_freq, vocab_size)` → trains WordPiece, returns vocabulary
- `tokenize_word(word, wp_vocab)` → tokenizes using longest-match
- `tokenize(text, wp_vocab)` → tokenizes full sentence
- `save(...)` / `load(...)` → persist results to disk

---

## How Every Notebook Works

Every notebook follows the exact same 4-part structure:
```
┌─────────────────────────────────────────────┐
│ CELL 1 — Setup                              │
│   Clone repo, install deps, import src/     │
│   ↑ Only cell you need to re-run on restart │
├─────────────────────────────────────────────┤
│ CELL 2 — Load Data                          │
│   Load word frequencies from data/          │
├─────────────────────────────────────────────┤
│ CELL 3..N — Work                            │
│   Call functions from src/                  │
│   Display results                           │
├─────────────────────────────────────────────┤
│ LAST CELL — Push to GitHub                  │
│   Save outputs + git push                   │
└─────────────────────────────────────────────┘
```

---

## How to Start a Session (for all teammates)

**Every single session — same 3 steps:**

### Step 1 — Open the right notebook
Go to [Google Drive](https://drive.google.com) → find the notebook 
you need → open in Colab.

### Step 2 — Run Cell 1 only
Cell 1 automatically:
- Clones the repo if it's a fresh session
- Pulls the latest changes if the repo exists
- Adds `src/` to the Python path
- Imports all modules
```python
# This is all Cell 1 does — run it and you're ready
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
print("✅ Ready")
```

### Step 3 — Run the rest of the cells normally

---

## What Lives on GitHub vs Colab Only

| File | GitHub | Colab only | Reason |
|------|--------|------------|--------|
| `src/*.py` | ✅ | | Code, not data |
| `*.ipynb` | ✅ | | Code, not data |
| `word_freq_sample.json` | ✅ | | Small (top 5000 words) |
| `bpe_merge_rules.json` | ✅ | | Small output file |
| `bpe_vocab.json` | ✅ | | Small output file |
| `wp_vocab.json` | ✅ | | Small output file |
| `corpus_clean.txt` | | ✅ | Too large (~500MB) |
| `word_frequencies.json` | | ✅ | Too large, regenerated each session |

> **Rule of thumb:** if it's code or a small JSON → GitHub.  
> If it's a large text file generated from the corpus → Colab only.

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
- HuggingFace — [BPE tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/5) | [WordPiece tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/6)
- Mielke et al. (2021) — [Between words and characters](https://arxiv.org/abs/2112.10508)
