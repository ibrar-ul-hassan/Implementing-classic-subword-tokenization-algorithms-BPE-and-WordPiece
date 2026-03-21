import os

# ── Paths ──
REPO_DIR  = "/content/tokenization-project"
DATA_DIR  = f"{REPO_DIR}/data"
SRC_DIR   = f"{REPO_DIR}/src"

# ── GitHub ──
GITHUB_USERNAME = "ibrar-ul-hassan"
REPO_NAME       = "Implementing-classic-subword-tokenization-algorithms-BPE-and-WordPiece"

# ── Training parameters ──
SAMPLE_VOCAB_SIZE  = 1100    # for quick testing (5000 words)
FULL_VOCAB_SIZE    = 20000   # for final training (full corpus)
SAMPLE_SIZE        = 5000    # top N words for sample

# ── Data files ──
WORD_FREQ_SAMPLE   = f"{DATA_DIR}/word_freq_sample.json"
BPE_MERGE_RULES    = f"{DATA_DIR}/bpe_merge_rules.json"
BPE_VOCAB          = f"{DATA_DIR}/bpe_vocab.json"
WP_VOCAB           = f"{DATA_DIR}/wp_vocab.json"
WP_MERGE_LOG       = f"{DATA_DIR}/wp_merge_log.json"
