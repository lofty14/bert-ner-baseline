"""NCBI Disease data: the CoNLL files from spyysalo/ncbi-disease are the single source.

Files are downloaded once into ``data/ncbi_disease/`` and verified against pinned
SHA-256 hashes. Empty records are dropped (5432/923/940 sentences).
"""

import sys
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit_ncbi_splits import (  # noqa: E402  (single parser shared with the audit script)
    CONLL_URL, EXPECTED_SHA256, LABEL_NAMES, SPLIT_FILES, audit_conll, read_conll,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ncbi_disease"
LABELS = list(LABEL_NAMES)  # ["O", "B-Disease", "I-Disease"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
EXPECTED_SIZES = {"train": 5432, "validation": 923, "test": 940}


def ensure_data(data_dir=DEFAULT_DATA_DIR):
    """Download missing split files and verify every file's SHA-256. Returns raw bytes per split."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_by_split = {}
    for split, filename in SPLIT_FILES.items():
        path = data_dir / filename
        if not path.exists():
            print(f"Downloading {CONLL_URL + filename} -> {path}")
            with urlopen(CONLL_URL + filename, timeout=60) as response:
                raw = response.read()
            if sha256(raw).hexdigest() != EXPECTED_SHA256[split]:
                raise ValueError(f"Downloaded {filename} has unexpected SHA-256; refusing to save it")
            path.write_bytes(raw)
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != EXPECTED_SHA256[split]:
            raise ValueError(f"{path}: SHA-256 {digest} != expected {EXPECTED_SHA256[split]}")
        raw_by_split[split] = raw
    return raw_by_split


def load_ncbi(data_dir=DEFAULT_DATA_DIR, audit=True):
    """Return ({split: [(tokens, tags), ...]}, audit_summary)."""
    raw_by_split = ensure_data(data_dir)
    splits = {split: read_conll(raw) for split, raw in raw_by_split.items()}
    for split, rows in splits.items():
        if len(rows) != EXPECTED_SIZES[split]:
            raise ValueError(f"{split}: {len(rows)} sentences, expected {EXPECTED_SIZES[split]}")
    summary = audit_conll(raw_by_split=raw_by_split, verbose=audit)
    return splits, summary
