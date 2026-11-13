"""Audit the NCBI Disease splits.

Two modes:
  * ``audit_conll(conll_dir)``: CoNLL files only (no Hugging Face needed) --
    sentence counts, BIO counts, orphan I-Disease tags, cross-split duplicate
    sentences and SHA-256 of each raw file.
  * ``audit_ncbi_splits(hf_dataset, conll_dir)``: compare an already loaded
    Hugging Face ``ncbi/ncbi_disease`` DatasetDict with the CoNLL files.
    The HF loader emits one trailing empty record per split (5433/924/941 vs.
    5432/923/940); empty HF records are ignored before the row-by-row
    comparison so they are no longer reported as differences.

CLI:
  python audit_ncbi_splits.py --no-hf [--conll_dir data/ncbi_disease]
  python audit_ncbi_splits.py          # also compares with HF (needs `datasets`)
"""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from urllib.request import urlopen


CONLL_URL = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/"
SPLIT_FILES = {"train": "train.tsv", "validation": "devel.tsv", "test": "test.tsv"}
LABEL_NAMES = ("O", "B-Disease", "I-Disease")
# SHA-256 of the spyysalo/ncbi-disease CoNLL files (master branch).
EXPECTED_SHA256 = {
    "train": "5f87b0de649eccff21e6d6e49e304aace808056ecae55225ea31602192517be9",
    "validation": "b4e1e7efcc2a35047bf2f3d281648357470e674f555d4a26a0049ac37d59cc61",
    "test": "8f088ce45dc4746592188f9e69de3d89d87f3bb8f0ed8c4a287d018123fa148d",
}


def read_conll(raw):
    """Parse raw CoNLL bytes into a list of (tokens, tags). Empty records are dropped."""
    rows = []
    tokens, tags = [], []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            if tokens:
                rows.append((tokens, tags))
                tokens, tags = [], []
            continue

        token, separator, tag = line.partition("\t")
        tag = tag.rstrip()
        if not separator or not token or tag not in LABEL_NAMES:
            raise ValueError(f"Invalid CoNLL row on line {line_number}: {line!r}")
        tokens.append(token)
        tags.append(tag)

    if tokens:
        rows.append((tokens, tags))
    return rows


def read_raw_split(split, conll_dir=None):
    """Return the raw bytes of one split, from ``conll_dir`` or from GitHub."""
    filename = SPLIT_FILES[split]
    if conll_dir is None:
        with urlopen(CONLL_URL + filename, timeout=60) as response:
            return response.read()
    return (Path(conll_dir) / filename).read_bytes()


def count_orphan_i(tags):
    """Count I-Disease tags not preceded by B-Disease/I-Disease."""
    orphans, previous = 0, "O"
    for tag in tags:
        orphans += tag == "I-Disease" and previous == "O"
        previous = tag
    return orphans


def audit_conll(conll_dir=None, verbose=True, raw_by_split=None):
    """Audit the CoNLL files alone. Returns a summary dict (also printed if verbose)."""
    summary = {"splits": {}, "cross_split_duplicates": {}}
    sentence_sets = {}
    for split in SPLIT_FILES:
        raw = raw_by_split[split] if raw_by_split else read_raw_split(split, conll_dir)
        rows = read_conll(raw)
        tag_counts = Counter(tag for _, tags in rows for tag in tags)
        digest = sha256(raw).hexdigest()
        summary["splits"][split] = {
            "sentences": len(rows),
            "tokens": sum(len(tokens) for tokens, _ in rows),
            "tag_counts": {label: tag_counts.get(label, 0) for label in LABEL_NAMES},
            "entities": tag_counts.get("B-Disease", 0),
            "orphan_i": sum(count_orphan_i(tags) for _, tags in rows),
            "within_split_duplicate_sentences": len(rows) - len({tuple(t) for t, _ in rows}),
            "sha256": digest,
            "sha256_ok": digest == EXPECTED_SHA256[split],
        }
        sentence_sets[split] = {tuple(tokens) for tokens, _ in rows}

    names = list(SPLIT_FILES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            summary["cross_split_duplicates"][f"{a}&{b}"] = len(sentence_sets[a] & sentence_sets[b])

    if verbose:
        print("NCBI Disease CoNLL audit")
        for split, s in summary["splits"].items():
            print(
                f"  {split:<10} sentences={s['sentences']:<5} tokens={s['tokens']:<6} "
                f"B={s['tag_counts']['B-Disease']:<5} I={s['tag_counts']['I-Disease']:<5} "
                f"O={s['tag_counts']['O']:<6} orphan_I={s['orphan_i']} "
                f"dup_in_split={s['within_split_duplicate_sentences']} "
                f"sha256={s['sha256'][:12]}... ({'OK' if s['sha256_ok'] else 'MISMATCH'})"
            )
        dups = ", ".join(f"{k}={v}" for k, v in summary["cross_split_duplicates"].items())
        print(f"  cross-split duplicate sentences (unique token sequences): {dups}")
    return summary


def audit_ncbi_splits(hf_dataset, conll_dir=None):
    """Print split statistics and return True when all ordered non-empty rows match."""
    all_match = True
    for split, filename in SPLIT_FILES.items():
        raw = read_raw_split(split, conll_dir)
        conll_rows = read_conll(raw)
        hf_rows = hf_dataset[split]
        hf_labels = hf_rows.features["ner_tags"].feature
        token_mismatches = label_mismatches = missing_rows = invalid_bio = 0
        empty_hf_records = 0
        tag_counts = Counter()
        first_difference = None

        hf_values = []
        for row_number, hf_row in enumerate(hf_rows):
            tokens = hf_row["tokens"]
            tags = [hf_labels.int2str(tag) for tag in hf_row["ner_tags"]]
            if len(tokens) != len(tags) or not set(tags).issubset(LABEL_NAMES):
                raise ValueError(f"Invalid Hugging Face row: {split}[{row_number}]")
            if not tokens:  # the HF loader yields a trailing empty record per split
                empty_hf_records += 1
                continue
            hf_values.append((tokens, tags))
            tag_counts.update(tags)
            invalid_bio += count_orphan_i(tags)

        for row_number, (hf_value, conll_row) in enumerate(zip_longest(hf_values, conll_rows)):
            if hf_value is None or conll_row is None:
                missing_rows += 1
            else:
                token_mismatches += hf_value[0] != conll_row[0]
                label_mismatches += hf_value[1] != conll_row[1]
            if hf_value != conll_row and first_difference is None:
                first_difference = (row_number, hf_value, conll_row)

        matches = first_difference is None
        all_match &= matches
        print(f"\n{split}: {'EXACT MATCH' if matches else 'DIFFERENT'}")
        print(f"  Sentences: Hugging Face={len(hf_rows)} ({empty_hf_records} empty ignored), "
              f"CoNLL={len(conll_rows)}")
        print(f"  Rows with different tokens: {token_mismatches}")
        print(f"  Rows with different BIO labels: {label_mismatches}")
        print(f"  Rows present in only one source: {missing_rows}")
        print(f"  Hugging Face BIO counts: {dict(tag_counts)}")
        print(f"  Annotated disease starts (B-Disease): {tag_counts['B-Disease']}")
        print(f"  Orphan I-Disease tags: {invalid_bio}")
        print(f"  CoNLL SHA-256: {sha256(raw).hexdigest()}")
        if not matches:
            row_number, hf_value, conll_row = first_difference
            print(f"  First difference at zero-based row {row_number} (empty records skipped):")
            print(f"    Hugging Face: {hf_value!r}\n    CoNLL: {conll_row!r}")

    return all_match


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conll_dir", default=None,
                        help="Directory with train.tsv/devel.tsv/test.tsv (default: download from GitHub)")
    parser.add_argument("--no-hf", dest="no_hf", action="store_true",
                        help="Only audit the CoNLL files; do not load the Hugging Face dataset")
    args = parser.parse_args()

    summary = audit_conll(args.conll_dir)
    ok = all(s["sha256_ok"] and s["orphan_i"] == 0 for s in summary["splits"].values())
    if not args.no_hf:
        from datasets import load_dataset  # optional dependency

        ok &= audit_ncbi_splits(load_dataset("ncbi/ncbi_disease"), args.conll_dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
