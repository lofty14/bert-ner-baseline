"""Compare an already loaded Hugging Face NCBI Disease dataset with CoNLL TSVs."""

from collections import Counter
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from urllib.request import urlopen


CONLL_URL = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/"
SPLIT_FILES = {"train": "train.tsv", "validation": "devel.tsv", "test": "test.tsv"}
LABEL_NAMES = ("O", "B-Disease", "I-Disease")


def read_conll(raw):
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


def audit_ncbi_splits(hf_dataset, conll_dir=None):
    """Print split statistics and return True when all ordered rows match."""
    all_match = True
    for split, filename in SPLIT_FILES.items():
        if conll_dir is None:
            with urlopen(CONLL_URL + filename, timeout=60) as response:
                raw = response.read()
        else:
            raw = (Path(conll_dir) / filename).read_bytes()

        conll_rows = read_conll(raw)
        hf_rows = hf_dataset[split]
        hf_labels = hf_rows.features["ner_tags"].feature
        token_mismatches = label_mismatches = missing_rows = invalid_bio = 0
        tag_counts = Counter()
        first_difference = None

        for row_number, (hf_row, conll_row) in enumerate(zip_longest(hf_rows, conll_rows)):
            hf_value = None
            if hf_row is not None:
                tokens = hf_row["tokens"]
                tags = [hf_labels.int2str(tag) for tag in hf_row["ner_tags"]]
                if len(tokens) != len(tags) or not set(tags).issubset(LABEL_NAMES):
                    raise ValueError(f"Invalid Hugging Face row: {split}[{row_number}]")
                hf_value = (tokens, tags)
                tag_counts.update(tags)
                previous = "O"
                for tag in tags:
                    invalid_bio += tag == "I-Disease" and previous == "O"
                    previous = tag

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
        print(f"  Sentences: Hugging Face={len(hf_rows)}, CoNLL={len(conll_rows)}")
        print(f"  Rows with different tokens: {token_mismatches}")
        print(f"  Rows with different BIO labels: {label_mismatches}")
        print(f"  Rows present in only one source: {missing_rows}")
        print(f"  Hugging Face BIO counts: {dict(tag_counts)}")
        print(f"  Annotated disease starts (B-Disease): {tag_counts['B-Disease']}")
        print(f"  Orphan I-Disease tags: {invalid_bio}")
        print(f"  CoNLL SHA-256: {sha256(raw).hexdigest()}")
        if not matches:
            row_number, hf_value, conll_row = first_difference
            print(f"  First difference at zero-based row {row_number}:")
            print(f"    Hugging Face: {hf_value!r}\n    CoNLL: {conll_row!r}")

    return all_match
