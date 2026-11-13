"""Model registry. Every checkpoint is pinned to a Hugging Face commit."""

import json

from transformers import AutoTokenizer

MODELS = {
    # `google-bert/bert-base-cased` is the canonical repo that `bert-base-cased` redirects to.
    "bert": {
        "checkpoint": "google-bert/bert-base-cased",
        "revision": "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
        "lowercase": False,
    },
    # BioBERT v1.2 ships no tokenizer_config.json, so BertTokenizer would default to
    # do_lower_case=True on a cased vocab. We force case preservation.
    "biobert": {
        "checkpoint": "dmis-lab/biobert-base-cased-v1.2",
        "revision": "67c9c25b46986521ca33df05d8540da1210b3256",
        "lowercase": False,
    },
    "pubmedbert": {
        "checkpoint": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "revision": "d673b8835373c6fa116d6d8006b33d48734e305d",
        "lowercase": True,
    },
}


def load_tokenizer(key):
    spec = MODELS[key]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["checkpoint"], revision=spec["revision"], use_fast=True, do_lower_case=spec["lowercase"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"{key}: a fast tokenizer is required for word_ids()")

    # Fatal check that the casing behaviour is what we expect.
    normalizer = json.loads(tokenizer.backend_tokenizer.to_str()).get("normalizer") or {}
    lowercases = bool(normalizer.get("lowercase"))
    pieces = tokenizer.tokenize("Hereditary")
    if lowercases != spec["lowercase"] or (pieces[0][0].isupper() == spec["lowercase"]):
        raise RuntimeError(
            f"{key}: tokenizer casing mismatch (expected lowercase={spec['lowercase']}, "
            f"normalizer={normalizer}, tokenize('Hereditary')={pieces})"
        )
    return tokenizer
