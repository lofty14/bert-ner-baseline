"""Unlabelled PubMed corpus for domain-adaptive (continued MLM) pre-training, with overlap audit.

    python -m ner.dapt_corpus search                 # one-off: PMIDs per publication year -> dapt/pmids.txt
    python -m ner.dapt_corpus build --limit 30000    # efetch titles+abstracts, remove NCBI Disease overlap
    python -m ner.dapt_corpus positive_control       # check that the sentence-level overlap check fires

`search` queries NCBI E-utilities (esearch) per publication month of every year with
    diseases category[MeSH Terms] AND hasabstract[text] AND english[Language] AND <yyyy/mm>[pdat]
takes the same number of PMIDs from every year (the remainder of total / n_years goes to the
earliest years, one each; within a year spread over the months, each drawn at random with a fixed
seed from the first 9999 esearch results of that month), shuffles the union with a fixed seed and
writes dapt/pmids.txt, so
"the first N lines" is a deterministic, year-balanced subset. The file is committed because
esearch results change as PubMed grows.

`build` fetches the first --limit PMIDs (efetch, XML, POST batches of 200; cached in
data/dapt/raw.jsonl, PMIDs already fetched -- including ones without an abstract -- are not
fetched again) and removes overlap with NCBI Disease in two passes:
  1. by PMID: every document of the original NCBI Disease train/dev/test corpora;
  2. by text: every dev/test sentence (CoNLL tokens joined) is normalised (lower case, all
     non-alphanumeric characters removed); those with >= 40 characters are fingerprints, and
     any corpus document whose normalised title+abstract contains one is dropped entirely.
About 2% of the remaining documents (deterministic PMID hash) are held out for validation.
Outputs: data/dapt/corpus_train.txt, corpus_valid.txt (one document per line: title + " " +
abstract) and audit.json.

Requests carry only tool=ncbi-ner-dapt (no e-mail / API key) and are throttled to <= 3/s with
retries and exponential back-off.
"""

import argparse
import bisect
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .data import load_ncbi

REPO_ROOT = Path(__file__).resolve().parent.parent
PMID_FILE = REPO_ROOT / "dapt" / "pmids.txt"
OUT_DIR = REPO_ROOT / "data" / "dapt"
ORIGINAL_DIR = REPO_ROOT / "data" / "ncbi_disease" / "original"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "ncbi-ner-dapt"
QUERY = "diseases category[MeSH Terms] AND hasabstract[text] AND english[Language] AND {date}[pdat]"
POOL = 9999  # esearch cannot page beyond the first 10,000 results
ORIGINAL_URL = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/original-data/"
ORIGINAL_FILES = {
    "train": "train/NCBItrainset_corpus.txt",
    "validation": "devel/NCBIdevelopset_corpus.txt",
    "test": "test/NCBItestset_corpus.txt",
}
FETCH_BATCH = 200
MIN_FINGERPRINT_CHARS = 40
VALID_PER_MILLE = 20  # ~2% held out


# ----------------------------------------------------------------------------- HTTP
_last_request = [0.0]


def eutils(endpoint, params, retries=6):
    """POST to an E-utilities endpoint; <= 3 requests/s, retries with exponential back-off."""
    body = urlencode({**params, "tool": TOOL}).encode()
    for attempt in range(retries):
        wait = 0.34 - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.time()
        try:
            with urlopen(Request(EUTILS + endpoint, data=body), timeout=120) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1 or (isinstance(e, HTTPError) and e.code not in (429, 500, 502, 503, 504)):
                raise
            delay = 2 ** attempt
            print(f"  {endpoint}: {e!r}; retry in {delay}s", flush=True)
            time.sleep(delay)


def fetch_url(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {path}")
        with urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- text helpers
def normalise(text):
    """Lower case, drop every non-alphanumeric character (spaces included)."""
    return "".join(c for c in text.lower() if c.isalnum())


def doc_text(doc):
    return (doc["title"] + " " + doc["abstract"]).strip()


def is_valid_doc(pmid):
    return int(sha256(f"dapt-valid:{pmid}".encode()).hexdigest()[:8], 16) % 1000 < VALID_PER_MILLE


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class FingerprintIndex:
    """Substring search of many fingerprints in many documents (one joined string, str.find)."""

    SEP = "\x00"  # never produced by normalise()

    def __init__(self, texts):
        self.starts, parts, pos = [], [], 0
        for text in texts:
            self.starts.append(pos)
            parts.append(text)
            pos += len(text) + 1
        self.blob = self.SEP.join(parts)

    def matches(self, fingerprints):
        """Return {doc index: set(fingerprint indices)} for every occurrence."""
        hits = {}
        for f_index, fp in enumerate(fingerprints):
            pos = self.blob.find(fp)
            while pos != -1:
                doc = bisect.bisect_right(self.starts, pos) - 1
                hits.setdefault(doc, set()).add(f_index)
                pos = self.blob.find(fp, pos + 1)
        return hits


def ncbi_fingerprints():
    """Normalised dev/test sentences of the CoNLL release with >= MIN_FINGERPRINT_CHARS characters."""
    splits, _ = load_ncbi(audit=False)
    fps, sources, short = {}, {}, 0
    for split in ("validation", "test"):
        for i, (tokens, _) in enumerate(splits[split]):
            fp = normalise("".join(tokens))
            if len(fp) < MIN_FINGERPRINT_CHARS:
                short += 1
                continue
            fps.setdefault(fp, len(fps))
            sources.setdefault(fp, []).append(f"{split}:{i}")
    fingerprints = sorted(fps, key=fps.get)
    return fingerprints, [sources[fp] for fp in fingerprints], short


def read_original_corpora():
    """PubTator-format NCBI Disease corpora -> {split: {pmid: {"title", "abstract"}}}."""
    out = {}
    for split, rel in ORIGINAL_FILES.items():
        text = fetch_url(ORIGINAL_URL + rel, ORIGINAL_DIR / Path(rel).name)
        docs = {}
        for line in text.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[1] in ("t", "a"):
                doc = docs.setdefault(parts[0], {"title": "", "abstract": ""})
                doc["title" if parts[1] == "t" else "abstract"] = parts[2].strip()
        out[split] = docs
    return out


# ----------------------------------------------------------------------------- search
def cmd_search(args):
    years = list(range(args.year_from, args.year_to + 1))
    base, extra = divmod(args.total, len(years))
    per_year = {y: base + (i < extra) for i, y in enumerate(years)}
    # esearch returns at most the first 10,000 PMIDs of a query, in its default order (most recently
    # added first), so "the first k of a year" would over-represent records added to PubMed long after
    # publication. Instead each year is queried per month (<year>/<mm>[pdat]; records with a year-only
    # date fall into January), up to POOL PMIDs per month are retrieved and the month's quota is drawn
    # from that pool with a fixed seed. A PMID can match two windows (e-pub vs. print date); a PMID
    # already taken is skipped, so every year keeps exactly per_year[year] PMIDs.
    all_ids, seen, counts = [], set(), {}
    for year in years:
        q_base, q_extra = divmod(per_year[year], 12)
        counts[year] = {"available": 0, "pool": 0, "taken": 0}
        for month in range(1, 13):
            quota = q_base + (month <= q_extra)
            raw = eutils("esearch.fcgi", {"db": "pubmed", "term": QUERY.format(date=f"{year}/{month:02d}"),
                                          "retmax": POOL, "retmode": "json"})
            result = json.loads(raw)["esearchresult"]
            pool = [i for i in result["idlist"] if i not in seen]
            if len(pool) < quota:
                raise SystemExit(f"{year}/{month:02d}: only {len(pool)} new PMIDs returned, wanted {quota}")
            ids = random.Random(f"{args.seed}-{year}-{month}").sample(pool, quota)
            counts[year]["available"] += int(result["count"])
            counts[year]["pool"] += len(result["idlist"])
            counts[year]["taken"] += len(ids)
            all_ids += ids
            seen.update(ids)
        c = counts[year]
        print(f"  {year}: {c['available']} available (month windows), pool {c['pool']}, took {c['taken']}", flush=True)
    random.Random(args.seed).shuffle(all_ids)
    PMID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PMID_FILE.write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    meta = {"query": QUERY, "years": [years[0], years[-1]], "total": len(all_ids), "per_year": counts,
            "sampling": f"per month: up to {POOL} PMIDs in esearch default order, quota drawn with "
                        "random.Random(f'{seed}-{year}-{month}')", "shuffle_seed": args.seed,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": file_sha256(PMID_FILE)}
    (PMID_FILE.parent / "pmids_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(all_ids)} PMIDs -> {PMID_FILE} (sha256 {meta['sha256'][:12]})")


# ----------------------------------------------------------------------------- fetch
def parse_efetch(xml_bytes):
    """-> {pmid: {"title", "abstract"}} from a PubmedArticleSet."""
    docs = {}
    for article in ET.fromstring(xml_bytes).iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        pmid = citation.findtext("PMID")
        art = citation.find("Article")
        title = " ".join("".join(art.find("ArticleTitle").itertext()).split()) if art.find("ArticleTitle") is not None else ""
        parts = []
        for node in art.findall("Abstract/AbstractText"):
            text = " ".join("".join(node.itertext()).split())
            if text:
                parts.append(text)
        docs[pmid] = {"title": title, "abstract": " ".join(parts)}
    return docs


def fetch_docs(pmids, cache_path):
    """Fetch missing PMIDs into the jsonl cache; return {pmid: record} for all requested PMIDs."""
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["pmid"]] = rec
    todo = [p for p in pmids if p not in cache]
    print(f"efetch: {len(pmids)} requested, {len(pmids) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), FETCH_BATCH):
            batch = todo[start:start + FETCH_BATCH]
            docs = parse_efetch(eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch),
                                                        "retmode": "xml", "rettype": "abstract"}))
            for pmid in batch:  # PMIDs efetch did not return are recorded too (status "missing")
                doc = docs.get(pmid)
                rec = {"pmid": pmid, "status": "missing", "title": "", "abstract": ""} if doc is None else \
                      {"pmid": pmid, "status": "ok" if doc["abstract"] else "no_abstract", **doc}
                cache[pmid] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  fetched {min(start + FETCH_BATCH, len(todo))}/{len(todo)}", flush=True)
    return {p: cache[p] for p in pmids}


# ----------------------------------------------------------------------------- build
def overlap_by_text(docs, fingerprints, sources):
    """docs: list of (pmid, text). Returns {pmid: [fingerprint sources]} for documents to drop."""
    index = FingerprintIndex([normalise(text) for _, text in docs])
    hits = index.matches(fingerprints)
    return {docs[d][0]: sorted(s for f in fs for s in sources[f]) for d, fs in sorted(hits.items())}


def cmd_build(args):
    out_dir = Path(args.out_dir)
    all_pmids = [p for p in PMID_FILE.read_text(encoding="utf-8").split() if p]
    pmids = all_pmids[: args.limit] if args.limit else all_pmids
    records = fetch_docs(pmids, out_dir / "raw.jsonl")
    dropped = {"missing": [p for p in pmids if records[p]["status"] == "missing"],
               "no_abstract": [p for p in pmids if records[p]["status"] == "no_abstract"]}
    docs = [(p, doc_text(records[p])) for p in pmids if records[p]["status"] == "ok"]

    # Pass 1: PMIDs of the original NCBI Disease corpora (all three splits).
    original = read_original_corpora()
    ncbi_pmids = {split: set(d) for split, d in original.items()}
    pass1 = {split: sorted(p for p, _ in docs if p in ids) for split, ids in ncbi_pmids.items()}
    removed1 = set().union(*map(set, pass1.values()))
    docs = [(p, t) for p, t in docs if p not in removed1]

    # Pass 2: dev/test sentence fingerprints inside the normalised document text.
    fingerprints, sources, short = ncbi_fingerprints()
    pass2 = overlap_by_text(docs, fingerprints, sources)
    docs = [(p, t) for p, t in docs if p not in pass2]

    train = [(p, t) for p, t in docs if not is_valid_doc(p)]
    valid = [(p, t) for p, t in docs if is_valid_doc(p)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("corpus_train.txt", train), ("corpus_valid.txt", valid)):
        (out_dir / name).write_text("".join(t.replace("\n", " ") + "\n" for _, t in rows), encoding="utf-8")

    sentence_sources = sorted({s for v in pass2.values() for s in v})
    audit = {
        "query": QUERY, "pmid_file": str(PMID_FILE.relative_to(REPO_ROOT)), "pmid_file_sha256": file_sha256(PMID_FILE),
        "requested": len(pmids), "retrieved": len(pmids) - len(dropped["missing"]),
        "dropped_missing": len(dropped["missing"]), "dropped_no_abstract": len(dropped["no_abstract"]),
        "pass1_pmid_overlap": {
            "ncbi_documents": {s: len(v) for s, v in ncbi_pmids.items()},
            "removed": {s: len(v) for s, v in pass1.items()}, "removed_total": len(removed1),
            "removed_pmids": pass1,
        },
        "pass2_text_overlap": {
            "fingerprints": len(fingerprints), "min_chars": MIN_FINGERPRINT_CHARS,
            "dev_test_sentences_too_short": short,
            "removed_documents": len(pass2), "sentences_involved": len(sentence_sources),
            "removed_pmids": {p: v for p, v in pass2.items()},
        },
        "final_documents": {"train": len(train), "valid": len(valid)},
        "words_approx": {"train": sum(len(t.split()) for _, t in train),
                         "valid": sum(len(t.split()) for _, t in valid)},
        "valid_rule": f"int(sha256('dapt-valid:<pmid>')[:8], 16) % 1000 < {VALID_PER_MILLE}",
        "sha256": {"corpus_train.txt": file_sha256(out_dir / "corpus_train.txt"),
                   "corpus_valid.txt": file_sha256(out_dir / "corpus_valid.txt")},
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print_summary(audit, out_dir)
    return audit


def print_summary(a, out_dir):
    p1, p2 = a["pass1_pmid_overlap"], a["pass2_text_overlap"]
    print("\nDAPT corpus audit")
    print(f"  requested {a['requested']} PMIDs, retrieved {a['retrieved']}, "
          f"dropped: {a['dropped_missing']} not returned, {a['dropped_no_abstract']} without abstract")
    print(f"  pass 1 (PMID in NCBI Disease): removed {p1['removed_total']} "
          f"(train {p1['removed']['train']}, dev {p1['removed']['validation']}, test {p1['removed']['test']}; "
          f"NCBI documents {p1['ncbi_documents']})")
    print(f"  pass 2 (dev/test sentence text, {p2['fingerprints']} fingerprints >= {p2['min_chars']} chars, "
          f"{p2['dev_test_sentences_too_short']} shorter sentences not used): removed {p2['removed_documents']} "
          f"documents, {p2['sentences_involved']} sentences involved")
    print(f"  final: {a['final_documents']['train']} train / {a['final_documents']['valid']} valid documents, "
          f"~{a['words_approx']['train']} / ~{a['words_approx']['valid']} words")
    print(f"  sha256 train {a['sha256']['corpus_train.txt'][:12]}, valid {a['sha256']['corpus_valid.txt'][:12]}")
    print(f"  -> {out_dir}/corpus_train.txt, corpus_valid.txt, audit.json")


def cmd_positive_control(args):
    """Run pass 2 on documents that must overlap: the original dev/test corpora themselves, and the
    PubMed (efetch) versions of a few test documents under fake PMIDs."""
    fingerprints, sources, _ = ncbi_fingerprints()
    original = read_original_corpora()
    report = {}
    for split in ("validation", "test"):
        docs = [(p, doc_text(d)) for p, d in original[split].items()]
        hits = overlap_by_text(docs, fingerprints, sources)
        report[split] = {"documents": len(docs), "hit": len(hits)}
    index = FingerprintIndex([normalise(doc_text(d)) for s in ("validation", "test") for d in original[s].values()])
    matched = index.matches(fingerprints)
    covered = {f for fs in matched.values() for f in fs}
    report["fingerprints_found_in_original_text"] = f"{len(covered)}/{len(fingerprints)}"
    sample = sorted(original["test"])[: args.n_pubmed]
    fetched = parse_efetch(eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(sample),
                                                   "retmode": "xml", "rettype": "abstract"}))
    docs = [(f"fake{i}", doc_text(fetched[p])) for i, p in enumerate(sample) if p in fetched]
    hits = overlap_by_text(docs, fingerprints, sources)
    report["pubmed_versions_of_test_docs"] = {"documents": len(docs), "hit": len(hits), "pmids": sample}
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="esearch PMIDs per year -> dapt/pmids.txt")
    s.add_argument("--year_from", type=int, default=1990)
    s.add_argument("--year_to", type=int, default=2012)
    s.add_argument("--total", type=int, default=30000)
    s.add_argument("--seed", type=int, default=9444)
    b = sub.add_parser("build", help="efetch + overlap removal -> data/dapt/")
    b.add_argument("--limit", type=int, default=None, help="use the first N PMIDs of dapt/pmids.txt (default: all)")
    b.add_argument("--out_dir", default=str(OUT_DIR))
    c = sub.add_parser("positive_control", help="check that the text-overlap pass detects known overlap")
    c.add_argument("--n_pubmed", type=int, default=10)
    args = p.parse_args(argv)
    {"search": cmd_search, "build": cmd_build, "positive_control": cmd_positive_control}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
