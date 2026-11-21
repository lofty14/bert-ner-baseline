"""Merge each member's executed notebook (parts/*.ipynb) into the single submission notebook.

Usage (from the repository root):
    python tools/merge_parts.py                 # writes Project13_final.ipynb
    python tools/merge_parts.py --check         # only report un-executed cells

The order, titles and owners come from parts/manifest.json. Outputs are copied as they are;
nothing is re-run (markers do not re-run the notebook either), so every part must be executed
and saved with its outputs before merging.
"""
import argparse
import copy
import json
import re
import uuid
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "parts"
HEADING = re.compile(r"^(#{1,6})(\s)", re.M)


def shift_headings(cell, shift):
    if cell.cell_type != "markdown" or shift == 0:
        return cell
    cell = copy.deepcopy(cell)
    cell.source = HEADING.sub(lambda m: "#" * min(6, len(m.group(1)) + shift) + m.group(2), cell.source)
    return cell


def min_heading(cells):
    levels = [len(m.group(1)) for c in cells if c.cell_type == "markdown" for m in HEADING.finditer(c.source)]
    return min(levels) if levels else 3


def unexecuted(nb):
    return [i for i, c in enumerate(nb.cells)
            if c.cell_type == "code" and c.source.strip() and c.get("execution_count") is None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Project13_final.ipynb"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((PARTS / "manifest.json").read_text())
    parts = []
    problems = []
    for i, p in enumerate(manifest["parts"], start=1):
        nb = nbformat.read(PARTS / p["file"], as_version=4)
        missing = unexecuted(nb)
        if missing:
            problems.append(f"{p['file']}: {len(missing)} code cell(s) not executed (cells {missing[:10]})")
        parts.append((i, p, nb))

    for line in problems:
        print("WARNING", line)
    if args.check:
        return

    toc = "\n".join(f"| {i} | {p['title']} | {p['owner']} | `parts/{p['file']}` |" for i, p, _ in parts)
    head = nbformat.v4.new_markdown_cell(
        f"# {manifest['title']}\n\n**{manifest['subtitle']}**\n\n"
        "| Part | Content | Owner | Source notebook |\n|---|---|---|---|\n" + toc + "\n\n"
        "Every part was executed by its owner and is included here with its saved outputs; markers do not need "
        "to re-run anything. Each part starts with its own setup cell, so it can also be run on its own "
        "(repository root or Google Colab). The shared code lives in the `ner/` package of this repository."
    )
    cells = [head]
    for i, p, nb in parts:
        cells.append(nbformat.v4.new_markdown_cell(
            f"## Part {i}. {p['title']}\n*Owner: {p['owner']}.*" + (f" {p['note']}" if p.get("note") else "")))
        shift = 3 - min_heading(nb.cells)
        cells.extend(shift_headings(c, shift) for c in nb.cells)

    for c in cells:  # parts may share cell ids; give every merged cell a fresh one
        c["id"] = uuid.uuid4().hex[:8]
    out = nbformat.v4.new_notebook()
    out.cells = cells
    out.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                    "language_info": {"name": "python"}}
    nbformat.write(out, args.out)
    print(f"wrote {args.out}: {len(parts)} parts, {len(cells)} cells, {len(problems)} part(s) with un-executed cells")


if __name__ == "__main__":
    main()
