# Submission notebook parts

Each member keeps one notebook here. `Project13_final.ipynb` (the file we submit) is generated from them:

```bash
python tools/merge_parts.py --check   # lists parts that still have un-executed cells
python tools/merge_parts.py           # rebuilds Project13_final.ipynb
```

Rules
- Replace your part with your final notebook, **run all cells and save with outputs** (markers do not re-run anything).
- Keep the first setup cell, so the part runs from the repository root or on Colab.
- Use `#` for your part title and `##` for sections; the merge script shifts headings to fit.
- Order, titles and owners are in `manifest.json`.
- Do not edit `Project13_final.ipynb` by hand; it is overwritten on every merge.
