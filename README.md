# Workshop paper and evaluation artifacts

- `latex/`: manuscript sources, tables, PDF figures, and the compiled paper.
- `code/`: Python generation and audit scripts, dependency manifest, and uv lockfile.
- `artifacts/`: canonical reasoning records, language evaluation scores, and generated
  Parquet analysis files.

## Generate and check

Requires Python 3.14 or later and uv. From this folder:

```bash
cd code
uv sync --locked
uv run python src/generate_workshop_artifacts.py
uv run python src/audit_paper_numbers.py
cd ../latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_neurips_2026.tex
```

PDF compilation requires a LaTeX installation with the packages used by the manuscript,
BibTeX, and latexmk. Generation overwrites automated tables in `latex/tables/`, plots in
`latex/figures/`, and Parquet files in `artifacts/`.

The generator produces 11 tables, five plots, and nine Parquet files. The pipeline
diagram, budget table, and the tables in `reasoning_draft.tex` and `language_draft.tex`
are maintained directly in LaTeX. Protocol B values supplied by a colleague are stored
in `code/src/protocol_b.py`. See `code/README.md` for evaluation and audit details.
