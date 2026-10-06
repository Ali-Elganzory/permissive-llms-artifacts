# Workshop artifact generation

Regenerates 11 LaTeX tables, five PDF plots, and nine Parquet analysis artifacts.
Run these commands from `code/`:

    uv sync
    uv run python src/generate_workshop_artifacts.py

Then check that the prose still reconciles with the regenerated tables, and compile:

    uv run python src/audit_paper_numbers.py
    cd ../latex
    latexmk -pdf -interaction=nonstopmode -halt-on-error main_neurips_2026.tex

`audit_paper_numbers.py` checks 27 selected numerical claims (effects, contrasts,
ranges, gaps, paired deltas) using CSV-derived reasoning and language aggregates, plus supplied Protocol B values,
and asserts that the expected text fragment appears verbatim in the LaTeX source.
It does not read back generated tables or independently validate every numerical claim,
confidence interval, or source evaluation. It exits nonzero on any checked mismatch. This
catches the failure mode where a claim is right against the per-sample data but wrong
against the paper's own tables, because a difference of two rounded means is not the
rounded difference of two exact means. Run it after any edit to a number or a table.

## What the generator does

1. Loads `../artifacts/canonical_reasoning_per_sample.csv` and the language evaluation CSV,
   and builds two reasoning frames,
   the 3x2 front-loading by web-substrate factorial and the wider Phase 1 model zoo.
2. Validates factor completeness, seed coverage, per-benchmark sample counts, duplicate
   keys, null values, score ranges, unique checkpoint IDs, and experiment membership.
3. Drops raw checkpoint identifiers before any artifact is written, so the generated
   Parquet files and LaTeX fragments carry only anonymous experimental factors.
4. Computes benchmark means, sample standard deviations, and per-seed suite averages
   directly from the CSV. Tables display means to one decimal and standard deviations
   to two. Main-table deltas subtract the displayed means, while statistical contrasts
   use unrounded scores.
5. Computes paired item-level bootstrap intervals with Benjamini-Hochberg correction for
   the front-loading, substrate, and interaction contrast families.
6. Writes Parquet frames to `../artifacts/`, LaTeX fragments to `../latex/tables/`, and
   figures to `../latex/figures/`, and removes superseded figures from earlier drafts.

## Two evaluation protocols

Protocol A is the unified greedy protocol with three decoding seeds. The generator
recomputes the factorial and wider Phase 1 results from the per-sample CSVs. The older
mixture-ablation results are maintained in existing LaTeX tables. Decoding-seed standard
deviations do not measure variability across independent training runs.
Protocol B is the reasoning post-training protocol, single seed, covering every
OpenThoughts3-30K experiment and the held-out reasoning experiments. These experiments
are conducted by a colleague. Their supplied values are maintained as transcribed
constants in `src/protocol_b.py`. Underlying evaluation records are not available in
this project. Import-time checks validate row lengths, available-score averages within
a tolerance, and interval ordering. The held-out intervals and significance labels are
supplied constants, not recomputed by this generator. The two protocols score the same checkpoint differently
and are never mixed in a table, a figure, or a sentence.

## Separately maintained artifacts

The generator does not regenerate `figures/figure_pipeline.tex`, `tables/budget.tex`,
`tables/reasoning_draft.tex`, or `tables/language_draft.tex`. These LaTeX sources contain
the pipeline diagram, compute budgets, and legacy reasoning and language results.
They are included when compiling the manuscript.

## Reasoning input

`../artifacts/canonical_reasoning_per_sample.csv` is the single per-sample source for the
factorial and Phase 1 reasoning results. Each checkpoint has one model ID and one
score per seed, benchmark, and sample. The boolean `in_factorial` and `in_zoo`
columns specify experiment membership, including checkpoints shared by both analyses.
`front_loading` specifies whether the instruction-and-reasoning subset is present.
The generator filters the membership columns directly. Tables, plots, and Parquet
summaries are computed directly from the same records.

## Layout

    src/generate_workshop_artifacts.py   entry point
    src/wsdata.py                        loading, validation, aggregation
    src/wsstats.py                       paired item-level bootstrap and BH correction
    src/wstables.py                      LaTeX table emitters
    src/wsfigures.py                     figure emitters
    src/protocol_b.py                    transcribed Protocol B results
    src/audit_paper_numbers.py           selected numerical prose checks
