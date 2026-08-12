# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Protein alpha / non-alpha classifier

## Mission

Develop a reproducible, research-only binary classifier that predicts `alpha` or `not_alpha` from an amino-acid sequence. Do not claim structural determination, biological function, clinical utility, or causality from model output.

## Governing documents

Read the relevant document in `docs/` before changing its subject area:

- `docs/source-data.md` — authoritative SCOP inputs, downloads, provenance, and source-to-sequence mapping
- `docs/data-contract.md` — schema, labels, validation, and ambiguous-residue filtering
- `docs/split-and-similarity-policy.md` — MMseqs2 clustering, similarity policy, representatives, and audits
- `docs/evaluation-protocol.md` — partitions, metrics, reporting, and model-comparison rules
- `docs/experiment-tracking.md` — MLflow and artifact requirements
- `docs/decisions.md` — durable scientific and operational decisions

## Non-negotiable rules

- The annotation-derived label is `alpha` only when `CL=1000000`; every other `CL` value is `not_alpha`.
- Omit sequences containing ambiguous or other noncanonical amino-acid symbols before MMseqs2. Never cluster, featurize, train on, or evaluate an omitted sequence.
- The 30% amino-acid sequence-similarity policy uses MMseqs2 with `--min-seq-id 0.3 --cov-mode 0`.
- ML datasets, training, validation, tuning, and test evaluation use only MMseqs2 cluster representatives.
- Models may use only declared sequence-derived features. Never use identifiers, coordinates, annotation fields, labels in headers, source metadata, or cluster/split IDs as features.
- Use Conda and the versioned `environment.yml`; do not use system Python or install packages outside the environment.
- Keep raw sequences, raw annotations, derived datasets, artifacts, and MLflow stores out of Git.

## Code layout

```
src/protein_alpha_classifier/
  pipelines/
    fetch_sources.py       # downloads SCOP annotation + FASTA, writes source manifest
    validate_input.py      # eligibility filtering, emits eligible FASTA + omission parquet
    cluster_sequences.py   # runs MMseqs2, emits cluster_assignments.parquet + representatives.fasta
    build_dataset.py       # joins representatives to annotations, writes dataset.parquet
    split_dataset.py       # stratified cluster-group split -> train/val/test parquets
    train_baselines.py     # fits baselines, 5-fold CV on train, logs everything to MLflow
    make_figures.py        # publication 2×3 figure -> figures/baseline_comparison.{pdf,png}
  features/
    composition.py         # AACompositionTransformer, DipeptideCompositionTransformer (stateless)
  models/
    baselines.py           # composition-LR and dipeptide-LR sklearn pipelines + BASELINE_CONFIGS

tests/                     # mirrors src layout; unit + integration tests

data/
  raw/scop/                # immutable downloaded files (gitignored)
  interim/                 # MMseqs2 outputs and intermediate FASTs (gitignored)
  processed/               # ML-ready dataset.parquet (gitignored)

figures/                   # output figures (gitignored except .gitkeep)
mlruns/                    # MLflow tracking store (gitignored)
```

Each pipeline is a `__main__`-runnable module. New pipelines go in `src/protein_alpha_classifier/pipelines/`. Feature extractors must be stateless pure functions; any learned transformation belongs in the sklearn `Pipeline` inside `models/`.

## Environment setup

```bash
conda env create -f environment.yml
conda activate protein-alpha-classifier
```

## Quick commands

```bash
# Lint and format
ruff check src tests
ruff format --check src tests

# All tests
pytest -q

# Single test
pytest -q tests/path/to/test_file.py::test_function_name
```

## Pipeline (run in order)

```bash
python -m protein_alpha_classifier.pipelines.fetch_sources \
  --annotations-url https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt \
  --fasta-url https://www.ebi.ac.uk/pdbe/scop/files/scop_fa_represeq_lib_latest.fa \
  --out-dir data/raw/scop

python -m protein_alpha_classifier.pipelines.validate_input \
  --fasta data/raw/scop/scop_fa_represeq_lib_latest.fa \
  --annotations data/raw/scop/scop-cla-latest.txt \
  --output-fasta data/interim/eligible_labeled_sequences.fasta \
  --invalid-records data/interim/omitted_ambiguous_sequences.parquet

python -m protein_alpha_classifier.pipelines.cluster_sequences \
  --input-fasta data/interim/eligible_labeled_sequences.fasta \
  --out-dir data/interim/mmseqs_30pct

python -m protein_alpha_classifier.pipelines.build_dataset \
  --representatives data/interim/mmseqs_30pct/representatives.fasta \
  --annotations data/raw/scop/scop-cla-latest.txt \
  --cluster-assignments data/interim/mmseqs_30pct/cluster_assignments.parquet \
  --output data/processed/dataset.parquet

python -m protein_alpha_classifier.pipelines.split_dataset \
  --dataset data/processed/dataset.parquet \
  --out-dir data/processed/splits \
  --test-frac 0.15 \
  --val-frac 0.15 \
  --seed 42
```

## Training baselines

```bash
python -m protein_alpha_classifier.pipelines.train_baselines \
  --train data/processed/splits/train.parquet \
  --val data/processed/splits/val.parquet \
  --split-manifest data/processed/splits/split_manifest.json \
  --source-manifest data/raw/scop/source_manifest.json
```

## Figures

```bash
python -m protein_alpha_classifier.pipelines.make_figures \
  --train data/processed/splits/train.parquet \
  --val   data/processed/splits/val.parquet \
  --out-dir figures/
```

Outputs `figures/baseline_comparison.pdf` and `figures/baseline_comparison.png`.

## Experiment tracking

Every training/evaluation run must log to MLflow per `docs/experiment-tracking.md`. To browse runs:

```bash
mlflow ui --backend-store-uri mlruns/
```

Required logged fields include the labeling rule, full MMseqs2 command, dataset checksums and counts, split/seed artifacts, and the trained sklearn pipeline artifact. Never overwrite or manually edit a completed run.

## Change workflow

1. Read this file and the relevant governing document before changing data, labels, clustering, splits, features, or evaluation.
2. For a change affecting a durable decision, data contract, similarity policy, partitioning, metric, or scientific claim, write a short plan and record the decision in `docs/decisions.md`.
3. Implement the smallest complete change with configuration and test updates.
4. Run formatting, linting, targeted tests, and the smoke experiment when applicable.
5. Report changed files, command outcomes, dataset/config versions, and remaining limitations.

## Definition of done

A model comparison is review-ready only with versioned source artifacts, auditable labels and filtering, the required MMseqs2 artifact, representative-only ML data, train-only preprocessing, declared baselines, MLflow artifacts, and passing required checks. Explicitly label every deviation.
