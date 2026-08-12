# Alpha-Helical Sequence Classifier

A reproducible, research-only binary classifier that predicts **alpha** or **not-alpha** from a raw amino-acid sequence, using the [SCOP](https://scop.mrc-lmb.cam.ac.uk/) structural database as ground truth.

> **Scope:** This is a research prototype. Predictions are statistical associations derived from sequence composition — they do not imply structural determination, biological mechanism, or clinical utility.

---

## What it does

Given an amino-acid sequence, the classifier outputs a probability that the sequence belongs to SCOP's all-alpha class (`CL=1000000`). Two logistic-regression baselines are provided, differing only in feature representation:

| Baseline | Features | Dimensionality |
|---|---|---|
| Amino-Acid Freq | Fractional single-residue counts | 20 |
| Dipeptide Freq | Fractional consecutive-pair counts | 400 |

Both models use L2 regularisation and balanced class weights to correct for the ~23 % alpha prevalence in training data.

---

## Data pipeline

```
SCOP annotation + FASTA
        │
        ▼
  validate_input          ← filter to canonical 20 AA, rewrite headers
        │
        ▼
  cluster_sequences       ← MMseqs2 at 30 % identity (--min-seq-id 0.3 --cov-mode 0)
        │
        ▼
  build_dataset           ← join representatives → annotations → labels
        │
        ▼
  split_dataset           ← 70 / 15 / 15 % stratified cluster-group split
        │
        ▼
  train_baselines         ← 5-fold StratifiedGroupKFold CV + val evaluation → MLflow
```

**Dataset snapshot (current build)**

| Partition | Sequences | Alpha | Not-Alpha |
|---|---|---|---|
| Train | 9,279 | 2,115 (22.8 %) | 7,164 (77.2 %) |
| Val | 1,987 | 454 (22.8 %) | 1,533 (77.2 %) |
| Test | 1,987 | 454 (22.8 %) | 1,533 (77.2 %) |

Primary metric: **AUPRC** (preferred over AUROC when the positive class is uncommon).

---

## Baseline performance

![Baseline comparison — ROC, PR curve, and normalised confusion matrix for both models](figures/baseline_comparison.png)

*Confusion matrices are row-normalised (recall view). Each cell shows the fraction of true instances predicted in that column.*

### Model comparison table

| Model | Val AUPRC | Val AUROC | Not-Alpha Recall | Alpha Recall |
|---|---|---|---|---|
| Amino-Acid Freq (composition LR) | 0.636 | 0.847 | 0.77 | 0.75 |
| Dipeptide Freq (dipeptide LR) | 0.655 | 0.858 | 0.79 | 0.77 |

*This table is updated with each new model. Test-set results are withheld until final comparison.*

---

## Setup

```bash
conda env create -f environment.yml
conda activate protein-alpha-classifier
```

Requires [MMseqs2](https://github.com/soedinglab/MMseqs2) (installed via `bioconda::mmseqs2`).

---

## Running the pipeline

```bash
# 1. Download SCOP sources
python -m protein_alpha_classifier.pipelines.fetch_sources \
  --annotations-url https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt \
  --fasta-url https://www.ebi.ac.uk/pdbe/scop/files/scop_fa_represeq_lib_latest.fa \
  --out-dir data/raw/scop

# 2. Filter to canonical sequences
python -m protein_alpha_classifier.pipelines.validate_input \
  --fasta data/raw/scop/scop_fa_represeq_lib_latest.fa \
  --annotations data/raw/scop/scop-cla-latest.txt \
  --output-fasta data/interim/eligible_labeled_sequences.fasta \
  --invalid-records data/interim/omitted_ambiguous_sequences.parquet

# 3. Cluster at 30 % identity
python -m protein_alpha_classifier.pipelines.cluster_sequences \
  --input-fasta data/interim/eligible_labeled_sequences.fasta \
  --out-dir data/interim/mmseqs_30pct

# 4. Build ML-ready dataset
python -m protein_alpha_classifier.pipelines.build_dataset \
  --representatives data/interim/mmseqs_30pct/representatives.fasta \
  --annotations data/raw/scop/scop-cla-latest.txt \
  --cluster-assignments data/interim/mmseqs_30pct/cluster_assignments.parquet \
  --output data/processed/dataset.parquet

# 5. Split
python -m protein_alpha_classifier.pipelines.split_dataset \
  --dataset data/processed/dataset.parquet \
  --out-dir data/processed/splits \
  --test-frac 0.15 --val-frac 0.15 --seed 42

# 6. Train baselines and log to MLflow
python -m protein_alpha_classifier.pipelines.train_baselines \
  --train data/processed/splits/train.parquet \
  --val   data/processed/splits/val.parquet \
  --split-manifest data/processed/splits/split_manifest.json \
  --source-manifest data/raw/scop/source_manifest.json

# 7. Regenerate publication figure
python -m protein_alpha_classifier.pipelines.make_figures \
  --train data/processed/splits/train.parquet \
  --val   data/processed/splits/val.parquet \
  --out-dir figures/
```

Browse MLflow runs:

```bash
mlflow ui --backend-store-uri mlruns/
```

---

## Repository layout

```
src/protein_alpha_classifier/
  pipelines/       fetch_sources, validate_input, cluster_sequences,
                   build_dataset, split_dataset, train_baselines, make_figures
  features/        AACompositionTransformer, DipeptideCompositionTransformer
  models/          BASELINE_CONFIGS, pipeline factories
tests/             unit + integration tests (pytest)
docs/              governing documents (data contract, evaluation protocol, decisions)
figures/           baseline_comparison.png / .pdf
```

## Development

```bash
ruff check src tests      # lint
ruff format --check src tests
pytest -q                 # all tests
```
