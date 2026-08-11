# Experiment tracking

## Scope

MLflow is the required experiment record for every training or evaluation run.

## Required MLflow records

Every run must log:

- Experiment name and run name
- Resolved configuration artifact
- Git commit or explicit `dirty_worktree` indicator
- Python/package environment and MMseqs2 version
- Annotation and FASTA URLs, download metadata, source-header release information, checksums, and parser version
- Dataset version/checksum; raw, omitted, eligible, representative, and ML-ready counts; and class counts at each stage
- The labeling rule: `CL=1000000 -> alpha; all other CL -> not_alpha`
- Full MMseqs2 command/configuration and confirmation that `--min-seq-id 0.3 --cov-mode 0` were used
- Random seed and split/fold assignments, or immutable artifact IDs for them
- Model class, hyperparameters, feature-extraction settings, and preprocessing settings
- Train/CV/test metrics with clear partition prefixes, for example `cv_mean_auprc` and `test_auprc`
- Confusion matrix, ROC/PR curves, calibration output when relevant, predictions keyed by `sequence_id`, and the trained sklearn pipeline/model artifact

## Artifact rules

- Never overwrite or manually edit a completed MLflow run.
- Do not log raw sequences or sensitive/proprietary annotations to a shared tracking server without approval.
- Store raw-source and derived-data paths/checksums in artifacts or metadata; keep raw data itself out of Git.
- Preserve the exact resolved configuration and the commands needed to reproduce the run.

## Implementation rule

Before changing tracked fields, artifact retention, run naming, or tracking infrastructure, read this document and update the tracking implementation and tests.
