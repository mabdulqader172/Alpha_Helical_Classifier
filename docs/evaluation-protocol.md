# Evaluation protocol

## Scope

This document defines partitions, cross-validation, preprocessing boundaries, baselines, metrics, and model-comparison reporting.

## Partitioning

- ML datasets contain representatives only; retain `cluster_id` for provenance/auditing and never as a model feature.
- Reserve an untouched test set before any model selection. Never use its metrics, predictions, or error analysis to choose features, preprocessing, hyperparameters, thresholds, or models.
- Use stratified group-aware cross-validation when the available class/group counts permit it. Otherwise document the compromise and report class and group counts per fold.
- For any split involving pre-representative records, apply the grouping requirements in `docs/split-and-similarity-policy.md`.
- Never present random sequence-level cross-validation as performance on dissimilar proteins.

## Leakage prevention

Fit all learned operations only on each fold's training portion. This includes feature selection, dimensionality reduction, scaling, imputation, vocabularies, encoding parameters, and threshold selection. Use an sklearn `Pipeline` where practical.

Models may use only declared sequence-derived features. Do not use UniProt IDs, coordinates, header-encoded labels, source IDs, `source_cl`, `label_source`, source checksums/URLs, cluster IDs, split IDs, or post-annotation metadata.

## Baselines

Start every comparison with these sequence-only baselines:

- Amino-acid composition plus regularized logistic regression
- Dipeptide composition plus regularized logistic regression

**Rationale (2026-08-11):** Majority-class and length-only baselines were removed because they provide minimal diagnostic value beyond AUPRC prevalence and composition features. Dipeptide composition captures local pairwise residue context that monomer composition cannot, making it a more informative second reference point.

Evaluate more complex models under the identical split and reporting protocol.

## Metrics and reporting

Declare a primary metric in the experiment configuration before evaluation. When alpha examples are uncommon, use AUPRC as the primary metric. As appropriate, also report AUROC, class-wise precision/recall/F1, balanced accuracy, confusion matrix at a prespecified threshold, and calibration diagnostics.

For every partition and final comparison, report:

- Number of sequences and clusters
- Class prevalence and sequence-length distribution
- Per-fold results plus mean and variation/uncertainty
- Holdout-test results exactly once
- Fixed threshold and its selection procedure
- Valid error slices by sequence length or declared metadata, without using undeclared predictors

## Interpretation

Interpret model scores and feature importance as predictive associations only. Do not infer structural mechanisms, biological function, clinical utility, or causal conclusions.

## Implementation rule

Before changing a split, fold, metric, threshold, baseline, preprocessing rule, or reported claim, read this document and update experiment configuration and tests.
