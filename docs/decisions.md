# Project decisions

This document records durable scientific and operational decisions. Do not revise a decision silently; add the date, rationale, effect, and migration/testing requirements when changing one.

## Label definition

**Decision:** The source annotation's `CL` value defines the label.

- `CL=1000000` is `alpha` (`label=1`).
- Every other `CL` value is `not_alpha` (`label=0`).

**Effect:** No label may be inferred from sequence content, identifiers, coordinates, another annotation level, a structure prediction, or helix fraction.

## Sequence eligibility

**Decision:** Retain only uppercase sequences composed entirely of `ACDEFGHIKLMNPQRSTVWY`.

**Effect:** Omit sequences containing `X`, `B`, `Z`, `J`, `U`, `O`, or any other noncanonical character before MMseqs2. Never map or impute those residues.

## Similarity and representatives

**Decision:** The 30% similarity requirement applies to amino-acid sequences and is implemented by MMseqs2 with:

```bash
--min-seq-id 0.3 --cov-mode 0
```

**Effect:** Use only MMseqs2 cluster representatives in ML-ready data and in all model development/evaluation. Preserve cluster assignments and commands as audits.

## Authoritative sources

**Decision:** The source annotation and FASTA are:

- [scop-cla-latest.txt](https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt)
- [scop_fa_represeq_lib_latest.fa](https://www.ebi.ac.uk/pdbe/scop/files/scop_fa_represeq_lib_latest.fa)

**Effect:** Download, retain, and checksum both files for every build. A `latest` URL is not itself a reproducible input version.

## Research-use scope

**Decision:** This is a research-only sequence classifier.

**Effect:** Do not claim structural determination, biological mechanism, causality, clinical utility, or biological function from predictions.

## Baseline model set

**Decision (2026-08-11):** Replace the original three baselines (majority-class, length-only LR, AA-composition LR) with two composition baselines:

- Amino-acid composition + L2-regularised logistic regression (20 features)
- Dipeptide composition + L2-regularised logistic regression (400 features)

**Rationale:** Majority-class and length-only baselines provide negligible diagnostic signal beyond class prevalence. Dipeptide composition captures ordered pairwise residue context that monomer composition cannot, making it a more informative upper reference point for simple sequence features.

**Effect:** `BASELINE_CONFIGS` in `models/baselines.py` now contains exactly these two entries. MLflow runs from the prior baseline set remain in history but must not be compared directly against new runs without labelling the version difference.

**Migration:** No data or split changes required. Re-run `train_baselines.py` to produce new MLflow runs under the updated baseline set.

## Change procedure

When changing any decision above:

1. Record the date, proposed replacement, rationale, and expected impact here.
2. Identify required source/data migrations, configuration changes, and backward-compatibility concerns.
3. Update the governing document, tests, and MLflow metadata together.
4. Do not compare results across materially different decision versions without clearly labeling the difference.
