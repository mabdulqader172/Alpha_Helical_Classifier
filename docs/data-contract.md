# Data contract

## Scope

This document defines the canonical ML-ready dataset, label semantics, validation, and sequence eligibility rules.

## Canonical table

The canonical ML-ready dataset is Parquet and contains at least:

| Column                      | Type    | Requirement                                                  |
| --------------------------- | ------- | ------------------------------------------------------------ |
| `sequence_id`               | string  | Unique and exactly `UNIPROTID_START_END`                     |
| `uniprot_id`                | string  | Prefix parsed from `sequence_id`                             |
| `start`                     | integer | 1-based inclusive interval start                             |
| `end`                       | integer | 1-based inclusive interval end; `end >= start`               |
| `sequence`                  | string  | Uppercase canonical amino-acid sequence                      |
| `label`                     | integer | `1` = alpha; `0` = not_alpha                                 |
| `label_name`                | string  | `alpha` or `not_alpha`                                       |
| `source_cl`                 | string  | Original extracted source `CL` value                         |
| `label_source`              | string  | Annotation URL/release/checksum and source-record identifier |
| `cluster_id`                | string  | MMseqs2 cluster identifier for provenance/auditing           |
| `is_cluster_representative` | boolean | Always `true` in ML-ready data                               |
| `dataset_version`           | string  | Source checksums plus preparation-config identifier          |

## Label semantics

The source annotation determines labels exactly:

- `CL=1000000` maps to `alpha` and `label=1`.
- Every other `CL` value maps to `not_alpha` and `label=0`.

Never derive a label from a sequence ID, sequence, coordinates, any other classification field, a structural prediction, or a helix-fraction rule. Retain `source_cl` and `label_source` for provenance only; they are prohibited model features.

## Sequence eligibility

The allowed alphabet, after uppercasing, is exactly:

```text
ACDEFGHIKLMNPQRSTVWY
```

Omit every sequence containing any character outside this alphabet. This includes ambiguous symbols such as `X`, `B`, `Z`, `J`, `U`, and `O`. Do not map, replace, impute, or retain such symbols.

Perform the omission before generating the MMseqs2 input FASTA. An omitted sequence must never enter MMseqs2, any cluster assignment, representative selection, feature extraction, splitting, model fitting, validation, or evaluation.

## Required validations

- `sequence_id` matches `^[A-Z0-9]+_[1-9][0-9]*_[1-9][0-9]*$`; never silently repair malformed identifiers.
- Start/end coordinates and sequence length conform to the documented interval convention.
- Label derivation follows the exact `CL` rule.
- No duplicate `sequence_id` or contradictory canonical sequence/interval labels exists without an explicit, tested resolution rule.
- Record every rejected sequence with `sequence_id`, rejection reason, invalid symbols, source checksums, and pipeline/config version.
- Log raw, omitted, eligible, representative, and ML-ready counts; labels/class balance; invalid-record counts; and length distributions.
- Assert that every final ML row has `is_cluster_representative=true`.

## Versioning and privacy

Keep raw FASTA, raw annotations, intermediate data, and final Parquet files out of Git. Version data through immutable checksums, a data registry, or release identifiers. Do not place raw sequences or proprietary annotations on shared tracking infrastructure without approval.

## Implementation rule

Before modifying ingestion, labels, filtering, schema, or dataset construction, read this document and add tests for the changed behavior.
