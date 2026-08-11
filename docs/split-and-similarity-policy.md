# Split and similarity policy

## Scope

This document governs MMseqs2 clustering, the 30% amino-acid sequence-similarity policy, representative selection, partitioning, and similarity audits.

## Policy

“No more than 30% similarity” means similarity between amino-acid sequences. Implement it with MMseqs2 using these required options:

```bash
--min-seq-id 0.3 --cov-mode 0
```

Do not reinterpret this as similarity of identifiers, annotations, structures, embeddings, nucleotide sequences, or metadata. Record all additional MMseqs2 options and the exact MMseqs2 version.

Only MMseqs2 cluster representatives may be used for ML-ready data, model fitting, cross-validation, tuning, or final evaluation. Nonrepresentative members may be retained only as auditable intermediate data.

## Required procedure

1. Use only the eligible labeled FASTA emitted after the data-contract validation and noncanonical-residue omission.
2. Run versioned MMseqs2 clustering with an argument list that includes the required options.

```bash
mmseqs easy-cluster eligible_labeled_sequences.fasta cluster_res tmp_dir \
  --min-seq-id 0.3 \
  --cov-mode 0
```

3. Save the MMseqs2 version, complete argument list, input FASTA checksum, configuration, stdout, stderr, and output paths.
4. Build `cluster_assignments.parquet` mapping every eligible original `sequence_id` to `cluster_id` and representative status.
5. Build `representatives.fasta` and ML-ready Parquet exclusively from representative rows.
6. Assert that every ML-ready row has `is_cluster_representative=true`.
7. Use `cluster_id` as the grouping variable for any split or audit that involves pre-representative records. A cluster may occur in only one partition.
8. Create a similarity audit artifact documenting the input checksum, exact command, MMseqs2 version, required options, cluster count, representative count, and representative-only assertion.

## Claims and limits

Do not silently mix representatives and nonrepresentative members. Do not claim a stronger strict all-pairs alignment guarantee than the implemented MMseqs2 procedure and the recorded audit establish.

## Implementation rule

Before changing MMseqs2 parameters, commands, representative selection, cluster handling, or split grouping, read this document; update the configuration, audit artifact, and tests.
