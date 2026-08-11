# Source data

## Scope

This document governs acquisition, versioning, parsing, and linkage of the external source annotation and amino-acid sequence files.

## Authoritative inputs

| Input                              | URL                                                                                                    | Purpose                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| SCOP classification annotations    | [scop-cla-latest.txt](https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt)                       | Source classification records, including `SCOPCLA` and extracted `CL` |
| SCOP representative-sequence FASTA | [scop_fa_represeq_lib_latest.fa](https://www.ebi.ac.uk/pdbe/scop/files/scop_fa_represeq_lib_latest.fa) | Amino-acid sequences for dataset preparation                          |

## Acquisition and provenance

For every dataset build:

1. Download both source files into a Gitignored raw-data location.
2. Preserve the exact downloaded files as immutable run artifacts.
3. Record each URL, download timestamp, HTTP metadata when available, SHA-256 checksum, source release/version in the file header, and download/parser code version.
4. Treat `latest` as a retrieval alias, not as a reproducible version. The checksum and retained source file identify the actual input.
5. Record any retrieval failure, empty file, duplicate record, parser failure, or source/FASTA mismatch as a preparation error or explicitly logged exclusion.

## Parsing and linkage

- Parse the annotation format with a versioned parser; retain the source record identifier, full `SCOPCLA` field, and extracted `CL` value.
- Parse FASTA with Biopython; do not hand-parse it through string splitting.
- Join annotations to sequences only via a documented and tested identifier/coordinate mapping.
- Never link records by row position or assume that source order is shared.
- Preserve the mapping rule, mapping result, unmapped source records, unmapped FASTA records, duplicates, and collision-resolution rule as artifacts.
- Store annotation provenance in `label_source` and the source file checksum(s) in the processed-data provenance.

## Required outputs

Source preparation must create or retain:

- Immutable raw annotation and FASTA files
- A source manifest with URLs, release/header information, checksums, timestamps, and parser version
- Parsed annotation records with original `SCOPCLA` and extracted `CL`
- A documented source-to-sequence mapping artifact
- Exclusion/error reports and counts

## Implementation rule

Before modifying source retrieval, parsing, record linkage, or provenance fields, read this document and update tests, the source manifest schema, and `docs/decisions.md` when the change alters a durable rule.
