# Native Pixie-32 implementation record

## Acceptance gates

- Keep all production v3 and worker modules unchanged.
- Implement native 1024-pixel images, 256 causal patches, 279 sequence positions, partial axial RoPE, and explicit v3 warm-start mapping.
- Require frozen holdouts, validated row identities and hashes, grouped splits, quantized deduplication, and new output folders.
- Refuse resume fingerprint mismatches, retain deterministic epoch plans and RNG, and keep paused runs resumable into fresh output folders.
- Run CPU behavioral tests and validate the provided 200-image sample. Run a bounded real-assets smoke only if required artifacts are supplied.
- Open a PR from the handoff branch and never merge or launch a long run.

## Work sequence

1. Read the handoff documents and inspect v3 interfaces.
2. Implement model, data, and trainer along disjoint module boundaries.
3. Add selected-GPU launch and stop commands, operator documentation, and measured test artifacts.
4. Run causality, warm-start, manifest integrity, holdout, split, and resume tests.
5. Independently review the integrated implementation and fix proven failures.
6. Publish a PR with measured output and unresolved hardware or asset limits.

## Evidence limits at start

The supplied sample contains images, caption files, caption TSV, R1 list, and edge list. Its documented provenance manifest and metadata JSONL are missing. No actual palette, tokenizer, CLIP weights, v3 checkpoint, or Eval-50 file was supplied. The available GPU has 8 GB VRAM, so 12 GB fit cannot be directly established here. Required artifacts were requested rather than discovered.

## Design choices

Model the Domain separates dataset identity, model configuration, and resume state. Prove It Works requires behavioral causality and resume checks rather than syntax-only validation. Explain the Number keeps tiny-fixture throughput separate from production model performance.

The implementation retains the planned 2x2 factorization and remaps learned spatial tables while adding partial rotary position information. This preserves more warm-start information than dropping every positional parameter. Comparing these choices in a real training run remains operator work.

## Handoff update

Update `b49864e` supplied the missing sample manifests, frozen Eval-50, and
exact asset shapes. The operator explicitly retains the real checkpoint,
palette, CLIP weights and tokenizer and will run the real GPU smoke. Input
flags and environment variables follow the updated task. CPU fixtures now
exercise the full checkpoint/CLIP shapes and real sample provenance. The
sample remains loader-only; no training on it was performed.
