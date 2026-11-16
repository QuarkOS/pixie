# Check frozen holdout bytes before a run

The dataset manifest hashes the raw Blind-50 and Eval-50 files. Formatting,
line endings and a UTF-8 BOM change those hashes even when the parsed phrases
are identical. Keep the original files as immutable byte sequences.

Run this read-only preflight with the original files used for the prepared cache:

```sh
"$PY" -m train.pixie32.check_holdouts --data "$DATASET_OUT" \
  --blind50 "$BLIND50" --eval50 "$EVAL50"
```

A successful check prints `"byte_identity": true`. It reads the manifest and
holdout files only. It does not load arrays, weights or a GPU and writes nothing.
It does not replace the trainer's full dataset validation.

Use the same explicit holdout flags for the trainer, cache builder and benchmarks.
The launcher already accepts `BLIND50` and `EVAL50` environment variables.
Repository copies may have the same phrases in a different JSON format, so do
not substitute them for the original manifest inputs. Do not reserialize the
originals, canonicalize hashes or edit a manifest to make a mismatch disappear.
This change does not alter either checked-in holdout file.

If the original bytes are unavailable, an exact continuation cannot pass the
existing identity guard. Recover the operator's original frozen files first.
A changed policy requires a separately reviewed dataset/run and cannot be
silently treated as a resume.

Run the CPU regression checks:

```sh
"$PY" -m unittest train.pixie32.test_holdout_bytes -v
```
