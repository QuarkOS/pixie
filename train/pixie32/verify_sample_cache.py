"""Exercise the real sample loader with explicitly synthetic palette/tokenizer assets."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

from train.pixie32.build_data_p32 import build
from train.pixie32.data import DEFAULT_BLIND50, DEFAULT_EVAL50, load_dataset


def synthetic_assets(output, texts):
    cube = np.array(np.meshgrid(*([np.arange(16, dtype=np.uint8) * 17] * 3), indexing="ij")).reshape(3, -1).T
    palette = np.concatenate([np.zeros((1, 3), np.uint8), cube])
    np.save(output / "synthetic-palette.npy", palette, allow_pickle=False)
    alphabet = sorted(set("".join(texts).lower()))
    pieces = sorted(set(alphabet + [char + "</w>" for char in alphabet]))
    vocab = {piece: i for i, piece in enumerate(pieces)}
    for i in range(len(vocab), 49406):
        vocab[f"<synthetic{i}>"] = i
    vocab["<|startoftext|>"], vocab["<|endoftext|>"] = 49406, 49407
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], end_of_word_suffix="</w>", unk_token="<|endoftext|>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
    tokenizer.post_processor = processors.TemplateProcessing(single="<|startoftext|> $A <|endoftext|>", special_tokens=[("<|startoftext|>", 49406), ("<|endoftext|>", 49407)])
    tokenizer.save(str(output / "synthetic-tokenizer.json"))
    assert tokenizer.get_vocab_size() == 49408


def verify(sample, out, blind50=DEFAULT_BLIND50, eval50=DEFAULT_EVAL50):
    if out.exists():
        raise ValueError("Sample cache report output must be a new folder")
    out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in (sample / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    synthetic_assets(out, [text for row in records for text in (row["caption"], row["subject"])])
    args = Namespace(data_root=sample, out=out / "dataset", r1=sample / "r1_list.txt", edge_list=sample / "edge_px_ge32_list.txt",
                     palette=out / "synthetic-palette.npy", tokenizer=out / "synthetic-tokenizer.json", blind50=blind50, eval50=eval50,
                     seed=0, probe_fraction=.05, iid_fraction=.05)
    manifest = build(args)
    data = load_dataset(args.out, blind50, eval50)
    train = data.arrays["split"] == 0
    report = {"purpose": "real 200-sample loader validation with synthetic assets; no training",
              "assets": {"palette": "synthetic uniform 4096-colour RGB cube plus transparent sentinel", "tokenizer": "synthetic BPE, 49408 entries, BOS49406/EOS49407"},
              "input_manifest_rows": len(records), "admitted_rows": len(data.ids), "split_counts": manifest["split_counts"],
              "drops": manifest["drops"], "token_shape": list(data.arrays["tokens"].shape), "caption_shape": list(data.arrays["text"].shape),
              "train_mean_weight": float(data.arrays["weight"][train].mean()), "edge_rows": sum(row["edge_heavy"] for row in data.records),
              "holdout_matches": sum(data.holdouts.matches(row["caption"], row["subject"], row["trimmed_caption"], row["trimmed_subject"]) for row in data.records),
              "style": manifest["style"], "dataset_sha256": data.fingerprint}
    (out / "sample-cache-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    args = parser.parse_args()
    verify(args.sample, args.out, args.blind50, args.eval50)
