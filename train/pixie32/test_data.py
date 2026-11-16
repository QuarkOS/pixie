"""Data-boundary behavior using explicit synthetic provenance and tokenizer."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from train.pixie32.build_data_p32 import Quantizer, build, encode, make_parser, relative_path
from train.pixie32.data import Holdouts, load_dataset, sha256

# Blind-50 / Eval-50 are not part of the public tree; tests that need the real
# files skip unless they are in <repo>/holdout/ or PIXIE_HOLDOUT_DIR.
import os as _os
import unittest as _unittest
from pathlib import Path as _Path
_HOLDOUT_DIR = _Path(_os.environ.get("PIXIE_HOLDOUT_DIR") or next(
    p for p in _Path(__file__).resolve().parents if (p / "train" / "pixie32").is_dir()) / "holdout")
needs_holdouts = _unittest.skipUnless(
    (_HOLDOUT_DIR / "blind-50.json").is_file() and (_HOLDOUT_DIR / "eval-50.json").is_file(),
    "Blind-50/Eval-50 holdout files are not shipped; set PIXIE_HOLDOUT_DIR")



class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        image_dir = self.source / "final/images/3080ti/000"
        image_dir.mkdir(parents=True)
        palette = np.zeros((4097, 3), np.uint8)
        ids = np.arange(4097)
        palette[:, 0] = ids % 256
        palette[:, 1] = ids // 256 * 16
        np.save(self.root / "palette.npy", palette)
        self.palette = palette
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "sprite": 2, "held": 3, "subject": 4, "eval": 5, "relic": 6}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.root / "tokenizer.json"))
        self.blind_phrases = ["held subject"] + [f"blind reserved phrase{i}" for i in range(49)]
        self.eval_phrases = ["eval relic"] + [f"eval reserved phrase{i}" for i in range(49)]
        (self.root / "blind.json").write_text(json.dumps(self.blind_phrases))
        (self.root / "eval.json").write_text(json.dumps(self.eval_phrases))
        self.metadata, self.provenance, self.paths = [], [], []
        for i in range(48):
            rid = f"3080ti-s{i:06d}-c0"
            relative = f"images/3080ti/000/{rid}.png"
            caption = f"a sprite object{i // 4} variant{i}"
            pixels = np.zeros((32, 32, 4), np.uint8)
            pixels[8:24, 8:24, :3] = palette[i + 1]
            pixels[8:24, 8:24, 3] = 255
            Image.fromarray(pixels).save(self.source / "final" / relative)
            (self.source / "final" / relative).with_suffix(".txt").write_text(caption)
            self.metadata.append(dict(id=rid, file_name=relative, text=caption, source="3080ti"))
            self.provenance.append(dict(id=rid, source="3080ti", caption=caption, subject=f"object{i // 4}", sheet_id=i,
                                        pass_extract_v2=True, final=True, final_path="final/" + relative, holdout=None, dup_of=None))
            self.paths.append("final/" + relative)
        self.write_metadata()
        (self.root / "r1.txt").write_text("\n".join(self.paths))
        (self.root / "edge.txt").write_text("\n".join(self.paths[::2]))
        self.args = Namespace(data_root=self.source, out=self.root / "dataset", r1=self.root / "r1.txt", edge_list=self.root / "edge.txt",
                              palette=self.root / "palette.npy", tokenizer=self.root / "tokenizer.json", blind50=self.root / "blind.json",
                              eval50=self.root / "eval.json", seed=17, probe_fraction=.1, iid_fraction=.1)

    def write_metadata(self):
        for file, rows in ((self.source / "manifest.jsonl", self.provenance), (self.source / "final/metadata.jsonl", self.metadata)):
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def change_caption(self, index, caption):
        self.metadata[index]["text"] = self.provenance[index]["caption"] = caption
        (self.source / "final" / self.metadata[index]["file_name"]).with_suffix(".txt").write_text(caption)

    def load(self):
        return load_dataset(self.args.out, self.args.blind50, self.args.eval50)

    def test_build_has_aligned_arrays_weights_and_disjoint_splits(self):
        result = build(self.args)
        data = self.load()
        self.assertEqual(result["rows"], 48)
        self.assertEqual(data.arrays["tokens"].shape, (48, 1024))
        self.assertEqual(data.ids, [row["id"] for row in data.records])
        self.assertTrue(np.all(data.arrays["text"][:, 0] == 49406))
        self.assertTrue(np.all(data.arrays["text"][:, -1] == 49407))
        split = data.arrays["split"]
        self.assertAlmostEqual(float(data.arrays["weight"][split == 0].mean()), 1., places=6)
        for subject in {row["subject"] for row in data.records}:
            bucket = [(row["edge_heavy"], float(weight)) for row, s, weight in zip(data.records, split, data.arrays["weight"]) if s == 0 and row["subject"] == subject]
            edge = [weight for flag, weight in bucket if flag]
            normal = [weight for flag, weight in bucket if not flag]
            if edge and normal:
                self.assertAlmostEqual(edge[0] / normal[0], .5, places=6)
        self.assertEqual(data.manifest["style"]["fill_thresholds"], [320, 720])
        self.assertEqual(data.manifest["inputs"]["tokenizer"], sha256(self.args.tokenizer))
        self.assertTrue(data.holdouts.matches("a HELD subject sprite"))

    def test_shuffled_ids_and_array_corruption_are_refused(self):
        build(self.args)
        ids = self.args.out / "ids.txt"
        original = ids.read_text()
        ids.write_text("\n".join(reversed(original.splitlines())) + "\n")
        with self.assertRaisesRegex(ValueError, "integrity mismatch: ids"):
            self.load()
        ids.write_text(original)
        array = self.args.out / "text.npy"
        value = np.load(array)
        value[[0, 1]] = value[[1, 0]]
        value[0, 1] = 7
        np.save(array, value)
        with self.assertRaisesRegex(ValueError, "integrity mismatch: text"):
            self.load()

    def test_array_id_manifest_must_match(self):
        build(self.args)
        file = self.args.out / "manifest.json"
        manifest = json.loads(file.read_text())
        manifest["array_ids_sha256"]["text.npy"] = "0" * 64
        file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "row identity mismatch"):
            self.load()

    def test_required_frozen_holdouts(self):
        self.args.eval50.write_text("")
        with self.assertRaisesRegex(ValueError, "JSON array"):
            build(self.args)
        self.assertFalse(self.args.out.exists())
        self.args.eval50.write_text(json.dumps(self.eval_phrases))
        build(self.args)
        self.args.eval50.write_text(json.dumps(["changed evaluation phrase"] + self.eval_phrases[1:]))
        with self.assertRaisesRegex(ValueError, "holdout byte hashes differ"):
            self.load()

    def test_caption_subject_and_tokenizer_decoded_holdouts_drop(self):
        self.change_caption(0, "a held subject artifact")
        self.provenance[1]["subject"] = "an eval relic object"
        self.change_caption(2, "maskword")
        tokenizer = Tokenizer.from_file(str(self.args.tokenizer))
        tokenizer.normalizer = normalizers.Replace("maskword", "held subject")
        tokenizer.save(str(self.args.tokenizer))
        self.write_metadata()
        result = build(self.args)
        self.assertEqual(result["drops"]["holdout"], 3)
        self.assertNotIn(self.metadata[2]["id"], self.load().ids)
        policy = Holdouts(self.args.blind50, self.args.eval50)
        self.assertFalse(policy.matches("held subjects"))

    def test_holdouts_normalize_punctuation_without_changing_word_boundaries(self):
        policy = Holdouts(self.args.blind50, self.args.eval50)
        for text in ("a HELD-subject artifact", "held_subject", "held\u2014subject",
                     "(held), 'subject'", "held\u3001subject", "held...subject"):
            with self.subTest(text=text):
                self.assertTrue(policy.matches(text))
        for text in ("held subjects", "withheld subject", "heldsubject", "held subjectivity"):
            with self.subTest(text=text):
                self.assertFalse(policy.matches(text))
        self.change_caption(0, "a held-subject artifact")
        self.provenance[1]["subject"] = "an eval_relic object"
        self.change_caption(2, "maskword")
        tokenizer = Tokenizer.from_file(str(self.args.tokenizer))
        tokenizer.normalizer = normalizers.Replace("maskword", "held subject")
        tokenizer.save(str(self.args.tokenizer))
        self.write_metadata()
        result = build(self.args)
        self.assertEqual(result["drops"]["holdout"], 3)
        self.assertTrue(set(row["id"] for row in self.metadata[:3]).isdisjoint(self.load().ids))

    def test_loader_rechecks_punctuation_matches_in_existing_cache(self):
        build(self.args)
        path = self.args.out / "records.json"
        records = json.loads(path.read_text())
        records[0]["trimmed_caption"] = "held\u2014subject"
        path.write_text(json.dumps(records))
        manifest_path = self.args.out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["records.json"] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Frozen holdout"):
            self.load()

    def test_missing_provenance_and_existing_outputs_are_refused(self):
        (self.source / "manifest.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "requires final/metadata"):
            build(self.args)
        self.write_metadata()
        build(self.args)
        manifest_hash = sha256(self.args.out / "manifest.json")
        with self.assertRaisesRegex(ValueError, "already exists"):
            build(self.args)
        self.assertEqual(sha256(self.args.out / "manifest.json"), manifest_hash)

    def test_quantized_dedup_and_provenance_path_alignment(self):
        first = self.source / "final" / self.metadata[0]["file_name"]
        second = self.source / "final" / self.metadata[1]["file_name"]
        second.write_bytes(first.read_bytes())
        result = build(self.args)
        self.assertEqual(result["drops"]["quantized_duplicate"], 1)
        self.args.out = self.root / "new-dataset"
        self.provenance[2]["final_path"] = self.paths[3]
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "differs from provenance"):
            build(self.args)

    def test_r1_paths_are_validated_and_sources_are_filtered(self):
        for value in ("../outside.png", "C:/outside.png", "/outside.png"):
            with self.assertRaisesRegex(ValueError, "relative"):
                relative_path(value)
        self.provenance[0]["source"] = "gameicons"
        self.write_metadata()
        result = build(self.args)
        self.assertEqual(result["drops"]["unapproved_source"], 1)
        self.assertNotIn(self.metadata[0]["id"], self.load().ids)

    def test_alpha_threshold_and_nearest_palette(self):
        rgba = np.zeros((32, 32, 4), np.uint8)
        rgba[0, 0] = [1, 0, 0, 128]
        rgba[0, 1] = [1, 0, 0, 127]
        tokens = Quantizer(self.palette)(rgba)
        self.assertEqual(int(tokens[0]), 1)
        self.assertEqual(int(tokens[1]), 0)

    @needs_holdouts
    def test_holdout_schema_and_default_files(self):
        for values in ([], self.eval_phrases[:-1], ["duplicate"] * 50,
                       ["Uppercase"] + self.eval_phrases[1:], [""] + self.eval_phrases[1:],
                       {"phrases": self.eval_phrases}):
            self.args.eval50.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                Holdouts(self.args.blind50, self.args.eval50)
        self.args.eval50.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            Holdouts(self.args.blind50, self.args.eval50)
        self.assertTrue(Holdouts().hashes["eval50"])

    def test_env_defaults_explicit_flags_and_clip_trimming(self):
        with patch.dict("os.environ", {"PALETTE": "palette-env.npy", "TOKENIZER": "tokenizer-env.json"}):
            parser = make_parser()
            base = ["--data-root", "source", "--r1", "r1.txt", "--out", "new"]
            args = parser.parse_args(base)
            self.assertEqual(args.palette, Path("palette-env.npy"))
            self.assertEqual(args.tokenizer, Path("tokenizer-env.json"))
            self.assertEqual(parser.parse_args(base + ["--palette", "explicit.npy"]).palette, Path("explicit.npy"))
        tokenizer = Tokenizer.from_file(str(self.args.tokenizer))
        short, decoded = encode(tokenizer, "a sprite")
        self.assertEqual(short.tolist(), [49406, 1, 2] + [49407] * 15)
        long, _ = encode(tokenizer, " ".join(["sprite"] * 24))
        self.assertEqual(long.tolist(), [49406] + [2] * 16 + [49407])
        self.assertEqual(decoded, "a sprite")

    def test_directional_and_text_captions_disable_flip(self):
        self.change_caption(0, "a sprite facing left")
        self.change_caption(1, "a sprite with written text")
        self.write_metadata()
        build(self.args)
        data = self.load()
        for index in (0, 1):
            row = data.ids.index(self.metadata[index]["id"])
            self.assertTrue(data.arrays["noflip"][row])

    def make_mixed_subject_sheets(self):
        for index, row in enumerate(self.provenance):
            row["sheet_id"] = index % 24
        self.write_metadata()

    def test_probe_sheet_keeps_other_subjects_in_train_and_iid_sheets_separate(self):
        self.make_mixed_subject_sheets()
        build(self.args)
        data = self.load()
        self.assertEqual(len(data.ids), 48)
        self.assertEqual(data.manifest["split_counts"], {"0": 40, "1": 4, "2": 4})
        self.assertNotIn("probe_sheet_other_subject", data.manifest["drops"])
        sheet_splits = {}
        for row, split in zip(data.records, data.arrays["split"]):
            sheet_splits.setdefault(row["sheet"], set()).add(int(split))
        self.assertIn({0, 2}, sheet_splits.values())
        self.assertTrue(all(splits == {1} for splits in sheet_splits.values() if 1 in splits))
        train_subjects = {row["subject"] for row, split in zip(data.records, data.arrays["split"]) if split == 0}
        probe_subjects = {row["subject"] for row, split in zip(data.records, data.arrays["split"]) if split == 2}
        self.assertFalse(train_subjects & probe_subjects)

    def test_loader_still_rejects_iid_sheet_and_probe_subject_leakage(self):
        self.make_mixed_subject_sheets()
        build(self.args)
        original = np.load(self.args.out / "split.npy")
        records = json.loads((self.args.out / "records.json").read_text())
        probe_rows = np.flatnonzero(original == 2)
        probe_sheets = {records[index]["sheet"] for index in probe_rows}
        shared_train = next(index for index, row in enumerate(records)
                            if original[index] == 0 and row["sheet"] in probe_sheets)
        for row, target, error in ((shared_train, 1, "Sheet leakage"),
                                   (int(probe_rows[0]), 0, "unseen-subject split leakage")):
            changed = original.copy()
            changed[row] = target
            np.save(self.args.out / "split.npy", changed)
            path = self.args.out / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["files"]["split.npy"] = sha256(self.args.out / "split.npy")
            path.write_text(json.dumps(manifest))
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.load()


if __name__ == "__main__":
    unittest.main()
