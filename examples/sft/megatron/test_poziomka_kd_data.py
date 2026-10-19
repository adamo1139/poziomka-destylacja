"""CPU checks for poziomka_kd_data on a synthetic cache: `python3 test_poziomka_kd_data.py`."""
import json
import tempfile
from pathlib import Path

import numpy as np

from poziomka_data import FORMAT, SPECIAL_IDS, MMapSFTDataset, sha256
from poziomka_kd_data import THINK_OFF_BLOCK, MMapKDDataset, label_coverage, mask_think_block

K = 4


def build(root, seq_length=64):
    """Two train shards + validation; teacher row j has token j as its top-1."""
    rng = np.random.default_rng(0)
    root, teacher = Path(root) / "cache", Path(root) / "teacher"
    shards, tshards = [], []
    for split, prefix, n in (("train", "train/shard_00000", 30), ("train", "train/shard_00001", 25),
                             ("validation", "validation/shard_00002", 9)):
        (root / split).mkdir(parents=True, exist_ok=True)
        (teacher / split).mkdir(parents=True, exist_ok=True)
        lengths = rng.integers(3, seq_length + 2, n)
        tokens = [np.concatenate([[1], rng.integers(5, 32000, L - 1)]) for L in lengths]
        flat = np.concatenate(tokens).astype("<u2")
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype("<u8")
        masks = np.ones(len(flat), dtype=np.uint8)
        masks[offsets[:-1]] = 0
        flat.tofile(root / f"{prefix}.tokens.bin")
        masks.tofile(root / f"{prefix}.masks.bin")
        offsets.tofile(root / f"{prefix}.offsets.bin")
        shards.append(dict(split=split, prefix=prefix, records=int(n), tokens=int(len(flat))))
        ids = rng.integers(5, 32000, (len(flat), K)).astype("<u2")
        ids[:, 0] = flat
        logp = np.sort(rng.normal(-3, 1, (len(flat), K)).astype("<f2"), axis=1)[:, ::-1]
        ids[offsets[:-1]], logp[offsets[:-1]] = 0, 0  # BOS rows are zeros
        files = {name: f"{prefix}.{name}.bin" for name in ("topk_ids", "topk_logp", "target_logp")}
        ids.tofile(teacher / files["topk_ids"])
        np.ascontiguousarray(logp).tofile(teacher / files["topk_logp"])
        tshards.append(dict(prefix=prefix, split=split, records=int(n), tokens=int(len(flat)), files=files))
    (root / "manifest.json").write_text(json.dumps(dict(
        format=FORMAT, special_ids=SPECIAL_IDS, seq_length=seq_length, shards=shards)))
    (teacher / "manifest.json").write_text(json.dumps(dict(
        format="teacher-topk-v1", k=K, cache_manifest_sha256=sha256(root / "manifest.json"),
        shards=tshards)))
    return root / "manifest.json", teacher


def check_alignment(pack):
    with tempfile.TemporaryDirectory() as tmp:
        manifest, teacher = build(tmp)
        kd = MMapKDDataset(manifest, "train", teacher, pack=pack, seed=7)
        sft = MMapSFTDataset(manifest, "train", pack=pack, seed=7)
        assert len(kd) == len(sft)
        for i in range(len(kd)):
            a, b = kd[i], sft[i]
            for key in b:
                assert (a[key] == b[key]).all(), key
            mask = a["loss_mask"] > 0
            # Every supervised label is the teacher's top-1 at the same position.
            assert (a["topk_ids"][mask, 0] == a["labels"][mask]).all()
            assert a["topk_ids"].shape == (64, K) and a["topk_logp"].dtype == np.float32
        assert label_coverage(kd, samples=len(kd)) == 1.0
        plain = MMapKDDataset(manifest, "validation", teacher, pack=pack, with_teacher=False)
        assert "topk_ids" not in plain[0]


def test_alignment_packed():
    check_alignment(pack=True)


def test_alignment_unpacked():
    check_alignment(pack=False)


def test_rejects_other_cache():
    with tempfile.TemporaryDirectory() as tmp:
        manifest, teacher = build(tmp)
        meta = json.loads((teacher / "manifest.json").read_text())
        meta["cache_manifest_sha256"] = "0" * 64
        (teacher / "manifest.json").write_text(json.dumps(meta))
        try:
            MMapKDDataset(manifest, "train", teacher, pack=True)
        except ValueError as e:
            assert "different cache" in str(e)
        else:
            raise AssertionError("mismatched teacher accepted")


def test_think_block_matches_apt4_and_masks_only_it():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(__file__).resolve().parents[3] / "resource/tokenizer/apt4/tokenizer.json"))
    # Dwie tury v13 bez rozumowania, jak w cache (labels = ids[1:]).
    text = ("<s><|im_start|>user\nJak?<|im_end|>\n<|im_start|>assistant\n<think></think>\nTak.<|im_end|>\n"
            "<|im_start|>user\nA czemu?<|im_end|>\n<|im_start|>assistant\n<think></think>\n**Bo** tak.<|im_end|>\n")
    ids = np.array(tok.encode(text, add_special_tokens=False).ids, dtype=np.int64)
    labels, mask = ids[1:], np.ones(len(ids) - 1, dtype=np.float32)
    assert mask_think_block(labels, mask) == 2
    assert int((mask == 0).sum()) == 2 * len(THINK_OFF_BLOCK)
    # Zamaskowane są dokładnie tokeny bloku, a token tuż po nim (początek odpowiedzi) nie.
    for start in np.flatnonzero(mask[:-1] - mask[1:] == 1) + 1:
        assert (labels[start:start + len(THINK_OFF_BLOCK)] == THINK_OFF_BLOCK).all()
        assert mask[start + len(THINK_OFF_BLOCK)] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
