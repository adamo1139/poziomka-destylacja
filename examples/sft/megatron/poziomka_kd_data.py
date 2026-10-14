"""SFT cache + teacher top-k rows (teacher-topk-v1 from polskie-dane-instruktazowe/destylacja).

The teacher directory mirrors the cache shard by shard: row j of {prefix}.topk_ids.bin and
{prefix}.topk_logp.bin is the teacher's distribution for token j of {prefix}.tokens.bin
(row of a record's BOS is zeros). MMapSFTDataset makes labels = ids[s+1:e], so the label at
window position i is token s+1+i and its teacher row is exactly s+1+i: the same slice.
Packing, shuffling and bins are inherited unchanged, so a KD run sees the same windows an
SFT run on this cache would.
"""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from poziomka_data import MMapSFTDataset, sha256

TEACHER_FORMAT = "teacher-topk-v1"


def check_teacher(teacher_dir, manifest_path):
    """Teacher top-k must be computed from this exact cache; returns the teacher manifest."""
    teacher_dir, manifest_path = Path(teacher_dir).resolve(), Path(manifest_path).resolve()
    teacher = json.loads((teacher_dir / "manifest.json").read_text())
    if teacher["format"] != TEACHER_FORMAT:
        raise ValueError(f"Teacher top-k format {teacher['format']}, expected {TEACHER_FORMAT}")
    if teacher["cache_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Teacher top-k was computed for a different cache manifest")
    cache = json.loads(manifest_path.read_text())
    want = {s["prefix"]: s["tokens"] for s in cache["shards"] if s["records"]}
    have = {s["prefix"]: s["tokens"] for s in teacher["shards"]}
    if want != have:
        raise ValueError("Teacher top-k shards differ from cache shards")
    k = teacher["k"]
    for shard in teacher["shards"]:
        for name, itemsize in (("topk_ids", 2), ("topk_logp", 2)):
            path = teacher_dir / shard["files"][name]
            if path.stat().st_size != shard["tokens"] * k * itemsize:
                raise ValueError(f"Incomplete teacher file: {path}")
    return teacher


class MMapKDDataset(MMapSFTDataset):
    """MMapSFTDataset plus topk_ids [seq, K] int64 and topk_logp [seq, K] float32 per window.

    `with_teacher=False` returns plain SFT samples (pipeline stages that never compute the
    loss don't need to read ~2 MB of teacher rows per sample).
    """

    def __init__(self, manifest_path, split, teacher_dir, *args, with_teacher=True, **kwargs):
        super().__init__(manifest_path, split, *args, **kwargs)
        self.teacher_dir = Path(teacher_dir).resolve()
        self.teacher = check_teacher(self.teacher_dir, manifest_path)
        self.k = self.teacher["k"]
        self.with_teacher = with_teacher
        self._files = {s["prefix"]: s["files"] for s in self.teacher["shards"]}
        self._teacher_maps = OrderedDict()

    def __getstate__(self):
        state = super().__getstate__()
        state["_teacher_maps"] = OrderedDict()
        return state

    def _teacher_rows(self, shard_id):
        if shard_id not in self._teacher_maps:
            shard = self.shards[shard_id]
            files = self._files[shard["prefix"]]
            self._teacher_maps[shard_id] = tuple(
                np.memmap(self.teacher_dir / files[name], dtype=dtype, mode="r",
                          shape=(shard["tokens"], self.k))
                for name, dtype in (("topk_ids", "<u2"), ("topk_logp", "<f2")))
            if len(self._teacher_maps) > self.max_subsequences:
                self._teacher_maps.popitem(last=False)
        self._teacher_maps.move_to_end(shard_id)
        return self._teacher_maps[shard_id]

    def _records(self, index):
        row = self._shuffled_row(index)
        return [int(r) for r in self.bins[row]] if self.pack else [row]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self.with_teacher:
            return sample
        # Padding rows: id 0 with log p 0 -- finite garbage, masked by loss_mask.
        ids = np.zeros((self.seq_length, self.k), dtype=np.int64)
        logp = np.zeros((self.seq_length, self.k), dtype=np.float32)
        filled = 0
        for record in self._records(index):
            shard_id, local_row = self._locate(record)
            offsets = self._shard_maps(shard_id)[2]
            start, end = int(offsets[local_row]), int(offsets[local_row + 1])
            size = end - start - 1
            teacher_ids, teacher_logp = self._teacher_rows(shard_id)
            ids[filled:filled + size] = teacher_ids[start + 1:end]
            logp[filled:filled + size] = teacher_logp[start + 1:end]
            filled += size
        sample["topk_ids"], sample["topk_logp"] = ids, logp
        return sample


def label_coverage(dataset, samples=4):
    """Share of supervised labels found among the teacher's top-k (expected ~0.99).

    A misaligned teacher (off-by-one rows, wrong cache) drops this to ~K/vocab.
    """
    hits = total = 0
    for i in range(min(samples, len(dataset))):
        s = dataset[i]
        mask = s["loss_mask"] > 0
        hits += int((s["topk_ids"][mask] == s["labels"][mask, None]).any(-1).sum())
        total += int(mask.sum())
    return hits / max(total, 1)
