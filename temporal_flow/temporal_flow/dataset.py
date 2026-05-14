from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class SequenceRecord:
    path: Path
    ref: Path
    frames: List[Path]


def find_sequences(root: str | Path) -> List[SequenceRecord]:
    root = Path(root)
    records: List[SequenceRecord] = []
    for seq_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        ref = seq_dir / "ref.png"
        if not ref.exists():
            # Fall back to first frame as reference.
            candidates = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
            if not candidates:
                continue
            ref = candidates[0]
        frames = sorted([p for p in seq_dir.iterdir() if p.name.startswith("frame_") and p.suffix.lower() in IMG_EXTS])
        if len(frames) >= 1:
            records.append(SequenceRecord(seq_dir, ref, frames))
    if not records:
        raise RuntimeError(f"No valid temporal-flow sequences found under {root}")
    return records


def read_gray_tensor(path: str | Path, width: int | None = None, height: int | None = None) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0)  # 1xHxW


class TemporalFlowClipDataset(Dataset):
    """Return short clips for self-supervised temporal residual flow training.

    Each item:
        ref:    1xHxW tensor
        frames: Kx1xHxW tensor
    """

    def __init__(self, root: str | Path, clip_length: int = 8, width: int | None = 350,
                 height: int | None = 350, samples_per_epoch: int | None = None):
        self.records = find_sequences(root)
        self.clip_length = int(clip_length)
        self.width = width
        self.height = height
        self.samples: List[Tuple[int, int]] = []
        for i, rec in enumerate(self.records):
            n = len(rec.frames)
            if n <= self.clip_length:
                self.samples.append((i, 0))
            else:
                for start in range(0, n - self.clip_length + 1):
                    self.samples.append((i, start))
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.samples)

    def __getitem__(self, idx: int):
        if self.samples_per_epoch is not None:
            rec_idx, start = random.choice(self.samples)
        else:
            rec_idx, start = self.samples[idx]
        rec = self.records[rec_idx]
        ref = read_gray_tensor(rec.ref, self.width, self.height)
        frame_paths = rec.frames[start:start + self.clip_length]
        if len(frame_paths) < self.clip_length:
            frame_paths = frame_paths + [frame_paths[-1]] * (self.clip_length - len(frame_paths))
        frames = torch.stack([read_gray_tensor(p, self.width, self.height) for p in frame_paths], dim=0)
        return {"ref": ref, "frames": frames, "seq": rec.path.name, "start": start}
