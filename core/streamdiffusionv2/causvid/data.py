from __future__ import annotations

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Minimal prompt dataset used by streamv2v CLI entrypoints."""

    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as handle:
            self.texts = [line.strip() for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]
