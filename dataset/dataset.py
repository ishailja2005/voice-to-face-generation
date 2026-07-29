"""Dataset utilities for paired voice embeddings and aligned face images."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VoiceFaceDataset(Dataset):
    """Loads the exact audio-face pairs listed in a preprocessing manifest."""

    def __init__(self, manifest_path=None, transform=None):
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_root = self.project_root / "data"
        self.manifest_path = Path(manifest_path) if manifest_path else self.data_root / "processed" / "manifest.csv"
        self.data = pd.read_csv(self.manifest_path)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def _resolve(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return self.data_root / path

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        embedding = np.load(self._resolve(row["embedding_path"])).astype(np.float32)
        image = Image.open(self._resolve(row["face_image_path"])).convert("RGB")
        return {
            "embedding": torch.from_numpy(embedding),
            "image": self.transform(image),
            "speaker_id": row["voxceleb2_id"],
        }
