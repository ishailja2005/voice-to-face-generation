import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch 
from torch.utils.data import Dataset
from torchvision import transforms

class VoiceFaceDataset(Dataset):

    def __init__(self, manifest_path="data/processed/manifest.csv",
                 embedding_dir="data/processed/embeddings",
                 face_dir="data/processed/faces",
                 transform=None,
    ):
        self.data= pd.read_csv(manifest_path)
        self.embedding_dir = embedding_dir
        self.face_dir = face_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        speaker_id = row["voxceleb2_id"]
        face_id = row["vggface2_id"]

        embedding_folder = os.path.join(
            self.embedding_dir,
            speaker_id
        )

        embedding_files = glob.glob(
            os.path.join(embedding_folder, "*.npy")
        )
        

        embedding = np.load(embedding_files[0])
        image_path = os.path.join(
            self.face_dir,
            f"{face_id}_aligned.jpg"
        )
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        embedding = torch.tensor(
            embedding,
            dtype=torch.float32
        )
        return {
            "embedding": embedding,
            "image": image
        }

