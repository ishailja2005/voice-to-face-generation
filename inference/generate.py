import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models.generator import Generator
generator = Generator()
generator.load_state_dict(
    torch.load("checkpoints/generator.pth", map_location="cpu")
)
generator.eval()
embedding_path = "data/processed/embeddings/id00040"
embedding_file = glob.glob(
    os.path.join(embedding_path, "*.npy")
)[0]
embedding = np.load(embedding_file)
embedding = torch.tensor(
    embedding,
    dtype=torch.float32
)
embedding = embedding.unsqueeze(0)
with torch.no_grad():

    generated = generator(embedding)
generated = generated.squeeze(0)
generated = (generated + 1) / 2
generated = transforms.ToPILImage()(generated)
os.makedirs("outputs", exist_ok=True)

generated.save(
    "outputs/generated_face.png"
)
print("Generated face saved successfully!")