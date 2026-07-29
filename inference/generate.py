import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.generator import Generator


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=Path, default=project_root / "data/processed/embeddings/id00040/zK-jkYHtVEg_00203.npy")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "checkpoints/generator.pth")
    parser.add_argument("--output", type=Path, default=project_root / "outputs/generated_face.png")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    embedding_path = args.embedding if args.embedding.is_absolute() else project_root / args.embedding
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}. Run training/train.py first.")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator = Generator(face_feature_dim=checkpoint.get("latent_dim", 512))
    generator.load_state_dict(checkpoint.get("model", checkpoint))
    generator.eval()
    embedding = torch.tensor(np.load(embedding_path), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        generated = generator(embedding).squeeze(0)
    image = transforms.ToPILImage()((generated + 1) / 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Generated face saved to {output_path}")


if __name__ == "__main__":
    main()
