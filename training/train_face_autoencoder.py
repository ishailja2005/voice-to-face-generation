"""Stage 1: learn a canonical face-feature space and its decoder."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.face_autoencoder import FaceDecoder, FaceEncoder
from training.dataloader import make_face_autoencoder_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--max-faces-per-identity",
        type=int,
        default=50,
        help="Number of raw VGGFace images used per matched identity; use 0 for all images.",
    )
    parser.add_argument(
        "--use-raw-faces",
        action="store_true",
        help="Experimental: use unaligned raw VGGFace images instead of the default aligned faces.",
    )
    parser.add_argument(
        "--aligned-face-root",
        type=Path,
        default=None,
        help="Folder created by preprocessing/align_faces.py. This is recommended for clearer output.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = FaceEncoder(args.latent_dim).to(device), FaceDecoder(args.latent_dim).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    project_root = Path(__file__).resolve().parents[1]
    aligned_face_root = args.aligned_face_root
    if aligned_face_root and not aligned_face_root.is_absolute():
        aligned_face_root = project_root / aligned_face_root
    loader = make_face_autoencoder_loader(
        args.batch_size,
        args.max_faces_per_identity,
        args.use_raw_faces,
        aligned_face_root,
    )
    loss_fn = nn.L1Loss()
    checkpoint = project_root / "checkpoints" / "face_autoencoder.pth"
    checkpoint.parent.mkdir(exist_ok=True)
    best_loss = float("inf")
    print(f"Training face autoencoder on {len(loader.dataset)} face images")

    for epoch in range(args.epochs):
        encoder.train(); decoder.train(); total = 0.0
        for images in loader:
            images = images.to(device)
            loss = loss_fn(decoder(encoder(images)), images)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        average_loss = total / len(loader)
        scheduler.step(average_loss)
        print(f"Face AE {epoch + 1}/{args.epochs} | L1: {average_loss:.4f}")
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "latent_dim": args.latent_dim},
                checkpoint,
            )

    print(f"Saved best face autoencoder checkpoint to {checkpoint} (L1: {best_loss:.4f})")


if __name__ == "__main__":
    main()
