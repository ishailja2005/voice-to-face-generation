"""Stage 2: train the voice encoder against a pretrained face autoencoder."""

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.face_autoencoder import FaceEncoder
from models.generator import Generator
from training.dataloader import make_train_loader
from training.loss import Speech2FaceLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--face-checkpoint", type=Path, default=Path("checkpoints/face_autoencoder.pth"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/generator.pth"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    face_checkpoint = args.face_checkpoint if args.face_checkpoint.is_absolute() else project_root / args.face_checkpoint
    output = args.output if args.output.is_absolute() else project_root / args.output
    if not face_checkpoint.exists():
        raise FileNotFoundError(
            f"Face autoencoder checkpoint not found: {face_checkpoint}. "
            "Run training/train_face_autoencoder.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_state = torch.load(face_checkpoint, map_location=device)
    latent_dim = face_state["latent_dim"]
    face_encoder = FaceEncoder(latent_dim).to(device)
    generator = Generator(face_feature_dim=latent_dim).to(device)
    face_encoder.load_state_dict(face_state["encoder"])
    generator.face_decoder.load_state_dict(face_state["decoder"])
    for module in (face_encoder, generator.face_decoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    optimizer = Adam(generator.voice_encoder.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    criterion = Speech2FaceLoss()
    loader = make_train_loader(args.batch_size)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        generator.voice_encoder.train()
        total_loss = 0.0
        for batch in loader:
            embeddings, images = batch["embedding"].to(device), batch["image"].to(device)
            with torch.no_grad():
                target_features = face_encoder(images)
            predicted_features = generator.voice_encoder(embeddings)
            generated_images = generator.face_decoder(predicted_features)
            loss, _, _ = criterion(predicted_features, target_features, generated_images, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        scheduler.step(average_loss)
        print(f"Speech2Face {epoch + 1}/{args.epochs} | Loss: {average_loss:.4f}")
        if average_loss < best_loss:
            best_loss = average_loss
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": generator.state_dict(), "latent_dim": latent_dim}, output)

    print(f"Saved best generator checkpoint to {output} (loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
