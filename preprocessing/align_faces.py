"""Create consistently aligned face crops from the raw VGGFace images."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image


def evenly_sample(paths, limit):
    if not limit or len(paths) <= limit:
        return paths
    return [paths[index * len(paths) // limit] for index in range(limit)]


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-root", type=Path, default=workspace_root / "raw_data" / "vggface2")
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "processed" / "manifest.csv")
    parser.add_argument("--output-root", type=Path, default=project_root / "data" / "aligned_faces")
    parser.add_argument("--max-faces-per-identity", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
    manifest = pd.read_csv(args.manifest)
    identity_ids = sorted(manifest["vggface2_id"].unique())
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    detected = skipped = 0

    for identity_id in identity_ids:
        source_images = sorted((args.face_root / identity_id).glob("*.jpg"))
        source_images += sorted((args.face_root / identity_id).glob("*.jpeg"))
        for source_path in evenly_sample(source_images, args.max_faces_per_identity):
            output_path = args.output_root / f"{identity_id}_{source_path.stem}.jpg"
            if output_path.exists() and not args.overwrite:
                rows.append({"identity_id": identity_id, "source_path": str(source_path), "aligned_path": str(output_path)})
                skipped += 1
                continue
            try:
                image = Image.open(source_path).convert("RGB")
                face = detector(image, save_path=str(output_path))
                if face is None:
                    continue
                rows.append({"identity_id": identity_id, "source_path": str(source_path), "aligned_path": str(output_path)})
                detected += 1
            except Exception:
                continue

    pd.DataFrame(rows).to_csv(args.output_root / "manifest.csv", index=False)
    print(f"Aligned {detected} faces; reused {skipped} existing crops.")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
