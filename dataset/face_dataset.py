"""Face-only dataset used to train a stronger, canonical face autoencoder."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceImageDataset(Dataset):
    """Loads several VGGFace images for every identity in the paired manifest."""

    def __init__(self, manifest_path=None, face_root=None, max_faces_per_identity=50, use_raw_faces=False, aligned_face_root=None):
        project_root = Path(__file__).resolve().parents[1]
        workspace_root = project_root.parent
        if aligned_face_root:
            aligned_face_root = Path(aligned_face_root)
            self.image_paths = sorted(aligned_face_root.glob("*.jpg"))
            if not self.image_paths:
                raise FileNotFoundError(f"No aligned face images found under {aligned_face_root}")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ])
            return
        if not use_raw_faces:
            processed_faces = project_root / "data" / "processed" / "faces"
            self.image_paths = sorted(processed_faces.glob("*.jpg"))
            if not self.image_paths:
                raise FileNotFoundError(f"No aligned face images found under {processed_faces}")
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ])
            return

        manifest_path = Path(manifest_path) if manifest_path else project_root / "data" / "processed" / "manifest.csv"
        face_root = Path(face_root) if face_root else workspace_root / "raw_data" / "vggface2"

        # Raw VGGFace images have varied pose and lighting; this option is experimental.
        identity_rows = manifest_path.read_text(encoding="utf-8").splitlines()[1:]
        identity_ids = sorted({row.split(",")[3] for row in identity_rows})
        self.image_paths = []
        for identity_id in identity_ids:
            images = sorted((face_root / identity_id).glob("*.jpg"))
            images += sorted((face_root / identity_id).glob("*.jpeg"))
            if max_faces_per_identity and len(images) > max_faces_per_identity:
                indices = [index * len(images) // max_faces_per_identity for index in range(max_faces_per_identity)]
                images = [images[index] for index in indices]
            self.image_paths.extend(images)

        if not self.image_paths:
            raise FileNotFoundError(f"No face images found under {face_root}")

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)
