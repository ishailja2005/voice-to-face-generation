from torch.utils.data import DataLoader

from dataset.face_dataset import FaceImageDataset
from dataset.dataset import VoiceFaceDataset


def make_train_loader(batch_size=8, shuffle=True, num_workers=0):
    return DataLoader(VoiceFaceDataset(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def make_face_autoencoder_loader(batch_size=8, max_faces_per_identity=50, use_raw_faces=False, aligned_face_root=None, num_workers=0):
    dataset = FaceImageDataset(
        max_faces_per_identity=max_faces_per_identity,
        use_raw_faces=use_raw_faces,
        aligned_face_root=aligned_face_root,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
