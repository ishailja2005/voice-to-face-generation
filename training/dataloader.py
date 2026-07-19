from torch.utils.data import DataLoader 
from dataset.dataset import VoiceFaceDataset
dataset = VoiceFaceDataset()
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)
