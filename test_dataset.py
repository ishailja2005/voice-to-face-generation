from dataset.dataset import VoiceFaceDataset

# Create dataset
dataset = VoiceFaceDataset()

print("=" * 50)
print("Dataset loaded successfully!")
print("Total samples:", len(dataset))
print("=" * 50)

# Load the first sample
sample = dataset[0]

print("Embedding Shape :", sample["embedding"].shape)
print("Image Shape     :", sample["image"].shape)

print("Embedding Type  :", type(sample["embedding"]))
print("Image Type      :", type(sample["image"]))