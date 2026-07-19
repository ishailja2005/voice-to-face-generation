import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from models.generator import Generator
from training.dataloader import train_loader
from training.loss import criterion

# Create model
generator = Generator()

# Put model in training mode
generator.train()

# Optimizer
optimizer = optim.Adam(
    generator.parameters(),
    lr=0.0002
)

epochs = 20

for epoch in range(epochs):

    epoch_loss = 0.0

    for batch in train_loader:

        embeddings = batch["embedding"]
        real_images = batch["image"]

        # Forward pass
        generated_images = generator(embeddings)

        # Compute loss
        loss = criterion(generated_images, real_images)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# Save trained model
torch.save(
    generator.state_dict(),
    "checkpoints/generator.pth"
)
print("Model saved successfully!")