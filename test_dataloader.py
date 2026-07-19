from training.dataloader import train_loader

for batch in train_loader:

    print(batch["embedding"].shape)
    print(batch["image"].shape)

    break