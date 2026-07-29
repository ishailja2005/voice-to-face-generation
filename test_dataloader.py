from training.dataloader import make_train_loader

for batch in make_train_loader():

    print(batch["embedding"].shape)
    print(batch["image"].shape)

    break
