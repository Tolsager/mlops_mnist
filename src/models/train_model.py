import argparse
import sys
import matplotlib.pyplot as plt

import torch
import os
import click

print(os.getcwd())
# from src.data.make_dataset import get_image_and_label_tensors
from src.data.make_dataset import MNISTDataset
from src.models.model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=3, help="number of epochs to train for")
@click.option("--batch_size", default=64, help="amount of samples in each batch")
@click.option("--device", default="cuda", help="the device to train the model on")
def train(lr, epochs, batch_size, device):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    train_dict = torch.load("data/processed/processed_data.pth")["train"]
    ds_train = MNISTDataset(train_dict)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    accuracies = []
    model.train()
    start_weights = list(model.parameters())
    for epoch in range(epochs):
        n_correct = 0
        n_samples = 0
        for images, labels in dl_train:
            images = images.to(device)
            optimizer.zero_grad()

            out = model(images)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            # metrics
            n_samples += labels.shape[0]
            _, predictions = out.topk(1, dim=1)
            predictions = predictions.view(-1)
            n_correct += (predictions == labels).sum()

        accuracy = (n_correct / n_samples).cpu().item()
        print(f"Accuracy: {accuracy}")
        accuracies.append(accuracy)

        plt.plot(accuracies)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training curve")
        plt.savefig("reports/figures/training_curve.png")

    checkpoint = {"device": device, "state_dict": model.state_dict()}
    torch.save(checkpoint, "models/checkpoint.pth")


cli.add_command(train)

if __name__ == "__main__":
    cli()
