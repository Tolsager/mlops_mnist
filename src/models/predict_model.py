import argparse
import sys

import torch
import click

from src.data.make_dataset import get_image_and_label_tensors, MNISTDataset
from src.models.model import MyAwesomeModel
@click.group()
def cli():
    pass
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()

    checkpoint = torch.load(model_checkpoint)
    device = checkpoint['device']
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    test_dict = torch.load('data/processed/processed_data.pth')['test']
    ds_test = MNISTDataset(test_dict)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dl_test:
            images = images.to(device)

            out = model(images)
            labels = labels.to(device)

            # metrics
            n_samples += labels.shape[0]
            _, predictions = out.topk(1, dim=1)
            predictions = predictions.view(-1)
            n_correct += (predictions == labels).sum()
        
        print(f"Accuracy: {n_correct / n_samples}")

cli.add_command(evaluate)
if __name__ == '__main__':
    cli()