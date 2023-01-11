# -*- coding: utf-8 -*-
import torch.nn.functional as F
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

import torch
import numpy as np
import os


def npz_to_tensors(path: str) -> dict:
    npz_object = np.load(path)
    images = npz_object['images']
    labels = npz_object['labels']

    # convert numpy arrays to tensors
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return {'images': images, 'labels': labels}



def get_image_and_label_tensors(data_path: str) -> dict[str: dict]:
    n_train_files = 5

    train_filenames = [f'train_{i}.npz' for i in range(n_train_files)]
    train_dicts = [npz_to_tensors(os.path.join(data_path, i)) for i in train_filenames]
    train_images = torch.concat([i['images'] for i in train_dicts])
    train_labels = torch.concat([i['labels'] for i in train_dicts])

    test_file = npz_to_tensors(os.path.join(data_path, 'test.npz'))
    test_images = test_file['images']
    test_labels = test_file['labels']

    train_mean = train_images.mean(dim=0)
    test_mean = test_images.mean(dim=0)

    train_images -= train_mean
    train_images = F.normalize(train_images, dim=0)
    test_images -= test_mean
    test_images = F.normalize(test_images, dim=0)
    
    train = {'images': train_images, 'labels': train_labels}
    test = {'images': test_images, 'labels': test_labels}

    return {'train': train, 'test': test}


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_dict = get_image_and_label_tensors(input_filepath)
    torch.save(data_dict, os.path.join(output_filepath, 'processed_data.pth'))

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        super().__init__()
        self.data_dict = data_dict

    def __getitem__(self, i):
        return self.data_dict['images'][i].type(torch.float32), self.data_dict['labels'][i]

    def __len__(self):
        return self.data_dict['images'].shape[0]

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
