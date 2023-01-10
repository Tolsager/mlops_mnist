import os
import pytest
import torch


@pytest.mark.skipif(
    not os.path.exists("data/processed/processed_data.pth"),
    reason="processed data not found",
)
def test_data():
    ds_path = os.path.join("data", "processed", "processed_data.pth")
    data = torch.load(ds_path)
    train_dict = data["train"]
    test_dict = data["test"]

    assert train_dict["labels"].shape[0] == 25_000
    assert train_dict["images"].shape[0] == 25_000

    assert test_dict["labels"].shape[0] == 5_000
    assert test_dict["images"].shape[0] == 5_000

    # check shape of train
    for i in range(25_000):
        assert train_dict["images"][i].shape == (28, 28)

    # check shape of test
    for i in range(5_000):
        assert test_dict["images"][i].shape == (28, 28)

    # check that all classes are represented in train and test
    assert len(torch.unique(train_dict["labels"])) == 10
    assert len(torch.unique(test_dict["labels"])) == 10
