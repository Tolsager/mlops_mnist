from src.models.model import MyAwesomeModel
import pytest
import torch


def test_forward():
    model = MyAwesomeModel()

    sample_input = torch.randn((1, 28, 28))
    out = model(sample_input)
    assert out.shape == (1, 10), "the out shape of the model doesn't match"

    sample_input2 = torch.randn((1, 1, 28, 28))
    with pytest.raises(ValueError):
        model(sample_input2)
