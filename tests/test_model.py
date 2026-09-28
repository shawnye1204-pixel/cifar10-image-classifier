import torch

from src.model import CNN


def test_output_shape():
    model = CNN()

    x = torch.randn(8, 3, 32, 32)

    output = model(x)

    assert output.shape == (8, 10)