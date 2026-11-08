import torch
from src.model import DigitNet


def test_dropout_modes_and_logits():
    x = torch.ones(8, 1, 28, 28)
    model = DigitNet(0.5)
    assert model(x).shape == (8, 10)
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))
    baseline = DigitNet(0)
    assert torch.equal(baseline(x), baseline(x))
