from src.models.unet import UNetModel
import torch


def test_forward():
    model = UNetModel(image_size=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y, f = model(x, t, return_features=True)
    assert y.shape == x.shape
    assert f.ndim == 2
