import torch

from models.modules.lgda import LGDA


def test_lgda_preserves_shape_and_warm_start_init():
    module = LGDA(channels=8, num_groups=4)
    assert module.alpha.item() == 1e-3
    assert module.beta.item() == 1.0

    x = torch.randn(2, 8, 16, 16)
    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_lgda_supports_backward_pass():
    module = LGDA(channels=8, num_groups=4)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_lgda_rejects_even_kernel_sizes():
    try:
        LGDA(channels=8, spatial_kernel_size=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for even spatial kernel")

    try:
        LGDA(channels=8, channel_kernel_size=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for even channel kernel")
