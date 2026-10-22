import torch

from models.modules.class_aware_fpn import ClassAwareGatedFPN


def _four_laterals():
    return [
        torch.randn(2, 32, 64, 64),
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 32, 8, 8),
    ]


def test_class_aware_gated_fpn_output_shape():
    module = ClassAwareGatedFPN(channels=32, num_classes=5, num_levels=4)
    laterals = _four_laterals()

    y = module(laterals)

    assert y.shape == (2, 32, 64, 64)


def test_class_aware_gated_fpn_backward():
    module = ClassAwareGatedFPN(channels=32, num_classes=5, num_levels=4)
    laterals = [t.clone().requires_grad_(True) for t in _four_laterals()]

    module(laterals).mean().backward()

    for t in laterals:
        assert t.grad is not None
        assert t.grad.shape == t.shape

    for p in module.parameters():
        assert p.grad is not None


def test_class_protos_and_scale_weights_are_parameters():
    module = ClassAwareGatedFPN(channels=32, num_classes=5, num_levels=4)

    assert isinstance(module.class_protos, torch.nn.Parameter)
    assert module.class_protos.shape == (5, 32)

    assert isinstance(module.scale_weights, torch.nn.Parameter)
    assert module.scale_weights.shape == (4,)


def test_num_levels_must_be_at_least_two():
    try:
        ClassAwareGatedFPN(channels=32, num_levels=1)
    except ValueError as e:
        assert "num_levels" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for num_levels < 2")


def test_wrong_number_of_laterals_raises():
    module = ClassAwareGatedFPN(channels=32, num_levels=4)
    try:
        module([torch.randn(2, 32, 8, 8)])
    except ValueError as e:
        assert "expected 4" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong lateral count")


def test_wrong_channel_count_raises():
    module = ClassAwareGatedFPN(channels=32, num_levels=4)
    bad = [
        torch.randn(2, 16, 64, 64),
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 32, 8, 8),
    ]
    try:
        module(bad)
    except ValueError as e:
        assert "channels" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for wrong channel count")
