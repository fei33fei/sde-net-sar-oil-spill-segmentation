import torch

from models.modules.spcm import (
    SPCM,
    StableStripSpatialAttention,
)


def test_spcm_preserves_spatial_shape_and_sets_output_channels():
    module = SPCM(
        in_channels=16,
        out_channels=8,
        pool_scales=(1, 2),
        strip_kernel_size=7,
        use_local_branch=True,
    )
    x = torch.randn(2, 16, 16, 16)

    y = module(x)

    assert y.shape == (2, 8, 16, 16)


def test_spcm_supports_without_local_branch():
    module = SPCM(
        in_channels=16,
        out_channels=8,
        pool_scales=(1, 2),
        use_local_branch=False,
    )
    x = torch.randn(2, 16, 16, 16, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_spcm_identity_branch_always_present_and_fusion_width():
    """Identity branch mirrors PPM's raw-x concat; it exists even without local."""
    for use_local in (True, False):
        module = SPCM(
            in_channels=16,
            out_channels=8,
            pool_scales=(1, 2),
            use_local_branch=use_local,
        )
        assert hasattr(module, "identity_branch")
        num = len(module.pool_branches) + 2 + (1 if use_local else 0)
        fusion_in = module.fusion.block[0].in_channels
        assert fusion_in == 8 * num


def test_strip_attention_rejects_even_kernel_size():
    try:
        StableStripSpatialAttention(channels=16, strip_kernel_size=6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for even strip kernel")
