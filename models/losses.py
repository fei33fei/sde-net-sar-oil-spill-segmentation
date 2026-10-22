"""Loss functions for segmentation with severe class imbalance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_xy_depthwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel Sobel; ``x`` is ``(B, C, H, W)``."""
    b, c, h, w = x.shape
    device, dtype = x.device, x.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    wx = kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    wy = ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, wx, padding=1, groups=c)
    gy = F.conv2d(x, wy, padding=1, groups=c)
    return gx, gy


class SobelEdgeMagnitudeLoss(nn.Module):
    """L1 between gradient magnitudes of pred probs and GT one-hot (Sobel).

    Matches the formulation ``|| sqrt(Gx^2+Gy^2+eps)_pred
    - sqrt(Gx^2+Gy^2+eps)_gt ||_1`` with fixed 3×3 Sobel kernels.
    """

    def __init__(
        self,
        num_classes: int = 5,
        ignore_index: int = 255,
        edge_eps: float = 1e-6,
        edge_class_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge_eps = edge_eps
        if edge_class_ids is not None:
            for i in edge_class_ids:
                if not 0 <= i < num_classes:
                    raise ValueError(f"edge_class_ids must be in [0, {num_classes}), got {i}")
        self.edge_class_ids = edge_class_ids

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_long = targets.long()
        valid = targets_long != self.ignore_index
        targets_safe = targets_long.clone()
        targets_safe[~valid] = 0
        one_hot = F.one_hot(targets_safe, self.num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()

        gx_p, gy_p = _sobel_xy_depthwise(probs)
        gx_g, gy_g = _sobel_xy_depthwise(one_hot)
        mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + self.edge_eps)
        mag_g = torch.sqrt(gx_g * gx_g + gy_g * gy_g + self.edge_eps)
        diff = (mag_p - mag_g).abs()

        if self.edge_class_ids is not None:
            diff = diff[:, self.edge_class_ids]

        mask = valid.unsqueeze(1).expand_as(diff).float()
        num = (diff * mask).sum()
        den = mask.sum().clamp(min=1.0)
        return num / den


class DiceLoss(nn.Module):
    """Per-class Dice loss averaged over classes.

    Handles ``ignore_index`` by zeroing out both prediction and target for
    ignored pixels before computing Dice per class.
    """

    def __init__(
        self,
        num_classes: int = 5,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_long = targets.long()

        valid = targets_long != self.ignore_index
        targets_safe = targets_long.clone()
        # Ignored pixels need a legal class id for one_hot; they are masked out
        # immediately afterwards and therefore do not affect Dice statistics.
        targets_safe[~valid] = 0

        one_hot = F.one_hot(targets_safe, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()

        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    """Weighted combination of CrossEntropy and Dice loss.

    ``loss = ce_weight * CE + dice_weight * Dice``

    Args:
        num_classes: Number of semantic classes.
        ce_weight: Scalar weight for CE term (default 1.0).
        dice_weight: Scalar weight for Dice term (default 1.0).
        class_weights: Optional per-class weights for CE (e.g. inverse
            frequency). Tensor of shape ``(num_classes,)``.
        ignore_index: Label index to ignore in both CE and Dice.
    """

    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # Class weights are optional and only affect the CE term; Dice remains
        # class-balanced by averaging per-class overlap.
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        return self.ce_weight * self.ce(logits, targets.long()) + \
               self.dice_weight * self.dice(logits, targets)


class CEDiceEdgeLoss(nn.Module):
    """Cross-entropy + Dice + Sobel edge magnitude L1.

    ``loss = ce_weight * CE + dice_weight * Dice + edge_weight * L_edge``

    The edge term matches gradient magnitude agreement between softmax
    probabilities and a masked one-hot target (fixed Sobel 3×3). Use
    ``edge_class_ids`` to restrict channels (e.g. ``[1]`` for oil only).

    Default weights mirror plain ``CEDiceLoss`` when ``edge_weight=0`` is
    *not* used; set ``edge_weight`` small (e.g. 0.1) if the edge term
    destabilises training.
    """

    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        edge_weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
        edge_eps: float = 1e-6,
        edge_class_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.edge = SobelEdgeMagnitudeLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            edge_eps=edge_eps,
            edge_class_ids=edge_class_ids,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        return (
            self.ce_weight * self.ce(logits, targets.long())
            + self.dice_weight * self.dice(logits, targets)
            + self.edge_weight * self.edge(logits, targets)
        )


class FocalCEDiceLoss(nn.Module):
    """Focal CrossEntropy + Dice loss.

    Replaces standard CE with Focal CE so that well-classified (easy) pixels
    contribute less gradient, forcing the model to focus on hard boundaries
    such as oil / look-alike transitions.

    ``loss = focal_weight * FocalCE + dice_weight * Dice``
    """

    def __init__(
        self,
        num_classes: int = 5,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None,
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def _focal_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self._focal_ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


class LookAlikeBoostCEDiceLoss(nn.Module):
    """CEDiceLoss with an extra CE term on look_alike pixels.

    ``loss = ce_weight * CE_all + dice_weight * Dice
             + la_boost_weight * mean(CE[label == la_class_id])``

    The base ``CEDiceLoss`` term already includes per-class ``class_weights``;
    the boost term adds *additional* gradient specifically on look_alike pixels
    without altering the loss on other classes.
    """

    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.5,
        la_boost_weight: float = 1.0,
        la_class_id: int = 2,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.base = CEDiceLoss(
            num_classes=num_classes,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.la_boost_weight = la_boost_weight
        self.la_class_id = la_class_id
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.base(logits, targets)
        ce_map = F.cross_entropy(
            logits, targets.long(),
            ignore_index=self.ignore_index,
            reduction='none',
        )
        la_mask = targets.long() == self.la_class_id
        if la_mask.any():
            la_loss = ce_map[la_mask].mean()
        else:
            la_loss = ce_map.new_tensor(0.0)
        return base_loss + self.la_boost_weight * la_loss


class OHEMCEDiceLoss(nn.Module):
    """CE + Dice with Online Hard Example Mining on the CE term.

    Only the hardest ``ohem_ratio`` fraction of valid pixels contribute to the
    CE scalar loss.  Dice is computed on *all* valid pixels as usual so that
    per-class overlap statistics remain meaningful.

    ``loss = ce_weight * OHEM_CE + dice_weight * Dice``
    """

    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.5,
        ohem_ratio: float = 0.7,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if not 0.0 < ohem_ratio <= 1.0:
            raise ValueError(f"ohem_ratio must be in (0, 1], got {ohem_ratio}")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ohem_ratio = ohem_ratio
        self.ignore_index = ignore_index
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None,
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def _ohem_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_map = F.cross_entropy(
            logits, targets.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        valid = targets.long() != self.ignore_index
        valid_losses = ce_map[valid]
        n = valid_losses.numel()
        if n == 0:
            return ce_map.new_tensor(0.0)
        k = max(1, int(self.ohem_ratio * n))
        hard_losses, _ = valid_losses.topk(k)
        return hard_losses.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
        return self.ce_weight * self._ohem_ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


def build_loss(lcfg: dict, num_classes: int) -> nn.Module:
    """Factory that builds a loss module from the ``model.loss`` config block."""
    loss_type = lcfg.get('type', 'CEDiceLoss')
    ignore_index = int(lcfg.get('ignore_index', 255))
    class_weights = lcfg.get('class_weights', None)

    if loss_type == 'FocalCEDiceLoss':
        return FocalCEDiceLoss(
            num_classes=num_classes,
            focal_weight=float(lcfg.get('focal_weight', 1.0)),
            dice_weight=float(lcfg.get('dice_weight', 1.0)),
            gamma=float(lcfg.get('gamma', 2.0)),
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    if loss_type == 'OHEMCEDiceLoss':
        return OHEMCEDiceLoss(
            num_classes=num_classes,
            ce_weight=float(lcfg.get('ce_weight', 1.0)),
            dice_weight=float(lcfg.get('dice_weight', 1.5)),
            ohem_ratio=float(lcfg.get('ohem_ratio', 0.7)),
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    if loss_type == 'LookAlikeBoostCEDiceLoss':
        return LookAlikeBoostCEDiceLoss(
            num_classes=num_classes,
            ce_weight=float(lcfg.get('ce_weight', 1.0)),
            dice_weight=float(lcfg.get('dice_weight', 1.5)),
            la_boost_weight=float(lcfg.get('la_boost_weight', 1.0)),
            la_class_id=int(lcfg.get('la_class_id', 2)),
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    if loss_type == 'CEDiceEdgeLoss':
        edge_ids = lcfg.get('edge_class_ids', None)
        if edge_ids is not None:
            edge_ids = [int(x) for x in edge_ids]
        return CEDiceEdgeLoss(
            num_classes=num_classes,
            ce_weight=float(lcfg.get('ce_weight', 1.0)),
            dice_weight=float(lcfg.get('dice_weight', 1.0)),
            edge_weight=float(lcfg.get('edge_weight', 1.0)),
            class_weights=class_weights,
            ignore_index=ignore_index,
            edge_eps=float(lcfg.get('edge_eps', 1e-6)),
            edge_class_ids=edge_ids,
        )

    return CEDiceLoss(
        num_classes=num_classes,
        ce_weight=float(lcfg.get('ce_weight', 1.0)),
        dice_weight=float(lcfg.get('dice_weight', 1.0)),
        class_weights=class_weights,
        ignore_index=ignore_index,
    )
