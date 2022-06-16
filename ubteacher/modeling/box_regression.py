from typing import Tuple

import torch

_DEFAULT_SCALE_CLAMP = 1000.0 / 16


__all__ = ["Box2BoxXYXYTransform"]


@torch.jit.script
class Box2BoxXYXYTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float],
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Follow the KL-Loss implementation (CVPR'19)
        https://github.com/yihui-he/KL-Loss/blob/1c67310c9f5a79cfa985fea241791ccedbdb7dcf/detectron/utils/boxes.py#L328-L353

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        tgt_l = target_boxes[:, 0]
        tgt_r = target_boxes[:, 2]
        tgt_d = target_boxes[:, 1]
        tgt_u = target_boxes[:, 3]

        src_l = src_boxes[:, 0]
        src_r = src_boxes[:, 2]
        src_d = src_boxes[:, 1]
        src_u = src_boxes[:, 3]

        src_widths = src_r - src_l + 1.0
        src_heights = src_u - src_d + 1.0

        # kind of weird to use (10,10,10,10), but we just follow KL-loss
        wx, wy, _, _ = self.weights
        s2t_dl = wx * (tgt_l - src_l) / src_widths
        s2t_dr = wx * (tgt_r - src_r) / src_widths
        s2t_dd = wy * (tgt_d - src_d) / src_heights
        s2t_du = wy * (tgt_u - src_u) / src_heights

        deltas = torch.stack((s2t_dl, s2t_dr, s2t_dd, s2t_du), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Follow the KL-Loss implementation (CVPR'19)
        https://github.com/yihui-he/KL-Loss/blob/1c67310c9f5a79cfa985fea241791ccedbdb7dcf/detectron/utils/boxes.py#L208

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        left = boxes[:, 0]
        right = boxes[:, 2]
        down = boxes[:, 1]
        up = boxes[:, 3]

        wx, wy, _, _ = self.weights
        dl = deltas[:, 0::4] / wx
        dr = deltas[:, 1::4] / wx
        dd = deltas[:, 2::4] / wy
        du = deltas[:, 3::4] / wy

        # Prevent sending too large values into torch.exp()
        # dw = torch.clamp(dw, max=self.scale_clamp)
        # dh = torch.clamp(dh, max=self.scale_clamp)
        # dl = np.maximum(np.minimum(dl, cfg.BBOX_XFORM_CLIPe), -cfg.BBOX_XFORM_CLIPe)
        # dr = np.maximum(np.minimum(dr, cfg.BBOX_XFORM_CLIPe), -cfg.BBOX_XFORM_CLIPe)
        # dd = np.maximum(np.minimum(dd, cfg.BBOX_XFORM_CLIPe), -cfg.BBOX_XFORM_CLIPe)
        # du = np.maximum(np.minimum(du, cfg.BBOX_XFORM_CLIPe), -cfg.BBOX_XFORM_CLIPe)

        # Prevent sending too large values into np.exp()       # TODO: find out cfg.BBOX_XFORM_CLIPe
        dl = torch.clamp(dl, max=self.scale_clamp, min=-self.scale_clamp)
        dr = torch.clamp(dr, max=self.scale_clamp, min=-self.scale_clamp)
        dd = torch.clamp(dd, max=self.scale_clamp, min=-self.scale_clamp)
        du = torch.clamp(du, max=self.scale_clamp, min=-self.scale_clamp)

        # pred_ctr_x = dl * widths[:, None] + left[:, None]
        # pred_ctr_y = dr * heights[:, None] + right[:, None]

        pred_l = dl * widths[:, None] + left[:, None]
        pred_r = dr * widths[:, None] + right[:, None]
        pred_d = dd * heights[:, None] + down[:, None]
        pred_u = du * heights[:, None] + up[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_l
        pred_boxes[:, 1::4] = pred_d
        pred_boxes[:, 2::4] = pred_r
        pred_boxes[:, 3::4] = pred_u
        return pred_boxes
