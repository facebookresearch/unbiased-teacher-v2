# most part of this file is modified from AdelaiDet
# https://github.com/aim-uofa/AdelaiDet

import math
from typing import Dict, List

import torch
from detectron2.layers import NaiveSyncBatchNorm, ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from torch import nn
from torch.nn import functional as F
from ubteacher.utils.comm import compute_locations

from .fcos_outputs import FCOSOutputs


__all__ = ["FCOS"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL

        self.kl_loss = cfg.MODEL.FCOS.KL_LOSS
        self.kl_loss_type = cfg.MODEL.FCOS.KL_LOSS_TYPE

        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = FCOSOutputs(cfg)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]

        if self.kl_loss:
            (
                pred_class_logits,
                pred_deltas,
                reg_pred,
                pred_centerness,
                top_feats,
                bbox_towers,
            ) = self.fcos_head(features, top_module, self.yield_proposal)
        else:
            (
                pred_class_logits,
                pred_deltas,
                pred_centerness,
                top_feats,
                bbox_towers,
            ) = self.fcos_head(features, top_module, self.yield_proposal)

        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(
        self,
        images,
        features,
        gt_instances=None,
        top_module=None,
        output_raw=False,
        nms_method="cls_n_ctr",
        ignore_near=False,
        branch="labeled",
    ):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]

        locations = self.compute_locations(features)

        raw_output = {}
        if self.kl_loss:
            (
                logits_pred,
                reg_pred,
                reg_pred_std,
                ctrness_pred,
                top_feats,
                bbox_towers,
            ) = self.fcos_head(features, top_module, self.yield_proposal)
            raw_output["reg_pred_std"] = reg_pred_std

        else:
            (
                logits_pred,
                reg_pred,
                ctrness_pred,
                top_feats,
                bbox_towers,
            ) = self.fcos_head(features, top_module, self.yield_proposal)
            reg_pred_std = None
        # accumlate feature pred for pseudo-labeling
        raw_output["logits_pred"] = logits_pred
        raw_output["reg_pred"] = reg_pred
        raw_output["top_feats"] = top_feats
        raw_output["bbox_towers"] = bbox_towers
        raw_output["locations"] = locations
        raw_output["ctrness_pred"] = ctrness_pred
        raw_output["image_sizes"] = images.image_sizes

        results = {}
        if self.yield_proposal:
            results["features"] = {f: b for f, b in zip(self.in_features, bbox_towers)}

        if self.training:

            if branch == "labeled":
                results, losses = self.fcos_outputs.losses(
                    logits_pred,
                    reg_pred,
                    ctrness_pred,
                    locations,
                    gt_instances,
                    reg_pred_std,
                    top_feats,
                    ignore_near,
                    branch=branch,
                )
            elif branch == "unlabeled":
                results, losses = self.fcos_outputs.pseudo_losses(
                    logits_pred,
                    reg_pred,
                    ctrness_pred,
                    locations,
                    gt_instances,
                    reg_pred_std,
                    top_feats,
                    ignore_near,
                    branch=branch,
                )
            elif branch == "raw":
                results = {}
                losses = {}
            else:
                raise ValueError("Unknown branch")

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred=logits_pred,
                        reg_pred=reg_pred,
                        ctrness_pred=ctrness_pred,
                        locations=locations,
                        image_sizes=images.image_sizes,
                        reg_pred_std=reg_pred_std,
                        top_feats=top_feats,
                        nms_method=nms_method,
                    )
            if output_raw:
                return results, losses, raw_output
            else:
                return results, losses

        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred=logits_pred,
                reg_pred=reg_pred,
                ctrness_pred=ctrness_pred,
                locations=locations,
                image_sizes=images.image_sizes,
                reg_pred_std=reg_pred_std,
                top_feats=top_feats,
                nms_method=nms_method,
            )
            if output_raw:
                return results, {}, raw_output
            else:
                return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level], feature.device
            )
            locations.append(locations_per_level)
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()

        self.reg_max = cfg.MODEL.FCOS.REG_MAX
        self.reg_discrete = cfg.MODEL.FCOS.REG_DISCRETE
        self.kl_loss = cfg.MODEL.FCOS.KL_LOSS

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {
            "cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS, cfg.MODEL.FCOS.USE_DEFORMABLE),
            "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS, cfg.MODEL.FCOS.USE_DEFORMABLE),
            "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS, False),
        }
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for _ in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(
                    conv_func(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(
                        ModuleListDial(
                            [
                                nn.BatchNorm2d(in_channels)
                                for _ in range(self.num_levels)
                            ]
                        )
                    )
                elif norm == "SyncBN":
                    tower.append(
                        ModuleListDial(
                            [
                                NaiveSyncBatchNorm(in_channels)
                                for _ in range(self.num_levels)
                            ]
                        )
                    )
                tower.append(nn.ReLU())
            self.add_module("{}_tower".format(head), nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1, padding=1
        )

        if self.reg_discrete:
            self.bbox_pred = nn.Conv2d(
                in_channels, 4 * (self.reg_max + 1), kernel_size=3, stride=1, padding=1
            )
        else:
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1, padding=1
            )

        if self.kl_loss:
            self.bbox_pred_std = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1, padding=1
            )

        self.ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if cfg.MODEL.FCOS.USE_SCALE:  # learnable scale
            self.scales = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )
        else:
            self.scales = None

        # initialize
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.share_tower,
            self.cls_logits,
            self.bbox_pred,
            self.ctrness,
        ]:
            for lay in modules.modules():
                if isinstance(lay, nn.Conv2d):
                    torch.nn.init.normal_(lay.weight, std=0.01)
                    torch.nn.init.constant_(lay.bias, 0)

        if self.kl_loss:
            torch.nn.init.normal_(
                self.bbox_pred_std.weight, std=0.0001
            )  # follows KL-Loss
            torch.nn.init.constant_(self.bbox_pred_std.bias, 0)  # follows KL-Loss

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        bbox_reg_std = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)

            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.

            if self.reg_discrete:
                # generalized focal loss use softmax
                bbox_reg.append(reg)
            else:
                bbox_reg.append(F.relu(reg))

            if self.kl_loss:
                reg_std = self.bbox_pred_std(bbox_tower)
                bbox_reg_std.append(reg_std)

            if top_module is not None:
                top_feats.append(top_module(bbox_tower))

        if self.kl_loss:  # additional box prediction std output
            return logits, bbox_reg, bbox_reg_std, ctrness, top_feats, bbox_towers
        else:
            return logits, bbox_reg, ctrness, top_feats, bbox_towers
