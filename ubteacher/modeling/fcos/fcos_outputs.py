# most part of this file is modified from AdelaiDet
# https://github.com/aim-uofa/AdelaiDet


import logging

import torch
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from ubteacher.layers import IOULoss, KLLoss, ml_nms, NLLoss
from ubteacher.utils.comm import reduce_sum

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets

    ctrness_pred: predicted centerness scores

"""


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.

    From generalized focal loss v2

    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
    )
    return torch.sqrt(ctrness)


def compute_iou_targets(pred, target):
    """
    reg_pred: (# of instnaces, 4) in normalized format
    reg_targets: (# of instances, 4) in normalized format

    ctrness_targets = compute_iou_targets(
        reg_pred.detach(),
        instances.reg_targets)

    """
    if len(target) == 0:
        return target.new_zeros(len(target))

    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(
        pred_right, target_right
    )
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
        pred_top, target_top
    )

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)

    return ious


class FCOSOutputs(nn.Module):
    def __init__(self, cfg):
        super(FCOSOutputs, self).__init__()

        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN

        self.pre_nms_thresh_test = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.FCOS.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.FCOS.THRESH_WITH_CTR

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES

        # box loss weight
        self.cls_loss_weight = cfg.SEMISUPNET.SOFT_CLS_LABEL
        self.cls_loss_method = cfg.SEMISUPNET.CLS_LOSS_METHOD

        # bin offset classification
        self.reg_discrete = cfg.MODEL.FCOS.REG_DISCRETE
        self.reg_max = cfg.MODEL.FCOS.REG_MAX
        self.fpn_stride = torch.tensor(cfg.MODEL.FCOS.FPN_STRIDES).cuda().float()
        self.dfl_loss_weight = cfg.MODEL.FCOS.DFL_WEIGHT
        self.unify_ctrcls = cfg.MODEL.FCOS.UNIFY_CTRCLS

        # kl loss
        self.kl_loss = cfg.MODEL.FCOS.KL_LOSS
        self.kl_loss_type = cfg.MODEL.FCOS.KL_LOSS_TYPE  # 'klloss' or 'nlloss'
        self.kl_loss_weight = cfg.MODEL.FCOS.KLLOSS_WEIGHT

        self.loc_fun_all = cfg.MODEL.FCOS.LOC_FUN_ALL

        # unsupervised regression loss
        self.reg_unsup_loss = cfg.SEMISUPNET.CONSIST_REG_LOSS

        #  KL loss  or IoU loss
        if self.kl_loss:
            if self.kl_loss_type == "klloss":
                self.kl_loc_loss_func = KLLoss()
            elif self.kl_loss_type == "nlloss":
                self.kl_loc_loss_func = NLLoss()
            else:
                raise ValueError

        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)

        # Quality estimation
        self.quality_est = cfg.MODEL.FCOS.QUALITY_EST

        # TS better classification
        self.cls_loss_pseudo_method = cfg.SEMISUPNET.CLS_LOSS_PSEUDO_METHOD
        self.tsbetter_cls_sigma = cfg.MODEL.FCOS.TSBETTER_CLS_SIGMA

        # TS better
        self.tsbetter_reg = cfg.SEMISUPNET.TS_BETTER
        self.tsbetter_reg_cert = cfg.SEMISUPNET.TS_BETTER_CERT

        # Ratio
        # self.fg_bg_ratio = cfg.MODEL.FCOS.FG_BG_RATIO

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

        self.integral = Integral(self.reg_max)

    # loss
    # supervised loss branch
    def losses(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        locations,
        gt_instances,
        reg_pred_std=None,
        top_feats=None,
        ignore_near=False,
        branch="",
    ):
        training_targets = self._get_ground_truth(locations, gt_instances, ignore_near)

        instances = Instances((0, 0))
        instances.labels = cat(
            [x.reshape(-1) for x in training_targets["labels"]],
            dim=0,
        )

        instances.box_weights = cat(
            [x.reshape(-1) for x in training_targets["box_weights"]],
            dim=0,
        )

        # ignore some samples during training
        instances.keep_locations = cat(
            [x.reshape(-1) for x in training_targets["keep_locations"]],
            dim=0,
        )

        instances.gt_inds = cat(
            [x.reshape(-1) for x in training_targets["target_inds"]],
            dim=0,
        )
        instances.im_inds = cat(
            [x.reshape(-1) for x in training_targets["im_inds"]], dim=0
        )
        instances.reg_targets = cat(
            [x.reshape(-1, 4) for x in training_targets["reg_targets"]],
            dim=0,
        )
        instances.locations = cat(
            [x.reshape(-1, 2) for x in training_targets["locations"]], dim=0
        )
        instances.fpn_levels = cat(
            [x.reshape(-1) for x in training_targets["fpn_levels"]], dim=0
        )

        instances.logits_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred],
            dim=0,
        )

        if self.reg_discrete:
            instances.reg_pred = cat(
                [
                    x.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
                    for x in reg_pred
                ],
                dim=0,
            )
        else:
            instances.reg_pred = cat(
                [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred],
                dim=0,
            )

        if self.kl_loss:
            assert reg_pred_std is not None
            instances.reg_pred_std = cat(
                [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred_std],
                dim=0,
            )

        instances.ctrness_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness_pred],
            dim=0,
        )

        if len(top_feats) > 0:
            instances.top_feats = cat(
                [
                    # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                    x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
                    for x in top_feats
                ],
                dim=0,
            )

        if branch == "labeled":
            return self.fcos_losses(instances)
        else:
            raise ValueError("Incorrect branch name")

    def fcos_losses(self, instances):

        losses = {}
        if instances.keep_locations.sum() > 0:  # some instances are not ignored
            instances = instances[instances.keep_locations]

        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()
        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot (N, 1000)
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        # classification loss (unifying branch or individual branch)
        if self.cls_loss_method == "focal":
            class_loss_all = sigmoid_focal_loss_jit(
                instances.logits_pred,
                class_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="none",
            )
            ## sum over class dimension
            weighted_class_loss = class_loss_all.sum(1)
            class_loss = weighted_class_loss.sum() / num_pos_avg

        ## only compute the centerness loss and regression loss for the foreground classes
        instances = instances[pos_inds]
        instances.pos_inds = pos_inds
        ##  process regression prediction (from discrete to continous)
        ##  we find this helps the unsupervised loss
        if self.reg_discrete and pos_inds.numel() > 0:  # offset bin classification
            pred_ltrb_discrete = instances.reg_pred
            pred_ltrb_scalar = self.integral(pred_ltrb_discrete)
            reg_pred = pred_ltrb_scalar
        else:
            reg_pred = instances.reg_pred

        # process target for centerness loss
        if self.quality_est == "centerness":
            ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        elif self.quality_est == "iou":
            # pos_decode_bbox_pred: xyxy, pos_decode_bbox_targets: xyxy
            ctrness_targets = compute_iou_targets(
                reg_pred.detach(), instances.reg_targets
            )

        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        iou_targets = compute_iou_targets(reg_pred.detach(), instances.reg_targets)

        if pos_inds.numel() > 0:
            # cetnerness loss
            ctrness_loss = (
                F.binary_cross_entropy_with_logits(
                    instances.ctrness_pred, ctrness_targets, reduction="sum"
                )
                / num_pos_avg
            )

            # regression loss
            if self.kl_loss:
                reg_pred_std = instances.reg_pred_std

                if self.kl_loss_type == "klloss":
                    kl_loss = self.kl_loss_weight * self.kl_loc_loss_func(
                        reg_pred,
                        reg_pred_std,
                        instances.reg_targets,
                        loss_denorm=loss_denorm,
                        weight=ctrness_targets,
                        iou_weight=iou_targets,
                        method=self.loc_fun_all,
                    )

                    iou_loss = (
                        self.loc_loss_func(
                            reg_pred, instances.reg_targets, ctrness_targets
                        )
                        / loss_denorm
                    )
                    reg_loss = self.kl_loss_weight * kl_loss + iou_loss

                elif self.kl_loss_type == "nlloss":
                    nlloss = self.kl_loss_weight * self.kl_loc_loss_func(
                        reg_pred,
                        reg_pred_std,
                        instances.reg_targets,
                        loss_denorm=loss_denorm,
                        weight=ctrness_targets,
                        iou_weight=iou_targets,
                        method=self.loc_fun_all,
                    )

                    iou_loss = (
                        self.loc_loss_func(
                            reg_pred, instances.reg_targets, ctrness_targets
                        )
                        / loss_denorm
                    )
                    reg_loss = self.kl_loss_weight * nlloss + iou_loss

            else:
                # IoU loss
                reg_loss = (
                    self.loc_loss_func(reg_pred, instances.reg_targets, ctrness_targets)
                    / loss_denorm
                )

        else:
            reg_loss = torch.tensor(0).cuda()
            ctrness_loss = torch.tensor(0).cuda()
            loss_denorm = 1.0

        if instances.keep_locations.sum() == 0:
            class_loss = class_loss * 0
            reg_loss = reg_loss * 0
            ctrness_loss = ctrness_loss * 0
            loss_denorm = 1.0

        losses_all = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_ctr": ctrness_loss,
        }

        losses.update(losses_all)
        extras = {"instances": instances, "loss_denorm": loss_denorm}
        return extras, losses

    # unsupervised loss branch
    def pseudo_losses(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        locations,
        gt_instances,
        reg_pred_std=None,
        top_feats=None,
        ignore_near=False,
        branch="",
    ):
        """
        Return the losses from a set of GFocal predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """
        assert branch == "unlabeled"

        extras = {}
        losses = {}

        # cls pseudo-labels --> cls and centerness, reg pseudo-labels --> reg
        return_list = {"cls": ["cls", "ctr"], "reg": ["reg"]}
        for labeltype in gt_instances.keys():
            training_target = self._get_ground_truth(
                locations, gt_instances[labeltype], ignore_near
            )
            instances = self.prepare_instance(
                training_targets=training_target,
                logits_pred=logits_pred,
                reg_pred=reg_pred,
                ctrness_pred=ctrness_pred,
                reg_pred_std=reg_pred_std,
                top_feats=top_feats,
            )
            extras_each, losses_each = self.fcos_pseudo_losses(
                instances, return_loss=return_list[labeltype], labeltype=labeltype
            )
            extras.update(extras_each)
            losses.update(losses_each)

        return extras, losses

    def fcos_pseudo_losses(self, instances, return_loss, labeltype=""):

        return_instances = instances
        losses = {}

        # compute pos_inds and num_pos_avg
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes
        labels = instances.labels.flatten()
        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # classification loss
        if "cls" in return_loss:
            class_loss = self.class_loss(
                instances, pos_inds, labels, num_classes, num_pos_avg
            )
            losses.update(class_loss)

        # only compute the centerness loss and regression loss for the foreground classes
        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        # prepare centerness ground-truth labels
        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        if pos_inds.numel() > 0:
            # centerness loss
            if "ctr" in return_loss:
                ctrness_loss = (
                    F.binary_cross_entropy_with_logits(
                        instances.ctrness_pred, ctrness_targets, reduction="sum"
                    )
                    / num_pos_avg
                )
                if self.unify_ctrcls:
                    ctrness_loss = ctrness_loss * 0
                losses["loss_fcos_ctr"] = ctrness_loss

            if "reg" in return_loss:
                # process regressiion prediction
                if self.reg_discrete and pos_inds.numel() > 0:
                    # offset bin classification, we find this slightly improves unsupervised loss
                    pred_ltrb_discrete = instances.reg_pred
                    pred_ltrb_scalar = self.integral(pred_ltrb_discrete)
                    reg_pred = pred_ltrb_scalar
                else:
                    # continous output
                    reg_pred = instances.reg_pred

                # regression loss
                if self.kl_loss:  # kl loss
                    assert instances.has("reg_pred_std")
                    reg_pred_std = instances.reg_pred_std
                    if self.reg_unsup_loss == "ts_locvar_better_nms_nll_l1":
                        loc_conf_student = 1 - instances.reg_pred_std.sigmoid()
                        loc_conf_teacher = 1 - instances.boundary_vars.sigmoid()
                        select = (loc_conf_teacher > self.tsbetter_reg_cert) * (
                            loc_conf_teacher > loc_conf_student + self.tsbetter_reg
                        )

                        losses["teacher_better_student"] = select.sum()

                        reg_student = reg_pred
                        reg_teacher = instances.reg_targets

                        if select.sum() > 0:
                            reg_loss = F.smooth_l1_loss(
                                reg_student[select], reg_teacher[select], beta=0.0
                            )
                        else:
                            reg_loss = torch.tensor(0).cuda()

                    else:
                        iou_targets = compute_iou_targets(
                            reg_pred.detach(), instances.reg_targets
                        )

                        reg_loss = self.kl_loss_weight * self.kl_loc_loss_func(
                            reg_pred,
                            reg_pred_std,
                            instances.reg_targets,
                            loss_denorm=loss_denorm,
                            weight=ctrness_targets,
                            iou_weight=iou_targets,
                            method=self.loc_fun_all,
                        )

                else:
                    raise ValueError


                losses["loss_fcos_loc"] = reg_loss

        else:
            if "ctr" in return_loss:
                losses["loss_fcos_ctr"] = torch.tensor(0).cuda()

            loss_denorm = 1.0

            if "reg" in return_loss:
                losses["loss_fcos_loc"] = torch.tensor(0).cuda()
                losses["teacher_better_student"] = torch.tensor(0).cuda()

        # final check for multiple gpu running
        extras = {
            "instances_" + labeltype: return_instances,
            "loss_denorm": loss_denorm,
        }
        return extras, losses

    # classification loss (for unsupervised branch)
    def class_loss(self, instances, pos_inds, labels, num_classes, num_pos_avg):

        losses = {}
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        ### Classification loss (unifying branch or individual branch)
        # unifying centerness and classification
        # we find this leads to worse results
        class_loss_all = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="none",
        )

        weighted_class_loss = class_loss_all.sum(1)
        class_loss = weighted_class_loss.sum() / num_pos_avg
        losses["loss_fcos_cls"] = class_loss

        return losses

    # other functions
    def _transpose(self, training_targets, num_loc_list):
        """
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        """
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(torch.cat(targets_per_level, dim=0))
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances, ignore_near=False):
        num_loc_list = [len(loc) for loc in locations]
        # compute locations to size ranges
        loc_to_size_range = []
        for lo, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(
                self.sizes_of_interest[lo]
            )
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[lo], -1)
            )
            # [prev_layer_size, this layer_size ]
            # [[-1,64], .... ,[64,128],...,[128,256], ...,[256,512],... [512,100000]]

        loc_to_size_range = torch.cat(
            loc_to_size_range, dim=0
        )  # size [L1+L2+...+L5, 2]
        locations = torch.cat(locations, dim=0)  # size [L1+L2+...+L5, 2]

        # compute the reg, label target for each element
        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list, ignore_near
        )

        training_targets["locations"] = [
            locations.clone() for _ in range(len(gt_instances))
        ]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i
            for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        # reg_targets is normalized for each level!
        #  this is ltrb format
        reg_targets = training_targets["reg_targets"]
        for la in range(len(reg_targets)):
            reg_targets[la] = reg_targets[la] / float(self.strides[la])

        return training_targets

    def get_sample_region(
        self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1
    ):
        """
        boxes: size:[# of GT boxes, 4(x1,y1,x2,y2)]
        strides: [8,16,32,64,128]
        num_loc_list: [15200, 3800, 950 ,247, 70]
        loc_xs: size[20267]
        loc_ys: size[20267]
        bitmasks:
        radius
        """

        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            # gt box center Size[number of bbox]
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)

        # compute the bbox region (it is shrinked into the region near the center point)
        # ! Center point region is not object-size variant
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride  # center x shift
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt: gt only has 1.5 pixel away from the center
            center_gt[beg:end, :, 0] = torch.where(
                xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax
            )
            beg = end
        #
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(
        self, locations, targets, size_ranges, num_loc_list, ignore_near=False
    ):

        labels = []
        reg_targets = []
        target_inds = []
        keep_locations = []
        box_weights = []
        boundary_vars = []

        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):  # image-wise operation
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # box weight weights
            if targets_per_im.has("scores") and self.cls_loss_weight:
                box_weights_per_im = targets_per_im.scores
            else:
                box_weights_per_im = torch.ones_like(targets_per_im.gt_classes)

            # box weight weights
            if targets_per_im.has("reg_pred_std"):
                boundary_var_per_im = targets_per_im.reg_pred_std
            else:
                boundary_var_per_im = torch.zeros_like(targets_per_im.gt_boxes.tensor)

            # no gt
            if bboxes.numel() == 0:
                # no bboxes then all labels are background
                labels.append(
                    labels_per_im.new_zeros(locations.size(0)) + self.num_classes
                )
                # no bboxes then all boxes weights are zeros
                box_weights.append(box_weights_per_im.new_zeros(locations.size(0)))
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                boundary_vars.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                keep_locations.append(torch.zeros(xs.shape[0]).to(bool).cuda())
                continue
            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.strides,
                    num_loc_list,
                    xs,
                    ys,
                    bitmasks=bitmasks,
                    radius=self.radius,
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            if ignore_near:
                # ignore all pixels inside the boxes
                is_ignore = reg_targets_per_im.min(dim=2)[0] > 0
                keep_location_bg = ~(is_ignore.sum(1) > 0)

                # keep all pixel inside the box
                keep_location_fg = is_in_boxes.sum(1) > 0
                keep_location = keep_location_bg + keep_location_fg
            else:
                # keep all
                keep_location = torch.ones(is_in_boxes.shape[0]).to(bool).cuda()

            # filter out these box is too small or too big for each scale
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = (max_reg_targets_per_im >= size_ranges[:, [0]]) & (
                max_reg_targets_per_im <= size_ranges[:, [1]]
            )

            # compute the area for each gt box
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # set points (outside box/small region) as background
            locations_to_gt_area[is_in_boxes == 0] = INF
            # set points with too large displacement or too small displacement as background
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(
                dim=1
            )

            # use the minial area as creteria to choose ground-truth boxes of regression for each point
            reg_targets_per_im = reg_targets_per_im[
                range(len(locations)), locations_to_gt_inds
            ]

            # regard object in different image as different instance
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            # TODO: background weight is 1.0 for now; we could try to use average score as background weights
            box_weights_per_im = box_weights_per_im[locations_to_gt_inds]
            box_weights_per_im[locations_to_min_area == INF] = 1.0

            boundary_var_per_im = boundary_var_per_im[locations_to_gt_inds]
            boundary_var_per_im[locations_to_min_area == INF] = 99999.0

            labels.append(labels_per_im)
            box_weights.append(box_weights_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)
            keep_locations.append(keep_location)
            boundary_vars.append(boundary_var_per_im)

        return {
            "labels": labels,
            "box_weights": box_weights,
            "reg_targets": reg_targets,
            "target_inds": target_inds,
            "keep_locations": keep_locations,
            "boundary_vars": boundary_vars,
        }

    def prepare_instance(
        self,
        training_targets,
        logits_pred,
        reg_pred,
        ctrness_pred,
        reg_pred_std=None,
        top_feats=None,
    ):
        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.
        instances = Instances((0, 0))
        instances.labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["labels"]
            ],
            dim=0,
        )

        # add soft weight for each labels
        instances.box_weights = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["box_weights"]
            ],
            dim=0,
        )

        # ignore some samples during training
        instances.keep_locations = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["keep_locations"]
            ],
            dim=0,
        )

        instances.gt_inds = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["target_inds"]
            ],
            dim=0,
        )
        instances.im_inds = cat(
            [x.reshape(-1) for x in training_targets["im_inds"]], dim=0
        )
        instances.reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4)
                for x in training_targets["reg_targets"]
            ],
            dim=0,
        )
        instances.locations = cat(
            [x.reshape(-1, 2) for x in training_targets["locations"]], dim=0
        )
        instances.fpn_levels = cat(
            [x.reshape(-1) for x in training_targets["fpn_levels"]], dim=0
        )

        if "boundary_vars" in training_targets:
            instances.boundary_vars = cat(
                [
                    # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                    x.reshape(-1, 4)
                    for x in training_targets["boundary_vars"]
                ],
                dim=0,
            )

        instances.logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in logits_pred
            ],
            dim=0,
        )

        if self.reg_discrete:
            instances.reg_pred = cat(
                [
                    # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                    x.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
                    for x in reg_pred
                ],
                dim=0,
            )
        else:
            instances.reg_pred = cat(
                [
                    # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                    x.permute(0, 2, 3, 1).reshape(-1, 4)
                    for x in reg_pred
                ],
                dim=0,
            )

        if self.kl_loss:
            assert reg_pred_std is not None
            instances.reg_pred_std = cat(
                [
                    # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                    x.permute(0, 2, 3, 1).reshape(-1, 4)
                    for x in reg_pred_std
                ],
                dim=0,
            )

        instances.ctrness_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.permute(0, 2, 3, 1).reshape(-1)
                for x in ctrness_pred
            ],
            dim=0,
        )

        if len(top_feats) > 0:
            instances.top_feats = cat(
                [
                    # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                    x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
                    for x in top_feats
                ],
                dim=0,
            )

        return instances

    def predict_proposals(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        locations,
        image_sizes,
        reg_pred_std=None,
        top_feats=None,
        nms_method="cls_n_ctr",
    ):

        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locations,
            "o": logits_pred,
            "r": reg_pred,
            "c": ctrness_pred,
            "s": self.strides,
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        if reg_pred_std is not None:
            bundle["r_std"] = reg_pred_std

        # each iteration = 1 scale
        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.

            l = per_bundle["l"]
            o = per_bundle["o"]

            if self.reg_discrete:  # discrete to scalar
                bs = per_bundle["r"].shape[0]
                imgw = per_bundle["r"].shape[2]
                imgh = per_bundle["r"].shape[3]
                reg_discre_raw = (
                    per_bundle["r"]
                    .permute(0, 2, 3, 1)
                    .reshape(-1, 4 * (self.reg_max + 1))
                )
                scalar_r = self.integral(reg_discre_raw).reshape(bs, imgw, imgh, 4)
                scalar_r = scalar_r.permute(0, 3, 1, 2)
                r = scalar_r * per_bundle["s"]

                r_cls = (per_bundle["r"], per_bundle["s"])
            else:
                r = per_bundle["r"] * per_bundle["s"]
                r_cls = None

            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None

            r_std = per_bundle["r_std"] if "r_std" in bundle else None

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, r_cls, c, image_sizes, r_std, t, nms_method
                )
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = (
                    l.new_ones(len(per_im_sampled_boxes), dtype=torch.long) * i
                )

        # nms
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map(
        self,
        locations,
        logits_pred,
        reg_pred,
        reg_pred_cls,
        ctrness_pred,
        image_sizes,
        reg_pred_std=None,
        top_feat=None,
        nms_method="cls_n_ctr",
    ):
        N, C, H, W = logits_pred.shape
        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()

        if reg_pred_cls is not None:
            box_reg_cls = (
                reg_pred_cls[0]
                .view(N, 4 * (self.reg_max + 1), H, W)
                .permute(0, 2, 3, 1)
            )
            box_reg_cls = box_reg_cls.reshape(N, -1, 4 * (self.reg_max + 1))
            scalar = reg_pred_cls[1]

        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        if reg_pred_std is not None:
            box_regression_std = reg_pred_std.view(N, 4, H, W).permute(0, 2, 3, 1)
            box_regression_std = box_regression_std.reshape(N, -1, 4)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)
        cls_confs = logits_pred

        if not self.thresh_with_ctr:
            if nms_method == "cls_n_ctr":
                logits_pred = logits_pred * ctrness_pred[:, :, None]
            elif nms_method == "cls":
                logits_pred = logits_pred
            elif nms_method == "ctr":
                logits_pred = ctrness_pred[:, :, None]

            elif nms_method == "cls_n_loc":
                assert box_regression_std is not None
                boundary_regression_std = 1 - box_regression_std.sigmoid()
                box_reg_std = boundary_regression_std.mean(2)
                logits_pred = logits_pred * box_reg_std[:, :, None]
            else:  # default cls + ctr
                logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):  # each image
            # select pixels larger than threshold (0.05)
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            # get the index of pixel and its class prediction
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]

            # for bin classification
            if reg_pred_cls is not None:
                per_box_reg_cls = box_reg_cls[i]
                per_box_reg_cls = per_box_reg_cls[per_box_loc]

            # for localization std
            if reg_pred_std is not None:
                per_box_regression_std = box_regression_std[i]
                per_box_regression_std = per_box_regression_std[per_box_loc]

            per_locations = locations[per_box_loc]

            # centerness
            per_centerness = ctrness_pred[i]
            per_centerness = per_centerness[per_box_loc]
            per_cls_conf = cls_confs[i]
            per_cls_conf = per_cls_conf[per_candidate_inds]

            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            # select top k
            per_pre_nms_top_n = pre_nms_top_n[i]

            # check whether per_candidate boxes is too many
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(
                    per_pre_nms_top_n, sorted=False
                )
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]

                if reg_pred_cls is not None:
                    per_box_reg_cls = per_box_reg_cls[top_k_indices]

                if reg_pred_std is not None:
                    per_box_regression_std = per_box_regression_std[top_k_indices]

                per_locations = per_locations[top_k_indices]
                per_centerness = per_centerness[top_k_indices]
                per_cls_conf = per_cls_conf[top_k_indices]

                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack(
                [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ],
                dim=1,
            )

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            if nms_method == "cls_n_ctr" or nms_method == "cls_n_loc":
                boxlist.scores = torch.sqrt(per_box_cls)
            elif nms_method == "cls" or nms_method == "ctr":
                boxlist.scores = per_box_cls
            else:  # default cls + ctr
                raise ValueError("Undefined nms criteria")

            if reg_pred_cls is not None:
                boxlist.reg_pred_cls = per_box_reg_cls
                boxlist.reg_pred_cls_scalar = (
                    torch.ones(per_box_reg_cls.shape[0]).cuda() * scalar
                )

            if reg_pred_std is not None:
                boxlist.reg_pred_std = per_box_regression_std

            # boxlist.scores = torch.sqrt(per_box_cls)
            # boxlist.scores = torch.sqrt(per_box_cls)

            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            boxlist.centerness = per_centerness
            boxlist.cls_confid = per_cls_conf

            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores, number_of_detections - self.post_nms_topk + 1
                )

                # torch.topk()
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
