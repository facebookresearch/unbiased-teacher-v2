import logging

import numpy as np
import torch
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from torch import nn


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for
    bezier control points.
    """
    scale_x, scale_y = (
        output_width / results.image_size[1],
        output_height / results.image_size[0],
    )
    results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)

    return results


@META_ARCH_REGISTRY.register()
class PseudoProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(
        self,
        batched_inputs,
        output_raw=False,
        nms_method="cls_n_ctr",
        ignore_near=False,
        branch="labeled",
    ):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0] and branch != "teacher_weak":
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0] and branch != "teacher_weak":
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if output_raw:
            proposals, proposal_losses, raw_pred = self.proposal_generator(
                images,
                features,
                gt_instances,
                output_raw=output_raw,
                nms_method=nms_method,
                ignore_near=ignore_near,
            )
        else:
            proposals, proposal_losses = self.proposal_generator(
                images,
                features,
                gt_instances,
                output_raw=output_raw,
                nms_method=nms_method,
                ignore_near=ignore_near,
            )

        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            if output_raw:
                return proposal_losses, raw_pred
            else:
                return proposal_losses

        if output_raw:
            # output raw will not rescale
            return proposals, raw_pred
        else:
            # standard output will rescale
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                ret = detector_postprocess(results_per_image, height, width)
                processed_results.append({"proposals": ret})
            return processed_results


@META_ARCH_REGISTRY.register()
class OneStageDetector(PseudoProposalNetwork):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Uses "instances" as the return key instead of using "proposal".
    """

    def forward(
        self,
        batched_inputs,
        output_raw=False,
        nms_method="cls_n_ctr",
        ignore_near=False,
        branch="labeled",
    ):
        # training
        if self.training:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            features = self.backbone(images.tensor)

            # pseudo-labels for classification and regression
            if (
                "instances_class" in batched_inputs[0]
                and "instances_reg" in batched_inputs[0]
            ):
                gt_instances_cls = [
                    x["instances_class"].to(self.device) for x in batched_inputs
                ]
                gt_instances_reg = [
                    x["instances_reg"].to(self.device) for x in batched_inputs
                ]
                gt_instances = {"cls": gt_instances_cls, "reg": gt_instances_reg}

            elif "instances" in batched_inputs[0] and branch != "teacher_weak":
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            elif "targets" in batched_inputs[0] and branch != "teacher_weak":
                log_first_n(
                    logging.WARN,
                    "'targets' in the model inputs is now renamed to 'instances'!",
                    n=10,
                )
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            if output_raw:
                proposals, proposal_losses, raw_pred = self.proposal_generator(
                    images,
                    features,
                    gt_instances,
                    output_raw=output_raw,
                    ignore_near=ignore_near,
                    branch=branch,
                )
            else:
                proposals, proposal_losses = self.proposal_generator(
                    images,
                    features,
                    gt_instances,
                    output_raw=output_raw,
                    ignore_near=ignore_near,
                    branch=branch,
                )

            if self.training:
                if output_raw:
                    return proposal_losses, raw_pred, proposals
                else:
                    return proposal_losses

        # inference
        if output_raw:
            proposal, raw_pred = super().forward(
                batched_inputs,
                output_raw=output_raw,
                nms_method=nms_method,
                branch=branch,
            )
            return proposal, raw_pred
        else:
            processed_results = super().forward(
                batched_inputs,
                output_raw=output_raw,
                nms_method=nms_method,
                branch=branch,
            )
            processed_results = [
                {"instances": r["proposals"]} for r in processed_results
            ]
            return processed_results

    def visualize_training(self, batched_inputs, proposals, branch):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            if branch == "labeled":
                img = input["image"]
                img = convert_image_to_rgb(img.permute(1, 2, 0), "BGR")
                v_gt = Visualizer(img, None)
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes.to("cpu")
                )
                anno_img = v_gt.get_image()
                box_size = min(len(prop.pred_boxes), max_vis_prop)
                v_pred = Visualizer(img, None)
                v_pred = v_pred.overlay_instances(
                    boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy()
                )
                prop_img = v_pred.get_image()
                vis_img = np.concatenate((anno_img, prop_img), axis=1)
                vis_img = vis_img.transpose(2, 0, 1)
                vis_name = (
                    branch
                    + " | Left: GT bounding boxes;      Right: Predicted proposals"
                )
            elif branch == "unlabeled":
                img_list = []
                img = input["image"]
                img = convert_image_to_rgb(img.permute(1, 2, 0), "BGR")

                # classification pseudo-set
                if "instances_class" in input:
                    v_gt = Visualizer(img, None)
                    v_gt = v_gt.overlay_instances(
                        boxes=input["instances_class"].gt_boxes.to("cpu")
                    )
                    anno_img = v_gt.get_image()
                    img_list.append(anno_img)

                # regression pseudo-set
                if "instances_reg" in input:
                    v_gt2 = Visualizer(img, None)
                    v_gt2 = v_gt2.overlay_instances(
                        boxes=input["instances_reg"].gt_boxes.to("cpu")
                    )
                    anno_reg_img = v_gt2.get_image()
                    img_list.append(anno_reg_img)

                box_size = min(len(prop.pred_boxes), max_vis_prop)
                v_pred = Visualizer(img, None)
                v_pred = v_pred.overlay_instances(
                    boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy()
                )
                prop_img = v_pred.get_image()
                img_list.append(prop_img)

                vis_img = np.concatenate(tuple(img_list), axis=1)
                vis_img = vis_img.transpose(2, 0, 1)

                vis_name = (
                    branch
                    + " | Left: Pseudo-Cls; Center: Pseudo-Reg; Right: Predicted proposals"
                )
            else:
                break
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
