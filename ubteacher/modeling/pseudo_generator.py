# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.structures import Boxes
from detectron2.structures.instances import Instances
from ubteacher.modeling.fcos.fcos_outputs import FCOSOutputs


class PseudoGenerator:
    def __init__(self, cfg):
        self.fcos_output = FCOSOutputs(cfg)

    def nms_from_dense(self, raw_output, nms_method):

        assert nms_method in ["cls", "ctr", "cls_n_ctr", "cls_n_loc"]

        logits_pred = raw_output["logits_pred"]
        reg_pred = raw_output["reg_pred"]
        top_feats = raw_output["top_feats"]
        locations = raw_output["locations"]
        ctrness_pred = raw_output["ctrness_pred"]
        image_sizes = raw_output["image_sizes"]

        reg_pred_std = None
        if "reg_pred_std" in raw_output:
            reg_pred_std = raw_output["reg_pred_std"]

        results = self.fcos_output.predict_proposals(
            logits_pred,
            reg_pred,
            ctrness_pred,
            locations,
            image_sizes,
            reg_pred_std,
            top_feats,
            nms_method,
        )
        return results

    # generate
    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            elif psedo_label_method == "thresholding_cls_ctr":
                proposal_bbox_inst = self.threshold_cls_ctr_bbox(
                    proposal_bbox_inst, thres=cur_threshold
                )

            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        # for fcos
        if isinstance(proposal_bbox_inst, dict):
            proposal_bbox_inst = proposal_bbox_inst["instances"]

        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]

        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)
            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
            new_proposal_inst.centerness = proposal_bbox_inst.centerness[valid_map]
            new_proposal_inst.cls_confid = proposal_bbox_inst.cls_confid[valid_map]
            if proposal_bbox_inst.has("reg_pred_std"):
                new_proposal_inst.reg_pred_std = proposal_bbox_inst.reg_pred_std[
                    valid_map
                ]

        return new_proposal_inst

    def threshold_cls_ctr_bbox(self, proposal_bbox_inst, thres=(0.5, 0.5)):
        # for fcos
        if isinstance(proposal_bbox_inst, dict):
            proposal_bbox_inst = proposal_bbox_inst["instances"]
        cls_map = proposal_bbox_inst.cls_confid > thres[0]
        ctr_map = proposal_bbox_inst.centerness > thres[1]
        valid_map = cls_map * ctr_map

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)
        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        new_proposal_inst.centerness = proposal_bbox_inst.centerness[valid_map]
        new_proposal_inst.cls_confid = proposal_bbox_inst.cls_confid[valid_map]
        if proposal_bbox_inst.has("reg_pred_std"):
            new_proposal_inst.reg_pred_std = proposal_bbox_inst.reg_pred_std[valid_map]

        return new_proposal_inst
