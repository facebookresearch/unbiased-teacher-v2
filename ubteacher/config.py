# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.config import CfgNode as CN


def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0

    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ubteacher"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.UNSUP_REG_LOSS_WEIGHT = 0.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

    _C.SEMISUPNET.PROBE = True
    _C.SEMISUPNET.PSEUDO_CTR_THRES = 0.5

    _C.SEMISUPNET.EMA_SCHEDULE = False

    _C.SEMISUPNET.PSEUDO_CLS_IGNORE_NEAR = False
    _C.SEMISUPNET.SOFT_CLS_LABEL = False
    _C.SEMISUPNET.CLS_LOSS_METHOD = "focal"
    _C.SEMISUPNET.CLS_LOSS_PSEUDO_METHOD = "focal"

    _C.SEMISUPNET.REG_FG_THRES = 0.5

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    fb_path = (
        "manifold://mobile_vision_dataset/tree/unbiased_teacher/COCO_supervision.txt"
    )
    local_path = "dataseed/COCO_supervision.txt"
    if os.path.isfile(local_path):
        _C.DATALOADER.RANDOM_DATA_SEED_PATH = local_path
    else:
        _C.DATALOADER.RANDOM_DATA_SEED_PATH = fb_path
    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True

    # ---------------------------------------------------------------------------- #
    # FCOS Head
    # ---------------------------------------------------------------------------- #
    # _C.MODEL.FCOS = CN()

    # # This is the number of foreground classes.
    # _C.MODEL.FCOS.NUM_CLASSES = 80
    # _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    # _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    # _C.MODEL.FCOS.PRIOR_PROB = 0.01
    # _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    # _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    # _C.MODEL.FCOS.NMS_TH = 0.6
    # _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    # _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    # _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    # _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    # _C.MODEL.FCOS.TOP_LEVELS = 2
    # _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
    # _C.MODEL.FCOS.USE_SCALE = True

    # # Multiply centerness before threshold
    # # This will affect the final performance by about 0.05 AP but save some time
    # _C.MODEL.FCOS.THRESH_WITH_CTR = False

    # # Focal loss parameters
    # _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    # _C.MODEL.FCOS.LOSS_GAMMA = 2.0
    # _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    # _C.MODEL.FCOS.USE_RELU = True
    # _C.MODEL.FCOS.USE_DEFORMABLE = False

    # # the number of convolutions used in the cls and bbox tower
    # _C.MODEL.FCOS.NUM_CLS_CONVS = 4
    # _C.MODEL.FCOS.NUM_BOX_CONVS = 4
    # _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
    # _C.MODEL.FCOS.CENTER_SAMPLE = True
    # _C.MODEL.FCOS.POS_RADIUS = 1.5
    # _C.MODEL.FCOS.LOC_LOSS_TYPE = "giou"
    # _C.MODEL.FCOS.YIELD_PROPOSAL = False

    # _C.MODEL.FCOS.NMS_CRITERIA_TRAIN = "cls"
    # _C.MODEL.FCOS.NMS_CRITERIA_TRAIN_REG = "cls_n_loc"
    # _C.MODEL.FCOS.NMS_CRITERIA_TEST = "cls_n_ctr"

    # ----------------------------------------------- #
    # Generalized Focal loss
    # ----------------------------------------------- #

    _C.MODEL.FCOS = CN()

    # This is the number of foreground classes.
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    _C.MODEL.FCOS.NMS_TH = 0.6
    _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    _C.MODEL.FCOS.TOP_LEVELS = 2
    _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
    _C.MODEL.FCOS.USE_SCALE = True

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    _C.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    _C.MODEL.FCOS.LOSS_GAMMA = 2.0
    _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    _C.MODEL.FCOS.USE_RELU = True
    _C.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.FCOS.NUM_CLS_CONVS = 4
    _C.MODEL.FCOS.NUM_BOX_CONVS = 4
    _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
    _C.MODEL.FCOS.CENTER_SAMPLE = True
    _C.MODEL.FCOS.POS_RADIUS = 1.5
    _C.MODEL.FCOS.LOC_LOSS_TYPE = "giou"
    _C.MODEL.FCOS.YIELD_PROPOSAL = False

    _C.MODEL.FCOS.NMS_CRITERIA_TRAIN = "cls"
    _C.MODEL.FCOS.NMS_CRITERIA_TEST = "cls_n_ctr"
    _C.MODEL.FCOS.NMS_CRITERIA_REG_TRAIN = "cls_n_loc"

    _C.MODEL.FCOS.REG_DISCRETE = False
    _C.MODEL.FCOS.DFL_WEIGHT = 0.0
    _C.MODEL.FCOS.LOC_FUN_ALL = "mean"

    _C.MODEL.FCOS.UNIFY_CTRCLS = False

    _C.MODEL.FCOS.REG_MAX = 16

    _C.MODEL.FCOS.QUALITY_EST = "centerness"  # 'iou'

    _C.MODEL.FCOS.TSBETTER_CLS_SIGMA = 0.0

    # pseudo-labeling (joint)
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.5
    _C.SEMISUPNET.BBOX_CTR_THRESHOLD = 0.5

    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG = "thresholding"
    _C.SEMISUPNET.BBOX_THRESHOLD_REG = 0.5
    _C.SEMISUPNET.BBOX_CTR_THRESHOLD_REG = 0.5

    # analysis config
    _C.SEMISUPNET.ANALYSIS_PRINT_FRE = 5000
    _C.SEMISUPNET.ANALYSIS_ACCUMLATE_FRE = 200

    _C.SEMISUPNET.TS_BETTER = 0.1
    _C.SEMISUPNET.TS_BETTER_CERT = 0.8

    # unsupervised loss
    _C.SEMISUPNET.CONSIST_CLS_LOSS = "mse_loss_raw"
    _C.SEMISUPNET.CONSIST_CTR_LOSS = "kl_loss"
    _C.SEMISUPNET.CONSIST_REG_LOSS = "mse_loss_all_raw"

    # horizontal flip
    _C.SEMISUPNET.RANDOM_FLIP_STRONG = False

    # predict uncertainty for box localization
    _C.MODEL.FCOS.KL_LOSS = False
    _C.MODEL.FCOS.KL_LOSS_TYPE = "klloss"  # "nlloss"
    _C.MODEL.FCOS.KLLOSS_WEIGHT = 0.1

    _C.SEMISUPNET.DYNAMIC_EMA = False
    _C.SEMISUPNET.DEMA_FINAL = 1.0

    _C.MODEL.ROI_BOX_HEAD.BBOX_PSEUDO_REG_LOSS_TYPE = "tsbetter"

    # regression
    _C.SEMISUPNET.T_CERT = 0.5

    # EMA scheduler
    # if false, then use the EMA_KEEP_RATE
    # if true, then use the rate_step
    _C.SEMISUPNET.EMA_SCHEDULER = False
    _C.SEMISUPNET.EMA_RATE_STEP = (0.9996,)
    _C.SEMISUPNET.EMA_INTVEL = (120000,)

    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0

    # use weak labeled
    _C.SEMISUPNET.USE_SUP_STRONG = "both"
