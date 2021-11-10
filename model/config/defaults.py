import os

from yacs.config import CfgNode as CN
_C = CN()

# MODEL
_C.MODEL = CN()
_C.MODEL.FLIP_AUG = False
_C.MODEL.MASK_ON = False
_C.MODEL.RELATION_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""
_C.MODEL.PRETRAINED_DETECTOR_CKPT = ""
_C.MODEL.ROOT_DIR = ""

# DATASET
_C.DATASETS = CN()
_C.DATASETS.RSCAN_BASE_DIR = ""
_C.DATASETS.SSG_BASE_DIR = ""
_C.DATASETS.RSCAN_OBJ_10_CLASSES = ["table", "chair",
                                    "picture",
                                    "bag", "lamp", "shelf", "pillow",
                                    "door", "window", "curtain"]
_C.DATASETS.RSCAN_OBJ_40_CLASSES = ["table", "bed", "chair", "toilet", "sofa", "desk", "picture",
                                    "armchair",
                                    "bag", "lamp", "shelf", "object", "box", "window", "curtain", "cabinet",
                                    "sink", "towel", "door", "light", "plant", "basket", "bench", "blanket",
                                    "bucket",
                                    "clothes", "commode", "cushion", "heater", "item", "kitchen_cabinet",
                                    "monitor",
                                    "pc", "pillow", "shoes", "stool", "trash_can", "vase", "wardrobe",
                                    "windowsill"]
_C.DATASETS.RSCAN_OBJ_79_CLASSES = ["table", "bed", "chair", "toilet", "sofa", "desk", "picture",
                                    "armchair",
                                    "bag", "lamp", "shelf", "object", "box", "window", "curtain", "cabinet",
                                    "sink", "towel", "door", "light", "plant", "basket", "bench", "blanket",
                                    "bucket",
                                    "clothes", "commode", "cushion", "heater", "item", "kitchen_cabinet",
                                    "monitor",
                                    "pc", "pillow", "shoes", "stool", "trash_can", "vase", "wardrobe",
                                    "windowsill",
                                    "backpack",
                                    "bath_cabinet",
                                    "bathtub",
                                    "bicycle",
                                    "blinds",
                                    "book",
                                    "clutter",
                                    "coffee_tabel",
                                    "couch",
                                    "couch_table",
                                    "counter",
                                    "decoration",
                                    "doorframe",
                                    "frame",
                                    "kitchen_appliance",
                                    "kitchen_counter",
                                    "microwave",
                                    "mirror",
                                    "nightstand",
                                    "ottoman",
                                    "oven",
                                    "pillar",
                                    "plank",
                                    "printer",
                                    "rack",
                                    "radiator",
                                    "refrigerator",
                                    "showcase",
                                    "side_table",
                                    "stand",
                                    "stove",
                                    "suitcase",
                                    "toilet_paper",
                                    "tv",
                                    "tv_stand",
                                    "washing_machine",
                                    "whiteboard",
                                    "floor",
                                    "wall"
                                    ]
_C.DATASETS.RSCAN_OBJ_27_CLASSES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                                    'window', 'counter', 'shelf', 'curtain', 'pillow', 'clothes', 'fridge', 'tv',
                                    'towel', 'plant', 'box', 'nightstand', 'toilet', 'sink', 'lamp',
                                    'bathtub', 'object', 'blanket']
_C.DATASETS.RSCAN_REL_CLASSES_KEYS = ["None", "supported_by", "left", "right", "front", "behind",
                                      "close_by",
                                      "bigger_than", "smaller_than",
                                      "higher_than", "lower_than", "same_symmetry_as", "same_as",
                                      "attached_to",
                                      "standing_on", "lying_on",
                                      "hanging_on", "connected_to", "leaning_against", "part_of",
                                      "belonging_to",
                                      "build_in", "standing_in",
                                      "cover", "lying_in", "hanging_in", "same_color", "same_shape",
                                      "brighter_than",
                                      "darker_than"]
_C.DATASETS.RSCAN_NEW_REL_CLASSES_KEYS = ["None", "supported by", "left", "front",
                                          "attached to",
                                          "standing on", "lying on",
                                          "hanging on", "connected to", "leaning against", "part of",
                                          "belonging to",
                                          "build in", "standing in",
                                          "cover", "lying in", "hanging in"]
_C.DATASETS.RSCAN_SUPPORT_REL_CLASSES_KEYS = ["None", "supported by",
                                              "attached to",
                                              "standing on", "lying on",
                                              "hanging on", "connected to", "leaning against", "part of",
                                              "belonging to",
                                              "build in", "standing in",
                                              "cover", "lying in", "hanging in"]
_C.DATASETS.RSCAN_REL_CLASSES_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                        23, 24, 25, 26, 27, 30, 38, 39]
_C.DATASETS.RSCAN_NEW_REL_CLASSES_VALUES = [0, 1, 2, 4, 14, 15, 16, 17, 18, 19, 20,
                                            21, 22,
                                            23, 24, 25, 26]
_C.DATASETS.RSCAN_SUPPORT_REL_CLASSES_IDX = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
_C.DATASETS.RSCAN_PROXI_REL_CLASSES_IDX = [2, 3]
_C.DATASETS.RSCAN_MAX_REL = 1

_C.DATASETS.TRAIN = ()
_C.DATASETS.VAL = ()
_C.DATASETS.TEST = ()

# DATALOADER
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SIZE_DIVISIBILITY = 0

# BACKBONE
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CKPT_DIR = "./log/detector/checkpoint.tar"

# GROUP NORM
_C.MODEL.GROUP_NORM = CN()
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# RPN options
_C.MODEL.RPN = CN()
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
_C.MODEL.RPN.NMS_THRESH = 0.7

# ROI HEAD
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.3
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.3
_C.MODEL.ROI_HEADS.MAX_GT_BOX_PER_SCAN = 128
_C.MODEL.ROI_HEADS.MAX_PRED_BOX_PER_SCAN = 256
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.01
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 256

# BOX HEAD
_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 26
_C.MODEL.ROI_BOX_HEAD.IN_FEATURE_DIM = 0
_C.MODEL.ROI_BOX_HEAD.OUT_FEATURE_DIM = 256

# RELATION HEAD
_C.MODEL.ROI_RELATION_HEAD = CN()
_C.MODEL.ROI_RELATION_HEAD.ABLATION_STUDY = False
_C.MODEL.ROI_RELATION_HEAD.NO_LSTM = False
_C.MODEL.ROI_RELATION_HEAD.NO_SUPP = False
_C.MODEL.ROI_RELATION_HEAD.NO_PROX = False
_C.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR_CKPT_DIR = "./log/pretrain/extractor/checkpoint.tar"
_C.MODEL.ROI_RELATION_HEAD.SUPPORT_CKPT_DIR = "./log/pretrain/support/checkpoint.tar"
_C.MODEL.ROI_RELATION_HEAD.CKPT_DIR = ""
_C.MODEL.ROI_RELATION_HEAD.PREDICTOR = "Motif"
_C.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 20
_C.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 64
_C.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
_C.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
_C.MODEL.ROI_RELATION_HEAD.EMBED_DIM = 200
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM = 512
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = 4096
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER = 1  # assert >= 1
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER = 1  # assert >= 1

_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER = CN()
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE = 0.1
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER = 4
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER = 2
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD = 8
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM = 2048
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM = 64
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM = 64

_C.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS = False
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION = False
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = False
_C.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
_C.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL = 4
_C.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = False

_C.MODEL.VGG = CN()

_C.MODEL.RESNETS = CN()

_C.MODEL.RETINANET = CN()

_C.MODEL.FBNET = CN()

# Solver
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.BASE_LR = 0.002
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.CLIP_NORM = 5.0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.SCHEDULE = CN()
_C.SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"
_C.SOLVER.SCHEDULE.PATIENCE = 2
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4
_C.SOLVER.SCHEDULE.COOLDOWN = 1
_C.SOLVER.SCHEDULE.FACTOR = 0.5
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 7
_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.GRAD_NORM_CLIP = 5.0
_C.SOLVER.PRINT_GRAD_FREQ = 5000
_C.SOLVER.TO_VAL = True
_C.SOLVER.PRE_VAL = True
_C.SOLVER.VAL_PERIOD = 2500
_C.SOLVER.UPDATE_SCHEDULE_DURING_LOAD = False

# Specific test options
_C.TEST = CN()
_C.TEST.BBOX_AUG = CN()
_C.TEST.BBOX_AUG.ENABLED = False
_C.TEST.BBOX_AUG.H_FLIP = False
_C.TEST.BBOX_AUG.SCALES = ()
_C.TEST.RELATION = CN()
_C.TEST.RELATION.MULTIPLE_PREDS = False
_C.TEST.RELATION.IOU_THRESHOLD = 0.5
_C.TEST.RELATION.REQUIRE_OVERLAP = False
_C.TEST.RELATION.LATER_NMS_PREDICTION_THRES = 0.3
_C.TEST.ALLOW_LOAD_FROM_CACHE = True

# Misc options
_C.OUTPUT_DIR = "."
_C.DETECTED_SGG_DIR = "."
_C.GLOVE_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.PATHS_DATA = os.path.join(os.path.dirname(__file__), "../data/datasets")

# Precision options
_C.DTYPE = "float32"


