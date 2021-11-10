import torch
import numpy as np
import scipy.linalg

from model.structure.box3d_list import Box3dList
from model.dataset.model_util_rscan import RScanDatasetConfig
from model.modeling.detector.utils.box_util import box3d_iou_depth, box3d_vol
from model.dataset.rscan_utils import flip_axis_to_camera, flip_axis_to_depth

from my_utils.my_nms import _box_nms
DC = RScanDatasetConfig()


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="pred_score"):
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxes = boxes[keep]
    return boxes, keep


def remove_small_boxes(boxlist, min_size):
    keep = np.argwhere((boxlist.area(DC) >= min_size) > 0)
    boxlist.clip(keep)
    return boxlist


def boxlist_iou(boxlist1, boxlist2):
    """
    Arguments:
      box1: (box3d_list) bounding boxes
      box2: (box3d_list) bounding boxes

    Returns:
      (tensor) iou, sized [N,M].
      (tensor) bird's eye view 2D bounding box IoU, sized [N,M]

    """
    N = len(boxlist1)
    M = len(boxlist2)

    boxes1 = boxlist1.bbox.detach().cpu().numpy() # (N,8,3)
    boxes2 = boxlist2.bbox.detach().cpu().numpy()  # (M,8,3)

    # boxes1 = flip_axis_to_camera(boxes1.reshape(-1, 3)).reshape(-1, 8, 3)
    # boxes2 = flip_axis_to_camera(boxes2.reshape(-1, 3)).reshape(-1, 8, 3)

    iou = np.zeros(shape=[N, M], dtype=np.float)
    iou2d = np.zeros(shape=[N, M], dtype=np.float)

    for n, box1 in enumerate(boxes1):
        for m, box2 in enumerate(boxes2):
            iou[n, m], iou2d[n, m] = box3d_iou_depth(box1, box2)

    iou = torch.as_tensor(iou).cuda()
    iou2d = torch.as_tensor(iou2d).cuda()

    return iou, iou2d


def boxlist_iou_tensor(boxlist1, boxlist2):

    N = boxlist1.shape[0]
    M = boxlist2.shape[0]

    boxes1 = boxlist1.detach().cpu().numpy()
    boxes2 = boxlist2.detach().cpu().numpy()

    # boxes1 = flip_axis_to_camera(boxes1.reshape(-1, 3)).reshape(-1, 8, 3)
    # boxes2 = flip_axis_to_camera(boxes2.reshape(-1, 3)).reshape(-1, 8, 3)

    iou = np.zeros(shape=[N, M], dtype=np.float)
    iou2d = np.zeros(shape=[N, M], dtype=np.float)

    for n, box1 in enumerate(boxes1):
        for m, box2 in enumerate(boxes2):
            iou[n, m], iou2d[n, m] = box3d_iou_depth(box1, box2)

    iou = torch.as_tensor(iou).to(boxlist1.device)
    iou2d = torch.as_tensor(iou2d).to(boxlist1.device)

    return iou, iou2d


def boxlist_iou_tensor_faster(boxlist1, boxlist2):
    N = boxlist1.shape[0]
    M = boxlist2.shape[0]

    # boxes1 = flip_axis_to_camera(boxes1.reshape(-1, 3)).reshape(-1, 8, 3)
    # boxes2 = flip_axis_to_camera(boxes2.reshape(-1, 3)).reshape(-1, 8, 3)

    iou = torch.zeros((N, M), dtype=torch.float64).to(boxlist1.device)
    iou2d = torch.zeros((N, M), dtype=torch.float64).to(boxlist1.device)

    for n, box1 in enumerate(boxlist1):
        for m, box2 in enumerate(boxlist2):
            iou[n, m], iou2d[n, m] = box3d_iou_simple(box1, box2)

    return iou, iou2d


def box3d_iou_simple(box1, box2):
    x1_1 = torch.min(box1[:, 0])
    x1_2 = torch.min(box2[:, 0])
    y1_1 = torch.min(box1[:, 1])
    y1_2 = torch.min(box2[:, 1])
    z1_1 = torch.min(box1[:, 2])
    z1_2 = torch.min(box2[:, 2])
    x2_1 = torch.max(box1[:, 0])
    x2_2 = torch.max(box2[:, 0])
    y2_1 = torch.max(box1[:, 1])
    y2_2 = torch.max(box2[:, 1])
    z2_1 = torch.max(box1[:, 2])
    z2_2 = torch.max(box2[:, 2])

    xx1 = torch.max(torch.as_tensor([x1_1, x1_2]))
    yy1 = torch.max(torch.as_tensor([y1_1, y1_2]))
    zz1 = torch.max(torch.as_tensor([z1_1, z1_2]))
    xx2 = torch.min(torch.as_tensor([x2_1, x2_2]))
    yy2 = torch.min(torch.as_tensor([y2_1, y2_2]))
    zz2 = torch.min(torch.as_tensor([z2_1, z2_2]))

    l = np.maximum(0, xx2-xx1)
    w = np.maximum(0, yy2-yy1)
    h = np.maximum(0, zz2-zz1)

    overlap = l*w*h
    overlap2d = l*w
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1) * (z2_1 - z1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2) * (z2_2 - z1_2)
    area2d_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2d_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return overlap/(area1+area2-overlap), overlap2d/(area2d_1 + area2d_2 - overlap2d)