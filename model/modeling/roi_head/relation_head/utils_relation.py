import array
import os
import zipfile
import itertools
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
from my_utils.misc import cat


def get_box_info(boxes, need_norm=True, proposal=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch.cat((boxes, center_box), 1)
    if need_norm:
        box_info = box_info / float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info


def get_box_pair_info(box1, box2):
    """
    input: 
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output: 
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:, :4].clone()
    unionbox[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox, need_norm=False)

    # intersection box
    intersextion_box = box1[:, :4].clone()
    intersextion_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    case1 = torch.nonzero(
        intersextion_box[:, 2].contiguous().view(-1) < intersextion_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch.nonzero(
        intersextion_box[:, 3].contiguous().view(-1) < intersextion_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersextion_box, need_norm=False)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch.cat((box1, box2, union_info, intersextion_info), 1)


def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    # boxes:(num_obj,num_class,4)
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    inters = inter[:, :, :, 0] * inter[:, :, :, 1]  # (N,N,nc)
    boxes_flat = boxes.view(-1, 4)  # (N*nc,4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (  # Xmax-Xmin
            boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0)  # Ymax-Ymin
    areas = areas_flat.view(boxes.size(0), boxes.size(1))  # (N,nc)
    union = -inters + areas[None] + areas[:, None]  # areas[None]+areas[:,None]->(N,N.nc)
    return inters / union  # (N,N,nc))


def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)
        return


def obj_prediction_nms(boxes_per_cls, pred_logits, nms_thresh=0.3):
    """
    boxes_per_cls:               [num_obj, num_cls, 4]
    pred_logits:                 [num_obj, num_category]
    return:
    pred_label:(num_obj,)
    """
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]

    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0), boxes_per_cls.size(0),
                                                  boxes_per_cls.size(1)).cpu().numpy() >= nms_thresh  # (N,N,nc):(Ture or False)

    prob_sampled = F.softmax(pred_logits, 1).cpu().numpy()
    prob_sampled[:, 0] = 0  # set bg to 0

    pred_label = torch.zeros(num_obj, device=pred_logits.device, dtype=torch.int64)

    for i in range(num_obj):
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0

    return pred_label


def block_orthogonal(tensor, split_sizes, gain=1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]
