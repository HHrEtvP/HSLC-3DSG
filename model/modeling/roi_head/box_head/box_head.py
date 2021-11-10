"""
#REMOVE:
box_head改编自Scene Graph Benchmark
box_head由3个部分组成
1.sampling
2.roi_box_feature_extractor
3.roi_box_predictor
sampling接收proposal(votenet输出,或者GTbox)和target(GT)，为每个proposal提供label信息
如果是PredCLS则直接跳过
roi_box_feature_extractor接收proposal，使用pointnet++为其中每个box提取特征
roi_box_predictor，根据特征和box输出box预测的label，如果是PredCLS则直接跳过,其实就算是SGCLS或者SGDet也可以跳过，因为votenet已经做了预测工作
要说finetune也不是不行，但是如果这里预测的结果反而更差呢？
原本还有一个post_processer,给box预测score，执行NMS等，该组件仅限SGDet
这里因为要执行大量过滤操作，感觉跟NMS有些重叠，甚至感觉大部分内容都可以在votenet中找到，所以我觉得这里暂时忽略也不是不行
但是在SGCLS中，由于不知道label，必须要走predictor，不需要走post_processor而已，因为是GTbox
#EDIT:
经过一些思考后，决定大改Boxhead的内容
Boxhead现在根据不同的情形有不同的任务：
1.SGDet和PredCLS:基本上什么都不做，接收boxlist，pclist和关系矩阵rel_mat和映射rel_obj_map，将这些内容装入，然后送入relation_head
解析输出或者GT可以用votenet的parse_prediction或者parse_groundtruth
2.SGCLS：目前存疑，或许也是直接跳过，或许在这里做predict
不过好在这里的很多东西，包括feature extractor和sampler倒是可以复用
BOXHEAD的cfg融合做好了
"""
import os
import torch
from torch import nn
import numpy as np

from model.config import cfg
from model.modeling.roi_head.box_head.sampling import make_roi_box_samp_processor
from model.structure.box3d_list import Box3dList
from model.structure.pc_list import PcList
from model.dataset.model_util_rscan import RScanDatasetConfig as DC


class BoxHead:
    def __init__(self, cfg, mode):
        self.sampler = make_roi_box_samp_processor(cfg)
        self.mode = mode
        self.config = {'remove_empty_box': False, 'use_3d_nms': True,
                       'nms_iou': 0.4, 'use_old_type_nms': False, 'cls_nms': True,
                       'per_class_proposal': False, 'conf_thresh': 0.05,
                       'dataset_config': DC()}
        self.max_gt_box = cfg.MODEL.ROI_HEADS.MAX_GT_BOX_PER_SCAN
        self.max_pred_box = cfg.MODEL.ROI_HEADS.MAX_PRED_BOX_PER_SCAN

    def parse(self, pc, rel_mat=None, rel_obj_id=None, rel_mat_idx=None, output=None, GT=None):
        boxlists = []
        gtlists = []
        pclist = PcList(xyz_tensor=pc)
        if self.mode == "predcls" or self.mode == "sgcls":
            with torch.no_grad():
                assert (output is None and GT is not None)
                gts = parse_groundtruths_after_detector(GT, config_dict=self.config)
                for i, scan in enumerate(gts):
                    cls_labs = []
                    bboxes = []
                    box_size = []
                    ctrs = []
                    per_scan_obj_id = []
                    for j, (classname, bbox, b_size, id, ctr) in enumerate(scan):
                        cls_labs.append(classname)
                        bboxes.append(bbox)
                        box_size.append(b_size)
                        ctrs.append(ctr)
                        per_scan_obj_id.append(id)
                    one_boxlist = Box3dList(
                        size=box_size,
                        label=cls_labs,
                        corners=bboxes,
                    )
                    one_boxlist.add_field("relation", rel_mat[i])
                    one_boxlist.add_field("rel_obj_id", rel_obj_id[i])
                    one_boxlist.add_field("rel_mat_idx", rel_mat_idx[i])
                    one_boxlist.add_field("obj_ids", per_scan_obj_id)
                    one_boxlist.add_field("obj_centers", torch.cat(ctrs, dim=0).view(-1, 3))
                    gtlists.append(one_boxlist)
                    boxlists.append(one_boxlist)
            return boxlists, pclist, gtlists
        elif self.mode == "sgdet":
            assert (output is not None and GT is not None)
            gts = parse_groundtruths_after_detector(GT, config_dict=self.config)
            for i, scan in enumerate(gts):
                cls_labs = []
                bboxes = []
                box_size = []
                ctrs = []
                per_scan_obj_id = []
                for j, (classname, bbox, b_size, id, ctr) in enumerate(scan):
                    cls_labs.append(classname)
                    bboxes.append(bbox)
                    box_size.append(b_size)
                    ctrs.append(ctr)
                    per_scan_obj_id.append(id)
                one_boxlist = Box3dList(
                    size=box_size,
                    label=cls_labs,
                    corners=bboxes,
                )
                one_boxlist.add_field("relation", rel_mat[i])
                one_boxlist.add_field("rel_obj_id", rel_obj_id[i])
                one_boxlist.add_field("rel_mat_idx", rel_mat_idx[i])
                one_boxlist.add_field("obj_ids", per_scan_obj_id)
                one_boxlist.add_field("obj_centers", torch.cat(ctrs, dim=0).view(-1, 3))
                gtlists.append(one_boxlist)
            preds = parse_predictions_after_detector(output, self.config)
            for i, scan in enumerate(preds):
                pred_cls = []
                bboxes = []
                box_sizes = []
                obj_scores = []
                ctrs = []
                for j, (cls, bbox, size, objscore, ctr) in enumerate(scan):
                    pred_cls.append(cls)
                    bboxes.append(bbox)
                    box_sizes.append(size)
                    obj_scores.append(objscore)
                    ctrs.append(ctr)
                one_boxlist = Box3dList(
                    size=box_sizes,
                    label=None,
                    corners=bboxes,
                )
                one_boxlist.add_field("pred_cls", torch.as_tensor(pred_cls).int().cuda())
                one_boxlist.add_field("pred_obj_score", obj_scores)
                one_boxlist.add_field("obj_centers", torch.cat(ctrs, dim=0).view(-1, 3))
                boxlists.append(one_boxlist)
            '''筛去不佳候选框，减少计算量'''
            boxlists, ious = self.sampler.assign_label_to_proposals(proposals=boxlists, targets=gtlists)
            return boxlists, pclist, gtlists, ious


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def parse_predictions_after_detector(end_points, config_dict):
    """
    process the output of votenet, execute NMS
    无事勿触碰
    """
    from model.dataset.rscan_utils import extract_pc_in_box3d, my_compute_box_3d
    from model.modeling.detector.utils.nms import nms_3d_faster_samecls

    pred_center = end_points['center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = torch.softmax(end_points['sem_cls_scores'], dim=-1)  # B,num_proposal,num_class
    # pred_sem_cls_prob = torch.max(sem_cls_probs, -1).item()  # B,num_proposal

    num_proposal = pred_center.shape[1]
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_depth = torch.zeros((bsize, num_proposal, 8, 3)).cuda()
    pred_center_upright_depth = pred_center
    pred_box_size = torch.zeros((bsize, num_proposal, 3)).cuda()
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            pred_box_size[i, j] = torch.from_numpy(box_size).cuda()
            corners_3d_upright_depth = my_compute_box_3d(size=box_size, heading_angle=heading_angle,
                                                         center=pred_center_upright_depth[i, j,
                                                                :].detach().cpu().numpy())
            pred_corners_3d_upright_depth[i, j] = torch.from_numpy(corners_3d_upright_depth)

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'][:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_depth[i, j, :, :]  # (8,3)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points['objectness_scores']
    obj_prob = torch.softmax(obj_logits, dim=-1)[:, :, 1] # (B,K)
    if config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_depth[i, j, :, 0].detach().cpu().numpy())
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_depth[i, j, :, 1].detach().cpu().numpy())
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_depth[i, j, :, 2].detach().cpu().numpy())
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_depth[i, j, :, 0].detach().cpu().numpy())
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_depth[i, j, :, 1].detach().cpu().numpy())
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_depth[i, j, :, 2].detach().cpu().numpy())
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                cur_list += [(ii, pred_corners_3d_upright_depth[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                             for j in range(pred_center.shape[1]) if
                             pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            # returns:(pred_cls, corners, size, obj_scores)
            batch_pred_map_cls.append(
                [(pred_sem_cls[i, j], pred_corners_3d_upright_depth[i, j], pred_box_size[i, j], obj_prob[i, j], pred_center_upright_depth[i, j]) \
                 for j in range(pred_center.shape[1]) if
                 pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls


def parse_groundtruths_after_detector(end_points, config_dict):
    """
    process groundtruth.
    无事勿触碰
    """
    from model.dataset.rscan_utils import extract_pc_in_box3d, my_compute_box_3d

    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    sem_cls_label = end_points['sem_cls_label']
    obj_ids = end_points['obj_ids']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = torch.zeros((bsize, K2, 8, 3)).cuda()
    gt_box_size = torch.zeros((bsize, K2, 3)).cuda()
    gt_center_upright_camera = center_label[:, :, 0:3]
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0 : continue
            heading_angle = config_dict['dataset_config'].class2angle(
                pred_cls=heading_class_label[i, j].detach().cpu().numpy(),
                residual=heading_residual_label[i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(
                pred_cls=int(size_class_label[i, j].detach().cpu().numpy()),
                residual=size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = my_compute_box_3d(
                center=gt_center_upright_camera[i, j, :].detach().cpu().numpy(), size=box_size,
                heading_angle=heading_angle)
            gt_corners_3d_upright_camera[i, j] = torch.from_numpy(corners_3d_upright_camera).cuda()
            gt_box_size[i, j] = torch.from_numpy(box_size).cuda()

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append(
            [(sem_cls_label[i, j], gt_corners_3d_upright_camera[i, j], gt_box_size[i, j], obj_ids[i, j], center_label[i, j]) for j in
             range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i, j] == 1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


def build_box_head(config, mode):
    head = BoxHead(config, mode)
    return head
