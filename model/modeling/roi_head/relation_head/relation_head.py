import torch
from torch import nn
from torch.autograd import Variable
import sys

from model.structure.pc_list import PcList
from model.modeling.roi_head.box_head.sampling import MySampling
from model.modeling.roi_head.relation_head.sampling import RelationSampling
from model.modeling.roi_head.box_head.roi_box_feature_extractor import box_feature_extractor, PointNet, PointNetCls
from model.modeling.roi_head.relation_head.Transformer.rel_predictor import PretrainPredictor
from model.modeling.roi_head.relation_head.Motif.rel_predictor import MotifPredictor
from model.modeling.roi_head.relation_head.Motif_aGCN.rel_predictor import MotifAGCNPredictor

from model.config import cfg

from model.modeling.roi_head.box_head.box_head import build_box_head
from my_utils.matcher import Matcher
from model.dataset.rscan_detection_dataset import RScanDetectionVotesDataset


class RelationHead(nn.Module):
    def __init__(self, cfg, mode):
        super(RelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.union_box_feature_extractor = None
        self.samp_processor = RelationSampling(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
            cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
            cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
            cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
            cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
            cfg.TEST.RELATION.REQUIRE_OVERLAP,
        )
        self.box_sampler = MySampling(
            proposal_matcher=Matcher(high_threshold=0.5, low_threshold=0.3, allow_low_quality_matches=False),
            batch_size=8
        )
        self.mode = mode

        self.feat_extractor = PointNetCls(k=26).cuda()
        for p in self.feat_extractor.parameters():
            p.requires_grad = False

        self.predictor = cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR

        if self.predictor == "SLAYOUT":
            self.rel_predictor = PretrainPredictor(config=cfg,
                                                   obj_feature_channels=cfg.MODEL.ROI_BOX_HEAD.OUT_FEATURE_DIM).cuda()
        elif self.predictor == "Motif":
            self.rel_predictor = MotifPredictor(config=cfg,
                                                obj_feature_channels=cfg.MODEL.ROI_BOX_HEAD.OUT_FEATURE_DIM).cuda()
        elif self.predictor == "aGCN":
            self.rel_predictor = MotifAGCNPredictor(config=cfg,
                                                    obj_feature_channels=cfg.MODEL.ROI_BOX_HEAD.OUT_FEATURE_DIM).cuda()

        self.obj_proj = nn.Linear(1024, 256)
        self.use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX

    def forward(self, proposals, pc, targets, ious=None):
        if self.training:
            if self.use_gt_box:  # PredCLS and SGCLS
                proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals,
                                                                                                        targets)
            else:  # SGDET
                proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals,
                                                                                                         targets,
                                                                                                         ious)
        else:
            # during inference, model still needs to know label and pair info(a bit stupid I know)
            if self.use_gt_box:
                proposals, gt_rel_labels, gt_rel_pair_idxs, gt_rel_binarys = self.samp_processor.gtbox_relsample(
                    proposals,
                    targets)
            else:
                proposals, gt_rel_labels, gt_rel_pair_idxs, gt_rel_binarys = self.samp_processor.detect_relsample(
                    proposals,
                    targets,
                    ious)
            rel_pair_idxs, rel_labels = self.samp_processor.prepare_test_pairs(proposals[0].bbox.device, proposals,
                                                                               gt_rel_pair_idxs, gt_rel_labels)
            rel_binarys = None

        # extract points, reorganize proposed bounding boxes.
        pc_batch, box_batch, box_idx = self.box_sampler.prep_box_head_batch(pc=pc, boxlist=proposals)

        # feature extract and initial object classification:
        obj_features, point_features, point_xyz, obj_dists = self.feature_extract(pc_batch=pc_batch, box_idx=box_idx,
                                                                                  need_padding=False,
                                                                                  num_box_per_scan=[p.size.shape[0] for
                                                                                                    p in
                                                                                                    proposals],
                                                                                  use_point_feature=not self.use_gt_box)
        if self.mode == "sgcls" or self.mode == "sgdet":
            for p, dist in zip(proposals, obj_dists):
                p.add_field('pred_cls', dist)

        '''Neural-Motif(Or any other general model)'''
        if self.predictor == "Motif" or self.predictor == "aGCN":
            obj_features = self.obj_proj(obj_features)
            refine_logits, relation_logits, add_losses, true_pair_idxs, obj_id_this_batch = self.rel_predictor(proposals,
                                                                                                               rel_pair_idxs,
                                                                                                               rel_labels,
                                                                                                               rel_binarys,
                                                                                                               roi_features=obj_features,
                                                                                                               union_features=None,
                                                                                                               logger=None)

        '''SLAYOUT(Or any other model that treats support and proximity relation differently)'''
        if self.predictor == "SLAYOUT":
            if self.training:
                refine_logits, support_pairs, support_rel_logits, support_rel_labels, proxi_pairs, proxi_rel_logits, proxi_rel_labels, obj_id_this_batch = self.rel_predictor(
                    proposals=proposals,
                    obj_features=obj_features,
                    point_features=point_features,
                    point_pos=point_xyz,
                    rel_pair_idxs=rel_pair_idxs,
                    rel_labels=rel_labels,
                    union_features=None,
                    logger=None)
            else:
                refine_logits, support_rel_dist, proxi_rel_dist, support_pairs, proxi_pairs, support_labels, \
                proxi_labels, obj_id_this_batch = self.rel_predictor(
                    proposals=proposals,
                    obj_features=obj_features,
                    point_features=point_features,
                    point_pos=point_xyz,
                    rel_pair_idxs=rel_pair_idxs,
                    rel_labels=rel_labels,
                    union_features=None,
                    logger=None)

        end_dict = {}
        end_dict['obj_features'] = obj_features
        end_dict['obj_labels'] = [p.get_field("labels") for p in proposals]
        end_dict['refine_logits'] = refine_logits
        end_dict['proposals'] = proposals
        end_dict['obj_id_this_batch'] = obj_id_this_batch
        end_dict['relation_labels'] = rel_labels
        end_dict['true_pair_idxs'] = rel_pair_idxs
        if self.predictor == "Motif" or self.predictor == "aGCN":
            end_dict['relation_logits'] = relation_logits
        elif self.predictor == "SLAYOUT":
            if self.training:
                end_dict['support_rel_logits'] = support_rel_logits
                end_dict['proxi_rel_logits'] = proxi_rel_logits
                end_dict['support_rel_labels'] = support_rel_labels
                end_dict['proxi_rel_labels'] = proxi_rel_labels
                end_dict['support_pairs'] = support_pairs
                end_dict['proxi_pairs'] = proxi_pairs
            else:
                end_dict['support_dists'] = support_rel_dist
                end_dict['proxi_dists'] = proxi_rel_dist
                end_dict['support_labels'] = support_labels
                end_dict['support_pairs'] = support_pairs
                end_dict['proxi_labels'] = proxi_labels
                end_dict['proxi_pairs'] = proxi_pairs

        return end_dict

    def feature_extract(self, pc_batch, box_idx, need_padding=False, num_box_per_scan=None, use_point_feature=False):
        total_obj_feature = []
        total_point_feature = []
        total_point_xyz = []
        '''feature extraction'''
        obj_dists, obj_feats_per_batch, _, point_feats_per_batch = self.feat_extractor(
            pc_batch.float().permute(0, 2, 1)
        )
        idx = box_idx
        keep = (idx == -1).nonzero().squeeze(-1)
        if len(keep) != 0:
            obj_feats_per_batch = obj_feats_per_batch[: 8 - len(keep)]
            point_feats_per_batch = point_feats_per_batch[: 8 - len(keep)]
        total_obj_feature.append(obj_feats_per_batch)
        total_point_feature.append(point_feats_per_batch)
        total_point_xyz.append(pc_batch)
        total_obj_feature = torch.cat(total_obj_feature, dim=0)
        total_point_feature = torch.cat(total_point_feature, dim=0)
        total_point_xyz = torch.cat(total_point_xyz, dim=0)
        '''Padding'''
        if need_padding:
            total_obj_feature = total_obj_feature.split(num_box_per_scan, dim=0)
            total_point_feature = total_point_feature.split(num_box_per_scan, dim=0)
            total_point_xyz = total_point_xyz.split(num_box_per_scan, dim=0)
            padded_total_obj_feature = []
            padded_total_point_feature = []
            padded_total_point_xyz = []
            for obj_feat, point_feat, point_xyz in zip(total_obj_feature, total_point_feature, total_point_xyz):
                padded_obj_feat = torch.zeros((self.cfg.MODEL.ROI_HEADS.MAX_GT_BOX_PER_SCAN,
                                               1024),
                                              device=torch.device('cuda:0'))
                padded_point_xyz = torch.zeros((self.cfg.MODEL.ROI_HEADS.MAX_GT_BOX_PER_SCAN,
                                                512,
                                                3),
                                               device=torch.device('cuda:0'))
                padded_obj_feat[:obj_feat.shape[0], :] = obj_feat
                padded_point_xyz[:point_feat.shape[0]] = point_xyz
                padded_total_obj_feature.append(padded_obj_feat)
                padded_total_point_xyz.append(padded_point_xyz)
                if use_point_feature:
                    padded_point_feat = torch.zeros((self.cfg.MODEL.ROI_HEADS.MAX_GT_BOX_PER_SCAN,
                                                     1024,
                                                     512),
                                                    device=torch.device('cuda:0'))
                    padded_point_feat[:point_feat.shape[0]] = point_feat
                    padded_total_point_feature.append(padded_point_feat)
            return padded_total_obj_feature, padded_total_point_feature, padded_total_point_xyz
        else:
            return total_obj_feature, total_point_feature, total_point_xyz, obj_dists.split(num_box_per_scan)


def generate_padding_GT(proposals, tgt_num):
    mask = torch.ones((len(proposals), tgt_num), device=torch.device('cuda:0')).byte()
    for i, proposal in enumerate(proposals):
        mask[i, :proposal.size.shape[0]] -= 1
        proposal.add_field('padding_mask', mask[i])

        padded_size = torch.zeros((tgt_num, 3), device=torch.device('cuda:0')) - 1
        padded_label = torch.zeros((tgt_num,), device=torch.device('cuda:0')) - 1
        padded_bbox = torch.zeros((tgt_num, 8, 3), device=torch.device('cuda:0')) - 1
        padded_ctr = torch.zeros((tgt_num, 3), device=torch.device('cuda:0')) - 1

        padded_size[:proposal.size.shape[0], :] = proposal.size
        padded_label[:proposal.get_field('labels').shape[0]] = proposal.get_field('labels')
        padded_bbox[:proposal.size.shape[0], :, :] = proposal.bbox
        padded_ctr[:proposal.size.shape[0], :] = proposal.get_field("obj_centers")

        proposal.bbox = padded_bbox
        proposal.size = padded_size
        proposal.add_field('padded_labels', padded_label)
        proposal.add_field('padded_centers', padded_ctr)

    return proposals


def build_pretrain_relation_head(cfg, mode):
    rel_head = RelationHead(cfg, mode).cuda()
    return rel_head

