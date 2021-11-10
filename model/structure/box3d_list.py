# -*- coding: UTF-8 -*-
import numpy as np
import torch
from model.dataset.model_util_rscan import RScanDatasetConfig


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def compute_bbox_3d(heading, centroid, size):
    heading = heading.cpu().numpy()
    centroid = centroid.cpu().numpy()
    size = size.cpu().numpy()
    heading_angle = np.arctan2(heading[1], heading[0])
    Rmat = rotz(-1 * heading_angle)
    l, w, h = size[:] / 2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corner_pts = np.dot(Rmat, np.vstack([x_corners, y_corners, z_corners]))
    corner_pts[:, 0] = corner_pts[:, 0] + centroid[0]
    corner_pts[:, 1] = corner_pts[:, 1] + centroid[1]
    corner_pts[:, 2] = corner_pts[:, 2] + centroid[2]
    return corner_pts


class Box3dList(object):
    def __init__(
            self,
            size=None,
            label=None,
            corners=None,
            center=None,
            is_empty=False
    ):
        self.extra_fields = {}
        self.triplet_extra_fields = {}
        self.DC = RScanDatasetConfig()
        self.withBG = False
        if size is not None:
            self.size = torch.cat([s.unsqueeze(0) for s in size], dim=0)
        if label is not None:
            self.add_field("labels", torch.as_tensor(label).to(self.size.device), is_triplet=False)
        if corners is None:
            self.bbox = None
        else:
            self.bbox = torch.cat([c.unsqueeze(0) for c in corners], dim=0)
        self.is_empty = is_empty
        self.device = self.size.device if isinstance(self.size, torch.Tensor) else torch.device("cpu")

    def to_device(self):
        self.size = self.size.to(self.device)
        self.bbox = self.bbox.to(self.device)
        for (k, v) in self.extra_fields.items():
            if isinstance(self.extra_fields[k], torch.Tensor):
                self.extra_fields[k] = self.extra_fields.pop(k).to(self.device)

    def area(self):
        l, w, h = self.bbox[:, 0, :] - self.bbox[:, 6, :]
        return (l * w).cpu().numpy()

    def vol(self):
        return self.size[:, 0]*self.size[:, 1]*self.size[:, 2]

    def add_field(self, field, field_data, is_triplet=False):
        self.extra_fields[field] = field_data
        if is_triplet:
            self.triplet_extra_fields[field] = field_data

    def add_tensor_field(self, field, data, dim):
        if self.has_field(field):
            if isinstance(self.extra_fields[field], torch.Tensor):
                self.extra_fields[field] = torch.cat([self.extra_fields[field], data], dim=dim)
                return True
            else:
                return False
        else:
            self.extra_fields[field] = data
            return False

    def list_field_append(self, field, append_data):
        if self.has_field(field):
            if isinstance(self.extra_fields[field], list):
                self.extra_fields[field].append(append_data)
                return True
            else:
                return False
        else:
            self.extra_fields[field] = []
            self.extra_fields[field].append(append_data)
            return True

    def get_field(self, field):
        if isinstance(self.extra_fields[field], list):
            return torch.as_tensor(self.extra_fields[field])
        else:
            return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def __getitem__(self, item):
        bbox = Box3dList(size=self.size[item], corners=self.bbox[item])
        for k, v in self.extra_fields.items():
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v[item][:, item], is_triplet=True)
            else:
                bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def copy(self):
        return Box3dList(size=self.size, corners=self.bbox)

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = Box3dList(size=self.size, corners=self.bbox)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                if field in self.triplet_extra_fields:
                    bbox.add_field(field, self.get_field(field), is_triplet=True)
                else:
                    bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def get_label_with_negative_val(self, index):
        pos_mask = index >= 0
        neg_mask = index < 0
        pos_idx = pos_mask.nonzero()
        neg_idx = neg_mask.nonzero()
        label_for_this_scan = []
        for i, idx in enumerate(index):
            if i in neg_idx:
                label_for_this_scan.append(-1)
                continue
            else:
                assert(i in pos_idx)
                fg_label_this_prp = self.extra_fields["labels"][idx]
                label_for_this_scan.append(fg_label_this_prp)
        return torch.as_tensor(label_for_this_scan).cuda()

    def with_BG(self):
        return self.withBG

    def extend_BG(self):
        if self.withBG:
            print("already contains background")
            return
        else:
            if 'labels' in self.extra_fields.keys():
                if isinstance(self.extra_fields['labels'], torch.Tensor):
                    self.extra_fields['labels'] = self.extra_fields['labels']+ 1
                else:
                    print("unsupported datatype")
            if 'pred_cls' in self.extra_fields.keys():
                if isinstance(self.extra_fields['pred_cls'], torch.Tensor):
                    self.extra_fields['pred_cls'] = self.extra_fields['pred_cls'] + 1
                else:
                    print("unsupported datatype")
            self.withBG = True

    def remove_BG(self):
        if not self.withBG:
            print("already don't contain background")
            return
        else:
            if 'labels' in self.extra_fields.keys():
                if isinstance(self.extra_fields['labels'], torch.Tensor):
                    self.extra_fields['labels'] = self.extra_fields['labels']- 1
                else:
                    print("unsupported datatype")
            if 'pred_cls' in self.extra_fields.keys():
                if isinstance(self.extra_fields['pred_cls'], torch.Tensor):
                    self.extra_fields['pred_cls'] = self.extra_fields['pred_cls'] - 1
                else:
                    print("unsupported datatype")
            self.withBG = False





