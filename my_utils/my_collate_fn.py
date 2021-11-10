from model.structure.pc_list import PcList
from model.structure.box3d_list import Box3dList
import torch
import numpy as np

def my_collate_fn(batch):
    transposed_batch = list(zip(*batch))
    ret_dict_box = transposed_batch[0]
    ret_dict_rel = transposed_batch[1]
    idxs = transposed_batch[2]
    batched_pc = torch.zeros([8, 40000, 3])
    batched_gt = {}
    batched_mat = []
    batched_obj_id = []
    batched_mat_idx = []
    for i, d in enumerate(batch):
        for k, v in d[0].items():
            if k not in batched_gt.keys():
                if isinstance(v, torch.Tensor):
                    batched_gt[k] = v.unsqueeze(0)
                else:
                    batched_gt[k] = [v]
                continue
            else:
                if isinstance(v, torch.Tensor):
                    batched_gt[k] = torch.cat([batched_gt[k], d[0][k].unsqueeze(0)], dim=0)
                else:
                    batched_gt[k].append(v)
        batched_pc[i, :, :] = torch.as_tensor(d[0]["point_clouds"]).float()
        batched_mat.append(d[1].unsqueeze(0))
        batched_obj_id.append(d[2].unsqueeze(0))
        batched_mat_idx.append(d[3].unsqueeze(0))
    for k, v in batched_gt.items():
        if k == 'scan_name':
            continue
        batched_gt.update({k: torch.as_tensor(np.array(v))})
    batched_mat = torch.cat(batched_mat, dim=0)
    batched_obj_id = torch.cat(batched_obj_id, dim=0)
    batched_mat_idx = torch.cat(batched_mat_idx, dim=0)
    return batched_gt, batched_mat, batched_obj_id, batched_mat_idx
