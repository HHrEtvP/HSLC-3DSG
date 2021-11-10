"""
编写collator函数
pytorch中的collator函数用于再处理dataset的__getitem__返回的数据
__getitem__通过sampler采样后会返回一个长度为batch_Size的列表，列表中包含了整个batch上所有的数据，也即__getitem__返回的数据
但是这么做不利于并行处理，假设__getitem__返回两个对象:img和label，前者是输入，后者是对应的标签，那么一次就只能处理1个样本数据，效率很低
collator函数的作用是将__getitem__返回的内容"转置"
那么就可以一次性输入整个batch上的img和label
跟NLP中的处理也是如出一辙，与其挨个处理每个句子的单词，不如一次性读入整个句子的单词
================================================================================================================================
输入的是一个batch上的数据
这个batch是用元组组织的，元组的每个元素分别是输入的数据
现在需要返回的是:
一个batch上的rel_dict_box，rel_dict_rel和idxs
在这里元组的内容如果是:(list(ret_dict_box), list(ret_dict_rel), list(idxs))
那么就不用改
但是如果是:list(ret_dict_box, ret_dict_rel, idxs)就需要改
这个我想还是实验的时候再看结果吧，出BUG了再改
"""

from model.structure.pc_list import PcList
from model.structure.box3d_list import Box3dList
import torch
import numpy as np

def my_collate_fn(batch):
    transposed_batch = list(zip(*batch))  # zip(*)是解压
    ret_dict_box = transposed_batch[0]  # (batch_size(box3d_list))
    ret_dict_rel = transposed_batch[1]  # (batch_size, (rel_mat, obj_idx))
    idxs = transposed_batch[2]  # (batch_size)
    batched_pc = torch.zeros([8, 40000, 3])  # 注意这里暂时改成了4
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
