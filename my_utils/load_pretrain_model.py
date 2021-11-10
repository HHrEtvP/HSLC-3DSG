import os
import numpy as np
import torch


def load_pretrain_model(ckpt_dir, multi_gpu=True):
    """
    loads VNet's checkpoint
    """
    model_dict = {}
    ckpt = torch.load(ckpt_dir)
    votenet_dict = ckpt["model_state_dict"]
    for k, v in list(votenet_dict.items()):
        k_list = k.split(".")
        k_list.insert(0, "detector")
        sep = "."
        new_k = sep.join(k_list)
        votenet_dict[new_k] = votenet_dict.pop(k)
    model_dict.update(votenet_dict)
    if multi_gpu:
        for k, v in list(model_dict.items()):
            model_dict["module."+k] = model_dict.pop(k)
    return model_dict, ckpt


def load_ckpt_from_single_gpu_to_multi_gpu(model_dict):
    for k, v in list(model_dict.items()):
        model_dict["module."+k] = model_dict.pop(k)
    return model_dict

if __name__ == "__main__":
    load_pretrain_model("")


