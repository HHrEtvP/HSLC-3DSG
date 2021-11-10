"""
原文件使用了CUDA拓展，这里先删去CUDA部分，用常规写法，后期根据性能再做优化
"""
import numpy as np

def _box_nms(boxes, score, nms_thresh):
    """
    stub，保留所有
    :param boxes:
    :param score:
    :param nms_thresh:
    :return:
    """
    keep = np.arange(np.shape(boxes)[0])
    return keep

