import torch


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.
    为每个prediction分配一个GT标签(或者根本分配不到)，GT可能会被同时分配到多个prediction

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.
    基于M*N的match_quality_matrix矩阵(M个GT,N个prediction)，该矩阵元素描述了i-j之间的匹配程度

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    返回一个数组长度N，元素是第n个prediction对应的GT m，如果没有对应的GT，则用-1填充
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): 超过这个阈值的可以被认为是匹配项
            low_threshold (float): 下限
            allow_low_quality_matches (bool): 如果为真，那么额外生成一些匹配，这些匹配的质量可能会比较低(因为对应矩阵元素的值比较低)
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]):(M*N)tensor，代表了M个GT和N个prediction之间的匹配程度
        等

        Returns:
            matches (Tensor[int64]): (N)tensor，N个prediction与其对应的GT
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        matched_vals, matches = match_quality_matrix.max(dim=0)  # 找出每个prediction对应最大的gt
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # 标记那些没有超过high_threshold的匹配
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS
        # 为那些匹配生成指派(质量较低)
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        生成额外指派，对每个GT找到对应最大的prediction，如果那个prediction没有指派，就将该GT指派给该prediction
        """
        _, max_idx = match_quality_matrix.max(dim=1)  # max_idx是每个GT对应的最大的prediction
        # Find highest quality match available, even if it is low, including ties
        # 无视匹配程度，只找最高的
        for idx, m in enumerate(max_idx):  # idx是GT的idx，m是该idxGT所对应的最大的prediction
            if matches[m] <= 0:  # 未指派
                matches[m] = idx
        return matches
