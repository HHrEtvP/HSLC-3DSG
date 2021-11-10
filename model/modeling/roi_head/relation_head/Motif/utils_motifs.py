import array
import os
import zipfile
import itertools
import six
import torch
import numpy as np
from six.moves.urllib.request import urlretrieve
from model.structure.box3d_list import Box3dList
from tqdm import tqdm
import sys
from my_utils.misc import cat, line2space
from model.modeling.roi_head.relation_head.utils_relation import nms_overlaps


def normalize_sigmoid_logits(orig_logits):
    orig_logits = torch.sigmoid(orig_logits)
    orig_logits = orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)
    return orig_logits


def generate_attributes_target(attributes, device, max_num_attri, num_attri_cat):
    """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
    assert max_num_attri == attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    attribute_targets = torch.zeros((num_obj, num_attri_cat), device=device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_num_attri):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1
    return attribute_targets, with_attri_idx


def transpose_packed_sequence_inds(lengths):
    """
    Get a TxB indices from sorted lengths. 
    Fetch new_inds, split by new_lens, padding to max(new_lens), and stack.
    original:
    5,5,5,5,5
    4,4,4,4
    3,3,3
    2,2
    1
    pad:
    5,5,5,5,5
    4,4,4,4,0
    3,3,3,0,0
    2,2,0,0,0
    1,0,0,0,0
    pack:
    5,4,3,2,1,5,4,3,2,5,4,3,5,4,5
    Returns:
        new_inds (np.array) [sum(lengths), ]
    """
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):  # 0,1,2,3,4
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer + 1)].copy())  # cum_add[:5]，[0,5,9,12,14]
        # cum_add[:4],[1,6,10,13](4)
        # cum_add[:3],[2,7,11](3)
        # cum_add[:2],[3,8](2)
        # cum_add[:1],[4](1)
        cum_add[:(length_pointer + 1)] += 1  # [1,6,10,13,15,15]
        # [2,7,11,14,15,15]
        # [3,8,12,14,15,15]
        # [4,9,12,14,15,15]
        # [5,9,12,14,15,15]
        new_lens.append(length_pointer + 1)  # append:5
        # append:4
        # append:3
        # append:2
        # append:1
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def sort_by_score(proposals, scores):

    num_rois = [len(b) for b in proposals]
    num_im = len(num_rois)
    scores = torch.as_tensor(scores).cpu()
    scores = scores.split(num_rois, dim=0)
    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores, num_rois)):
        ordered_scores.append(score + 2.0 * float(num_roi * 2 * num_im + i))
    ordered_scores = cat(ordered_scores, dim=0)
    _, perm = torch.sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)  # [5,4,3,2,1]
    inds, ls_transposed = transpose_packed_sequence_inds(num_rois)
    inds = inds.astype(float)
    inds = torch.LongTensor(inds).to(scores[0].device)
    ls_transposed = torch.LongTensor(ls_transposed)

    # packed = torch.nn.utils.rnn.PackedSequence(inds, ls_transposed)
    # pre_pack = torch.nn.utils.rnn.pad_packed_sequence(packed)

    perm = perm[inds]  # (batch_num_box, )

    _, inv_perm = torch.sort(perm)
    return perm, inv_perm, ls_transposed


def to_onehot(vec, num_classes, fill=0):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec.long() + num_classes * arange_inds] = 1
    return onehot_result


def get_dropout_mask(dropout_probability, tensor_shape, device):
    """
    once get, it is fixed all the time
    """
    binary_mask = (torch.rand(tensor_shape) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().to(device).div(1.0 - dropout_probability)
    return dropout_mask


def center_x(proposals):
    assert proposals[0].mode == 'xyxy'
    boxes = cat([p.bbox for p in proposals], dim=0)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)


def volume(proposals):
    v = []
    for p in proposals:
        v.append(p.vol())
    v = torch.cat(v, dim=0)
    return v


def encode_box_info(proposals):
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                          w * h / (wid * hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)


def pseudo_encode_box_info(proposals):
    boxes_info = []
    for proposal in proposals:
        for i in range(proposal.bbox.shape[0]):
                boxes_info.append(proposal.bbox[i])
    return torch.cat(boxes_info, dim=0).view((-1, 24))


def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    for i, token in enumerate(names):
        token = line2space(token)
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    return vectors


def load_word_vectors(root, wv_type, dim):
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')
    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


if __name__ == "__main__":
    new_inds, new_len = transpose_packed_sequence_inds([5, 4, 3, 2, 1])
    print(new_inds)
    print(new_len)
