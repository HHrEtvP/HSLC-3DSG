import os
import torch


def getDirList(p):
    p = str(p)
    if p == "":
        return []
    p = p.replace("/", "\\")
    if p[-1] != "\\":
        p = p + "\\"
    a = os.listdir(p)
    b = [x for x in a if os.path.isdir(p + x)]
    return b


def getFileList(p):
    p = str(p)
    if p == "":
        return []
    p = p.replace("/", "\\")
    if p[-1] != "\\":
        p = p + "\\"
    a = os.listdir(p)
    b = [x for x in a if os.path.isfile(p + x)]
    return b


def list2str(list):
    list = [str(i)+' ' for i in list]
    out = ''.join(list)
    return out


def cat(tensors, dim=0):
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_3rscan_statics(cfg):
    statics = {"obj_classes": cfg.DATASETS.RSCAN_OBJ_27_CLASSES,
               "rel_classes": cfg.DATASETS.RSCAN_NEW_REL_CLASSES_KEYS}
    return statics


def line2space(str):
    if isinstance(str, list):
        new_list = []
        for s in str:
            s = line2space(s)
            new_list.append(s)
        return new_list
    elif str.find("_")>0:
        seperator = " "
        str = seperator.join(str.split("_"))
        return str
    else:
        return str


def space2line(str):
    if str.find(" ")>0:
        seperator = "_"
        str = seperator.join(str.split(" "))
        return str
    else:
        return str


if __name__ == "__main__":
    print(line2space("trash_can_lol"))
    print(space2line("trash can lol"))


