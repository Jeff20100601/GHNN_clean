import os
from settings import settings
import numpy as np
import torch
import argparse

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]),  int(line_split[2])

def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(int(line_split[3])/settings['time_scale'])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)

def make_batch(a, b, c, d, e, f, g, h, j, k, l, m, n, valid1 = None, valid2 = None):
    if valid1 is None and valid2 is None:
        # For item i in a range that is a length of l
        for i in range(0, len(a), n):
                yield [a[i:i + n], b[i:i + n], c[i:i + n], d[i:i + n], e[i:i + n],
                      f[i:i + n], g[i:i + n], h[i:i + n], j[i:i + n], k[i:i + n], l[i:i + n], m[i:i + n]]
    else:
        # For item i in a range that is a length of l
        for i in range(0, len(a), n):
            yield [a[i:i + n], b[i:i + n], c[i:i + n], d[i:i + n], e[i:i + n],
                   f[i:i + n], g[i:i + n], h[i:i + n], j[i:i + n], k[i:i + n], l[i:i + n], m[i:i + n],
                   valid1[i:i + n], valid2[i:i + n]]

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def isListEmpty(inList):
    if isinstance(inList, list):
        return all( map(isListEmpty, inList) )
    return False

def get_sorted_s_r_embed(s_hist, s, r, ent_embeds, s_hist_dt):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    s_hist_dt_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])
        s_hist_dt_sorted.append(s_hist_dt[idx.item()])

    flat_s = []
    len_s = []
    for hist in s_hist_sorted:
        for neighs in hist:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh)
    s_tem = s[s_idx]
    r_tem = r[s_idx]
    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split, s_hist_dt_sorted

def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")
