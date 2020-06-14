import torch.nn as nn
import torch.nn.functional as F
from utils_ct import *
from settings import settings

class MeanAggregator(nn.Module):
    def __init__(self, h_dim, dropout, seq_len, gcn=False):
        super(MeanAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.gcn = gcn
        if gcn:
            self.gcn_layer = nn.Linear(h_dim, h_dim)

    def forward(self, s_hist, s, r, o, ent_embeds, rel_embeds, s_hist_dt):

        s_idx, s_len_non_zero, s_tem, r_tem, embeds_stack, len_s, embeds_split, s_hist_dt_sorted_truncated = \
            get_sorted_s_r_embed(s_hist, s, r, ent_embeds, s_hist_dt)

        # To get mean vector at each time
        curr = 0
        rows = []
        cols = []
        for i, leng in enumerate(len_s):  # lens stores the number of neighbors of each timestamp for all subjects
            rows.extend([i] * leng)
            cols.extend(list(range(curr, curr + leng)))
            curr += leng
        rows = to_device(torch.LongTensor(rows))
        cols = to_device(torch.LongTensor(cols))
        idxes = torch.stack([rows, cols], dim=0)

        mask_tensor = to_device(torch.sparse.FloatTensor(idxes, torch.ones(len(rows), device=idxes.device)))
        #mask_tensor = to_device(mask_tensor)
        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)
        embeds_mean = embeds_sum / to_device(torch.Tensor(len_s)).view(-1, 1)

        if self.gcn:
            embeds_mean = self.gcn_layer(embeds_mean)
            embeds_mean = F.relu(embeds_mean)

        # split embds_mean to each subjects with non_zero history
        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())

        # cat aggregation, subject embedding, relation embedding together.
        s_embed_seq_tensor = to_device(
            torch.zeros(len(s_len_non_zero), self.seq_len, self.h_dim + 2 * settings['embd_rank']))
        for i, embeds in enumerate(embeds_split):
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                 rel_embeds[r_tem[i]].repeat(len(embeds), 1)), dim=1)
        s_hist_dt_seq_tensor = to_device(torch.zeros(len(s_len_non_zero), self.seq_len))

        for i, dts in enumerate(s_hist_dt_sorted_truncated):
            s_hist_dt_seq_tensor[i, torch.arange(len(dts))] = to_device(
                torch.tensor(dts, dtype=s_hist_dt_seq_tensor.dtype))

        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor, s_len_non_zero, batch_first=True)
        s_packed_dt = torch.nn.utils.rnn.pack_padded_sequence(s_hist_dt_seq_tensor, s_len_non_zero, batch_first=True)

        return s_packed_input, s_packed_dt, s_idx, len(s_len_non_zero)


