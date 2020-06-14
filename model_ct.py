import torch.nn as nn
from Aggregator_ct import MeanAggregator
from utils_ct import *
from settings import settings
import torch

class HawkesLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_size, args):
        super(HawkesLSTMCell, self).__init__()
        self.input_g = nn.Linear(input_dim + hidden_size, hidden_size)
        self.forget_g = nn.Linear(input_dim + hidden_size, hidden_size)
        self.output_g = nn.Linear(input_dim + hidden_size, hidden_size)
        self.input_target = nn.Linear(input_dim + hidden_size, hidden_size)
        self.forget_target = nn.Linear(input_dim + hidden_size, hidden_size)
        self.z_gate = nn.Linear(input_dim + hidden_size, hidden_size)
        self.decay_layer = nn.Sequential(
            nn.Linear(input_dim + hidden_size, hidden_size),
            nn.Softplus(beta=args.softrelu_scale))

    def forward(self,x, h_t, c_t, c_target):
        """
        Compute the updated LSTM paramters.

        Args:s
            x: (ent_emb, rel_emb, aggregated_emb)
            h_t: cont. hidden state at timestamp t
            c_t: cont. cell state at timestamp t
            c_target: target cell state

        Returns:
            h_i: just-updated hidden state
            h_t: hidden state just before next event
            cell_i: just-updated cell state
            c_t: cell state decayed to before next event
            c_target_i: cell state target before the next event
            output: LSTM output
            decay_i: rate of decay for the cell state
        """
        v = torch.cat((x, h_t), dim=1)
        inpt = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        input_target = torch.sigmoid(self.input_target(v))
        forget_target = torch.sigmoid(self.forget_target(v))
        output = torch.sigmoid(self.output_g(v))  # compute output gate
        # Not-quite-c
        z_i = torch.tanh(self.z_gate(v))
        # Compute the decay parameter
        decay = self.decay_layer(v)
        # Update the cell state to c(t_i+)
        c_i = forget * c_t + inpt * z_i
        # Update the cell state target
        c_target = forget_target * c_target + input_target * z_i

        return c_i, c_target, output, decay

class GHNN(nn.Module):
    def __init__(self, num_e, num_rels, num_t, args, dropout=0):
        super(GHNN, self).__init__()
        self.num_e = num_e
        self.num_t = num_t
        self.h_dim = args.n_hidden
        self.num_rels = num_rels
        self.args = args

        self.rel_embeds = nn.Parameter(torch.zeros(2*num_rels, args.embd_rank))
        nn.init.xavier_uniform_(self.rel_embeds)
        self.ent_embeds = nn.Parameter(torch.zeros(self.num_e, args.embd_rank))
        nn.init.xavier_uniform_(self.ent_embeds)

        self.dropout = nn.Dropout(dropout)
        self.sub_encoder = HawkesLSTMCell(2 * args.embd_rank + self.h_dim, self.h_dim, args)
        self.obj_encoder = HawkesLSTMCell(2 * args.embd_rank + self.h_dim, self.h_dim, args)
        self.aggregator_s = MeanAggregator(self.h_dim, dropout, args.max_hist_len, gcn=False)
        self.aggregator_o = self.aggregator_s
        self.linear_h = nn.Linear(args.n_hidden, args.embd_rank, bias=False)
        self.linear_inten_layer = nn.Linear(self.h_dim + 2*args.embd_rank, args.embd_rank, bias= False)
        self.Softplus = nn.Softplus(beta= args.softrelu_scale)

        '''
        layers for time prediction
        '''
        self.criterion_time = nn.CrossEntropyLoss()
        self.criterion_link = nn.CrossEntropyLoss()

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state and the cell state.
        The initial cell state target is equal to the initial cell state.
        The first dimension is the batch size.

        Returns:
            (hidden, cell_state)
        """
        (h0, c0, c_target0) = (to_device(torch.zeros(batch_size, self.h_dim)),
                               to_device(torch.zeros(batch_size, self.h_dim)),
                               to_device(torch.zeros(batch_size, self.h_dim)))
        return h0, c0, c_target0

    def forward(self, input, mode_tp, mode_lk):
        # extract input
        if mode_lk == 'Training':
            quadruples, s_history_event_tp, s_history_event_lk, o_history_event_tp, o_history_event_lk, \
            s_history_dt_tp, s_history_dt_lk, o_history_dt_tp, o_history_dt_lk, dur_last_tp, sub_synchro_dt_tp, obj_synchro_dt_tp = input
        elif mode_lk in ['Valid', 'Test']:
            quadruples, s_history_event_tp, s_history_event_lk, o_history_event_tp, o_history_event_lk, \
            s_history_dt_tp, s_history_dt_lk, o_history_dt_tp, o_history_dt_lk, dur_last_tp, sub_synchro_dt_tp, obj_synchro_dt_tp,\
            val_subcentric_fils_lk, val_objcentric_fils_lk= input
        else:
            raise ValueError('Not implemented')

        #prepare model input
        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        #some batches don't include recurrent events.
        if isListEmpty(s_history_event_tp) or isListEmpty(o_history_event_tp):
            error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error = [None]*9
        else:
            # Aggregating concurrent events
            s_packed_input_tp, s_packed_dt_tp, s_idx_tp, s_nonzero_tp = \
                    self.aggregator_s(s_history_event_tp, s, r, o, self.ent_embeds,
                                self.rel_embeds[:self.num_rels], s_history_dt_tp)

            o_packed_input_tp, o_packed_dt_tp, o_idx_tp, o_nonzero_tp = \
                self.aggregator_o(o_history_event_tp, o, r, s, self.ent_embeds,
                                  self.rel_embeds[self.num_rels:], o_history_dt_tp)
            # compute hidden state
            sub_cell_tp, sub_cell_target_tp, sub_gate_output_tp, sub_decay_tp, _ = self.compute_sequence_states(
                s_packed_input_tp, s_packed_dt_tp, self.sub_encoder, s_idx_tp)

            obj_cell_tp, obj_cell_target_tp, obj_gate_output_tp, obj_decay_tp, _ = self.compute_sequence_states(
                o_packed_input_tp, o_packed_dt_tp, self.obj_encoder, o_idx_tp)

            #extract recurrent events
            dur_last_tp = to_device(torch.tensor(dur_last_tp))
            dur_non_zero_idx_tp = (dur_last_tp > 0).nonzero().squeeze()
            dur_last_nonzero_tp = dur_last_tp[dur_non_zero_idx_tp]

            #add synchro_dt_tp to synchronize the concatenated intensity from subject centric and object centeric
            sub_synchro_dt_tp = to_device(torch.tensor(sub_synchro_dt_tp, dtype=torch.float))
            sub_synchro_non_zero_idx_tp = (sub_synchro_dt_tp >= 0).nonzero().squeeze()
            sub_synchro_dt_nonzero_tp = sub_synchro_dt_tp[sub_synchro_non_zero_idx_tp]
            assert(torch.all(torch.eq(sub_synchro_non_zero_idx_tp, dur_non_zero_idx_tp)))
            obj_synchro_dt_tp = to_device(torch.tensor(obj_synchro_dt_tp, dtype=torch.float))
            obj_synchro_non_zero_idx_tp = (obj_synchro_dt_tp >= 0).nonzero().squeeze()
            obj_synchro_dt_nonzero_tp = obj_synchro_dt_tp[obj_synchro_non_zero_idx_tp]
            assert(torch.all(torch.eq(obj_synchro_non_zero_idx_tp, dur_non_zero_idx_tp)))

            if mode_tp == 'MSE':
                dur_last_nonzero_tp = dur_last_nonzero_tp.type(torch.float)
                sub_inten_tp = self.compute_inten_tpre(sub_cell_tp, sub_cell_target_tp, sub_decay_tp,
                                                             sub_gate_output_tp,
                                                             s, o, r, dur_non_zero_idx_tp,
                                                             self.rel_embeds[:self.num_rels], sub_synchro_dt_nonzero_tp)
                obj_inten_tp = self.compute_inten_tpre(obj_cell_tp, obj_cell_target_tp,
                                                                           obj_decay_tp,
                                                                           obj_gate_output_tp, o, s,
                                                                           r, dur_non_zero_idx_tp,
                                                                           self.rel_embeds[self.num_rels:],
                                                                           obj_synchro_dt_nonzero_tp)
                dt_tp, error_tp, density_tp, mae_tp, den1_tp, den2_tp, tpred, abs_error = self.predict_t(sub_inten_tp,
                                                                                                      obj_inten_tp,
                                                                                                      dur_last_nonzero_tp)

            else:
                raise ValueError('Not implemented')

        # some batches don't include (s,p) or (o,p) history events.
        if isListEmpty(s_history_event_lk) or isListEmpty(o_history_event_lk):
            sub_rank, obj_rank, cro_entr_lk = [None] * 3
            if mode_lk == 'Training':
                return cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, obj_rank, cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error
            else:
                raise ValueError('Not implemented')
        else:
            # Aggregating concurrent events
            s_packed_input_lk, s_packed_dt_lk, s_idx_lk, s_nonzero_lk = \
                    self.aggregator_s(s_history_event_lk, s, r, o,
                                self.ent_embeds, self.rel_embeds[:self.num_rels], s_history_dt_lk)

            o_packed_input_lk, o_packed_dt_lk, o_idx_lk, o_nonzero_lk = \
                self.aggregator_o(o_history_event_lk, o, r, s, self.ent_embeds,
                                  self.rel_embeds[self.num_rels:], o_history_dt_lk)

            # compute hidden state
            _, _, _, _, sub_hidden_lk = self.compute_sequence_states(
                s_packed_input_lk, s_packed_dt_lk, self.sub_encoder, s_idx_lk)

            _, _, _, _, obj_hidden_lk = self.compute_sequence_states(
                o_packed_input_lk, o_packed_dt_lk, self.obj_encoder, o_idx_lk)

            #compute intensity
            if mode_lk == 'Training':
                sub_cro_entr_loss = self.predict_link(sub_hidden_lk, s, o, r, self.rel_embeds[:self.num_rels], mode_lk)
                obj_cro_entr_loss = self.predict_link(obj_hidden_lk, o, s, r, self.rel_embeds[self.num_rels:], mode_lk)
                cro_entr_lk = (sub_cro_entr_loss + obj_cro_entr_loss) / 2
                return cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error
            elif mode_lk in ['Valid', 'Test']:
                sub_cro_entr_loss, sub_rank = self.predict_link(sub_hidden_lk, s, o, r, self.rel_embeds[:self.num_rels], mode_lk,
                                                        val_fils =  val_subcentric_fils_lk)
                obj_cro_entr_loss, obj_rank = self.predict_link(obj_hidden_lk, o, s, r, self.rel_embeds[self.num_rels:], mode_lk,
                                                       val_fils = val_objcentric_fils_lk)
                cro_entr_lk = (sub_cro_entr_loss + obj_cro_entr_loss) / 2
                return sub_rank, obj_rank, cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error
            else:
                raise ValueError('Not implemented')

    def predict_link(self, hiddens_ti, actor1, actor2, r, rel_embeds, mode_lk, val_fils = None):
            # for subject centric: actor1 is subejct and actor2 is object.
            inten_raw = self.linear_inten_layer(
                self.dropout(torch.cat((self.ent_embeds[actor1], self.linear_h(hiddens_ti), rel_embeds[r]), dim=1)))
            intens = self.Softplus(inten_raw.mm(self.ent_embeds.transpose(0, 1)))  # shape of pred_intens: num_batch*num_e
            cro_entr_loss = self.criterion_link(intens, actor2)
            ranks = []
            if mode_lk == 'Training':
                return cro_entr_loss
            elif mode_lk in ['Valid', 'Test']:
                ground = intens.gather(1, actor2.view(-1, 1))  # clone the score of the ground truth object, shape: [n_batch, 1]
                assert(len(val_fils) == intens.shape[0])
                for i in range(len(val_fils)):
                    if self.args.filtering:
                        intens[i, :][val_fils[i]] = 0
                    intens[i, actor2[i]] = ground[i]
                    pred_comp1 = (intens[i,:] > ground[i]).sum().item() + 1 # obejcts whose score larger than ground truth
                    pred_comp2 = ((intens[i,:] == ground[i]).sum().item() - 1)/2
                    ranks.append(pred_comp1 + pred_comp2)
                return cro_entr_loss, ranks
            else:
                raise ValueError('Not implemented')

    def compute_sequence_states(self, packed_input, packed_dt, encoder, sort_idx):
        #### Computes the LSTM network parameters for the next interval :math:`(t_i,t_{i+1}]` from the input at time :math:`t_{i}`.
        """
        Args:
          packed_dt (PackedSequence): time until next event
              Shape: seq_len * batch
          packed_input (PackedSequence): concatenated input sequence
              Shape: seq_len * batch * input_size
          h0: initial hidden state
          c0: initial cell state
          c_target: initial target cell state
        """
        truncated_size = packed_input.batch_sizes[0]
        iteration_size = len(sort_idx)
        h_0, c_0, c_target_0 = self.init_hidden(truncated_size)
        max_seq_length = len(packed_input.batch_sizes)
        beg_index = 0
        h_t, c_t, c_target = h_0, c_0, c_target_0
        cells = []
        cell_targets = []
        decays = []
        gate_outputs = []
        hiddens = [] #only for link prediction

        for i in range(max_seq_length):
            batch_size = packed_input.batch_sizes[i].item()  # the batch size for current step in the packed sequences.
            batch_size_next = packed_input.batch_sizes[i + 1].item() if i + 1 < max_seq_length else 0
            h_t = h_t[:batch_size]
            c_t = c_t[:batch_size]
            c_target = c_target[:batch_size]

            dt = packed_dt.data[beg_index:(beg_index + batch_size)]
            batch_input = packed_input.data[beg_index:(beg_index + batch_size)]

            # Update the hidden states and LSTM parameters following the equations
            cell_i, c_target, output, decay_i = encoder(batch_input, h_t, c_t, c_target)
            c_t = c_target + (cell_i - c_target) * torch.exp(-decay_i * dt[:, None])
            # compute the c(t) at t -> t_{i+1}, dt[:, None] for broadcasting.
            h_t = output * torch.tanh(c_t)  # decayed hidden state just before next event

            beg_index += batch_size  # move the starting index for the data in the PackedSequence
            if batch_size_next != batch_size:#if processing number of next batch is not equal to current batch, which means some sequences are finished, we store the results of those sequences.
                cells.insert(0, cell_i[batch_size_next: batch_size])
                cell_targets.insert(0, c_target[batch_size_next:batch_size])
                gate_outputs.insert(0, output[batch_size_next:batch_size])
                decays.insert(0, decay_i[batch_size_next:batch_size])
                hiddens.insert(0, h_t[batch_size_next:batch_size])
        cell_sorted_truncated = torch.cat(cells, dim=0)
        cell_target_sorted_truncated = torch.cat(cell_targets, dim=0)
        output_sorted_truncated = torch.cat(gate_outputs, dim=0)
        decay_sorted_truncated = torch.cat(decays, dim=0)
        hidden_t_sorted_truncated = torch.cat(hiddens, dim=0)

        # pad the states of events without history with zeros and resorted it.
        _, ori_idx = sort_idx.sort()
        cell_full = torch.cat((cell_sorted_truncated, to_device(torch.zeros(iteration_size - truncated_size,
                        self.h_dim))), dim=0)[ori_idx]
        cell_target_full = torch.cat((cell_target_sorted_truncated, to_device(torch.zeros(iteration_size - truncated_size,
                        self.h_dim))), dim=0)[ori_idx]
        output_full = torch.cat((output_sorted_truncated, to_device(torch.zeros(iteration_size - truncated_size,
                    self.h_dim))), dim=0)[ori_idx]
        decay_full = torch.cat((decay_sorted_truncated, to_device(torch.zeros(iteration_size - truncated_size,
                  self.h_dim))), dim=0)[ori_idx]
        hidden_full = torch.cat((hidden_t_sorted_truncated, to_device(torch.zeros(iteration_size - truncated_size,
                  self.h_dim))), dim=0)[ori_idx]

        return cell_full, cell_target_full, output_full, decay_full, hidden_full

    def compute_inten_tpre(self, cell, cell_target, decay, gate_output, actors, another_actors, r, non_zero_idx,
                           rel_embeds, synchro_dt_nonzero_tp):
        '''
        The last history timestamp of subject centric event sequence and object centric event sequence might be same,
        but not necessary. Therefore, to let subject centric intensity and object centric intensity have the same start
        point for decaying, we synchronize these two intensity functions here using extra_time_to_start_timestamp.

        actors: the centric actor
        another_actor:  the second actor
        '''
        hmax = settings['time_horizon']
        timestep  = settings['CI']
        n_samples = int(hmax / timestep) + 1 #add 1 to accomodate zero
        dt = to_device(torch.linspace(0, hmax, n_samples).repeat(non_zero_idx.shape[0], 1).transpose(0, 1)) + synchro_dt_nonzero_tp[None, :] #shape: [n_sample*n_batch], decay shape: n_batch*n_hidden, target shape: n_sample*n_batch*n_hidden
        cell_t = cell_target[non_zero_idx] + (cell[non_zero_idx] - cell_target[non_zero_idx]) * \
                 torch.exp(-decay[non_zero_idx][None, :, :] * dt[:, :, None]) #shape: n_sample*n_batch*n_hidden
        inten_raw = self.linear_inten_layer(self.dropout(torch.cat((self.ent_embeds[actors[non_zero_idx]].repeat(n_samples, 1, 1),
            self.linear_h(gate_output[non_zero_idx] * torch.tanh(cell_t)), rel_embeds[r[non_zero_idx]].repeat(n_samples,1,1)), dim = 2))) #shape: n_sample*batch_size*self.args.embd_rank
        o = self.ent_embeds[another_actors[non_zero_idx]].repeat(n_samples, 1, 1) #shape: n_sample*batch_size*self.args.embd_rank
        intens = self.Softplus((inten_raw * o).sum(dim=2))  #pointwise multiplication and then sum over each embd_rank is equivalent to dot product between two vector. shape: n_sample*n_batch
        return intens

    def predict_t(self, sub_inten_t, obj_inten_t, gt_t):
        timestep = settings['CI']
        hmax = settings['time_horizon']
        n_samples = int(hmax / timestep) + 1  # add 1 to accomodate zero
        dt = to_device(torch.linspace(0, hmax, n_samples).repeat(gt_t.shape[0], 1).transpose(0, 1))
        intens = (sub_inten_t + obj_inten_t) / 2
        integral_ = torch.cumsum(timestep * intens, dim=0)
        density = (intens * torch.exp(-integral_))  # shape: n_samples*n_batch

        t_pit = dt * density
        # trapeze method
        estimate_dt = (timestep * 0.5 * (t_pit[1:] + t_pit[:-1])).sum(dim=0)  # shape: n_batch
        mse = nn.MSELoss()
        error_dt = mse(estimate_dt, gt_t)
        with torch.no_grad():
            abs_error = (estimate_dt - gt_t).abs()
            mae = abs_error.mean()
        return dt, error_dt, density, mae, intens, torch.exp(-integral_), estimate_dt.detach(), abs_error

