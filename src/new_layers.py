import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import time


def orthonormal_initializer(output_size, input_size):
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine


class KG_score(nn.Module):
    def __init__(self, hidden_feats):
        super(KG_score, self).__init__()
        self.hidden_feats = hidden_feats
        self.feature_out = nn.Linear(hidden_feats, 1)

    def forward(self, input_head, input_dep):
        batch_size, len, dim = input_head.size()
        input_head = input_head.view(batch_size, len, 1, dim)
        input_dep = input_dep.view(batch_size, 1, len, dim)
        input_dep = input_dep.expand(-1, len, -1, -1)
        input_head = input_head.expand(-1, -1, len, -1)
        combined_score = input_head + input_dep
        score_table = self.feature_out(torch.tanh(combined_score))
        return score_table


class NEG_loss(nn.Module):
    def __init__(self, num_sample, out_embedding):

        super(NEG_loss, self).__init__()

        self.out_embedding = out_embedding

        self.num_classes = self.out_embedding.num_embeddings

        self.embed_dim = self.out_embedding.embedding_dim

        self.num_sample = num_sample


    def forward(self, input_latent, target_labels, batch_samples,gpu):

        batch_size, sent_len, _ = input_latent.shape

        output = self.out_embedding(target_labels)

        latent_for_output = input_latent.view(batch_size * sent_len, 1, self.embed_dim)
        output_for_latent = output.permute(0, 2, 1)
        output_for_latent = output_for_latent.view(batch_size, 1, self.embed_dim, sent_len)
        output_for_latent = output_for_latent.expand(batch_size, sent_len, self.embed_dim, sent_len)
        output_for_latent = output_for_latent.contiguous().view(batch_size*sent_len,self.embed_dim,sent_len)
        output_rep = torch.bmm(latent_for_output, output_for_latent)
        output_rep = output_rep.view(batch_size, sent_len, sent_len, 1)

        batch_sampled = self.out_embedding(batch_samples)
        batch_sampled = batch_sampled.view(batch_size, sent_len, self.num_sample - 1, self.embed_dim)
        output_for_sample = output.view(batch_size, sent_len, 1, self.embed_dim)
        batch_sampled = torch.cat((batch_sampled, output_for_sample), dim=2)

        batch_sampled_for_latent = batch_sampled.permute(0, 3, 1, 2)
        batch_sampled_for_latent = batch_sampled_for_latent.view(batch_size, 1, self.embed_dim, sent_len * self.num_sample)
        batch_sampled_for_latent = batch_sampled_for_latent.expand(batch_size, sent_len, self.embed_dim, sent_len * self.num_sample)
        batch_sampled_for_latent = batch_sampled_for_latent.contiguous().view(batch_size * sent_len, self.embed_dim, sent_len * self.num_sample)
        sample_rep = torch.bmm(latent_for_output, batch_sampled_for_latent)
        sample_rep = sample_rep.view(batch_size, sent_len, sent_len, self.num_sample)
        max_sample = torch.max(sample_rep, dim=3, keepdim=True)[0]

        ex_output_rep = torch.exp(output_rep - max_sample)
        ex_sample_rep = torch.exp(sample_rep - max_sample)
        # ex_output_rep = torch.exp(output_rep)
        # ex_sample_rep = torch.exp(sample_rep)
        ex_sum = torch.sum(ex_sample_rep, dim=3, keepdim=True)
        recons_prob = ex_output_rep / ex_sum
        #print recons_prob

        recons_likelihood = torch.log(recons_prob)
        recons_likelihood = recons_likelihood.view(batch_size, sent_len, sent_len)
        return recons_likelihood

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


class DmLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(DmLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self._all_weights = []
        for layer in range(num_layers):
            layer_params = (self.fcells[layer].weight_ih, self.fcells[layer].weight_hh, \
                            self.fcells[layer].bias_ih, self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih, self.bcells[layer].weight_hh, \
                                self.bcells[layer].bias_ih, self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            if self.bidirectional:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '_reverse')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '_reverse')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + 2 * self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))
            else:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)

    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = masks.transpose(0, 1)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = DmLSTM._forward_rnn(cell=self.fcells[layer], \
                                                                       input=input, masks=masks, initial=initial, drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = DmLSTM._forward_brnn(cell=self.bcells[layer], \
                                                                               input=input, masks=masks, initial=initial, drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)
