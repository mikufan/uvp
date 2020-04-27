import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
# from memory_profiler import profile
from new_layers import *


def get_kl_temp(curr_iteration, kl_anneal_rate, max_temp=1.0):
    temp = np.exp(kl_anneal_rate * curr_iteration) - 1.
    return float(np.minimum(temp, max_temp))


def gaussian(mean, logvar):
    '''
    .normal: sample from (0, 1)
    '''
    return mean + torch.exp(0.5 * logvar) * logvar.data.new(mean.size()).normal_()


def kl_normal2_normal2(mean1, log_var1, mean2, log_var2):
    '''
    args:
        mean_1: seq_len * dim
        log_var1: seq_len * dim
        mean2: seq_en * dim
        log_var2: seq_len * dim
    '''
    # pdb.set_trace()
    return 0.5 * log_var2 - 0.5 * log_var1 + (torch.exp(log_var1) + (mean1 - mean2) ** 2) / (
            2 * torch.exp(log_var2) + 1e-10) - 0.5


def compute_KL_div_g(mean_q, log_var_q):
    '''
    args:
        mean_q: batch_size * seq_len * dim
        log_var_q: batch_size * seq_len * dim
    '''
    kl_divergence = compute_KL_div2(mean_q, log_var_q)
    return kl_divergence


def compute_KL_div_m(struct_distr, prior, masks, distr_mask):
    batch_size, sent_len, _ = struct_distr.shape
    struct_distr = struct_distr * distr_mask
    prior = prior * distr_mask
    kl_zeros = masks.new(batch_size, sent_len, sent_len)
    kl_zeros.fill_(0)
    distr_zeros = torch.eq(kl_zeros, struct_distr)
    prior_zeros = torch.eq(kl_zeros, prior)
    masked_struct_distr = struct_distr.masked_fill(distr_zeros, 1)
    masked_prior = prior.masked_fill(prior_zeros, 1)
    log_distr = torch.log(masked_struct_distr)
    log_prior = torch.log(masked_prior)
    kl = masked_struct_distr * (log_distr - log_prior)
    kl = kl.sum(dim=2)
    return kl.mean()


def compute_KL_div2(mean, log_var):
    return - 0.5 * (1 + log_var - mean.pow(2) - log_var.exp())


def compute_uniform(masks):
    batch_size, sent_len = masks.shape
    distr_mask = masks.clone()
    distr_mask = distr_mask.unsqueeze(1)
    distr_mask = distr_mask.repeat(1, sent_len, 1)
    for i in range(sent_len):
        distr_mask[:, i, i] = 0
    uniform_sum = torch.sum(distr_mask, dim=2, keepdim=True)
    uniform_prior = distr_mask / uniform_sum
    for s in range(batch_size):
        for i in range(sent_len):
            if masks[s, i] == 0:
                uniform_prior[s, i, :] = 0
                distr_mask[s, i, :] = 0
    return uniform_prior, distr_mask


class VAEG(nn.Module):
    def __init__(self, word_vocab_size, t_dim, embed_init, hsize, zsize, anneal_rate=1e-3, struct_rate=1,
                 train_emb=False,
                 structured_prior=False, gpu=-1):
        '''
        args:
            word_vocab_size: reconstruction size
            t_dim: token_representation_dimensionality
            embed_init: reconstruction weight
            hsize: rnn input dim
            zsize: latent vector dim
        '''
        super(VAEG, self).__init__()
        self.to_latent_gaussian = gaussian_layer(input_size=hsize, output_size=zsize)
        self.t_dim = t_dim
        self.word_embed = nn.Embedding(word_vocab_size, t_dim)
        self.train_emb = train_emb
        self.gpu = gpu
        if embed_init is not None:
            # self.word_embed.weight.data.copy_(torch.from_numpy(embed_init))
            self.word_embed.weight.data = embed_init
            print("Initialized with pretrained word embedding")

        if not train_emb:
            self.word_embed.weight.requires_grad = False
            print("Word Embedding not trainable")

        self.x2token = nn.Linear(t_dim, word_vocab_size, bias=False)
        self.x2token.weight = self.word_embed.weight
        # self.x2token = nn.Linear(t_dim, 100, bias=False)

        self.structured_prior = structured_prior
        self.anneal_rate = anneal_rate
        self.struct_rate = struct_rate

        self.z2x = nn.Linear(zsize, t_dim)

    # @profile
    def _forward(self, latent_z, z_mean, z_var, reconstruction_labels, arc_logits, masks, curr_iteration,
                 batch_sen=None):

        batch_size, sent_len = reconstruction_labels.shape
        # reconstruction given the latent representation

        recons_likelihood, struct_distr = self.structured_z2x(latent_z, arc_logits, masks, reconstruction_labels,
                                                              batch_sen)

        # print recons_likelihood

        recons_likelihood = recons_likelihood.permute(0, 2, 1)

        dec_likelihood = recons_likelihood * struct_distr

        dec_loss = -dec_likelihood.sum(dim=2)

        dec_loss = dec_loss.mean()

        # print dec_loss

        # consider the annealing for KL divergence
        kl_temp = get_kl_temp(curr_iteration, self.anneal_rate)

        # padding in kl
        z_mean = z_mean * masks.view(batch_size, sent_len, 1)
        z_var = z_var * masks.view(batch_size, sent_len, 1)
        tmp_g = compute_KL_div_g(z_mean, z_var)
        uniform_prior, distr_mask = compute_uniform(masks)
        tmp_m = compute_KL_div_m(struct_distr, uniform_prior, masks, distr_mask)
        kl_div = tmp_g.mean() + self.struct_rate * tmp_m
        # kl_div =kl_temp*tmp_g.mean() +self.struct_rate*tmp_m
        loss = dec_loss + kl_temp * kl_div

        return loss, dec_loss, kl_div

    def predict(self, lstm_out):

        z, mean_qs, logvar_qs = self.to_latent_gaussian.forward(lstm_out, self.training)

        return z

    # @profile
    def structured_z2x(self, latent_z, arc_logits, masks, reconstruction_labels, batch_sen=None):

        batch_size, sent_len, zsize = latent_z.shape

        # predict the struct distribution
        struct_distr = F.softmax(arc_logits, dim=2)
        # mask the padded part for each sentence
        sent_mask = masks.unsqueeze(2)
        sent_mask = sent_mask.repeat(1, 1, sent_len)
        struct_distr = struct_distr * sent_mask
        # reconstruction for each head
        x_pred = self.z2x(latent_z)
        token_pred = self.x2token(x_pred)
        token_likelihood = F.log_softmax(token_pred, dim=2)
        reconstruction_labels = reconstruction_labels.view(batch_size, 1, sent_len)
        reconstruction_labels = reconstruction_labels.repeat(1, sent_len, 1)
        recons_likelihood = torch.gather(token_likelihood, dim=2, index=reconstruction_labels)

        return recons_likelihood, struct_distr


class gaussian_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(gaussian_layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        '''
        nn to estimate gaussian mean
        '''
        self.q_mean2_mlp = nn.Linear(input_size, output_size)
        '''
        nn to estimate gaussian var
        '''
        self.q_logvar2_mlp = nn.Linear(input_size, output_size)

    def calc_density(self, mean, logvar, target):
        constant = mean.data.new(mean.size())
        constant.fill_(-0.5 * np.log(2) - 0.5 * np.log(np.pi))
        density = constant - 0.5 * logvar - (target - mean) ** 2 / 2 * torch.exp(logvar)
        return density

    def forward(self, inputs, masks, predict=False):
        """
        args:
            inputs: batch_size * sent_len * input_size
            masks: batch_size * sent_len * input_size
        """
        batch_size, batch_len, _ = inputs.size()
        mean_qs = self.q_mean2_mlp(inputs)
        logvar_qs = self.q_logvar2_mlp(inputs)
        # if no_latent:
        #     mean_qs = inputs.new(1)
        #     logvar_qs = inputs.new(1)
        #     mean_qs.fill_(0.0)
        #     logvar_qs.fill_(1.0)

        if not predict:
            z = gaussian(mean_qs, logvar_qs) * masks
        else:
            z = mean_qs * masks

        return z, mean_qs, logvar_qs

    def decode_forward(self, inputs, target_embedding):
        mean_qs = self.q_mean2_mlp(inputs)
        logvar_qs = self.q_logvar2_mlp(inputs)
        g_density = self.calc_density(mean_qs, logvar_qs, target_embedding)
        recons_likelihood = torch.sum(g_density, dim=2)
        return recons_likelihood
