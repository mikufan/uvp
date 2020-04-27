import shutil
from tqdm import tqdm
import sys
from utils import read_conll
import torch
import utils, time, random, decoder
import vg
import numpy as np
import os
import pdb
import torch.nn.functional as F
import eisner_layer
from new_layers import *
from torch import optim
import new_decoder
import pickle


# from memory_profiler import profile


class MSTParserLSTMModel(nn.Module):
    def __init__(self, word_dict, pos_dict, options):
        super(MSTParserLSTMModel, self).__init__()

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims

        # self.zdims = 2 * self.ldims
        self.zdims = options.z_dim

        self.ExtnrEmbPath = options.external_embedding

        self.word_dict = word_dict
        self.pos_dict = pos_dict

        self.gpu = options.gpu

        self.dropout = options.dropout

        self.external_embedding, self.edim = None, 0
        if self.ExtnrEmbPath is not None:

            external_embedding_fp = open(self.ExtnrEmbPath, 'r')
            external_embedding_fp.readline()
            '''
            self.external_embedding: a dictionary
            '''
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line
                                       in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.ex_dict = {word: i + 2 for i, word in enumerate(self.external_embedding)}
            self.ex_dict['*PAD*'] = 0
            self.ex_dict['*UNKNOWN*'] = 1
            np_emb = np.zeros((len(word_dict), self.edim), dtype=np.float32)
            for word, i in self.word_dict.iteritems():
                if word in self.external_embedding:
                    np_emb[i] = self.external_embedding[word]

            self.elookup = nn.Embedding(len(word_dict), self.edim, padding_idx=0)
            self.elookup.weight.data.copy_(torch.from_numpy(np_emb))
            self.elookup.weight.requires_grad = False

            print 'Load external embedding. Vector dimensions', self.edim

            assert self.wdims == self.edim, 'The dimension for word embeddings must be consistent'

        '''
        add lstm
        '''
        self.hidden_units = options.hidden_units

        self.lstm = nn.LSTM(self.wdims + self.pdims, self.ldims, bidirectional=True,
                            batch_first=True, num_layers=options.lstm_layers, dropout=options.dropout)

        self.dmlstm = DmLSTM(
            input_size=self.wdims,
            hidden_size=self.ldims,
            num_layers=options.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=self.dropout,
            dropout_out=self.dropout,
        )

        self.wlookup = nn.Embedding(len(word_dict), self.wdims, padding_idx=0)  # word_emb
        if self.pdims > 0:
            self.plookup = nn.Embedding(len(pos_dict), self.pdims, padding_idx=0)  # pos_emb
        else:
            self.plookup = None

        word_init = np.zeros((len(word_dict), self.wdims), dtype=np.float32)
        self.wlookup.weight.data.copy_(torch.from_numpy(word_init))
        if self.pdims > 0:
            tag_init = np.random.randn(len(pos_dict), self.pdims).astype(np.float32)
            self.plookup.weight.data.copy_(torch.from_numpy(tag_init))

        self.input_drop = nn.Dropout(self.dropout)

        # self.mlp_head = nn.Linear(self.zdims, self.hidden_units)
        # self.mlp_dep = nn.Linear(self.zdims, self.hidden_units)

        self.mlp_arc_head = nn.Linear(2 * self.ldims, self.hidden_units)
        self.mlp_arc_dep = nn.Linear(2 * self.ldims, self.hidden_units)

        self.half_mlp = nn.Linear(self.ldims, self.hidden_units)

        self.reset_parameters(self.mlp_arc_head)
        self.reset_parameters(self.mlp_arc_dep)

        self.biaffine = Biaffine(self.hidden_units, self.hidden_units, 1, (True, False))
        self.kg_scoring = KG_score(self.hidden_units)

        # self.to_latent = nn.Linear(2 * self.ldims, self.zdims)

        self.to_hidden = nn.Linear(self.hidden_units, self.zdims)

        '''
        set up for vae
        '''

        self.best_val = 0.0

        self.best_test = 0.0

        self.anneal_rate = options.anneal_rate
        self.struct_rate = options.struct_rate

        '''
        VAE
        '''
        embed_init = self.wlookup.weight.data.clone() if self.ExtnrEmbPath is None else self.elookup.weight.data + \
                                                                                        self.wlookup.weight.data.clone()
        self.vg = vg.VAEG(len(word_dict), self.wdims, embed_init, self.hidden_units, self.zdims,
                          struct_rate=self.struct_rate, gpu=self.gpu)

        # Modification added
        self.all_counter = 0
        self.correct_counter = 0

        self.alpha = options.alpha

    def predict(self, batch_data):
        words, pos, parents = [s[0] for s in batch_data], [s[1] for s in batch_data], [s[2] for s in batch_data]
        # get latent variables for prediction
        encode_latent, decode_latent, latent_mean, latent_var, mask, parents_v, words_v = \
            self.initial_forward(words, parents, predict=True)
        arc_logits = self.parser_forward(encode_latent, mask, None, predict=False, labelled=False)
        encoding_score = F.log_softmax(arc_logits, dim=2)
        arc_score = encoding_score
        arc_score = arc_score.permute(0, 2, 1)
        arc_score = arc_score.cpu().detach().numpy()
        heads = eisner_layer.batch_parse(arc_score)

        batch_size, sent_len = words_v.shape
        for s in range(batch_size):
            for i in range(sent_len):
                if i == 0:
                    continue
                else:
                    self.all_counter += 1
                    if heads[s, i] == parents[s][i]:
                        self.correct_counter += 1

    def self_train_predict(self, batch_data, labelled_set):
        words, pos, parents, sen = [s[0] for s in batch_data], [s[1] for s in batch_data], \
                                   [s[2] for s in batch_data], [s[3][0] for s in batch_data]
        # get latent variables for prediction
        predicted_one_batch = []
        predicted_words = list(words)
        predicted_pos = list(pos)

        encode_latent, decode_latent, latent_mean, latent_var, mask, parents_v, words_v = \
            self.initial_forward(words, pos, parents, predict=True)
        arc_logits = self.l_forward(encode_latent, mask, None, predict=False, labelled=False)
        encoding_score = F.log_softmax(arc_logits, dim=2)
        arc_score = encoding_score
        arc_score = arc_score.permute(0, 2, 1)
        arc_score = arc_score.cpu().detach().numpy()
        heads = eisner_layer.batch_parse(arc_score)
        batch_size, sent_len = words_v.shape
        for s in range(batch_size):
            if sen[s] in labelled_set:
                continue
            one_sentence = []
            one_sentence.append(predicted_words[s])
            one_sentence.append(predicted_pos[s])
            one_sentence.append(list(heads[s]))
            predicted_one_batch.append(one_sentence)
        return predicted_one_batch

    def set_batch_variable(self, batch_size, batch_words, batch_parents):
        max_length = 0
        for s in range(batch_size):
            length = len(batch_words[s])
            if max_length < length:
                max_length = length
        batch_words_v = torch.zeros((batch_size, max_length), dtype=torch.long)
        batch_mask = torch.zeros((batch_size, max_length))
        batch_parents_v = torch.LongTensor(batch_size, max_length)
        batch_parents_v.fill_(-1)
        for s in range(batch_size):
            s_length = len(batch_words[s])
            for i in range(s_length):
                batch_words_v[s, i] = batch_words[s][i]
                batch_mask[s, i] = 1
                batch_parents_v[s, i] = batch_parents[s][i]
        if self.gpu >= 0 and torch.cuda.is_available():
            batch_words_v = batch_words_v.cuda()
            batch_mask = batch_mask.cuda()
            batch_parents_v = batch_parents_v.cuda()

        return batch_words_v, batch_mask, batch_parents_v

    def drop_input_independent(self, word_embeddings):
        batch_size, seq_length, _ = word_embeddings.size()
        word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - self.dropout)
        word_masks = torch.bernoulli(word_masks)
        word_masks.requires_grad = False

        scale = 3.0 / (2.0 * word_masks.data.cpu().numpy() + 1e-12)
        if self.gpu > -1 and torch.cuda.is_available():
            word_masks = word_masks * torch.cuda.FloatTensor(scale)
        word_masks = word_masks.unsqueeze(dim=2)
        word_embeddings = word_embeddings * word_masks
        return word_embeddings

    def drop_sequence_sharedmask(self, inputs):
        batch_size, sent_len, hidden_size = inputs.size()
        drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - self.dropout)
        drop_masks = torch.bernoulli(drop_masks)
        drop_masks.requires_grad = False
        drop_masks = drop_masks / (1 - self.dropout)
        drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, sent_len).permute(0, 2, 1)
        inputs = inputs * drop_masks

        return inputs

    def orthonormal_initializer(self, output_size, input_size):
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

    def reset_parameters(self, input_layer):

        W = self.orthonormal_initializer(input_layer.out_features, input_layer.in_features)
        input_layer.weight.data.copy_(torch.from_numpy(W))

        b = np.zeros(input_layer.out_features, dtype=np.float32)
        input_layer.bias.data.copy_(torch.from_numpy(b))

    def initial_forward(self, batch_words, batch_parents, predict=False):
        batch_size = len(batch_words)
        # Input to batch tensors
        batch_words_v, batch_mask, batch_parents_v = \
            self.set_batch_variable(batch_size, batch_words, batch_parents)
        # print 'this batch has the size of '+str(batch_words_v.size())
        word_input = self.wlookup(batch_words_v)
        # Add external embedding if available
        if self.ExtnrEmbPath is not None:
            exword_input = self.elookup(batch_words_v)
            lex_input = word_input + exword_input
        else:
            lex_input = word_input
        # Indepedent dropout
        if self.training:
            lex_input = self.drop_input_independent(lex_input)

        batch_input = lex_input
        # build masks
        _, sentence_length, input_dim = batch_input.shape
        batch_input_mask = batch_mask.unsqueeze(2)
        batch_input_mask = batch_input_mask.expand(-1, -1, self.ldims)
        # batch_input = self.input_drop(batch_input)
        # batch_input = batch_input * batch_input_mask
        # go through LSTM
        # hidden_out, _ = self.lstm(batch_input)
        hidden_out, _ = self.dmlstm(batch_input, batch_input_mask, None)
        hidden_out = hidden_out.permute(1, 0, 2)
        batch_z_mask = batch_mask.unsqueeze(2)
        batch_z_mask = batch_z_mask.expand(-1, -1, self.zdims)
        decode_hidden = self.mlp_arc_head(hidden_out)
        m = nn.LeakyReLU(0.1)
        decode_hidden = m(decode_hidden)
        # decode_hidden = self.to_hidden(decode_hidden)
        decode_latent, latent_mean, latent_var = self.vg.to_latent_gaussian(decode_hidden, batch_z_mask, predict)
        encode_latent = hidden_out

        return encode_latent, decode_latent, latent_mean, latent_var, batch_mask, batch_parents_v, batch_words_v

    def parser_forward(self, l_latent_z, masks, parents, predict=False, labelled=True):
        batch_size, sent_len, _ = l_latent_z.shape
        if self.training:
            l_latent_z = self.drop_sequence_sharedmask(l_latent_z)
        # go through mlps for dep and head respectively
        dep_hidden = self.mlp_arc_dep(l_latent_z)
        head_hidden = self.mlp_arc_head(l_latent_z)
        # else:
        # dep_hidden = self.mlp_dep(l_latent_z)
        # head_hidden = self.mlp_head(l_latent_z)
        m1 = nn.LeakyReLU(0.1)
        m2 = nn.LeakyReLU(0.1)
        dep_hidden = m1(dep_hidden)
        head_hidden = m2(head_hidden)
        if self.training:
            dep_hidden = self.drop_sequence_sharedmask(dep_hidden)
            head_hidden = self.drop_sequence_sharedmask(head_hidden)
        # get score for all dependency arcs
        # start = time.time()
        arc_logit = self.biaffine(dep_hidden, head_hidden)
        # end = time.time()
        # print 'time cost '+str(end-start)

        arc_logit = arc_logit.squeeze(3)
        arc_logit = arc_logit.permute(0, 2, 1)
        if not predict:
            # mask paddings
            batch_zeros = torch.zeros((batch_size, sent_len))
            if self.gpu >= 0 and torch.cuda.is_available():
                batch_zeros = batch_zeros.cuda()
            logit_mask = torch.eq(masks, batch_zeros)
            logit_mask = logit_mask.unsqueeze(1)
            logit_mask = logit_mask.repeat(1, sent_len, 1)
            for i in range(sent_len):
                logit_mask[:, i, i] = 1
            arc_logit = arc_logit.masked_fill(logit_mask, -10000)

            if labelled:
                # compute losses
                if batch_size > 0:
                    l_loss = F.cross_entropy(arc_logit.contiguous().view(batch_size * sent_len, -1), parents.view(-1),
                                             ignore_index=-1)
                else:
                    l_loss = torch.sum(batch_zeros)
                return l_loss
            else:
                return arc_logit
        else:
            arc_logit = arc_logit.permute(0, 2, 1)
            arc_scores = arc_logit.cpu().detach().numpy()
            heads = eisner_layer.batch_parse(arc_scores)

            return heads


def get_optim(opt, options, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=options.learning_rate)
    elif opt == 'adam':
        return optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=options.learning_rate,
                          betas=(options.beta1, options.beta2), eps=options.eps)


class MSTParserLSTM(object):
    def __init__(self, word_dict, pos_dict, options):
        self.model = MSTParserLSTMModel(word_dict, pos_dict, options)
        if options.gpu > -1 and torch.cuda.is_available():
            torch.cuda.set_device(options.gpu)
            self.model.cuda(options.gpu)
        self.trainer = get_optim(options.optim, options, self.model.parameters())
        self.options = options
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.batch_size = options.batch_size
        self.test_batch_size = options.test_batch_size

        self.gpu = options.gpu

        self.clip = options.clip
        self.update_every = options.update_every
        self.decay = options.decay

    def update_lr(self, step):
        self.decay = self.decay ** (step // self.decay_step)
        self.trainer.param_groups[0]['lr'] = self.trainer.param_groups[0]['lr'] * self.decay

    def test_predict(self, path, epoch):
        test = open(path, 'r')
        testData = list(read_conll(test))
        data_list = utils.construct_parsing_data_list(testData, self.word_dict, self.pos_dict)
        batch_test_data = utils.construct_sorted_batch_data(data_list, self.test_batch_size)
        tot_batch = len(batch_test_data)
        for batch_id, one_batch in tqdm(enumerate(batch_test_data), mininterval=2,
                                        desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
            self.model.predict(one_batch)

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def set_labelled_data(self, data_list, specify_labelled=False, specify_split=False, split_idx=0):
        labelled_data_list = list()
        unlabelled_data_list = list()
        if not specify_labelled and not specify_split:
            random.shuffle(data_list)

            lprop = self.options.lprop

            max_idx_l = int(len(data_list) * lprop) - 1

            for i in range(0, max_idx_l):
                one_sentence = data_list[i]
                sen_idx = one_sentence[3][0]
                self.labelled_set.add(sen_idx)
        elif specify_split:
            for i in range(0, split_idx):
                one_sentence = data_list[i]
                sen_idx = one_sentence[3][0]
                self.labelled_set.add(sen_idx)

        else:
            lprop = self.options.lprop
            max_idx_l = int(len(data_list) * lprop) - 1
            factor = len(data_list) // max_idx_l
            for i in range(len(data_list)):
                one_sentence = data_list[i]
                sen_idx = one_sentence[3][0]
                if sen_idx % factor == 0:
                    self.labelled_set.add(sen_idx)
        for i in range(len(data_list)):
            one_sentence = data_list[i]
            sen_idx = one_sentence[3][0]
            if sen_idx in self.labelled_set:
                labelled_data_list.append(one_sentence)

        return labelled_data_list

    # @profile
    def training(self, batch_data, epoch, step):
        self.model.train()
        l_loss_value = 0.0
        loss_value = 0.0
        recon_loss = 0.0
        kl = 0.0
        struct_kl = 0.0
        print 'start labelled training'
        # Training for the labelled data

        tot_batch = len(batch_data)
        # random.shuffle(batch_data)
        for batch_id, one_batch in tqdm(enumerate(batch_data), mininterval=2,
                                        desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
            # print 'size for this batch '+ str(len(batch_words[0]))
            batch_words, batch_pos, batch_parents, batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                               [s[2] for s in one_batch], [s[3][0] for s in one_batch]
            # skip if no label data in this batch for parsing only configuration
            encoding_latent, decoding_latent, latent_mean, latent_var, batch_mask, batch_parents_v, batch_words_v = \
                self.model.initial_forward(batch_words, batch_pos, batch_parents)
            arc_logits = self.model.parser_forward(encoding_latent, batch_mask, None, labelled=False)
            # compute the reconstruction loss for unlabelled data
            batch_loss, reconstruction_loss, kl_loss = self.model.vg._forward(decoding_latent, latent_mean, latent_var,
                                                                              batch_words_v,
                                                                              arc_logits, batch_mask, epoch, batch_sen)
            batch_loss.backward()
            step += 1
            if batch_id % self.update_every == 0 or batch_id == tot_batch:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=self.clip)
            self.trainer.step()
            loss_value += batch_loss.data.cpu().numpy()
            recon_loss += batch_loss.data.cpu().numpy()
            kl += kl_loss.data.cpu().numpy()
            self.trainer.zero_grad()

        if self.gpu > -1 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print 'loss in this epoch ' + str(loss_value / tot_batch)
        print 'reconstruction_loss ' + str(recon_loss / tot_batch)
        print 'kl ' + str(kl / tot_batch)
        #print 'structure kl ' + str(struct_kl / tot_batch)
        return step
