from collections import Counter
import re
import torch
import numpy as np
import random
from itertools import groupby


class ConllEntry(object):
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.cpos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation,
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    #relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            #relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys())


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                # tokens.append(line.strip())
                pass
            else:
                '''
                modified for conllu in UD, 3-4 switched
                '''
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def isfinite(a):
    return (a != np.inf) & (a != -np.inf) & (a != np.nan) & (a != -np.nan)


def amax(a, axis=None, keepdim=False):
    if isinstance(axis, tuple):
        for x in reversed(axis):
            a, index = a.max(x, keepdim=keepdim)
    else:
        a, index = a.max(axis, keepdim=True)
    return a


def asum(a, axis=None, keepdim=False):
    if isinstance(axis, tuple):
        for x in reversed(axis):
            a = a.sum(x, keepdim=keepdim)
    else:
        a = a.sum(axis, keepdim=keepdim)
    return a


def logsumexp(a, axis=None):
    a_max = amax(a, axis=axis, keepdim=True)
    a_max[~isfinite(a_max)] = 0
    res = torch.log(asum(torch.exp(a - a_max), axis=axis, keepdim=True)) + a_max
    if isinstance(axis, tuple):
        for x in reversed(axis):
            res.squeeze_(x)
    else:
        res.squeeze_(axis)
    return res


def construct_sorted_batch_data(data_list, batch_size):
    data_list.sort(key=lambda x: len(x[0]))
    grouped = [list(g) for k, g in groupby(data_list, lambda s: len(s[0]))]
    batch_data = []
    for group in grouped:
        sub_batch_data = get_batch_data(group, batch_size)
        batch_data.extend(sub_batch_data)
    return batch_data


def get_batch_data(grouped_data, batch_size):
    batch_data = []
    len_datas = len(grouped_data)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(grouped_data[start_idx:end_idx])
    return batch_data


def memoize(func):
    mem = {}

    def helper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return helper


@memoize
def constituent_index(sentence_length, multiroot):
    counter_id = 0
    basic_span = []
    id_2_span = {}
    for left_idx in range(sentence_length):
        for right_idx in range(left_idx, sentence_length):
            for dir in range(2):
                id_2_span[counter_id] = (left_idx, right_idx, dir)
                counter_id += 1

    span_2_id = {s: id for id, s in id_2_span.items()}

    for i in range(sentence_length):
        if i != 0:
            id = span_2_id.get((i, i, 0))
            basic_span.append(id)
        id = span_2_id.get((i, i, 1))
        basic_span.append(id)

    ijss = []
    ikcs = [[] for _ in range(counter_id)]
    ikis = [[] for _ in range(counter_id)]
    kjcs = [[] for _ in range(counter_id)]
    kjis = [[] for _ in range(counter_id)]

    for l in range(1, sentence_length):
        for i in range(sentence_length - l):
            j = i + l
            for dir in range(2):
                ids = span_2_id[(i, j, dir)]
                for k in range(i, j + 1):
                    if dir == 0:
                        if k < j:
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir + 1)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir)]
                            kjis[ids].append(idri)
                            # one complete span,one incomplete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                    else:
                        if k < j and ((not (i == 0 and k != 0) and not multiroot) or multiroot):
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir - 1)]
                            kjis[ids].append(idri)
                        if k > i:
                            # one incomplete span,one complete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                ijss.append(ids)

    return span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span


def construct_parsing_data_list(sentences, words, pos):
    data_list = list()
    sen_idx = 0
    for s in sentences:
        s_word, s_pos, s_parent = set_parsing_data_list(s, words, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append(s_parent)
        s_data_list.append([sen_idx])
        data_list.append(s_data_list)
        sen_idx += 1

    return data_list


def construct_batch_data(data_list, batch_size):
    random.shuffle(data_list)
    batch_data = []
    len_datas = len(data_list)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1
    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(data_list[start_idx:end_idx])
    return batch_data


def construct_unbalanced_batch_data(data_list, batch_size):
    random.shuffle(data_list)
    batch_data = []
    len_datas = len(data_list)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1
    for i in range(num_batch):
        over_size = False
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        for j in range(start_idx, end_idx):
            if len(data_list[j][0]) > 120 and batch_size > 80:
                over_size = True
                break
        if over_size:
            mid_idx_1 = start_idx + (end_idx - start_idx) / 4
            mid_idx_2 = start_idx + 2 * (end_idx - start_idx) / 4
            mid_idx_3 = start_idx + 3 * (end_idx - start_idx) / 4
            batch_data.append(data_list[start_idx:mid_idx_1])
            batch_data.append(data_list[mid_idx_1:mid_idx_2])
            batch_data.append(data_list[mid_idx_2:mid_idx_3])
        else:
            batch_data.append(data_list[start_idx:end_idx])
    return batch_data

    return batch_data


# coding=utf-8
import re
import random
from collections import Counter
from itertools import groupby
import numpy as np
import time


class ConllEntry:
    def __init__(self, id, form, lemma, cpos, pos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        # self.pred_parent_id = None
        # self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  str(self.parent_id) if self.parent_id is not None else None, self.relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def memoize(func):
    mem = {}

    def helper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return helper


@memoize
def constituent_index(sentence_length, multiroot):
    counter_id = 0
    basic_span = []
    id_2_span = {}
    for left_idx in range(sentence_length):
        for right_idx in range(left_idx, sentence_length):
            for dir in range(2):
                id_2_span[counter_id] = (left_idx, right_idx, dir)
                counter_id += 1

    span_2_id = {s: id for id, s in id_2_span.items()}

    for i in range(sentence_length):
        if i != 0:
            id = span_2_id.get((i, i, 0))
            basic_span.append(id)
        id = span_2_id.get((i, i, 1))
        basic_span.append(id)

    ijss = []
    ikcs = [[] for _ in range(counter_id)]
    ikis = [[] for _ in range(counter_id)]
    kjcs = [[] for _ in range(counter_id)]
    kjis = [[] for _ in range(counter_id)]

    for l in range(1, sentence_length):
        for i in range(sentence_length - l):
            j = i + l
            for dir in range(2):
                ids = span_2_id[(i, j, dir)]
                for k in range(i, j + 1):
                    if dir == 0:
                        if k < j:
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir + 1)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir)]
                            kjis[ids].append(idri)
                            # one complete span,one incomplete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                    else:
                        if k < j and ((not (i == 0 and k != 0) and not multiroot) or multiroot):
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir - 1)]
                            kjis[ids].append(idri)
                        if k > i:
                            # one incomplete span,one complete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                ijss.append(ids)

    return span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span


class data_sentence:
    def __init__(self, id, entry_list):
        self.id = id
        self.entries = entry_list
        self.size = len(entry_list)

    def set_data_list(self, words, pos):
        word_list = list()
        pos_list = list()
        for entry in self.entries:
            if entry.norm in words.keys():
                word_list.append(words[entry.norm])
            else:
                word_list.append(words['<UNKNOWN>'])
            if entry.pos in pos.keys():
                pos_list.append(pos[entry.pos])
            else:
                pos_list.append(pos['<UNKNOWN-POS>'])
        return word_list, pos_list

    def set_parsing_data_list(self, words, pos):
        word_list = list()
        pos_list = list()
        parent_list = list()
        for entry in self.entries:
            if words.get(entry.norm) is not None:
                word_list.append(words[entry.norm])
            else:
                word_list.append(words['<UNKNOWN>'])
            if pos.get(entry.pos) is not None:
                pos_list.append(pos[entry.pos])
            else:
                pos_list.append(pos['<UNKNOWN-POS>'])
            parent_list.append(entry.parent_id)
        return word_list, pos_list, parent_list

    def __str__(self):
        return '\t'.join([e for e in self.entries])


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-CPOS', 'ROOT-POS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                continue
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5],
                                         int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def read_data(conll_path, isPredict):
    sentences = []
    if not isPredict:
        wordsCount = set()
        posCount = set()
        s_counter = 0
        with open(conll_path, 'r') as conllFP:
            for sentence in read_conll(conllFP):
                for node in sentence:
                    if isinstance(node, ConllEntry):
                        wordsCount.add(node.norm)
                        posCount.add(node.pos)
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                s_counter += 1
        wordsCount.add('<UNKNOWN>')
        posCount.add('<UNKNOWN-POS>')
        return {w: i for i, w in enumerate(wordsCount)}, {p: i for i, p in enumerate(posCount)}, sentences
    else:
        with open(conll_path, 'r') as conllFP:
            s_counter = 0
            for sentence in read_conll(conllFP):
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                s_counter += 1
        return sentences


def construct_sample_batch_data(data_list, batch_size):
    data_list.sort(key=lambda x: len(x[0]))
    grouped = [list(g) for k, g in groupby(data_list, lambda s: len(s[0]))]
    batch_data = []
    for group in grouped:
         sub_batch_data = get_batch_data(group, batch_size)
         batch_data.extend(sub_batch_data)
    return batch_data


# def construct_imbalanced_batch_data(data_list, batch_size, order):
#     data_list.sort(key=lambda x: len(x[0]))
#     grouped = [list(g) for k, g in groupby(data_list, lambda s: len(s[0]))]
#     batch_data = []
#     for group in grouped:
#         sub_batch_data = get_imbalanced_batch_data(group, batch_size, order)
#         batch_data.extend(sub_batch_data)
#     return batch_data


def get_batch_data(grouped_data, batch_size):
    batch_data = []
    len_datas = len(grouped_data)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(grouped_data[start_idx:end_idx])
    return batch_data


def get_imbalanced_batch_data(grouped_data, batch_size):
    batch_data = []
    sample = grouped_data[0][0]
    actual_batch_size = batch_size
    if len(sample) < 30 and batch_size > 200:
        None
    len_datas = len(grouped_data)
    num_batch = len_datas // actual_batch_size
    if not len_datas % actual_batch_size == 0:
        num_batch += 1
    for i in range(num_batch):
        start_idx = i * actual_batch_size
        end_idx = min(len_datas, (i + 1) * actual_batch_size)
        batch_data.append(grouped_data[start_idx:end_idx])
    return batch_data


def set_parsing_data_list(sentence, words, pos):
    word_list = list()
    pos_list = list()
    parent_list = list()
    for entry in sentence:
        if words.get(entry.norm) is not None:
            word_list.append(words[entry.norm])
        else:
            word_list.append(words['*UNKNOWN*'])
        if pos.get(entry.pos) is not None:
            pos_list.append(pos[entry.pos])
        else:
            pos_list.append(pos['*UNKNOWN*'])
        parent_list.append(entry.parent_id)

    return word_list, pos_list, parent_list


def get_dict(words, pos):
    word_dict = {word: ind + 2 for ind, word in enumerate(words)}
    pos_dict = {pos: ind + 2 for ind, pos in enumerate(pos)}
    word_dict['*PAD*'] = 0
    pos_dict['*PAD*'] = 0

    word_dict['*UNKNOWN*'] = 1
    pos_dict['*UNKNOWN*'] = 1

    return word_dict, pos_dict

def get_weights_to_sample(words,word_dict):
    words_list = []
    idx_2_words = {}
    word_weights = np.zeros((len(word_dict),),dtype=int)
    for w in word_dict.keys():
        idx = word_dict[w]
        words_list.append(idx)
        idx_2_words[idx] = w
    words_list.sort()
    for i,idx in enumerate(words_list):
        if idx == 0:
            word_weights[idx] = 1
        elif idx == 1:
            word_weights[idx] = 1
        else:
            w = idx_2_words[idx]
            weight = words[w]
            word_weights[idx] = weight
    return word_weights

def construct_neg_sample_dict(data_list,weights_to_sample,num_sample,gpu):
    start = time.time()
    neg_sample_dict = {}
    print 'Start sampling negative samples'
    for s in data_list:
        sentence_words = s[0]
        sen_idx = s[3][0]
        sentence_sample = []
        sent_len = len(sentence_words)
        start = time.time()
        for i in range(sent_len):
            weights = torch.from_numpy(weights_to_sample).float()
            if gpu > 0 and torch.cuda.is_available():
                weights = weights.cuda()
            word_idx = sentence_words[i]
            weights[word_idx] = 0
            word_sample = torch.multinomial(weights, num_sample - 1).data.detach()
            word_sample = word_sample.unsqueeze(0)
            sentence_sample.append(word_sample)
        sentence_sample = torch.cat(sentence_sample,dim=0)
        print 'Sampling completed for sentence '+ str(sen_idx)
        neg_sample_dict[sen_idx] = sentence_sample
    end = time.time()
    print 'Time cost in negative sampling '+str(end- start)
    return neg_sample_dict






