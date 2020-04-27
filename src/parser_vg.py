from optparse import OptionParser
import pickle, utils, mstlstm_vg, os, os.path, time
import torch
from utils import read_conll
import numpy as np
import random
import vg
import pdb

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="../data/wsj10/wsj10_tr")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="../data/wsj10/wsj10_d")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                      default="../data/wsj10/wsj10_te")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=100)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=100)

    parser.add_option("--hidden", type="int", dest="hidden_units", default=200)

    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.002)
    parser.add_option("--outdir", type="string", dest="output", default="../results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--beta1", type="float", default=0.9)
    parser.add_option("--beta2", type="float", default=0.9)
    parser.add_option("--eps", type="float", default=1e-12)

    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=200)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.33)

    parser.add_option("--z_dim", type="int", dest="z_dim", default=100)
    parser.add_option('--anneal_rate', type='float', default=1e-3, help='the annealling rate for KL')
    parser.add_option('--struct_rate', type='float', default=1e-3,
                      help='hyper parameter to adjust the importance of strcuture KL')

    # Extra modification for new model
    parser.add_option('--t_batch', type=int, dest='test_batch_size', default=500, help='batch size set for test ')
    parser.add_option('--alpha', type=float, default=0.8,
                      help='hyperparameter that adjust the importance of labelled data loss')
    parser.add_option('--clip', type=float, default=5.0, help='gradients to be clipped')
    parser.add_option('--update_every', type=int, default=1, help="the frequecy of updating parameters")
    parser.add_option('--decay', type=float, default=0.75, help='learning rate decay')

    parser.add_option('--batch_size', type='int', dest='batch_size', default=50)
    parser.add_option('--gpu', type='int', dest='gpu', default=0)
    parser.add_option('--num_threads', type='int', dest='num_threads', default=4)
    parser.add_option('--do_test', action='store_true', default=False,
                      help='whether to evaluate on test set after evaluation on validation set')

    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding

    print 'pytorch version:', torch.__version__

    torch.set_num_threads(options.num_threads)

    # Random seed set for tuning
    # torch.manual_seed(666)
    # torch.cuda.manual_seed(666)
    # random.seed(666)
    # np.random.seed(666)

    if options.gpu >= 0 and torch.cuda.is_available():
        print 'To use gpu' + str(options.gpu)
        torch.backends.cudnn.eabled = True

    '''
    train
    '''
    print 'Preparing vocab'
    words, w2i, pos = utils.vocab(options.conll_train)

    print 'Finished collecting vocab'

    '''
    Construct dicts with padding and OOV
    '''

    word_dict, pos_dict = utils.get_dict(words, pos)

    idx_2_words = None

    print 'Initializing lstm mstparser:'
    '''
    Batchify traning data
    '''
    with open(options.conll_train, 'r') as conllFP:
        sentencesData = list(read_conll(conllFP))
    data_list = utils.construct_parsing_data_list(sentencesData, word_dict, pos_dict)

    parser = mstlstm_vg.MSTParserLSTM(word_dict, pos_dict, options)
    batch_data = utils.construct_batch_data(data_list, options.batch_size)
    step = 0
    label_step = 0
    for epoch in xrange(options.epochs):
        print 'Starting epoch', epoch
        '''
        train here
        '''
        step = parser.training(batch_data, epoch, step)
        parser.model.all_counter = 0
        parser.model.correct_counter = 0
        conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
        # devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
        # utils.write_conll(devpath, parser.predict(options.conll_dev, epoch))
        parser.test_predict(options.conll_dev, epoch)
        print str(parser.model.correct_counter) + '/' + str(parser.model.all_counter)
        print 'UAS: ' + str(float(parser.model.correct_counter) / parser.model.all_counter)
        curr_val = (float(parser.model.correct_counter) / parser.model.all_counter)
        if parser.model.best_val < curr_val:
            parser.model.best_val = curr_val
        print 'Best results in validation: ' + str(parser.model.best_val)
        if options.do_test:
            if parser.model.best_val == curr_val:
                parser.model.correct_counter = 0
                parser.model.all_counter = 0
                parser.test_predict(options.conll_test, epoch)
                curr_test = (float(parser.model.correct_counter) / parser.model.all_counter)
                parser.model.best_test = curr_test
            print 'Test result: ' + str(parser.model.best_test)
