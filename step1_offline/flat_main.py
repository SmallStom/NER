import argparse
import sys
import logging

import torch
import collections
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric
from fastNLP.core.callback import WarmupCallback, GradientClipCallback, EarlyStopCallback
from fastNLP import LRScheduler
from fastNLP.core import Trainer
from fastNLP.core import Callback

from load_data import *
from paths import *
from utils import get_peking_time
from models import Lattice_Transformer_SeqLabel
import utils
utils.configure_logging()

parser = argparse.ArgumentParser()
parser.add_argument('--status', default='train', choices=['train','test','trace'])
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--test_train', default=False)
parser.add_argument('--lexicon_name', default='yj', choices=['lk'])
parser.add_argument('--update_every', default=1, type=int)
parser.add_argument('--char_min_freq', default=1, type=int)
parser.add_argument('--bigram_min_freq', default=1, type=int)
parser.add_argument('--lattice_min_freq', default=1, type=int)
parser.add_argument('--only_train_min_freq', default=True)
parser.add_argument('--only_lexicon_in_train', default=False)
parser.add_argument('--word_min_freq', default=1, type=int)

# hyper of training
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--batch', default=10, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=8e-4, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='uniform', help='norm|uniform')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--norm_embed', default=True)
parser.add_argument('--norm_lattice_embed', default=True)
parser.add_argument('--warmup', default=0.01, type=float)

# hyper of model
parser.add_argument('--hidden', default=128, type=int)
parser.add_argument('--ff', default=384, type=int)
parser.add_argument('--layer', default=1, type=int)
parser.add_argument('--head', default=8, type=int)
parser.add_argument('--head_dim', default=16, type=int)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')
parser.add_argument('--embed_dropout', default=0.5, type=float)
parser.add_argument('--gaz_dropout', default=0.5, type=float)
parser.add_argument('--output_dropout', default=0.3, type=float)
parser.add_argument('--pre_dropout', default=0.5, type=float)
parser.add_argument('--post_dropout', default=0.3, type=float)
parser.add_argument('--ff_dropout', default=0.15, type=float)
parser.add_argument('--ff_dropout_2', default=0.15, type=float)
parser.add_argument('--attn_dropout', default=0.001, type=float)

args = parser.parse_args()
if args.test_batch == -1:
    args.test_batch = args.batch // 2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logging.info('device={}'.format(device))

datasets, vocabs, embeddings = prepare_inputs(ip_step1_path, yangjie_rich_pretrain_unigram_path,
                                              yangjie_rich_pretrain_bigram_path)

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff

for k, v in args.__dict__.items():
    logging.info('{}:{}'.format(k, v))

max_seq_len = max(*map(lambda x: max(x['seq_len']), datasets.values()))

for k, v in datasets.items():
    v.set_input('lattice', 'bigrams', 'seq_len', 'target')
    v.set_input('lex_num', 'pos_s', 'pos_e')
    v.set_target('target', 'seq_len')

dropout = collections.defaultdict(float)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

for k,v in dropout.items():
    logging.info('dropout k={},v={}'.format(k,v))


model = Lattice_Transformer_SeqLabel(embeddings['lattice'],
                                     len(vocabs['lattice']),
                                     50,
                                     embeddings['bigram'],
                                     len(vocabs['bigram']),
                                     50,
                                     args.hidden,
                                     len(vocabs['label']),
                                     args.head,
                                     args.layer,
                                     args.learn_pos,
                                     args.pre,
                                     args.post,
                                     args.ff,
                                     dropout,
                                     max_seq_len=max_seq_len)
logging.info('model:{}'.format(model))

for n, p in model.named_parameters():
    logging.info('{}:{}:{}'.format(n, p.data_ptr(), p.data.data_ptr()))

with torch.no_grad():
    logging.info('{}init pram{}'.format('*' * 15, '*' * 15))
    for n, p in model.named_parameters():
        if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    logging.info('xavier uniform init : {}'.format(n))
                elif args.init == 'norm':
                    logging.info('xavier norm init : {}'.format(n))
                    nn.init.xavier_normal_(p)
            except:
                logging.info(n)
                exit(1208)
    logging.info('{}init pram{}\n'.format('*' * 15, '*' * 15))

loss = LossInForward()
f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len', encoding_type='bio')
acc_metric = AccuracyMetric(pred='pred', target='target', seq_len='seq_len')
acc_metric.set_metric_name('label_acc')
metrics = [f1_metric, acc_metric]

# if args.self_supervised:
#     chars_acc_metric = AccuracyMetric(pred='chars_pred', target='chars_target', seq_len='seq_len')
#     chars_acc_metric.set_metric_name('chars_acc')
#     metrics.append(chars_acc_metric)

embedding_param = list(model.bigram_embed.parameters()) + list(model.lattice_embed.parameters())
embedding_param_ids = list(map(id, embedding_param))
non_embedding_param = list(filter(lambda x: id(x) not in embedding_param_ids, model.parameters()))

param_ = [{'params': non_embedding_param}, {'params': embedding_param,'lr': args.lr * args.embed_lr_rate}]
if args.optim == 'adam':
    optimizer = optim.AdamW(param_, lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == 'sgd':
    optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05 * ep)))
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)

callbacks = [lrschedule_callback, clip_callback]
if args.warmup > 0: callbacks.append(WarmupCallback(warmup=args.warmup))

save_model_path = './flat_main_seq_label.pt'
if args.status == 'train':
    trainer = Trainer(datasets['train'],
                      model,
                      optimizer,
                      loss,
                      args.batch // args.update_every,
                      update_every=args.update_every,
                      n_epochs=args.epoch,
                      dev_data=datasets['dev'],
                      metrics=metrics,
                      device=device,
                      callbacks=callbacks,
                      dev_batch_size=args.test_batch,
                      test_use_tqdm=False,
                      print_every=5,
                      check_code_level=-1)
    trainer.train()
    torch.save(model.state_dict(), save_model_path)
if args.status == 'trace':
    example_instance = datasets['train'][0]
    lattice = torch.tensor(example_instance['lattice']).view(1, -1)
    bigrams = torch.tensor(example_instance['bigrams']).view(1, -1)
    seq_len = torch.tensor(example_instance['seq_len']).view(-1)
    lex_num = torch.tensor(example_instance['lex_num']).view(-1)
    pos_s = torch.tensor(example_instance['pos_s']).view(1, -1)
    pos_e = torch.tensor(example_instance['pos_e']).view(1, -1)
    target = torch.tensor(example_instance['target']).view(1, -1)

    logging.info('raw_chars={}'.format(example_instance['raw_chars']))
    logging.info('lattice={}'.format(lattice))
    logging.info('bigrams={}'.format(bigrams))
    logging.info('seq_len={}'.format(seq_len))
    logging.info('lex_num={}'.format(lex_num))
    logging.info('pos_s={}'.format(pos_s))
    logging.info('pos_e={}'.format(pos_e))
    logging.info('target={}'.format(target))

    model.load_state_dict(torch.load(save_model_path))
    model = model.cpu()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    instance2 = datasets['train'][10]
    lattice2 = torch.tensor(instance2['lattice']).view(1, -1)
    bigrams2 = torch.tensor(instance2['bigrams']).view(1, -1)
    seq_len2 = torch.tensor(instance2['seq_len']).view(-1)
    lex_num2 = torch.tensor(instance2['lex_num']).view(-1)
    pos_s2 = torch.tensor(instance2['pos_s']).view(1, -1)
    pos_e2 = torch.tensor(instance2['pos_e']).view(1, -1)
    target2 = torch.tensor(instance2['target']).view(1, -1)

    logging.info('raw_chars2={}'.format(instance2['raw_chars']))
    logging.info('lattice2={}'.format(lattice2))
    logging.info('bigrams2={}'.format(bigrams2))
    logging.info('seq_len2={}'.format(seq_len2))
    logging.info('lex_num2={}'.format(lex_num2))
    logging.info('pos_s2={}'.format(pos_s2))
    logging.info('pos_e2={}'.format(pos_e2))
    logging.info('target2={}'.format(target2))
    check_inputs = [(lattice2, bigrams2, seq_len2, lex_num2, pos_s2, pos_e2, target2)]

    with torch.no_grad():
        # logging.info('model output: {}'.format(model(lattice, bigrams, seq_len, lex_num, pos_s, pos_e, target)))
        # traced_script_module = torch.jit.trace(model, (lattice, bigrams, seq_len, lex_num, pos_s, pos_e, target),
        #                                        check_trace=True,
        #                                        check_inputs=check_inputs)
        traced_script_module = torch.jit.script(model, (lattice, bigrams, seq_len, lex_num, pos_s, pos_e))
        logging.info('model output: {}'.format(traced_script_module.forward(lattice2, bigrams2, seq_len2, lex_num2, pos_s2, pos_e2, target2)))
        # logging.info(traced_script_module.graph)
        # logging.info(traced_script_module.code)
        traced_script_module.save("flat_main_seq_label_script.pt")

if args.status == 'test':
    device = torch.device('cpu')
    #device = torch.device('cuda:0')
    model.load_state_dict(torch.load(save_model_path, map_location=device))

    from fastNLP import Tester
    tester = Tester(datasets['test'], model, metrics=metrics, batch_size=1)
    logging.info('test start={}'.format(get_peking_time()))
    tester.test()
    logging.info('test end={}'.format(get_peking_time()))
