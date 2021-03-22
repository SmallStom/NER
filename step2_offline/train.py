import argparse
import sys
import logging

import torch
import collections
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric, ClassifyFPreRecMetric
from fastNLP.core.callback import WarmupCallback, GradientClipCallback, EarlyStopCallback
from fastNLP import LRScheduler
from fastNLP.core import Trainer
from fastNLP.core import Callback

from load_data import *
from paths import *
from utils import get_peking_time
from models import step2_clf
import utils
from my_metric import MyCLFMetric
utils.configure_logging()

parser = argparse.ArgumentParser()
parser.add_argument('--status', default='train', choices=['train','test','trace'])
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--test_train', default=False)
parser.add_argument('--update_every', default=1, type=int)

# hyper of training
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--batch', default=20, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=8e-4, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='norm', help='norm|uniform')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--warmup', default=0.01, type=float)
parser.add_argument('--hidden', default=258, type=int) #128
args = parser.parse_args()

if args.test_batch == -1:
    args.test_batch = args.batch // 2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if args.status == 'trace':
    device = 'cpu'
logging.info('device={}'.format(device))

datasets, vocabs, embeddings = prepare_inputs(ip_step2_path, char_embedding_path)

for k, v in args.__dict__.items():
    logging.info('{}:{}'.format(k, v))

model = step2_clf(embeddings['char'], len(vocabs['char']), 200, len(vocabs['entity']), 50, args.hidden, 5, device)
logging.info('model:{}'.format(model))

for n, p in model.named_parameters():
    logging.info('{}:{}:{}'.format(n, p.data_ptr(), p.data.data_ptr()))

with torch.no_grad():
    logging.info('{}init pram{}'.format('*' * 15, '*' * 15))
    for n, p in model.named_parameters():
        if 'chars_embed' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    logging.info('xavier uniform init : {}'.format(n))
                elif args.init == 'norm':
                    # logging.info('xavier norm init : {}'.format(n))
                    # nn.init.xavier_normal_(p)
                    logging.info('norm init : {}'.format(n))
                    nn.init.normal_(p, std=0.02)
            except:
                logging.info(n)
                exit(1208)
    logging.info('{}init pram{}\n'.format('*' * 15, '*' * 15))

loss = LossInForward()
# acc_metric = AccuracyMetric(pred='pred', target='target')
# acc_metric = ClassifyFPreRecMetric(pred='pred', target='target',only_gross=False)
my_metric = MyCLFMetric(pred='pred', target='target')
metrics = [my_metric]

embedding_param = list(model.chars_embed.parameters()) + list(model.entity_label_embed.parameters())
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

save_model_path = './ner_clf.pt'
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
    example = datasets['train'][0]
    left_chars = torch.tensor(example['left_chars']).view(1, -1)
    right_chars = torch.tensor(example['right_chars']).view(1, -1)
    entity_chars = torch.tensor(example['entity_chars']).view(1, -1)
    entity_label = torch.tensor(example['entity_label']).view(1, -1)
    target = torch.tensor(example['target']).view(1, -1)

    logging.info('left_chars={}'.format(left_chars))
    logging.info('right_chars={}'.format(right_chars))
    logging.info('entity_chars={}'.format(entity_chars))
    logging.info('entity_label={}'.format(entity_label))
    logging.info('target={}'.format(target))


    model.load_state_dict(torch.load(save_model_path))
    model = model.cpu()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        traced_script_module = torch.jit.script(model, (left_chars, right_chars, entity_chars, entity_label))
        logging.info('model output: {}'.format(traced_script_module.forward(left_chars, right_chars, entity_chars, entity_label)))
        traced_script_module.save("ner_clf_script.pt")

if args.status == 'test':
    device = torch.device('cpu')
    #device = torch.device('cuda:0')
    model.load_state_dict(torch.load(save_model_path, map_location=device))

    from fastNLP import Tester
    tester = Tester(datasets['test'], model, metrics=metrics, batch_size=1)
    logging.info('test start={}'.format(get_peking_time()))
    tester.test()
    logging.info('test end={}'.format(get_peking_time()))
