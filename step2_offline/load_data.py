from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
from fastNLP.io import Loader
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary

import logging
import utils
from utils import tokenize, entity_label_tokenize
from utils import Trie, build_pretrain_embedding
from functools import partial
import numpy as np
import pickle
import os
from fastNLP import cache_results
from fastNLP.embeddings import StaticEmbedding
from paths import *

import torch.nn.init


class myConllLoader(Loader):

    def __init__(self):
        super(myConllLoader, self).__init__()

    def _load(self, path: str = None):
        logging.info(path)
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '': continue
                splits = line.strip().split('\t')
                if len(splits) == 4:
                    raw_targets = [int(i) for i in splits[3].strip().lstrip('[').rstrip(']').split(' ')]
                elif len(splits) == 3:
                    raw_targets = [0, 0, 0, 0, 0]
                else:
                    logging.error('data format error')
                raw_query = splits[0]
                raw_entity = splits[1]
                left_context = raw_query[0:raw_query.find(raw_entity)]
                right_context = raw_query[raw_query.find(raw_entity) + len(raw_entity):]
                if left_context == '': left_context = '-'
                if right_context == '': right_context = '-'
                raw_entity_label = splits[2]
                if left_context and right_context and raw_entity and raw_entity_label:
                    ds.append(
                        Instance(left_context=tokenize(left_context),
                                 right_context=tokenize(right_context),
                                 raw_entity=tokenize(raw_entity),
                                 raw_entity_label=entity_label_tokenize(raw_entity_label),
                                 target=raw_targets))
        return ds


@cache_results(_cache_fp='cache/ip_step2', _refresh=False)
def load_ip_step2(path, char_embedding_path=None):
    train_path = os.path.join(path, 'train1.txt')
    dev_path = os.path.join(path, 'dev1.txt')
    test_path = os.path.join(path, 'test1.txt')

    # 播放徐秉龙的故事\t徐秉龙\talbum_film\t[0 0 0 0 1]
    loader = myConllLoader()
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    char_vocab = Vocabulary()
    entity_vocab = Vocabulary()
    logging.info('dev instance:{}'.format(len(datasets['dev'])))
    logging.info('test instance:{}'.format(len(datasets['test'])))
    logging.info('train instance:{}'.format(len(datasets['train'])))

    char_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='left_context')
    char_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='right_context')
    char_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='raw_entity')

    entity_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='raw_entity_label')

    char_vocab.index_dataset(datasets['train'],
                             datasets['dev'],
                             datasets['test'],
                             field_name='left_context',
                             new_field_name='left_chars')
    char_vocab.index_dataset(datasets['train'],
                             datasets['dev'],
                             datasets['test'],
                             field_name='right_context',
                             new_field_name='right_chars')
    char_vocab.index_dataset(datasets['train'],
                             datasets['dev'],
                             datasets['test'],
                             field_name='raw_entity',
                             new_field_name='entity_chars')
    entity_vocab.index_dataset(datasets['train'],
                               datasets['dev'],
                               datasets['test'],
                               field_name='raw_entity_label',
                               new_field_name='entity_label')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['entity'] = entity_vocab

    embeddings = {}
    if char_embedding_path is not None:
        bi_voc = dict()
        for k, v in char_vocab:
            bi_voc[k] = v
        embed_weight = build_pretrain_embedding(char_embedding_path, bi_voc, embedd_dim=200)
        embeddings['char'] = embed_weight

    return datasets, vocabs, embeddings


def prepare_inputs(ip_step2_path, char_embedding_path):
    datasets, vocabs, embeddings = load_ip_step2(ip_step2_path, char_embedding_path)
    for k, v in datasets.items():
        v.set_input('left_chars', 'right_chars', 'entity_chars', 'entity_label', 'target')
        v.set_target('target')
    return datasets, vocabs, embeddings


if __name__ == '__main__':
    utils.configure_logging()
    datasets, vocabs, embeddings = prepare_inputs(ip_step2_path, char_embedding_path)
    logging.info('vocabs:{}'.format(vocabs))

    for k, v in vocabs.items():
        with open('./{}.dict'.format(k), 'w') as f:
            for (k1, v1) in v:
                f.write('{}\t{}\n'.format(k1, v1))

    with open('./test_file.txt', 'w') as f:
        for it in datasets['test']:
            f.write('{}\n'.format(' '.join(list(it['raw_chars']))))

    num = 2
    logging.info('left_context:{}'.format(list(datasets['train'][:num]['left_context'])))
    logging.info('right_context:{}'.format(list(datasets['train'][:num]['right_context'])))
    logging.info('raw_entity:{}'.format(list(datasets['train'][:num]['raw_entity'])))
    logging.info('raw_entity_label:{}'.format(list(datasets['train'][:num]['raw_entity_label'])))
    logging.info('target:{}'.format(list(datasets['train'][:num]['target'])))

    logging.info('left_chars:{}'.format(list(datasets['train'][:num]['left_chars'])))
    logging.info('right_chars:{}'.format(list(datasets['train'][:num]['right_chars'])))
    logging.info('entity_chars:{}'.format(list(datasets['train'][:num]['entity_chars'])))
    logging.info('entity_label:{}'.format(list(datasets['train'][:num]['entity_label'])))