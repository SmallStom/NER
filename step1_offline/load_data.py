from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
import logging
import utils
from utils import Trie, build_pretrain_embedding
from functools import partial
import numpy as np
import pickle
import os
from fastNLP import cache_results
from fastNLP.embeddings import StaticEmbedding
from paths import *

import torch.nn.init


@cache_results(_cache_fp='cache/ip_step1', _refresh=False)
def load_ip_step1(path,
                  char_embedding_path=None,
                  bigram_embedding_path=None,
                  char_min_freq=1,
                  bigram_min_freq=1,
                  only_train_min_freq=False):
    # from fastNLP.io.loader import ConllLoader
    from my_load_data_utils import myConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path, 'train_crfpp_char.txt')
    dev_path = os.path.join(path, 'dev_crfpp_char.txt')
    test_path = os.path.join(path, 'test_crfpp_char.txt')

    # loader = ConllLoader(['chars','target'])
    loader = myConllLoader(headers=['raw_chars', 'raw_entity', 'raw_target'], indexes=[1, 3, 4], sep='\t')
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='raw_chars', new_field_name='raw_bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='raw_chars', new_field_name='raw_bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='raw_chars', new_field_name='raw_bigrams')

    datasets['train'].add_seq_len('raw_chars')
    datasets['dev'].add_seq_len('raw_chars')
    datasets['test'].add_seq_len('raw_chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    entity_vocab = Vocabulary()
    logging.info(datasets.keys())
    logging.info('dev instance:{}'.format(len(datasets['dev'])))
    logging.info('test instance:{}'.format(len(datasets['test'])))
    logging.info('train instance:{}'.format(len(datasets['train'])))

    char_vocab.from_dataset(datasets['train'],
                            field_name='raw_chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],
                              field_name='raw_bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='raw_target')
    entity_vocab.from_dataset(datasets['train'], field_name='raw_entity')

    char_vocab.index_dataset(datasets['train'],
                             datasets['dev'],
                             datasets['test'],
                             field_name='raw_chars',
                             new_field_name='chars')
    bigram_vocab.index_dataset(datasets['train'],
                               datasets['dev'],
                               datasets['test'],
                               field_name='raw_bigrams',
                               new_field_name='bigrams')
    label_vocab.index_dataset(datasets['train'],
                              datasets['dev'],
                              datasets['test'],
                              field_name='raw_target',
                              new_field_name='target')
    entity_vocab.index_dataset(datasets['train'],
                               datasets['dev'],
                               datasets['test'],
                               field_name='raw_entity',
                               new_field_name='entity')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['entity'] = entity_vocab

    embeddings = {}
    if bigram_embedding_path is not None:
        bi_voc = dict()
        for k, v in bigram_vocab:
            bi_voc[k] = v
        bigram_embed_weight = build_pretrain_embedding(bigram_embedding_path, bi_voc, embedd_dim=50)
        embeddings['bigram'] = bigram_embed_weight
    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list', _refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))

    return w_list


@cache_results(_cache_fp='cache/add_chinese_ner_with_lexicon', _refresh=True)
def add_chinese_ner_with_lexicon(datasets,
                                 vocabs,
                                 embeddings,
                                 w_list,
                                 word_embedding_path=None,
                                 word_char_mix_embedding_path=None,
                                 lattice_min_freq=1,
                                 only_train_min_freq=0):

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        return result

    def concat(ins):
        chars = ins['raw_chars']
        lexicons = ins['raw_lexicons']
        result = chars + list(map(lambda x: x[2], lexicons))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s
        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e
        return pos_e

    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)
    import copy
    for k, v in datasets.items():
        v.apply_field(partial(get_skip_path, w_trie=w_trie), 'raw_chars', 'raw_lexicons')
        v.add_seq_len('raw_lexicons', 'lex_num')
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'raw_lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'raw_lexicons', 'lex_e')
        v.apply(concat, new_field_name='lattice')
        v.set_input('lattice')
        v.apply_field(copy.deepcopy, 'lattice', 'raw_lattice')
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s', 'pos_e')
        v.set_input('entity')

    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'],
                               field_name='lattice',
                               no_create_entry_dataset=[v for k, v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab

    if word_char_mix_embedding_path is not None:
        la_voc = dict()
        for k, v in lattice_vocab:
            la_voc[k] = v

        lattice_embed_weight = build_pretrain_embedding(word_char_mix_embedding_path, la_voc, embedd_dim=50)
        embeddings['lattice'] = lattice_embed_weight
    vocabs['lattice'].index_dataset(*(datasets.values()), field_name='lattice', new_field_name='lattice')
    return datasets, vocabs, embeddings


def prepare_inputs(ip_step1_path, yangjie_rich_pretrain_unigram_path, yangjie_rich_pretrain_bigram_path):
    datasets, vocabs, embeddings = load_ip_step1(ip_step1_path, yangjie_rich_pretrain_unigram_path,
                                                 yangjie_rich_pretrain_bigram_path, 1, 1, True)
    w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path)
    datasets, vocabs, embeddings = add_chinese_ner_with_lexicon(datasets, vocabs, embeddings, w_list,
                                                                yangjie_rich_pretrain_word_path,
                                                                yangjie_rich_pretrain_char_and_word_path, 1, 1)
    for k, v in datasets.items():
        v.set_input('lattice', 'bigrams', 'seq_len', 'target', 'lex_num', 'pos_s', 'pos_e')
        v.set_target('target', 'seq_len')
    return datasets, vocabs, embeddings


if __name__ == '__main__':
    utils.configure_logging()
    datasets, vocabs, embeddings = prepare_inputs(ip_step1_path, yangjie_rich_pretrain_unigram_path,
                                                  yangjie_rich_pretrain_bigram_path)
    logging.info('vocabs:{}'.format(vocabs))
    logging.info('lattice embeddings:{}'.format(embeddings['lattice']))

    # for k, v in vocabs.items():
    #     with open('./{}.dict'.format(k), 'w') as f:
    #         for (k1, v1) in v:
    #             f.write('{}\t{}\n'.format(k1, v1))

    with open('./test.txt', 'w') as f:
        for it in datasets['test']:
            f.write('{}\n'.format(' '.join(list(it['raw_chars']))))

    logging.info('raw_chars:{}'.format(list(datasets['train'][:5]['raw_chars'])))
    logging.info('chars:{}'.format(list(datasets['train'][:5]['chars'])))
    logging.info('seq_len:{}'.format(list(datasets['train'][:5]['seq_len'])))
    logging.info('lex_num:{}'.format(list(datasets['train'][:5]['lex_num'])))
    logging.info('raw_entity:{}'.format(list(datasets['train'][:5]['raw_entity'])))
    logging.info('entity:{}'.format(list(datasets['train'][:5]['entity'])))
    logging.info('raw_target:{}'.format(list(datasets['train'][:5]['raw_target'])))
    logging.info('target:{}'.format(list(datasets['train'][:5]['target'])))
    logging.info('raw_bigrams:{}'.format(list(datasets['train'][:5]['raw_bigrams'])))
    logging.info('bigrams:{}'.format(list(datasets['train'][:5]['bigrams'])))
    logging.info('raw_lattice:{}'.format(list(datasets['train'][:5]['raw_lattice'])))
    logging.info('lattice:{}'.format(list(datasets['train'][:5]['lattice'])))
    logging.info('raw_lexicons:{}'.format(list(datasets['train'][:5]['raw_lexicons'])))
    logging.info('lex_s:{}'.format(list(datasets['train'][:5]['lex_s'])))
    logging.info('lex_e:{}'.format(list(datasets['train'][:5]['lex_e'])))
    logging.info('pos_s:{}'.format(list(datasets['train'][:5]['pos_s'])))
    logging.info('pos_e:{}'.format(list(datasets['train'][:5]['pos_e'])))
    # 输入model的是bigrams，lattice，pos_s，pos_e三个字段

    # 打印数据细节
    avg_seq_len = 0
    avg_lex_num = 0
    avg_seq_lex = 0
    train_seq_lex = []
    dev_seq_lex = []
    test_seq_lex = []
    train_seq = []
    dev_seq = []
    test_seq = []
    for k, v in datasets.items():
        max_seq_len = 0
        max_lex_num = 0
        max_seq_lex = 0
        max_seq_len_i = -1
        for i in range(len(v)):
            if max_seq_len < v[i]['seq_len']:
                max_seq_len = v[i]['seq_len']
                max_seq_len_i = i
            max_seq_len = max(max_seq_len, v[i]['seq_len'])
            max_lex_num = max(max_lex_num, v[i]['lex_num'])
            max_seq_lex = max(max_seq_lex, v[i]['lex_num'] + v[i]['seq_len'])

            avg_seq_len += v[i]['seq_len']
            avg_lex_num += v[i]['lex_num']
            avg_seq_lex += (v[i]['seq_len'] + v[i]['lex_num'])
            if k == 'train':
                train_seq_lex.append(v[i]['lex_num'] + v[i]['seq_len'])
                train_seq.append(v[i]['seq_len'])
                if v[i]['seq_len'] > 200:
                    logging.info('train里这个句子char长度已经超了200了')
                    logging.info(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
                else:
                    if v[i]['seq_len'] + v[i]['lex_num'] > 400:
                        logging.info('train里这个句子char长度没超200，但是总长度超了400')
                        logging.info(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
            if k == 'dev':
                dev_seq_lex.append(v[i]['lex_num'] + v[i]['seq_len'])
                dev_seq.append(v[i]['seq_len'])
            if k == 'test':
                test_seq_lex.append(v[i]['lex_num'] + v[i]['seq_len'])
                test_seq.append(v[i]['seq_len'])

        logging.info('{} 最长的句子是:{}'.format(k, list(map(lambda x: vocabs['char'].to_word(x),
                                                       v[max_seq_len_i]['chars']))))
        logging.info('{} max_seq_len:{}'.format(k, max_seq_len))
        logging.info('{} max_lex_num:{}'.format(k, max_lex_num))
        logging.info('{} max_seq_lex:{}'.format(k, max_seq_lex))