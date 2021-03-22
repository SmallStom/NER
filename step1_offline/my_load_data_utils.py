from fastNLP.io import Loader
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary
from paths import *


def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()
        if start != '':
            if start.startswith('B') or start.startswith('E'):
                pass
            else:
                sample.append(start.split(sep)) if sep else sample.append(start.split())
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith('B') or line.startswith('E'):
                continue
            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        yield line_idx, res
                    except Exception as e:
                        if dropna:
                            logger.warning('Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            continue
                        raise ValueError('Invalid instance which ends at line: {}'.format(line_idx))
            elif line.startswith('#'):
                continue
            else:
                sample.append(line.split(sep)) if sep else sample.append(line.split())
        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                yield line_idx, res
            except Exception as e:
                if dropna:
                    return
                logger.error('invalid instance ends at line: {}'.format(line_idx))
                raise e


class myConllLoader(Loader):

    def __init__(self, headers, sep=None, indexes=None, dropna=True):
        super(myConllLoader, self).__init__()
        if not isinstance(headers, (list, tuple)):
            raise TypeError('invalid headers: {}, should be list of strings'.format(headers))
        self.headers = headers
        self.dropna = dropna
        self.sep = sep
        if indexes is None:
            self.indexes = list(range(len(self.headers)))
        else:
            if len(indexes) != len(headers):
                raise ValueError
            self.indexes = indexes

    def _load(self, path):
        ds = DataSet()
        for idx, data in _read_conll(path, sep=self.sep, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds


# 第一步，定义具体的loader读取对应格式的输入数据,(如果没有内置已实现的loader，则实现自定义的loader类，具体方法：
# instance是采用dict的形式存放field信息，代表一个具体的sample语料
# dataset里面存放instance
#
# loader = myConllLoader(headers=['raw_words', 'ner'], indexes=[0, 1])
# paths = {
#     'train': "/Users/wangming/.fastNLP/dataset/weibo_NER/train.conll",
#     'dev': "/Users/wangming/.fastNLP/dataset/weibo_NER/dev.conll",
#     "test": "/Users/wangming/.fastNLP/dataset/weibo_NER/test.conll"
# }
# datasets = loader.load(paths).datasets
# # print(*list(datasets.keys()))
# # print(datasets['train'][0:3])

# word_vocab = Vocabulary()
# label_vocab = Vocabulary(padding=None, unknown=None)

# word_vocab.from_dataset(datasets['train'],
#                         field_name='raw_words',
#                         no_create_entry_dataset=[datasets['dev'], datasets['test']])
# label_vocab.from_dataset(datasets['train'], field_name='ner')
# print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))

# word_vocab.index_dataset(*list(datasets.values()), field_name='raw_words', new_field_name='raw_words_index')
# label_vocab.index_dataset(*list(datasets.values()), field_name='ner', new_field_name='ner_index')

# print(datasets['train'][0:3])
# print(word_vocab.idx2word[791])