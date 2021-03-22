import torch.nn.functional as F
import torch
import random
import numpy as np
from fastNLP import Const
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import Tester
import os
import math
import logging
import torch.nn as nn
from typing import Optional

def configure_logging(level=logging.INFO):
    format = '%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(process)d] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=level, format=format, datefmt=datefmt)

def load_pretrain_emb(embedding_path, embedd_dim):
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if not embedd_dim + 1 == len(tokens):
                continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim=100):
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, embedd_dim)
    vocab_size = len(word_vocab)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_vocab), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     embedding_path:%s, pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" %
          (embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb


def should_mask(name, t=''):
    if 'bias' in name:
        return False
    if 'embedding' in name:
        splited = name.split('.')
        if splited[-1] != 'weight':
            return False
        if 'embedding' in splited[-2]:
            return False
    if 'c0' in name:
        return False
    if 'h0' in name:
        return False

    if 'output' in name and t not in name:
        return False

    return True


def get_init_mask(model):
    init_masks = {}
    for name, param in model.named_parameters():
        if should_mask(name):
            init_masks[name + '.mask'] = torch.ones_like(param)
            # logger.info(init_masks[name+'.mask'].requires_grad)

    return init_masks


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 100)
    torch.manual_seed(seed + 200)
    torch.cuda.manual_seed_all(seed + 300)


def get_parameters_size(model):
    result = {}
    for name, p in model.state_dict().items():
        result[name] = p.size()

    return result


def prune_by_proportion_model(model, proportion, task):
    # print('this time prune to ',proportion*100,'%')
    for name, p in model.named_parameters():
        # print(name)
        if not should_mask(name, task):
            continue

        tensor = p.data.cpu().numpy()
        index = np.nonzero(model.mask[task][name + '.mask'].data.cpu().numpy())
        # print(name,'alive count',len(index[0]))
        alive = tensor[index]
        # print('p and mask size:',p.size(),print(model.mask[task][name+'.mask'].size()))
        percentile_value = np.percentile(abs(alive), (1 - proportion) * 100)
        # tensor = p
        # index = torch.nonzero(model.mask[task][name+'.mask'])
        # # print('nonzero len',index)
        # alive = tensor[index]
        # print('alive size:',alive.shape)
        # prune_by_proportion_model()

        # percentile_value = torch.topk(abs(alive), int((1-proportion)*len(index[0]))).values
        # print('the',(1-proportion)*len(index[0]),'th big')
        # print('threshold:',percentile_value)

        prune_by_threshold_parameter(p, model.mask[task][name + '.mask'], percentile_value)
        # for


def prune_by_proportion_model_global(model, proportion, task):
    # print('this time prune to ',proportion*100,'%')
    alive = None
    for name, p in model.named_parameters():
        # print(name)
        if not should_mask(name, task):
            continue

        tensor = p.data.cpu().numpy()
        index = np.nonzero(model.mask[task][name + '.mask'].data.cpu().numpy())
        # print(name,'alive count',len(index[0]))
        if alive is None:
            alive = tensor[index]
        else:
            alive = np.concatenate([alive, tensor[index]], axis=0)

    percentile_value = np.percentile(abs(alive), (1 - proportion) * 100)

    for name, p in model.named_parameters():
        if should_mask(name, task):
            prune_by_threshold_parameter(p, model.mask[task][name + '.mask'], percentile_value)


def prune_by_threshold_parameter(p, mask, threshold):
    p_abs = torch.abs(p)

    new_mask = (p_abs > threshold).float()
    # print(mask)
    mask[:] *= new_mask


def one_time_train_and_prune_single_task(
        trainer,
        PRUNE_PER,
        optimizer_init_state_dict=None,
        model_init_state_dict=None,
        is_global=None,
):

    from fastNLP import Trainer

    trainer.optimizer.load_state_dict(optimizer_init_state_dict)
    trainer.model.load_state_dict(model_init_state_dict)
    # print('metrics:',metrics.__dict__)
    # print('loss:',loss.__dict__)
    # print('trainer input:',task.train_set.get_input_name())
    # trainer = Trainer(model=model, train_data=task.train_set, dev_data=task.dev_set, loss=loss, metrics=metrics,
    #                   optimizer=optimizer, n_epochs=EPOCH, batch_size=BATCH, device=device,callbacks=callbacks)

    trainer.train(load_best_model=True)
    # tester = Tester(task.train_set, model, metrics, BATCH, device=device, verbose=1,use_tqdm=False)
    # print('FOR DEBUG: test train_set:',tester.test())
    # print('**'*20)
    # if task.test_set:
    #     tester = Tester(task.test_set, model, metrics, BATCH, device=device, verbose=1)
    #     tester.test()
    if is_global:

        prune_by_proportion_model_global(trainer.model, PRUNE_PER, trainer.model.now_task)

    else:
        prune_by_proportion_model(trainer.model, PRUNE_PER, trainer.model.now_task)


# def iterative_train_and_prune_single_task(get_trainer,ITER,PRUNE,is_global=False,save_path=None):
def iterative_train_and_prune_single_task(get_trainer,
                                          args,
                                          model,
                                          train_set,
                                          dev_set,
                                          test_set,
                                          device,
                                          save_path=None):
    '''

    :param trainer:
    :param ITER:
    :param PRUNE:
    :param is_global:
    :param save_path: should be a dictionary which will be filled with mask and state dict
    :return:
    '''

    from fastNLP import Trainer
    import torch
    import math
    import copy
    PRUNE = args.prune
    ITER = args.iter
    trainer = get_trainer(args, model, train_set, dev_set, test_set, device)
    optimizer_init_state_dict = copy.deepcopy(trainer.optimizer.state_dict())
    model_init_state_dict = copy.deepcopy(trainer.model.state_dict())
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # if not os.path.exists(os.path.join(save_path, 'model_init.pkl')):
        #     f = open(os.path.join(save_path, 'model_init.pkl'), 'wb')
        #     torch.save(trainer.model.state_dict(),f)

    mask_count = 0
    model = trainer.model
    task = trainer.model.now_task
    for name, p in model.mask[task].items():
        mask_count += torch.sum(p).item()
    init_mask_count = mask_count
    logger.info('init mask count:{}'.format(mask_count))
    # logger.info('{}th traning mask count: {} / {} = {}%'.format(i, mask_count, init_mask_count,
    #                                                             mask_count / init_mask_count * 100))

    prune_per_iter = math.pow(PRUNE, 1 / ITER)

    for i in range(ITER):
        trainer = get_trainer(args, model, train_set, dev_set, test_set, device)
        one_time_train_and_prune_single_task(trainer, prune_per_iter, optimizer_init_state_dict, model_init_state_dict)
        if save_path is not None:
            f = open(os.path.join(save_path, task + '_mask_' + str(i) + '.pkl'), 'wb')
            torch.save(model.mask[task], f)

        mask_count = 0
        for name, p in model.mask[task].items():
            mask_count += torch.sum(p).item()
        logger.info('{}th traning mask count: {} / {} = {}%'.format(i, mask_count, init_mask_count,
                                                                    mask_count / init_mask_count * 100))


def get_appropriate_cuda(task_scale='s'):
    if task_scale not in {'s', 'm', 'l'}:
        logger.info('task scale wrong!')
        exit(2)
    import pynvml
    pynvml.nvmlInit()
    total_cuda_num = pynvml.nvmlDeviceGetCount()
    for i in range(total_cuda_num):
        logger.info(i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilizationInfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        logger.info(i, 'mem:', memInfo.used / memInfo.total, 'util:', utilizationInfo.gpu)
        if memInfo.used / memInfo.total < 0.15 and utilizationInfo.gpu < 0.2:
            logger.info(i, memInfo.used / memInfo.total)
            return 'cuda:' + str(i)

    if task_scale == 's':
        max_memory = 2000
    elif task_scale == 'm':
        max_memory = 6000
    else:
        max_memory = 9000

    max_id = -1
    for i in range(total_cuda_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilizationInfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if max_memory < memInfo.free:
            max_memory = memInfo.free
            max_id = i

    if id == -1:
        logger.info('no appropriate gpu, wait!')
        exit(2)

    return 'cuda:' + str(max_id)

    # if memInfo.used / memInfo.total < 0.5:
    #     return


def print_mask(mask_dict):

    def seq_mul(*X):
        res = 1
        for x in X:
            res *= x
        return res

    for name, p in mask_dict.items():
        total_size = seq_mul(*p.size())
        unmasked_size = len(np.nonzero(p))

        print(name, ':', unmasked_size, '/', total_size, '=', unmasked_size / total_size * 100, '%')

    print()


def check_words_same(dataset_1, dataset_2, field_1, field_2):
    if len(dataset_1[field_1]) != len(dataset_2[field_2]):
        logger.info('CHECK: example num not same!')
        return False

    for i, words in enumerate(dataset_1[field_1]):
        if len(dataset_1[field_1][i]) != len(dataset_2[field_2][i]):
            logger.info('CHECK {} th example length not same'.format(i))
            logger.info('1:{}'.format(dataset_1[field_1][i]))
            logger.info('2:'.format(dataset_2[field_2][i]))
            return False

        # for j,w in enumerate(words):
        #     if dataset_1[field_1][i][j] != dataset_2[field_2][i][j]:
        #         print('CHECK', i, 'th example has words different!')
        #         print('1:',dataset_1[field_1][i])
        #         print('2:',dataset_2[field_2][i])
        #         return False

    logger.info('CHECK: totally same!')

    return True


def get_now_time():
    import time
    from datetime import datetime, timezone, timedelta
    dt = datetime.utcnow()
    # print(dt)
    tzutc_8 = timezone(timedelta(hours=8))
    local_dt = dt.astimezone(tzutc_8)
    result = ("_{}_{}_{}__{}_{}_{}".format(local_dt.year, local_dt.month, local_dt.day, local_dt.hour, local_dt.minute,
                                           local_dt.second))

    return result


def get_bigrams(words):
    result = []
    for i, w in enumerate(words):
        if i != len(words) - 1:
            result.append(words[i] + words[i + 1])
        else:
            result.append(words[i] + '<end>')

    return result

def seq_len_to_rel_distance(max_seq_len, dvc):
    '''
    :param seq_len: seq_len batch
    :return: L*L rel_distance
    '''
    index = torch.arange(0, max_seq_len)
    assert index.size(0) == max_seq_len
    assert index.dim() == 1
    index = index.repeat(max_seq_len, 1)
    offset = torch.arange(0, max_seq_len).unsqueeze(1)
    offset = offset.repeat(1, max_seq_len)
    index = index - offset
    index = index.to(dvc)
    return index

def better_init_rnn(rnn, coupled=False):
    import torch.nn as nn
    if coupled:
        repeat_size = 3
    else:
        repeat_size = 4
    # print(list(rnn.named_parameters()))
    if hasattr(rnn, 'num_layers'):
        for i in range(rnn.num_layers):
            nn.init.orthogonal_(getattr(rnn, 'weight_ih_l' + str(i)).data)
            weight_hh_data = torch.eye(rnn.hidden_size)
            weight_hh_data = weight_hh_data.repeat(1, repeat_size)
            with torch.no_grad():
                getattr(rnn, 'weight_hh_l' + str(i)).set_(weight_hh_data)
            nn.init.constant_(getattr(rnn, 'bias_ih_l' + str(i)).data, val=0)
            nn.init.constant_(getattr(rnn, 'bias_hh_l' + str(i)).data, val=0)

        if rnn.bidirectional:
            for i in range(rnn.num_layers):
                nn.init.orthogonal_(getattr(rnn, 'weight_ih_l' + str(i) + '_reverse').data)
                weight_hh_data = torch.eye(rnn.hidden_size)
                weight_hh_data = weight_hh_data.repeat(1, repeat_size)
                with torch.no_grad():
                    getattr(rnn, 'weight_hh_l' + str(i) + '_reverse').set_(weight_hh_data)
                nn.init.constant_(getattr(rnn, 'bias_ih_l' + str(i) + '_reverse').data, val=0)
                nn.init.constant_(getattr(rnn, 'bias_hh_l' + str(i) + '_reverse').data, val=0)

    else:
        nn.init.orthogonal_(rnn.weight_ih.data)
        weight_hh_data = torch.eye(rnn.hidden_size)
        weight_hh_data = weight_hh_data.repeat(repeat_size, 1)
        with torch.no_grad():
            rnn.weight_hh.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        print('rnn param size:{},{}'.format(rnn.weight_hh.size(), type(rnn)))
        if rnn.bias:
            nn.init.constant_(rnn.bias_ih.data, val=0)
            nn.init.constant_(rnn.bias_hh.data, val=0)

    # print(list(rnn.named_parameters()))


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None, initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf

# def seq_len_to_mask(seq_len:torch.Tensor, max_len:Optional[int]):
#     r"""

#     将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
#     转变 1-d seq_len到2-d mask.

#     .. code-block::

#         >>> seq_len = torch.arange(2, 16)
#         >>> mask = seq_len_to_mask(seq_len)
#         >>> print(mask.size())
#         torch.Size([14, 15])
#         >>> seq_len = np.arange(2, 16)
#         >>> mask = seq_len_to_mask(seq_len)
#         >>> print(mask.shape)
#         (14, 15)
#         >>> seq_len = torch.arange(2, 16)
#         >>> mask = seq_len_to_mask(seq_len, max_len=100)
#         >>>print(mask.size())
#         torch.Size([14, 100])

#     :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
#     :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
#         区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
#     :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
#     """
#     # if isinstance(seq_len, np.ndarray):
#     #     assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
#     #     max_len = int(max_len) if max_len else int(seq_len.max())
#     #     broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
#     #     mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

#     # elif isinstance(seq_len, torch.Tensor):
#     #     assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
#     #     batch_size = seq_len.size(0)
#     #     max_len = int(max_len) if max_len else seq_len.max().item()
#     #     # logging.info('max_len={}'.format(max_len))

#     #     broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
#     #     mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
#     # else:
#     #     raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

#     batch_size = seq_len.size(0)
#     max_len = int(max_len) if max_len else seq_len.max().item()
#     broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
#     mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))

#     return mask

def seq_len_to_mask(seq_len:torch.Tensor,max_len:Optional[int]=None):
    batch_size = seq_len.size(0)
    max_len = int(max_len) if max_len is not None else seq_len.max().item()
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    return mask

def get_peking_time():
    import time
    import datetime
    import pytz

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y_%m_%d_%H_%M_%S_%f')
    return t


def norm_static_embedding(x, norm=1):
    with torch.no_grad():
        x.embedding.weight /= (torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm

def norm_static_embedding_v2(x, norm=1):
    with torch.no_grad():
        x /= (torch.norm(x, dim=1, keepdim=True) + 1e-12)
        x *= norm


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'.format(model._get_name(),
                                                                                total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'.format(model._get_name(),
                                                                             total_nums * type_size * 2 / 1000 / 1000))


def size2MB(size_, type_size=4):
    num = 1
    for s in size_:
        num *= s

    return num * type_size / 1000 / 1000


def get_pos_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化
    如果是1，那么从-max_len到max_len的相对位置编码矩阵就按-max_len,max_len来初始化
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(2 * max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len, max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(2 * max_seq_len + 1, -1)
    if embedding_dim % 2 == 1: emb = torch.cat([emb, torch.zeros(2 * max_seq_len + 1, 1)], dim=1)
    if padding_idx is not None: emb[padding_idx, :] = 0
    return emb


import collections
from fastNLP import cache_results
def get_skip_path(chars,w_trie):
    sentence = ''.join(chars)
    result = w_trie.get_lexicon(sentence)

    return result

# @cache_results(_cache_fp='cache/get_skip_path_trivial',_refresh=True)
def get_skip_path_trivial(chars,w_list):
    chars = ''.join(chars)
    w_set = set(w_list)
    result = []
    # for i in range(len(chars)):
    #     result.append([])
    for i in range(len(chars)-1):
        for j in range(i+2,len(chars)+1):
            if chars[i:j] in w_set:
                result.append([i,j-1,chars[i:j]])

    return result


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        '''

        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self,sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j,sentence[i:j+1]])

        return result

from fastNLP.core.field import Padder
import numpy as np
import torch
from collections import defaultdict
class LatticeLexiconPadder(Padder):

    def __init__(self, pad_val=0, pad_val_dynamic=False,dynamic_offset=0, **kwargs):
        '''

        :param pad_val:
        :param pad_val_dynamic: if True, pad_val is the seq_len
        :param kwargs:
        '''
        self.pad_val = pad_val
        self.pad_val_dynamic = pad_val_dynamic
        self.dynamic_offset = dynamic_offset

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # 与autoPadder中 dim=2 的情况一样
        max_len = max(map(len, contents))

        max_len = max(max_len,1)#avoid 0 size dim which causes cuda wrong

        max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                            content_i in contents])

        max_word_len = max(max_word_len,1)
        if self.pad_val_dynamic:
            # print('pad_val_dynamic:{}'.format(max_len-1))

            array = np.full((len(contents), max_len, max_word_len), max_len-1+self.dynamic_offset,
                            dtype=field_ele_dtype)

        else:
            array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
        for i, content_i in enumerate(contents):
            for j, content_ii in enumerate(content_i):
                array[i, j, :len(content_ii)] = content_ii
        array = torch.tensor(array)

        return array

from fastNLP.core.metrics import MetricBase

def get_yangjie_bmeso(label_list,ignore_labels=None):
    def get_ner_BMESO_yj(label_list):
        def reverse_style(input_string):
            target_position = input_string.index('[')
            input_len = len(input_string)
            output_string = input_string[target_position:input_len] + input_string[0:target_position]
            # print('in:{}.out:{}'.format(input_string, output_string))
            return output_string

        # list_len = len(word_list)
        # assert(list_len == len(label_list)), "word list size unmatch with label list"
        list_len = len(label_list)
        begin_label = 'b-'
        end_label = 'e-'
        single_label = 's-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = label_list[i].lower()
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        # print stand_matrix
        return stand_matrix

    def transform_YJ_to_fastNLP(span):
        span = span[1:]
        span_split = span.split(']')
        # print('span_list:{}'.format(span_split))
        span_type = span_split[1]
        # print('span_split[0].split(','):{}'.format(span_split[0].split(',')))
        if ',' in span_split[0]:
            b, e = span_split[0].split(',')
        else:
            b = span_split[0]
            e = b

        b = int(b)
        e = int(e)

        e += 1

        return (span_type, (b, e))
    yj_form = get_ner_BMESO_yj(label_list)
    # print('label_list:{}'.format(label_list))
    # print('yj_from:{}'.format(yj_form))
    fastNLP_form = list(map(transform_YJ_to_fastNLP,yj_form))
    return fastNLP_form

class SpanFPreRecMetric_YJ(MetricBase):
    r"""
    别名：:class:`fastNLP.SpanFPreRecMetric` :class:`fastNLP.core.metrics.SpanFPreRecMetric`

    在序列标注问题中，以span的方式计算F, pre, rec.
    比如中文Part of speech中，会以character的方式进行标注，句子 `中国在亚洲` 对应的POS可能为(以BMES为例)
    ['B-NN', 'E-NN', 'S-DET', 'B-NN', 'E-NN']。该metric就是为类似情况下的F1计算。
    最后得到的metric结果为::

        {
            'f': xxx, # 这里使用f考虑以后可以计算f_beta值
            'pre': xxx,
            'rec':xxx
        }

    若only_gross=False, 即还会返回各个label的metric统计值::

        {
            'f': xxx,
            'pre': xxx,
            'rec':xxx,
            'f-label': xxx,
            'pre-label': xxx,
            'rec-label':xxx,
            ...
        }

    :param tag_vocab: 标签的 :class:`~fastNLP.Vocabulary` 。支持的标签为"B"(没有label)；或"B-xxx"(xxx为某种label，比如POS中的NN)，
        在解码时，会将相同xxx的认为是同一个label，比如['B-NN', 'E-NN']会被合并为一个'NN'.
    :param str pred: 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用 `pred` 取数据
    :param str target: 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用 `target` 取数据
    :param str seq_len: 用该key在evaluate()时从传入dict中取出sequence length数据。为None，则使用 `seq_len` 取数据。
    :param str encoding_type: 目前支持bio, bmes, bmeso, bioes
    :param list ignore_labels: str 组成的list. 这个list中的class不会被用于计算。例如在POS tagging时传入['NN']，则不会计算'NN'这
        个label
    :param bool only_gross: 是否只计算总的f1, precision, recall的值；如果为False，不仅返回总的f1, pre, rec, 还会返回每个
        label的f1, pre, rec
    :param str f_type: `micro` 或 `macro` . `micro` :通过先计算总体的TP，FN和FP的数量，再计算f, precision, recall; `macro` :
        分布计算每个类别的f, precision, recall，然后做平均（各类别f的权重相同）
    :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` .
        常用为beta=0.5, 1, 2. 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
    """
    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type='bio', ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        from fastNLP.core import Vocabulary
        from fastNLP.core.metrics import _bmes_tag_to_spans,_bio_tag_to_spans,\
            _bioes_tag_to_spans,_bmeso_tag_to_spans
        from collections import defaultdict

        encoding_type = encoding_type.lower()

        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        self.encoding_type = encoding_type
        # print('encoding_type:{}'self.encoding_type)
        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        elif self.encoding_type == 'bmesoyj':
            self.tag_to_span_func = get_yangjie_bmeso
            # self.tag_to_span_func =
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len):
        from fastNLP.core.utils import _get_func_signature
        """evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param pred: [batch, seq_len] 或者 [batch, seq_len, len(tag_vocab)], 预测的结果
        :param target: [batch, seq_len], 真实值
        :param seq_len: [batch] 文本长度标记
        :return:
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        """get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = self._compute_f_pre_rec(sum(self._true_positives.values()),
                                                  sum(self._false_negatives.values()),
                                                  sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):
        """

        :param tp: int, true positive
        :param fn: int, false negative
        :param fp: int, false positive
        :return: (f, pre, rec)
        """
        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre + rec + 1e-13)

        return f, pre, rec

if __name__ == '__main__':
    # a = get_peking_time()
    a = get_pos_embedding(1,4)
    print(a)
    print(a.view(-1))

    a_sum = a.sum(dim=-1, keepdim=True)
    print(a_sum)

    a = a / a_sum
    print(a)

    a = a.unsqueeze(0)
    print(a.dim())
