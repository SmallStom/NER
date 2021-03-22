import torch
import torch.nn as nn
import torch.nn.init as init

import logging

def initial_parameter(net, initial_method=None):
    r"""A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    net.apply(weights_init)

def my_1d_tolist(x):
    result: List[float] = []
    for i in x:
        result.append(i.item())
    return result

@torch.jit.script
def my_2d_tolist(x):
    result: List[List[float]] = []
    for i in x:
        result.append(my_1d_tolist(i))
    return result

class ConditionalRandomField(nn.Module):

    def __init__(self, num_tags, include_start_end_trans=False, allowed_transitions=None, initial_method=None):
        r"""

        :param int num_tags: 标签的数量
        :param bool include_start_end_trans: 是否考虑各个tag作为开始以及结尾的分数。
        :param List[Tuple[from_tag_id(int), to_tag_id(int)]] allowed_transitions: 内部的Tuple[from_tag_id(int),
                                   to_tag_id(int)]视为允许发生的跃迁，其他没有包含的跃迁认为是禁止跃迁，可以通过
                                   allowed_transitions()函数得到；如果为None，则所有跃迁均为合法
        :param str initial_method: 初始化方法。见initial_parameter
        """
        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.full((num_tags + 2, num_tags + 2), fill_value=-10000.0, dtype=torch.float)
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0
        self._constrain = nn.Parameter(constrain, requires_grad=False)

        initial_parameter(self, initial_method)

    def _normalizer_likelihood(self, logits, mask):
        r"""Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.

        :param logits:FloatTensor, max_len x batch_size x num_tags
        :param mask:ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha = alpha + self.start_scores.view(1, -1)

        flip_mask = mask.eq(int(False))

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = torch.logsumexp(tmp, 1).masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                    alpha.masked_fill(mask[i].eq(int(True)).view(batch_size, 1), 0)

        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return torch.logsumexp(alpha, 1)

    def _gold_score(self, logits, tags, mask):
        r"""
        Compute the score for the gold path.
        :param logits: FloatTensor, max_len x batch_size x num_tags
        :param tags: LongTensor, max_len x batch_size
        :param mask: ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_socre [L-1, B]
        mask = mask.eq(int(True))
        flip_mask = mask.eq(int(False))
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]].masked_fill(flip_mask[1:, :], 0)
        # emit_score [L, B]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags].masked_fill(flip_mask, 0)
        # score [L-1, B]
        score = trans_score + emit_score[:seq_len - 1, :]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]
            score = score + st_scores + ed_scores
        # return [B,]
        return score

    def forward(self, feats, tags, mask):
        r"""
        用于计算CRF的前向loss，返回值为一个batch_size的FloatTensor，可能需要mean()求得loss。

        :param torch.FloatTensor feats: batch_size x max_len x num_tags，特征矩阵。
        :param torch.LongTensor tags: batch_size x max_len，标签矩阵。
        :param torch.ByteTensor mask: batch_size x max_len，为0的位置认为是padding。
        :return: torch.FloatTensor, (batch_size,)
        """
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._gold_score(feats, tags, mask)

        return all_path_score - gold_path_score

    # def viterbi_decode(self, logits, mask, unpad=False):
    def viterbi_decode(self, logits, mask):
        r"""给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数

        :param torch.FloatTensor logits: batch_size x max_len x num_tags，特征矩阵。
        :param torch.ByteTensor mask: batch_size x max_len, 为0的位置认为是pad；如果为None，则认为没有padding。
        :param bool unpad: 是否将结果删去padding。False, 返回的是batch_size x max_len的tensor; True，返回的是
            List[List[int]], 内部的List[int]为每个sequence的label，已经除去pad部分，即每个List[int]的长度是这
            个sample的有效长度。
        :return: 返回 (paths, scores)。
                    paths: 是解码后的路径, 其值参照unpad参数.
                    scores: torch.FloatTensor, size为(batch_size,), 对应每个最优路径的分数。

        """
        batch_size, seq, n_tags = logits.size()
        logits = logits.transpose(0, 1).data  # L, B, H
        # mask = mask.transpose(0, 1).data.eq(True)  # L, B
        mask = mask.transpose(0, 1)

        # dp
        vpath = logits.new_zeros((seq, batch_size, n_tags), dtype=torch.long)
        vscore = logits[0]
        transitions = self._constrain.data.clone()
        transitions[:n_tags, :n_tags] += self.trans_m.data
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data
            transitions[:n_tags, n_tags + 1] += self.end_scores.data

        vscore += transitions[n_tags, :n_tags]
        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
        for i in range(1, seq):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = logits[i].view(batch_size, 1, n_tags)
            score = prev_score + trans_score + cur_score
            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            vscore = best_score.masked_fill(mask[i].eq(int(False)).view(batch_size, 1), 0) + \
                     vscore.masked_fill(mask[i].view(batch_size, 1), 0)

        if self.include_start_end_trans:
            vscore += transitions[:n_tags, n_tags + 1].view(1, -1)

        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq, dtype=torch.long, device=logits.device)
        lens = (mask.long().sum(0) - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % seq

        ans = logits.new_empty((seq, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(seq - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        paths = ans
        # logging.info('ans={}'.format(ans))
        # if unpad:
        #     paths = []
        #     for idx, seq in enumerate(lens):
        #         # paths.append(ans[idx, :seq + 1].tolist())
        #         logging.info(ans[idx, :seq + 1])
        #         paths.append(my_1d_tolist(ans[idx, :seq + 1]))
        # else:
        #     paths = ans
        return paths, ans_score