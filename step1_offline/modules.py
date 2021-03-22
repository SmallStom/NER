import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import logging

from utils import seq_len_to_mask

class Four_Pos_Fusion_Embedding(nn.Module):

    def __init__(self, pe_ss, pe_se, pe_es, pe_ee, max_seq_len, hidden_size):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.f1 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        # B prepare relative position encoding
        max_seq_len = pos_s.size(1)
        pe_ss = self.pe_ss[(pos_ss).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_se = self.pe_se[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        pe_2 = torch.cat([pe_ss, pe_ee], dim=-1)
        rel_pos_embedding = F.relu(self.f1(pe_2))

        return rel_pos_embedding


class MultiHead_Attention_Lattice_rel(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 attn_dropout=None):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        # self.max_seq_len = max_seq_len
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(attn_dropout)

        # self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        # self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

    def forward(self, key, query, value, seq_len, lex_num, rel_pos_embedding):
        batch = key.size(0)

        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)
        rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        # batch * seq_len * seq_len * n_head * d_head
        rel_pos_embedding = torch.reshape(rel_pos_embedding, [batch, max_seq_len, max_seq_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * seq_len
        key = key.transpose(-1, -2)

        # u_for_c: 1(batch broadcast) * n_head * 1(seq_len) * d_head
        # u_for_c = self.u.unsqueeze(0).unsqueeze(-2)

        # query_and_u_for_c = query + u_for_c
        query_and_u_for_c = query
        # query_and_u_for_c: batch * n_head * seq_len * d_head
        # key: batch * n_head * d_head * seq_len
        A_C = torch.matmul(query_and_u_for_c, key)
        # after above, A_C: batch * n_head * seq_len * seq_len

        # rel_pos_embedding_for_b: batch * num_head * query_len * per_head_size * key_len
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)

        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        query_for_b_and_v_for_d = query_for_b
        # after above, query_for_b_and_v_for_d: batch * num_head * seq_len * 1 * d_head

        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        # after above, B_D: batch * n_head * seq_len * key_len
        attn_score_raw = A_C + B_D

        # 后续会对transformer的输出做截断，只选取char部分的输出
        mask = seq_len_to_mask(seq_len + lex_num).unsqueeze(1).unsqueeze(1)
        # mask = seq_len_to_mask(seq_len + lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)
        # attn_score: batch * n_head * seq_len * key_len
        # value: batch * n_head * seq_len * d_head
        value_weighted_sum = torch.matmul(attn_score, value)
        # after above, value_weighted_sum: batch * n_head * seq_len * d_head

        result = value_weighted_sum.transpose(1, 2).contiguous().reshape(batch, max_seq_len, self.hidden_size)
        # after above, result: batch * seq_len * hidden_size (hidden_size=n_head * d_head)

        return result


class Positionwise_FeedForward(nn.Module):

    def __init__(self, sizes, dropout):
        super().__init__()
        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))
        self.dropout = nn.Dropout(dropout['ff'])
        self.dropout_2 = nn.Dropout(dropout['ff_2'])

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0: output = F.relu(output)
            w = getattr(self, 'w' + str(i))
            output = w(output)
            if i == 0: output = self.dropout(output)
            if i == 1: output = self.dropout_2(output)
        return output


class Transformer_Encoder_Layer(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 learnable_position,
                 layer_preprocess_sequence,
                 layer_postprocess_sequence,
                 dropout=None,
                 ff_size=-1):
        super().__init__()
        self.attn = MultiHead_Attention_Lattice_rel(hidden_size,
                                                    num_heads,
                                                    attn_dropout=dropout['attn'])

    def forward(self, inp, seq_len, lex_num, rel_pos_embedding):
        output = inp
        output = self.attn(output, output, output, seq_len, lex_num=lex_num, rel_pos_embedding=rel_pos_embedding)
        return output

class Transformer_Encoder(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_layers,
                 learnable_position,
                 layer_preprocess_sequence,
                 layer_postprocess_sequence,
                 dropout=None,
                 ff_size=-1,
                 max_seq_len=-1,
                 pe_ss=None,
                 pe_se=None,
                 pe_es=None,
                 pe_ee=None):

        super().__init__()
        self.num_layers = num_layers
        self.four_pos_fusion_embedding = Four_Pos_Fusion_Embedding(pe_ss, pe_se, pe_es, pe_ee,
                                                                   max_seq_len, hidden_size)
        # for i in range(self.num_layers):
        #     setattr(
        #         self, 'layer_{}'.format(i),
        #         Transformer_Encoder_Layer(hidden_size,
        #                                   num_heads,
        #                                   learnable_position,
        #                                   layer_preprocess_sequence,
        #                                   layer_postprocess_sequence,
        #                                   dropout,
        #                                   ff_size))
        self.layer_0 = Transformer_Encoder_Layer(hidden_size,
                                          num_heads,
                                          learnable_position,
                                          layer_preprocess_sequence,
                                          layer_postprocess_sequence,
                                          dropout,
                                          ff_size)
        # self.layer_preprocess = Layer_Process(layer_preprocess_sequence, hidden_size)

    def forward(self, inp, seq_len, lex_num, pos_s, pos_e):
        rel_pos_embedding = self.four_pos_fusion_embedding(pos_s, pos_e)
        output = self.layer_0(inp, seq_len, lex_num=lex_num, rel_pos_embedding=rel_pos_embedding)
        # output = inp
        # for i in range(self.num_layers):
        #     now_layer = getattr(self, 'layer_{}'.format(i))
        #     output = now_layer(output,
        #                        seq_len,
        #                        lex_num=lex_num,
        #                        rel_pos_embedding=rel_pos_embedding)
        # output = self.layer_preprocess(output)
        return output