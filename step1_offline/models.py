import torch.nn as nn
import torch
from modules import Transformer_Encoder
from utils import get_pos_embedding
from crf import ConditionalRandomField
from typing import Optional

import collections
import math
import copy
import logging
from utils import seq_len_to_mask


class Lattice_Transformer_SeqLabel(nn.Module):

    def __init__(self,
                 lattice_weight,
                 lattice_num,
                 lattice_dim,
                 bigram_weight,
                 bigram_num,
                 bigram_dim,
                 hidden_size,
                 label_size,
                 num_heads,
                 num_layers,
                 learnable_position,
                 layer_preprocess_sequence,
                 layer_postprocess_sequence,
                 ff_size=-1,
                 dropout=None,
                 max_seq_len=-1):
        super().__init__()
        self.lattice_embed = nn.Embedding(lattice_num, lattice_dim)
        self.lattice_embed.weight.data.copy_(torch.from_numpy(lattice_weight))
        self.bigram_embed = nn.Embedding(bigram_num, bigram_dim)
        self.bigram_embed.weight.data.copy_(torch.from_numpy(bigram_weight))

        pe_ss = nn.Parameter(get_pos_embedding(max_seq_len, hidden_size, rel_pos_init=0),
                             requires_grad=learnable_position)
        pe_se = nn.Parameter(get_pos_embedding(max_seq_len, hidden_size, rel_pos_init=0),
                             requires_grad=learnable_position)
        pe_es = nn.Parameter(get_pos_embedding(max_seq_len, hidden_size, rel_pos_init=0),
                             requires_grad=learnable_position)
        pe_ee = nn.Parameter(get_pos_embedding(max_seq_len, hidden_size, rel_pos_init=0),
                             requires_grad=learnable_position)

        # self.bigram_size = self.bigram_embed.embedding.weight.size(1)
        # char_input_size = self.lattice_embed.embedding.weight.size(1) + self.bigram_embed.embedding.weight.size(1)
        # lex_input_size = self.lattice_embed.embedding.weight.size(1)

        self.bigram_size = bigram_dim
        char_input_size = bigram_dim + lattice_dim
        lex_input_size = lattice_dim

        self.embed_dropout = nn.Dropout(p=dropout['embed'])
        self.gaz_dropout = nn.Dropout(p=dropout['gaz'])
        self.output_dropout = nn.Dropout(p=dropout['output'])

        self.char_proj = nn.Linear(char_input_size, hidden_size)
        self.lex_proj = nn.Linear(lex_input_size, hidden_size)

        self.encoder = Transformer_Encoder(hidden_size,
                                           num_heads,
                                           num_layers,
                                           learnable_position=learnable_position,
                                           layer_preprocess_sequence=layer_preprocess_sequence,
                                           layer_postprocess_sequence=layer_postprocess_sequence,
                                           dropout=dropout,
                                           ff_size=ff_size,
                                           max_seq_len=max_seq_len,
                                           pe_ss=pe_ss,
                                           pe_se=pe_se,
                                           pe_es=pe_es,
                                           pe_ee=pe_ee)
        self.output = nn.Linear(hidden_size, label_size)
        self.crf = ConditionalRandomField(label_size, include_start_end_trans=True)
        # self.crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    # 训练时用
    # TODO 参数类型
    def forward(self, lattice: torch.Tensor, bigrams: torch.Tensor, seq_len: torch.Tensor, lex_num: torch.Tensor,
                pos_s: torch.Tensor, pos_e: torch.Tensor, target: Optional[torch.Tensor]):
        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)
        bigrams_embed = self.bigram_embed(bigrams)
        bigrams_embed = torch.cat([
            bigrams_embed,
            torch.zeros(size=[batch_size, max_seq_len_and_lex_num -
                                max_seq_len, self.bigram_size]).to(bigrams_embed)
        ],
                                    dim=1)
        raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)

        raw_embed_char = self.embed_dropout(raw_embed_char)
        raw_embed = self.gaz_dropout(raw_embed)

        embed_char = self.char_proj(raw_embed_char)
        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num)
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        lex_mask = (seq_len_to_mask(seq_len + lex_num) ^ char_mask)
        embed_lex.masked_fill_(~(lex_mask).unsqueeze(-1), 0)

        embedding = embed_char + embed_lex
        encoded = self.encoder(embedding, seq_len, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)
        encoded = self.output_dropout(encoded)

        # 这里只获取transformer输出的char部分
        encoded = encoded[:, :max_seq_len, :]
        pred = self.output(encoded)
        mask = seq_len_to_mask(seq_len)

        # script使用
        # pred, path = self.crf.viterbi_decode(pred, mask)
        # return pred

        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}
            return result