import torch.nn as nn
import torch
from typing import Optional
import torch.nn.functional as F

import math
import copy
import logging


class step2_clf(nn.Module):

    def __init__(self, chars_weight, chars_num, chars_dim, entity_label_num, entity_label_dim, hidden_size,
                 number_of_labels, device):
        super().__init__()
        self.chars_embed = nn.Embedding(chars_num, chars_dim)
        self.chars_embed.weight.data.copy_(torch.from_numpy(chars_weight))
        self.entity_label_embed = nn.Embedding(entity_label_num, entity_label_dim)
        self.entity_label_embed.weight.data.uniform_(-1, 1)

        # 统一映射层
        # rep_dim = char_dim * 3 + dict_dim
        self.proj = nn.Linear(chars_dim * 3 + entity_label_dim, hidden_size)

        # 5个子任务的映射层（这里直接用parameter来构建5个映射矩阵的方式有点问题，直接用5个映射层）
        self.task_proj0 = nn.Linear(hidden_size, 2)
        self.task_proj1 = nn.Linear(hidden_size, 2)
        self.task_proj2 = nn.Linear(hidden_size, 2)
        self.task_proj3 = nn.Linear(hidden_size, 2)
        self.task_proj4 = nn.Linear(hidden_size, 2)

        # tasks_proj = {}
        self.number_of_labels = number_of_labels
        # # logging.info('number_of_labels={}'.format(number_of_labels))
        # for i in range(self.number_of_labels):
        #     # output_weights_tasks[i] = nn.Parameter(torch.zeros(size=[hidden_size, 2], requires_grad=True))
        #     # output_bias_tasks[i] = nn.Parameter(torch.zeros(size=[2], requires_grad=True))
        #     proj = nn.Linear(hidden_size, 2)
        #     tasks_proj[i] = proj
        # self.tasks_proj = tasks_proj

    # TODO 训练 和 trace 的时候手工更改一下forward函数，暂时没有找到好方法
    def forward(self, left_chars: torch.Tensor, right_chars: torch.Tensor, entity_chars: torch.Tensor,
                entity_label: torch.Tensor, target: Optional[torch.Tensor]):
        left_embed = torch.sum(self.chars_embed(left_chars), dim=1)
        right_embed = torch.sum(self.chars_embed(right_chars), dim=1)

        entity_embed_mean = torch.mean(self.chars_embed(entity_chars), dim=1, keepdim=False)
        entity_label_embed = torch.sum(self.entity_label_embed(entity_label), dim=1)

        mention = torch.cat([entity_embed_mean, entity_label_embed], dim=-1)
        context = torch.cat([left_embed, right_embed], dim=-1)
        representation = torch.cat([mention, context], dim=-1)
        project_output = self.proj(representation)

        one_hot_labels = F.one_hot(target, num_classes=2)
        all_probs_in_batch = []
        all_labels_loss_in_batch = []
        for i in range(self.number_of_labels):
            if i == 0:
                logits = self.task_proj0(project_output)
            elif i == 1:
                logits = self.task_proj1(project_output)
            elif i == 2:
                logits = self.task_proj2(project_output)
            elif i == 3:
                logits = self.task_proj3(project_output)
            elif i == 4:
                logits = self.task_proj4(project_output)


            # logits = self.tasks_proj[i](project_output)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            per_label_loss_in_batch = -torch.sum(
                torch.squeeze(one_hot_labels[:, i:i + 1, :], dim=1).float() * log_probs, dim=-1)
            all_labels_loss_in_batch.append(per_label_loss_in_batch)
            all_probs_in_batch.append(probs[:, 1])

        # avr_loss
        loss = torch.mean(torch.stack(all_labels_loss_in_batch, dim=1))

        # 把all_probs_in_batch转化为0 1
        all_probs_in_batch = torch.stack(all_probs_in_batch, dim=1)
        # logging.info('all_probs_in_batch size={}'.format(all_probs_in_batch.size()))
        # pred_label = torch.where(all_probs_in_batch > 0.5, torch.ones_like(all_probs_in_batch),
        #                          torch.zeros_like(all_probs_in_batch))
        # logging.info('pred_label={}'.format(pred_label))
        pred_label_list = list([])
        for i in range(self.number_of_labels):
            pred_label = all_probs_in_batch[:, i:i + 1]
            if i == 4:
                pred_label = torch.where(pred_label > 0.35, torch.ones_like(pred_label), torch.zeros_like(pred_label))
            else:
                pred_label = torch.where(pred_label > 0.5, torch.ones_like(pred_label), torch.zeros_like(pred_label))

            # true_label = target[:,i:i+1]
            # logging.info('pred_label size={}'.format(pred_label.size()))
            # logging.info('true_label size={}'.format(true_label.size()))

            pred_label_list.append(pred_label[:, 0])
        # total_pred = torch.squeeze(torch.stack(pred_label_list, dim=1), dim=2)
        total_pred = torch.stack(pred_label_list, dim=1)
        # logging.info('total_pred={}'.format(total_pred))

        if self.training:
            return {'loss': loss}
        else:
            return {'pred': total_pred}

    # script时用
    def forward2(self, left_chars: torch.Tensor, right_chars: torch.Tensor, entity_chars: torch.Tensor,
                 entity_label: torch.Tensor):
        left_embed = torch.sum(self.chars_embed(left_chars), dim=1)
        right_embed = torch.sum(self.chars_embed(right_chars), dim=1)
        entity_embed_mean = torch.mean(self.chars_embed(entity_chars), dim=1, keepdim=False)
        entity_label_embed = torch.sum(self.entity_label_embed(entity_label), dim=1)

        mention = torch.cat([entity_embed_mean, entity_label_embed], dim=-1)
        context = torch.cat([left_embed, right_embed], dim=-1)
        representation = torch.cat([mention, context], dim=-1)
        project_output = self.proj(representation)

        all_probs_in_batch = []
        for i in range(self.number_of_labels):
            # script的时候需要一个占位变量
            logits = torch.zeros(2)
            if i == 0:
                logits = self.task_proj0(project_output)
            elif i == 1:
                logits = self.task_proj1(project_output)
            elif i == 2:
                logits = self.task_proj2(project_output)
            elif i == 3:
                logits = self.task_proj3(project_output)
            elif i == 4:
                logits = self.task_proj4(project_output)
            probs = F.softmax(logits, dim=-1)
            all_probs_in_batch.append(probs[:, 1])

        # 把all_probs_in_batch转化为0 1
        all_probs_in_batch = torch.stack(all_probs_in_batch, dim=1)
        pred_label_list = list([])
        for i in range(self.number_of_labels):
            pred_label = all_probs_in_batch[:, i:i + 1]
            if i == 4:
                pred_label = torch.where(pred_label > 0.35, torch.ones_like(pred_label), torch.zeros_like(pred_label))
            else:
                pred_label = torch.where(pred_label > 0.5, torch.ones_like(pred_label), torch.zeros_like(pred_label))

            # true_label = target[:,i:i+1]
            # logging.info('pred_label size={}'.format(pred_label.size()))
            # logging.info('true_label size={}'.format(true_label.size()))

            pred_label_list.append(pred_label[:, 0])
        # total_pred = torch.squeeze(torch.stack(pred_label_list, dim=1), dim=2)
        total_pred = torch.stack(pred_label_list, dim=1)
        # logging.info('total_pred={}'.format(total_pred))
        # pred_label = torch.where(all_probs_in_batch > 0.5, torch.ones_like(all_probs_in_batch),
        #                          torch.zeros_like(all_probs_in_batch))
        # logging.info('pred_label={}'.format(pred_label))
        # for i in range(self.number_of_labels):
        #     pred_label = all_probs_in_batch[:, i:i+1]
        return total_pred