from collections import defaultdict
from fastNLP.core.metrics import MetricBase


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


class MyCLFMetric(MetricBase):

    def __init__(self, pred=None, target=None, f_type='macro', beta=1):
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta**2
        super().__init__()
        self._init_param_map(pred=pred, target=target)
        self._tp, self._fp, self._fn = defaultdict(int), defaultdict(int), defaultdict(int)
        # tp: truth=T, classify=T; fp: truth=T, classify=F; fn: truth=F, classify=T

    def evaluate(self, pred, target):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计
        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        """
        target = target.to(pred)
        labels = target.size(1)
        for i in range(labels):
            pred_temp = pred[:, i:i + 1]
            target_temp = target[:, i:i + 1]

            self._tp[i] += ((pred_temp == 1) & (target_temp == 1)).sum().item()
            self._fp[i] += ((pred_temp == 1) & (target_temp == 0)).sum().item()
            self._fn[i] += ((pred_temp == 0) & (target_temp == 1)).sum().item()

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {}
        if self.f_type == 'macro':
            tags = list(self._tp.keys())
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._tp[tag]
                fn = self._fn[tag]
                fp = self._fp[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if tag != '':  # tag!=''防止无tag的情况
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
            f, pre, rec = _compute_f_pre_rec(self.beta_square, sum(self._tp.values()), sum(self._fn.values()),
                                             sum(self._fp.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._tp = defaultdict(int)
            self._fp = defaultdict(int)
            self._fn = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result