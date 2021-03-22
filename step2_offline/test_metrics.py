import unittest
import torch
from my_metric import MyCLFMetric
class TestMyCLFMetric(unittest.TestCase):
    def test_MyCLFMetric(self):
        a = torch.rand((4,20))
        b = torch.rand((4,20))
        print(a)
        print(b)

        pred = torch.where(a > 0.5, torch.ones_like(a), torch.zeros_like(a))
        target = torch.where(b > 0.5, torch.ones_like(b), torch.zeros_like(b))

        print(pred)
        print(target)
        pred_dict = {"pred": pred}
        target_dict = {'target': target}
        metric = MyCLFMetric()

        metric(pred_dict=pred_dict, target_dict=target_dict)
        print(metric.get_metric())

if __name__ == '__main__':
    unittest.main()