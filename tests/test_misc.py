import os
import sys
import numpy as np
import unittest

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from torch.optim import SGD, Adam

from utils.misc import compute_margins


class TestMargins(unittest.TestCase):
    def test_compute_margins(self):
        inp = torch.tensor([[1., 4., 2.], [2., 3., 1.]])
        tgt = torch.LongTensor([1, 2])

        out = compute_margins(inp, tgt)
        chk = torch.tensor([[-2.],[2.]])
        self.assertAlmostEqual((out-chk).norm().item(),0)

if __name__ == '__main__':
    unittest.main()

