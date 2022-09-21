import os
import sys
import numpy as np
import unittest

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from torch.optim import SGD

from core.layers import ConstrainedLinear

class TestConstrainedLinear(unittest.TestCase):
    def test_lip(self):
        """Test if linear layer can enforce Lipschitz constraint"""
        lip_cond = 0.8
        num_inp_features = 10
        num_out_features = 20
        num_iter = 5000

        f = ConstrainedLinear(10,
                              20,
                              bias=False,
                              lip_cond=lip_cond,
                              dist_cond=np.inf)

        x = torch.randn(1, num_inp_features, requires_grad=True)
        y = torch.randn(1, num_inp_features, requires_grad=True)
        opt = SGD([x, y], lr=0.1)

        for it in range(num_iter+1):
            opt.zero_grad()
            x_o, y_o = f(x), f(y)
            loss = -1.0 * torch.norm(x_o-y_o, p=2)/torch.norm(x-y, p=2)
            loss.backward()
            opt.step()

        result = -loss.item()
        self.assertAlmostEqual(
            result, lip_cond, 3, 'should be {:.2f}'.format(lip_cond))


if __name__ == '__main__':
    unittest.main()

