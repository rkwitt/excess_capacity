import os
import sys
import numpy as np
import unittest

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from torch.optim import SGD, Adam

from core.layers import ConstrainedLinear, ConstrainedConv2d

class TestConstrainedLinear(unittest.TestCase):
    def test_lip(self):
        lip_cond = 0.3
        num_inp_features = 128
        num_out_features = 64
        num_iter = 10000

        f = ConstrainedLinear(num_inp_features,
                              num_out_features,
                              bias=False,
                              lip_cond=lip_cond,
                              dist_cond=np.inf)

        x = torch.randn(1, num_inp_features, requires_grad=True)
        opt = Adam([x], lr=0.5)

        for it in range(num_iter+1):
            opt.zero_grad()
            x_o = f(x)
            loss = -1.0 * torch.norm(x_o, p=2)/torch.norm(x, p=2)
            loss.backward()
            opt.step()

        result = -loss.item()
        self.assertAlmostEqual(
            result, lip_cond, 2, 'should be {:.2f}'.format(lip_cond))


class TestConstrainedConv2d(unittest.TestCase):
    def test_lip(self):
        lip_cond = 0.3
        n_inp_channels = 32
        n_out_channels = 64
        num_iter = 10000

        f = ConstrainedConv2d(n_inp_channels,
                              n_out_channels,
                              kernel_size=3,
                              padding=1,
                              stride=1,
                              bias=False,
                              padding_mode='circular',
                              dist_cond=np.inf,
                              lip_cond=lip_cond)

        x = torch.randn(1, 32, 32, 32, requires_grad=True)
        opt = Adam([x], lr=0.01)

        for it in range(num_iter+1):
            opt.zero_grad()
            x_o = f(x)
            loss = -1.0 * torch.norm(x_o, p=2)/torch.norm(x, p=2)
            loss.backward()
            opt.step()

        result = -loss.item()
        self.assertAlmostEqual(
            result, lip_cond, 2, 'should be {:.2f}'.format(lip_cond))


if __name__ == '__main__':
    unittest.main()

