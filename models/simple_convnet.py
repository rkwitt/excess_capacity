import numpy as np
import torch
import torch.nn as nn
from typing import (Any, Callable, List, Optional, Type, Union)

from core.layers import ConstrainedConv2d, ConstrainedLinear, GroupSort, SimplexClassifier

__all__ = ['simple_convnet6', 'simple_convnet11']


class SimpleConvnet(nn.Module):
    """A simple network with convolutional layers and ReLU activations. 
    Input dimensionality is progressively reduced by convolutions with 
    stride 2. The classification layer has inputs of spatial dim 1, corresponding 
    to matrix multiplication along the channels.
        
    This toy network assumes that the spatial input size is 32.
    """

    def __init__(self,
                 block_depth: int,
                 n_classes: int,
                 n_channels: int = 512,
                 strides: bool = True,
                 simplex: bool = False,
                 lip_cond: List = [np.inf, np.inf, np.inf, np.inf],
                 dist_cond: List = [np.inf, np.inf, np.inf, np.inf]) -> None:
        
        super(SimpleConvnet, self).__init__()
        self.strides = strides
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        ) if not self.strides else nn.Identity()
         
        self.block_depth = block_depth
        self.blocks = [self._make_block(3,
                                        n_channels,
                                        lip_cond=lip_cond[0:2], 
                                        dist_cond=dist_cond[0:2],
                                        strides=self.strides)
                    ]
        self.blocks += [self._make_block(n_channels,
                                         n_channels,
                                         lip_cond=lip_cond[1:2]*2,
                                         dist_cond=dist_cond[1:2]*2,
                                         strides=self.strides
                                         ) for _ in range(4)
                    ]
        if simplex == False:
            self.cls = nn.Sequential(
                ConstrainedConv2d(
                    n_channels,
                    n_classes,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=False,
                    padding_mode='circular',
                    lip_cond=lip_cond[-1],
                    dist_cond=dist_cond[-1]),
                nn.Flatten())
        else:
            if dist_cond[-1] == 0:
                self.cls = nn.Sequential(
                    nn.Flatten(),
                    SimplexClassifier(n_channels,
                                      n_classes,
                                      bias=False,
                                      lip_cond=lip_cond[-1],
                                      dist_cond=dist_cond[-1])
                )
            else:
                self.cls = nn.Sequential(
                    nn.Flatten(),
                    ConstrainedLinear(n_channels,
                                      n_classes,
                                      bias=False,
                                      lip_cond=lip_cond[-1],
                                      dist_cond=dist_cond[-1])
                )

        self.model = nn.Sequential(*self.blocks, self.cls)

    def forward(self, x):
       return self.model(x)       
    
    def _project_submodule(self, module: nn.Module, n_iter: int):
        if isinstance(module, ConstrainedConv2d):
            module.project(n_iter)

    def project(self, n_iter):
        f = lambda x: self._project_submodule(x, n_iter)
        self.apply(f)

    def get_layer_attr(self, attr):
        """Get dictionary containing lip/dist of each layer, which is required
        for tracking these attributes"""
        j = 1
        dic = {}
        for layer in self.model.modules():
            if isinstance(layer, ConstrainedConv2d):
                dic['conv'+str(j)] = getattr(layer, attr)()
                j += 1
        for layer in self.cls:
            if isinstance(layer, ConstrainedConv2d) or isinstance(layer, ConstrainedLinear):
                dic['cls'] = getattr(layer, attr)()
        return dic

    def _make_block(self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: int = 3,
                    dist_cond: float = [np.inf, np.inf],
                    lip_cond: float = [np.inf, np.inf],
                    strides: bool = True) -> None:
        """Make block with an initial (ConstrainedConv2d, Pool, ReLU) sequence
        of layers, followed by block_depth * (ConstrainedConv2d, ReLU) sequence
        of layers. Striding only affects the initial sequence; any remaining 
        convolution layers have stride of 1.
        """
        stride = 2 if strides == True else 1
        block = [
            ConstrainedConv2d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=1,
                              stride=stride,
                              bias=False,
                              padding_mode='circular',
                              dist_cond=dist_cond[0],
                              lip_cond=lip_cond[0]),
            self.pool,
            nn.ReLU()
        ]

        for _ in range(self.block_depth-1):
            block += [
                ConstrainedConv2d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  padding=1,
                                  stride=1,
                                  bias=False,
                                  padding_mode='circular',
                                  dist_cond=dist_cond[1],
                                  lip_cond=lip_cond[1]),
                nn.ReLU()
            ]

        return nn.Sequential(*block)

    def reset_uv(self):
        for module in self.modules():
            if hasattr(module, 'u'):
                module.u = None
            if hasattr(module, 'v'):
                module.v = None

    def spectral_complexity(self) -> float:
        spec_comp = 0.
        for module in self.modules():
            if isinstance(module, ConstrainedConv2d):
                spec_comp += (module.dist() / module.lip())**(2/3)
        spec_comp = spec_comp**(3/2)
        spec_comp *= self.lip()
        return spec_comp

    def spectral_complexity_bound(self) -> float:
        return 0.

    def lip(self) -> float:
        lip = [layer.lip() for layer in self.modules()
               if isinstance(layer, ConstrainedConv2d)]
        return torch.tensor(lip).prod().item()


def simple_convnet6(n_classes: int, n_channels: int = 512, **kwargs: Any):
    return SimpleConvnet(1, n_classes, n_channels, **kwargs)

def simple_convnet11(n_classes: int, n_channels: int = 512, **kwargs: Any):
    return SimpleConvnet(2, n_classes, n_channels, **kwargs)
