"""Implementation of a pre-act ResNet without striding.

For the shortcuts that go across layers which change spatial dimensionality
and double the feature maps, we use a simple pooling max-pooling strategy 
(PoolingShortcut, implemented in core/layers.py) which effectively downsamples
the input and doubles feature maps by concatenation across channels.
"""

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset
from torch.nn.modules.container import Sequential
from typing import (Any, Callable, List, Optional, Type, Union,
                    no_type_check_decorator)

from core.layers import ConstrainedConv2d, ConstrainedLinear, SimplexClassifier, PoolingShortcut


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            lip_cond: float = 1.,
            dist_cond: float = 100.,
            proj_mode='orthogonal') -> ConstrainedConv2d:
    """3x3 convolution with padding"""
    return ConstrainedConv2d(in_planes,
                             out_planes,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False,
                             lip_cond=lip_cond,
                             dist_cond=dist_cond,
                             proj_mode=proj_mode)


def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            lip_cond=1.,
            dist_cond=100.,
            proj_mode='orthogonal') -> ConstrainedConv2d:
    """1x1 convolution"""
    return ConstrainedConv2d(in_planes,
                             out_planes,
                             kernel_size=1,
                             stride=stride,
                             bias=False,
                             lip_cond=lip_cond,
                             dist_cond=dist_cond,
                             proj_mode=proj_mode)


class PreActBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 activation: Callable = nn.ReLU,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 lip_cond: float = np.inf,
                 dist_cond: float = np.inf,
                 proj_mode: str = 'orthogonal'
                 ) -> None:
        super(PreActBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes,
                             planes,
                             lip_cond=lip_cond,
                             dist_cond=dist_cond,
                             proj_mode=proj_mode)
        self.pool = nn.MaxPool2d(kernel_size=3,
                                 stride=stride,
                                 padding=1) if stride > 1 else nn.Identity()
        self.bn1 = norm_layer(inplanes)
        self.act = activation()
        self.conv2 = conv3x3(planes,
                             planes,
                             lip_cond=lip_cond,
                             dist_cond=dist_cond,
                             proj_mode=proj_mode)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.lipC = self._lipC()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.act(out)        
        
        if self.downsample is not None:
            identity = self.downsample(out)
        
        out = self.conv1(out)
        out = self.pool(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += identity
        return out

    def _lipC(self) -> float:
        """Return Lipschitz constant from the specified constraints."""
        if self.downsample is None:
            return 1 + self.conv1.lipC * self.conv2.lipC
        else:
            return self.downsample[0].lipC + self.conv1.lipC * self.conv2.lipC

    def lip(self) -> float:
        """Compute actual Lipschitz constant from each module's current weight parameters."""
        if self.downsample is None:
            return 1 + self.conv1.lip() * self.conv2.lip()
        else:
            return self.downsample[0].lip() + self.conv1.lip() * self.conv2.lip()


class PreActResNet(nn.Module):

    def __init__(self,
                 block: PreActBasicBlock,
                 layers: List[int],
                 downsample_fn: Callable = PoolingShortcut, 
                 activation: Callable = nn.ReLU,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 lip_cond=[np.inf, np.inf, np.inf, np.inf],
                 dist_cond=[np.inf, np.inf, np.inf, np.inf],
                 proj_mode='orthogonal'
                 ) -> None:
        super(PreActResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity

        self.norm = norm_layer
        self.lip_cond = lip_cond
        self.dist_cond = dist_cond
        self.inplanes = 64
        self.proj_mode = proj_mode
        self.base_width = 64
        self.act = activation()
        self.conv1 = ConstrainedConv2d(3,
                                       self.inplanes,
                                       kernel_size=7,
                                       stride=1,
                                       padding=3,
                                       bias=False,
                                       lip_cond=lip_cond[0],
                                       dist_cond=dist_cond[0],
                                       proj_mode=self.proj_mode)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.layer1 = self._make_layer(block=block,
                                       downsample_fn=downsample_fn,
                                       activation=activation,
                                       planes=self.base_width,
                                       blocks=layers[0],
                                       lip_cond=lip_cond[1:3],
                                       dist_cond=dist_cond[1:3])
        self.layer2 = self._make_layer(block=block,
                                       downsample_fn=downsample_fn,
                                       activation=activation,
                                       planes=2*self.base_width,
                                       blocks=layers[1],
                                       stride=2,
                                       lip_cond=lip_cond[1:3],
                                       dist_cond=dist_cond[1:3])
        self.layer3 = self._make_layer(block=block,
                                       downsample_fn=downsample_fn,
                                       activation=activation,
                                       planes=4*self.base_width,
                                       blocks=layers[2],
                                       stride=2,
                                       lip_cond=lip_cond[1:3],
                                       dist_cond=dist_cond[1:3])
        self.layer4 = self._make_layer(block=block,
                                       downsample_fn=downsample_fn,
                                       activation=activation,
                                       planes=8*self.base_width,
                                       blocks=layers[3],
                                       stride=2,
                                       lip_cond=lip_cond[1:3],
                                       dist_cond=dist_cond[1:3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dist_cond[3] == 0:
            self.fc = SimplexClassifier(8*self.base_width * block.expansion,
                                        num_classes,
                                        bias=False,
                                        lip_cond=lip_cond[3],
                                        dist_cond=dist_cond[3])
        else:
            self.fc = ConstrainedLinear(8*self.base_width * block.expansion,
                                        num_classes,
                                        bias=False,
                                        lip_cond=lip_cond[3],
                                        dist_cond=dist_cond[3])
              

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.lipC = self._lipC()
        self.spectral_complexity_bound = self._spectral_complexity_bound()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self,
                    block: PreActBasicBlock,
                    downsample_fn: Callable,
                    activation: Callable,
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    lip_cond=[np.inf, np.inf],
                    dist_cond=[np.inf, np.inf],
                    ) -> nn.Sequential:
        norm_layer = self.norm
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                downsample_fn(self.inplanes,
                              planes * block.expansion, 1,
                              lip_cond=lip_cond[1],
                              dist_cond=dist_cond[1],
                              proj_mode=self.proj_mode),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(inplanes=self.inplanes,
                            planes=planes,
                            stride=stride,
                            downsample=downsample,
                            activation=activation,
                            norm_layer=norm_layer,
                            lip_cond=lip_cond[0],
                            dist_cond=dist_cond[0],
                            proj_mode=self.proj_mode))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes,
                                planes=planes,
                                activation=activation,
                                norm_layer=norm_layer,
                                lip_cond=lip_cond[0],
                                dist_cond=dist_cond[0],
                                proj_mode=self.proj_mode))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_layer_attr(self, attr):
        """Gets dictionary containing lip/dist of each layer; this is required to track 
        these attributes. CAUTION: currently hard-coded for ResNet18
        """
        dic = { 'conv1': getattr(self.conv1,attr)()}
        for layer in range(1,5):
            for block in range(2):
                current_block = getattr(self,"layer{}".format(layer))[block]
                if current_block.downsample is not None:
                    dic["sc{}{}".format(layer,block+1)] = getattr(current_block.downsample[0],attr)()
                for conv in range(1,3):
                    dic["conv{}{}{}".format(layer,block+1,conv)] = getattr(getattr(current_block,"conv{}".format(conv)),attr)()
        dic["fc"] = getattr(self.fc,attr)()
        return dic 

    def _lipC(self) -> float:
        lipC = 1
        for module in self.children():
            if hasattr(module, 'lipC'):
                lipC *= module.lipC
            elif type(module) == Sequential:
                for block in module:
                    lipC *= block.lipC
        return lipC

    def lip(self) -> float:
        lip = 1
        for module in self.children():
            if hasattr(module, 'lip'):
                lip *= module.lip()
            elif type(module) == Sequential:
                for block in module:
                    lip *= block.lip()
        return lip

    def _spectral_complexity_bound(self) -> float:
        spec_comp = 0.
        for module in self.modules():
            if isinstance(module, ConstrainedConv2d) or isinstance(module, ConstrainedLinear):
                spec_comp += (module.dstC / module.lipC)**(2/3)
        spec_comp = spec_comp**(3/2)
        spec_comp *= self.lipC
        return spec_comp

    def spectral_complexity(self) -> float:
        spec_comp = 0.
        for module in self.modules():
            if isinstance(module, ConstrainedConv2d) or isinstance(module, ConstrainedLinear):
                spec_comp += (module.dist() / module.lip())**(2/3)
        spec_comp = spec_comp**(3/2)
        spec_comp *= self.lip()
        return spec_comp

    def generalization_bound(self,
                             dataset: Dataset,
                             margin_range: list = list(range(1, 101))) -> list:
        """see Lemma A.8, Bartlett et al., 2017"""
        n = len(dataset)
        if type(dataset) == data.dataset.Subset:
            points = dataset.dataset.data[dataset.indices]
        else:
            points = dataset.data
        b = np.linalg.norm(points)
        w = np.array([module.weight.numel() for module in self.modules() if hasattr(module, 'weight')]).max()
        spectral_complexity_bound = self.spectral_complexity_bound
        
        sqrt_r_range = [spectral_complexity_bound * 2 *b * np.sqrt(np.log(2 * w**2)) / margin for margin in margin_range]
        bound = [2* (4/n**(3/2) + 18*np.log(n)/n * sqrt_r) for sqrt_r in sqrt_r_range]
        return bound

    def _project_submodule(self, module: nn.Module, n_iter: int):
        if isinstance(module, ConstrainedConv2d) or isinstance(module, ConstrainedLinear):
            module.project(n_iter)

    def project(self, n_iter):
        f = lambda x: self._project_submodule(x, n_iter)
        self.apply(f)

    def reset_uv(self):
        for module in self.modules():
            if hasattr(module, 'u'):
                module.u = None
            if hasattr(module, 'v'):
                module.v = None


def _preactresnet(
    arch: str,
    block: PreActBasicBlock,
    layers: List[int],
    **kwargs: Any
) -> PreActResNet:
    model = PreActResNet(block, layers, **kwargs)
    return model


def nostride_preactresnet18(**kwargs: Any) -> PreActResNet:
    return _preactresnet('resnet18', PreActBasicBlock, [2, 2, 2, 2], **kwargs)
