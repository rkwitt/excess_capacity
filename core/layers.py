import os
import numpy as np
import configparser

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from . import projections
from . import cyclicpad


class ConstrainedLinear(nn.Linear):
    """Implementation of a constrained linear layer, implementing a map 
    of the form x |-> Ax (without bias). The constraints can come (1) in 
    the form of a constraint on the Lipschitz constant, (2) in the form 
    of a (2,1)-group norm distance constraint to the initial weight of 
    the map, or (3) in the form of both.

    Caution: currently the Lipschitz constraint can only be enforced if 
    stride=1!

    Attributes
    ----------
        lipC: float
            Lipschitz constant constraint 
        dstC: float
            distance to initialization constraint
        projection_method: projections.Dykstra, projections.Halpern
            method to use for iterated projections (i.e., when using multiple constraints)
        proj_mode: 'orthogonal' | 'radial'
            how to project onto constraint set
        is_initialized: bool
            whether the layer is initialized or not
    """
    def __init__(
        self, 
        in_features: int,   # Nr. of input features
        out_features: int,  # Nr. of output features
        bias: bool = True,  # Use bias -> Affine map
        device = None,        # Device
        dtype = None,         # see PyTorch doc. on nn.Linear
        lip_cond = np.inf,    # Constraint on Lipschitz constant
        dist_cond = np.inf,   # Constraint on (2,1) group norm distance to initialization
        proj_mode = 'orthogonal'
        ) -> None:
        """
        Arguments
        ---------
            The arguments in_features, out_features, bias, device and dtype correspond
            to the arguments of a standard PyTorch nn.Linear layer. The additional 
            arguments are:

            lip_cond: float 
                Lipschitz constant constraint (if np.inf, then no constraint 
                is enforced)
            dist_cond: float
                (2,1)-group norm distance constraint (if np.inf, then no constraint 
                is enforced)
            proj_mode: 'orthogonal' | 'radial'
                projection type
        """
        
        self.lipC = lip_cond
        self.dstC = dist_cond
        self.projection_method = projections.Dysktra # currently fixed!
        self.proj_mode = proj_mode
        self.is_initialized = False
        
        # initialize default nn.Linear
        nn.Linear.__init__(
            self, 
            in_features, 
            out_features, 
            bias=bias,
            device=device, 
            dtype=dtype
        )
        
        self.register_buffer("init_weight", None)
        
    def extra_repr(self):
        return "in_feature={}, out_features={}, bias={}, lip_cond={}, dist_cond={}".format(
            self.in_features, 
            self.out_features, 
            self.bias is not None, 
            self.lipC, self.dstC
        )

    def lip(self) -> float:
        """Computes Lipschitz constant via power iteration
        
        Returns
        -------
            lipC: float
                computed Lipschitz constant
        """
        w = self.weight
        _, s, v = projections.power_it(w, 100)
        return s.item()

    def dist(self) -> float:
        """Compute (2,1) group norm distance to initialization. Only works
        if layer is already initialized.
        
        Returns
        -------
            dstC: float
                computed distance to initialized weight 
        """
        assert self.is_initialized, 'Module not initialized'
        delta = self.weight - self.init_weight
        dist = delta.norm(dim=1, p=2).norm(p=1)
        return dist.item()

    def _initialize(self):
        """Initialize constrained linear layer. That is, make sure that the 
        desired Lipschitz constraint is satisfied initially.
        """
        with torch.no_grad():
            lip = self.lip()
            if lip > self.lipC:
                self.weight.data.mul_(self.lipC / lip)
            self.register_buffer("init_weight", self.weight.data.detach().clone())
        self.dstProj = projections.DistProj(self.dstC, self.init_weight, mode = self.proj_mode)
        self.lipProj = projections.LipProj(self.lipC, input_type='fc')
        self.is_initialized = True

    def project(self, n_iter = 10):
        """Project to satisfy constraints.
        
        Arguments
        ---------
            n_iter: int
                number of projection iterations to perform
        """
        assert self.is_initialized == True, 'Module not initialized'
        proj = self.projection_method([self.dstProj, self.lipProj])
        self.weight.data = proj(self.weight.data, n_iter)

    def forward(self, input: Tensor) -> Tensor:
        """Forward data through layer."""
        if not self.is_initialized:
            self._initialize()
        return F.linear(input, self.weight, self.bias)


class ConstrainedConv2d(nn.Conv2d):
    """Implementation of a constrained 2D convolution layer, implementing a
    map of the form x |-> Wx (without bias), where W corresponds to an 
    appropriately instantiated weight matrix, depending on kernel size and
    stride. The constraints can come (1) in the form of a constraint on the 
    Lipschitz constant, (2) in the form  of a (2,1)-group norm distance constraint 
    to the initial weight of the map, or (3) in the form of both.

    Attributes
    ----------
        lipC: float
            Lipschitz constaint constraint 
        dstC: float
            distance to initialization constraint
        projection_method: projections.Dykstra, projections.Halpern
            method to use for iterated projections (i.e., when using multiple constraints)
        proj_mode: 'orthogonal' | 'radial'
            how to project onto constraint set
        is_initialized: bool
            whether the layer is initialized or not
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_mode = 'circular',
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        n_iters: int = 10,
        lip_cond: float = np.inf,
        dist_cond: float = np.inf,
        proj_mode='orthogonal'
        ) -> None: 
        """
        Arguments
        ---------
        The arguments in_channels, out_channels, kernel_size, stride, padding, padding_mode, 
        dilation and groups correspond to the arguments of a standard PyTorch nn.Conv2d layer.
        Additional arguments are:

            lip_cond: float 
                Lipschitz constant constraint (if np.inf, then no constraint 
                is enforced)
            dist_cond: float
                (2,1)-group norm distance constraint (if np.inf, then no constraint 
                is enforced)
            proj_mode: 'orthogonal' | 'radial'
                projection type
        """   
        self.lipC = lip_cond
        self.dstC = dist_cond
        self.projection_method = projections.Dysktra
        self.is_initialized = False
        self.u = None
        self.proj_mode = proj_mode
        
        # initialize nn.Conv2d layer 
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )   

    def lip(self) -> float:
        """Computes the Lipschitz constant of the map.
        
        Returns
        -------
            lipC: float
                Lipschitz constant
        """
        if self.u == None:
            n_iter = 100
            u = torch.randn(self. out_dims, device = self.weight.device)
        else:
            n_iter = 5
            u = self.u.detach().clone()
        if self.padding_mode == 'circular':
            self.u, s, self.v= cyclicpad.cyclic_power_iteration(
                self.weight, 
                self.inp_dims, 
                self.out_dims, 
                u, 
                self.kernel_size[-1], 
                self.stride[-1], 
                self.padding[-1], 
                n_iter = n_iter
            )
        else:
            self.u, s, self.v = projections.conv_power_it(
                self.weight, 
                self.inp_dims, 
                self.out_dims, 
                u, 
                self.kernel_size[-1], 
                self.stride[-1], 
                self.padding[-1], 
                n_iter = n_iter
            )
        return s.item()

    def dist(self) -> float:
        """Compute (2,1)-group norm distance of the current parametrization to its initialization.
        
        Returns
        -------
            dstC: float
        """
        assert self.is_initialized, 'Module not initialized'
        delta = self.weight - self.init_weight
        dist = delta.norm(dim=1, p=2).norm(p=1)
        return dist.item()
    
    def _initialize(self, input: torch.Tensor):
        """Initialize layer by doing a bit of bookkeeping."""
        with torch.no_grad():
            out = self._conv_forward(input, self.weight, self.bias)       
            self.inp_dims = input.shape 
            self.out_dims = out.shape
            if self.inp_dims[2] == 1 and self.inp_dims[3] == 1:
                with torch.no_grad():
                    if self.padding_mode == 'zeros':
                        self.weight.data = self.weight.data[:,:,self.kernel_size[0]//2+1:self.kernel_size[1]//2+2, self.kernel_size[1]//2+1: self.kernel_size[1]//2+2]
                    elif self.padding_mode == 'circular':
                        self.weight.data = self.weight.data.sum([2,3], keepdim=True)
                self.kernel_size = (1,1)
                self.padding = (0,0)
            if self.inp_dims[2] == 2 and self.inp_dims[3] == 2:
                with torch.no_grad():
                    self.weight.data = self.weight.data[:,:,:2,:2]
                self.kernel_size = (2,2)
                self.padding = (1,1)

            self.output_padding = self.inp_dims[-1] - ((self.out_dims[-1]-1)*self.stride[-1]-2*self.padding[-1]+1*(self.weight.shape[-1]-1)+1)

            # params required for power iteration with cyclicpad
            self.radial_params = {
                'inp_dims': self.inp_dims,
                'out_dims': self.out_dims,
                'us': None,
                'kernel_size': self.kernel_size[0],
                'padding': self.padding[0],
                'n_iter': 10
            }

            lip = self.lip()
            if lip > self.lipC:
                self.weight.data.mul_(self.lipC / lip)
            self.register_buffer("init_weight", self.weight.data.detach().clone())
            #TODO case kernelsize < inpdim/#patches

            if self.proj_mode == 'radial':
                self.embed = lambda x: x
            else:
                self.embed = lambda x: projections.embed_plane(x, 
                    self.weight.shape[:2] + self.inp_dims[2:])

            self.dstProj = projections.DistProj(
                self.dstC, 
                self.embed(self.init_weight), 
                mode = self.proj_mode)
            self.clip = projections.ProjOnPlane(self.weight.shape)
            self.lipProj = projections.LipProj(self.lipC, input_type='conv', mode=self.proj_mode, radial_params = self.radial_params)
            self.is_initialized = True   

    def project(self, n_iter = 10) -> None:
        """Project parametrization onto constraint set(s).
        
        Arguments
        ---------
            n_iter: int 
                Number of iterations to run the iterative projection onto 
                the constraint sets.
        """
        assert self.is_initialized, 'Module not initialized'
        x = self.embed(self.weight)
        proj = self.projection_method([self.dstProj, self.lipProj, self.clip])
        x = proj(x, n_iter)
        self.weight.data = x[:, :, :self.weight.shape[2], :self.weight.shape[3]]
    
    def forward(self, input: Tensor) -> Tensor:
        if not self.is_initialized:
            self._initialize(input)
        # TODO currently hard-coded cyclic pad, as lazy solution to invalid dimensions when ks = 2
        return cyclicpad.conv2d_cyclic_pad(input, self.weight, self.bias, stride=self.stride)

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, lip_cond={lipC}, dist_cond={dstC}".format(
            **self.__dict__
        )

class PoolingShortcut(nn.Module):
    """Implementation of shortcut connection via pooling.
    
    Note: Input to __init__ is never used, but required for calling the class 
    consistently to conv1x1 (as done in the original PyTorch ResNet implementation).
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        lip_cond=np.inf,
        dist_cond=0,
        proj_mode='orthogonal',
    ):  
        super(PoolingShortcut, self).__init__()
        self.lip =  lambda: (2 * torch.ones(1)).sqrt().item()
        self.lipC = (2 * torch.ones(1)).sqrt().item()
        self.dist = lambda: 0
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
       
    def forward(self, x):
        if x.shape[-1]>1:
            a = self.pool1(x)
            b = self.pool2(x)[:, :, 0:int(x.shape[2]/2), 0:int(x.shape[3]/2)]  
        else:
            a, b = x, x
        return torch.cat((a,b), dim=1)


class SimplexClassifier(ConstrainedLinear):
    """Fixed (simplex) classifier with weights on the unit (#classes-1)-simplex
    
    Note: Weights are never updated and the Lipschitz constant is fix; hence,
    also the distance to the initialization is always zero.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None, 
        lip_cond=np.inf, 
        dist_cond=np.inf,
        config_file='config.ini'
        ) -> None:
        
        config = configparser.ConfigParser()
        config.read(config_file)

        self.prototype_file = os.path.join(
            config['GLOBAL']['PrototypeFolder'],
            "prototypes_{}_{}.pt".format(out_features, in_features))
        self.default_simplex_clf_lipC = np.sqrt(out_features/(out_features-1))

        super(SimplexClassifier, self).__init__(
            in_features, 
            out_features, 
            bias=bias,
            device=device, 
            dtype=dtype, 
            lip_cond=lip_cond, 
            dist_cond=dist_cond)

    def lip(self) -> float:
        """Returns fix Lipschitz constant of the map."""
        if self.lipC > self.default_simplex_clf_lipC:
            return self.default_simplex_clf_lipC
        return self.lipC

    def dist(self) -> float:
        """Returns the distance to initialization (always 0)"""
        return 0.
        
    def project(self, n_iter = 10):
        """No need to project here"""
        pass
 
    def _initialize(self):
        self.is_initialized = True
   
    def forward(self, input: Tensor) -> Tensor:
        if not self.is_initialized:
            self._initialize()
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self) -> None:
        """
        Sets weight vectors to the vertices of a (K-1)-simplex (with K=#classes), 
        scaled such that desired Lipschitz constant is satisfied. By default, the weights 
        in the prototype file are the vertices for a unit simplex.
        """
        if not os.path.exists(self.prototype_file):
            raise Exception('Prototype file {} does not exist!'.format(self.prototype_file))

        self.weight.data = torch.load(self.prototype_file) * self.lip()/self.default_simplex_clf_lipC
        self.weight.requires_grad = False
        with torch.no_grad():
            self.register_buffer("init_weight", self.weight.data.detach().clone())


class GroupSort(nn.Module):
    """Implementation of the group-sort activation function from
    
    Cem Anil, James Lucas, Roger Grosse
    Sorting out Lipschitz function approximation
    ICML 2019

    see https://arxiv.org/abs/1811.05381
    """
    def forward(self, x):
        a, b = x.split(x.size(1) // 2, 1)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=1)
