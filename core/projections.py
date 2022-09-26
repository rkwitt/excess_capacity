import torch
import torch.nn.functional as F
import scipy.sparse.linalg
import itertools

from . import cyclicpad


class Dysktra(object):
    """
    Implements Dykstra's (iterative) projection algorithm. Given a list of orthogonal projections
    onto convex sets, it converges to the orthogonal projection onto the intersection (if nonempty).
    For reference, see
    
    Combettes, P. L. and Pesquet J.-C.  
    "Proximal splitting methods in signal processing" 
    In: Fixed-Point Algorithms for Inverse Problems in Science and Engineering 
    pp. 185–212. 
    Springer, New York, 2011 
    """
    
    def __init__(self, projections: list): 
        """Initialize with list of callable projection objects.

        Arguments
        ---------
            projections: list
                list of callable projection objects (i.e., LipProj | ProjOnPlane | DistProj)
        """       
        self.projections = projections

    def __call__(self, x: torch.Tensor, n_iter: int = 10):
        p = [torch.zeros_like(x)] * len(self.projections)
        x = [x] * len(self.projections)
        for _ in range(n_iter):
            for i, proj in enumerate(self.projections):
                x[i] = proj(x[i-1] + p[i])
                p[i] = x[i-1] - x[i] + p[i]
        return x[-1]


class Halpern(object):
    """Implements Halpern's Iteration method. Given a list of orthogonal projections
    onto convex sets, it converges to the orthogonal projection onto the intersection 
    (if nonempty); for reference, see
    
    Bauschke H. H.
    "The Approximation of Fixed Points of Compositions of Nonexpansive Mappings in Hilbert Space",
    In: Journal of Mathematical Analysis and Applications v202 
    pp. 150-159
    1996
    """
    def __init__(self, projections: list):
        """Initialize with list of callable projection objects.

         Arguments
        ---------
            projections: list
                list of callable projection objects (i.e., LipProj | ProjOnPlane | DistProj)
        """          
        self.projections = projections
        self.coeff = lambda n: 1/(1+n)

    def __call__(self, x: torch.Tensor, n_iter: int = 10):
        anchor = x.detach().clone()
        for i in range(n_iter * len(self.projections)):
            coeff = self.coeff(i)
            proj = self.projections[i % len(self.projections)]
            x = coeff * anchor + (1 - coeff) * proj(x)
        return x


class AlternatingProjections(object):
    """Implements a trivial alternating projections method."""

    def __init__(self, projections: list):
        """Initialize with list of callable projection objects.

         Arguments
        ---------
            projections: list
                list of callable projection objects (i.e., LipProj | ProjOnPlane | DistProj)
        """
        self.projections = projections

    def __call__(self, x: torch.Tensor, n_iter: int = 10):
        anchor = x.detach().clone()
        for i in range(n_iter * len(self.projections)):
            proj = self.projections[i % len(self.projections)]
            x = proj(x)
        return x


class LipProj(object):
    def __init__(
        self, 
        lipC, 
        n_iter: int = 10,
        mode: str = 'orthogonal', 
        input_type: str = 'conv', 
        svd_mode: str = 'svd', 
        sv_ratio: int = 8, 
        radial_params: dict ={}
    ):
        assert input_type in ['conv', 'fc'], 'Type {} not implemented'.format(input_type)
        assert mode in ['orthogonal', 'radial'], 'Mode {} not implemented'.format(mode)
        self.lipC = lipC
        self.n_iter = n_iter
        self.mode = mode
        self.input_type = input_type
        self.svd_mode = svd_mode
        self.sv_ratio = sv_ratio
        self.radial_params = radial_params

        if self.mode == 'orthogonal':
            if self.input_type == 'conv':
                self.transform = lambda x: torch.fft.fft2(x, dim=[2,3]).permute(2,3,0,1)
                self.inv_transform = lambda x: torch.fft.ifft2(x.permute(2,3,0,1), dim=[2,3]).real
            elif self.input_type == 'fc':
                self.transform = lambda x: x
                self.inv_transform = lambda x: x

    def _svd(self, x: torch.Tensor):
        if self.svd_mode == 'svd':
            U, S, V = torch.linalg.svd(x, full_matrices = False)
            U = torch.moveaxis(U,-1,0)
            S = torch.moveaxis(S,-1,0)
            V = torch.moveaxis(V,-2,0)
        elif self.svd_mode == 'scipy':
            x_cpu = x.detach().clone().cpu()
            dims = x.shape
            #TODO check if dims are right for non square matrices
            k = max(dims[3]//self.sv_ratio, 1)   
            U = torch.zeros(dims[0], dims[1], dims[2], k, dtype=x.dtype)
            S = torch.zeros(dims[0], dims[1], k)
            V = torch.zeros(dims[0], dims[1], k, dims[3], dtype=x.dtype)
            for i,j in itertools.product(range(dims[0]),range(dims[1])):
                u, s, vt = scipy.sparse.linalg.svds(x_cpu[i,j], k = k)
                U[i,j] += u
                S[i,j] += s
                V[i,j] += vt
            U = torch.moveaxis(U,-1,0).to(x.device)
            S = torch.moveaxis(S,-1,0).to(x.device)
            V = torch.moveaxis(V,-2,0).to(x.device)
        elif self.svd_mode == 'power_it':
            U, S, V = get_singular_values(x, self.lipC, self.n_iter)
        return U, S, V

    def _orthogonal_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the orthogonal projection of (the weight) of a network layer onto 
        the operator norm ball with a specified radius. Supports fully connected layers 
        and convolutional layers with circular padding and kernel size equal to the 
        input dimensions. see 

            Sedghi, Gupta & Long
            The Singular Values of Convolutional Layers
            https://arxiv.org/pdf/1805.10408.pdf

        Note:aAssumes weight tensor x to be permuted via x.permute(2,3,0,1) and to 
        be of kernel size equal to the input dimension.
        """
        transform_coefficients = self.transform(x)
        U, S, V = self._svd(transform_coefficients)
        factor = F.relu(S - self.lipC)
        clipped_transform_coefficients = transform_coefficients - torch.einsum('a...i,a...,a...j->...ij', U, factor, V)
        x_proj = self.inv_transform(clipped_transform_coefficients)
        return x_proj
 
    def _radial_projection(self, x: torch.Tensor) -> torch.Tensor:
        lip = self._get_lipschitz(x)
        if lip <= self.lipC:
            x_proj = x
        else:           
            x_proj = x * self.lipC / lip
        return x_proj

    def _get_lipschitz(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_type == 'conv':
            # TODO get outdims, indims
            u, lip, v = cyclicpad.cyclic_power_iteration(x, **self.radial_params)
            self.radial_params['us'] = u
        elif self.input_type == 'fc':
            _, lip, _ = power_it(x, self.n_iter)
        return lip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'orthogonal':
            return self._orthogonal_projection(x)
        elif self.mode =='radial':
            return self._radial_projection(x)


class ProjOnPlane(object):
    """Implements projecting a tensor of size (a0,...,a3) onto the hyperplane of 
    tensors whose last a0-b0,...,a3-b3 coordinates are zero.
    """

    def __init__(self, plane: list):
        self.plane = plane
        #self.mask = embed_plane(torch.ones(plane), in_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = embed_plane(torch.ones(self.plane, device=x.device), x.shape)
        return x * mask


class DistProj(object):
    def __init__(self,
                 r: float,
                 x0: torch.Tensor = None,
                 n_iter: int = 100,
                 mode: str = 'orthogonal'):
        assert r > 0., 'r={}, but r > 0 required'.format(r)
        assert mode in ['orthogonal',
                        'radial'], 'Mode {} not implemented'.format(mode)
        self.r = r
        self.x0 = x0
        self.n_iter = n_iter
        self.mode = mode

    def __call__(self, x: torch.Tensor):
        if self.x0 == None:
            self.x0 = torch.zeros_like(x)
        delta = x - self.x0
        if delta.norm(dim=1, p=2).norm(p=1) <= self.r:
            delta_proj = delta
        else:
            if self.mode == 'orthogonal':
                delta_proj = self._orthogonal_projection(delta)
            elif self.mode == 'radial':
                delta_proj = self._radial_projection(delta)
        return delta_proj + self.x0

    def _orthogonal_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection onto (2,1)-group norm constraint set. 
        see

        Liu, Ji & Ye 
        Multi-Task Feature Learning Via Efficient 2,1-Norm Minimization
        https://arxiv.org/pdf/1205.2631.pdf
        """
        x_norm = x.norm(dim=1, keepdim=True)    
        a = torch.zeros(1, device=x.device)
        b = x_norm.max()
        f = lambda l: self._dual(l, x_norm) 
        lambd = bisect(f, a, b)
        x_proj = (x_norm>lambd).float() * (1 - lambd/x_norm) * x
        #avoid problems caused by 0-divison
        x_proj[x.abs()<1e-8] = 0
        return x_proj

    def _dual(self, lambd, x_norm):
        x_max = F.relu(x_norm - lambd)
        return x_max.sum() - self.r

    def _radial_projection(self, x: torch.Tensor) -> torch.Tensor:  
        norm = x.norm(dim=1, p=2).norm(p=1)          
        x_proj = x * self.r / norm
        return x_proj


def embed_plane(x: torch.tensor, shape: list) -> torch.Tensor:
    assert len(x.shape) == len(shape)
    assert len(shape) == 4
    # Workaround for when inp dim < kernel size. Needs to be resolved
    if any([x.shape[i] > shape[i] for i in range(len(shape))]):
        return x
    y = torch.zeros(shape, device=x.device)
    y[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
    return y


def conv_power_it(W: torch.Tensor,
                  inp_dims: list,
                  out_dims: list,
                  us: torch.Tensor = None,
                  stride: int = 1,
                  padding: int = 1,
                  n_iter: int = 25):
    eps = 1e-12
    output_padding = inp_dims[-1] - ((out_dims[-1]-1)*stride-2*padding+1*(W.shape[-1]-1)+1)
    if inp_dims[-1] == 2:
        output_padding = 0

    #no grad required to avoid memory leaks
    with torch.no_grad():
        if us == None:
            us = torch.randn(out_dims, device = W.device)
        else:
            us = us.to(W.device)
        u = F.normalize(us.view(-1), dim=0, eps=1e-12).view(us.shape)

        # TODO implement padding mode = circular for F.transpose2d, F.conv2d 
        # as done in nn.Conv2d._conv_forward
        for _ in range(n_iter):
            vs = F.conv_transpose2d(
                    u, W, bias=None, stride=stride, padding=padding,
                    output_padding = output_padding)
            v = F.normalize(vs.contiguous().view(-1), dim=0, eps=eps).view(vs.shape)
            us = F.conv2d(v, W, bias=None, stride=stride, padding=padding)
            u = F.normalize(us.contiguous().view(-1), dim=0, eps=eps).view(us.shape)

        weight_v = F.conv2d(v, W, bias=None, stride=stride, padding=padding)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)

    return u, sigma, v


def bisect(f, a, b, n_iter=1000):
    assert torch.sign(f(a)) != torch.sign(
        f(b)), "invalid signs, fa = {} fb = {}".format(f(a), f(b))
    eps = 1e-8
    c = (a+b)/2
    fc = f(c)
    i = 0
    while abs(fc) > eps and i < n_iter:
        i += 1
        c = (a+b)/2
        fc = f(c)
        if torch.sign(fc) == torch.sign(f(a)):
            a = c
        else:
            b = c
    return c


def power_it(W: torch.Tensor, n_iter: int):
    v = torch.randn(*W.shape[:-2], W.shape[-1], dtype=W.dtype).to(W.device)
    v = F.normalize(v, dim=-1)

    for _ in range(n_iter):
        us = torch.einsum('...ij,...j->...i', W, v)
        u = F.normalize(us, dim=-1)
        vs = torch.einsum('...ji,...j->...i', W.conj(), u)
        v = F.normalize(vs, dim=-1)

    s = torch.einsum('...i,...ij,...j->...', u.conj(), W, v).abs()
    return u.unsqueeze(0), s.unsqueeze(0), v.unsqueeze(0)


def get_singular_values(W: torch.Tensor, r: float = 1e-12, n_iter=100):
    tmp = W.detach().clone()
    u, s, v = power_it(tmp, n_iter)
    while s[-1].max() > r:
        tmp = tmp - torch.einsum('...i,...,...j->...ij',
                                 u[-1], s[-1], v[-1].conj())
        u_i, s_i, v_i = power_it(tmp, n_iter)
        u = torch.cat((u, u_i), dim=0)
        s = torch.cat((s, s_i), dim=0)
        v = torch.cat((v, v_i), dim=0)
    return u, s, v.conj()
