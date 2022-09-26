"""This implementation of cyclic padding is taken (with minor modifications) from:?

https://github.com/ColinQiyangLi/LConvNet/blob/abc14e1a17ae175bee02f57e35394a0c53fd725f/lconvnet/layers/utils.py#L60
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


__all__  = ['conv2d_cyclic_pad', 'cyclic_power_iteration']


def cyclic_pad_2d(x: torch.Tensor,
                  pads: List[int],
                  even_h: bool = False,
                  even_w: bool = False) -> torch.Tensor:
    """Implementation of cyclic padding for 2-D image input."""
    pad_change_h = -1 if even_h else 0
    pad_change_w = -1 if even_w else 0
    pad_h, pad_w = pads
    if pad_h != 0:
        v_pad = torch.cat((x[..., :, -pad_h:, :], x,
                           x[..., :, :pad_h+pad_change_h, :]), dim=-2)
    elif pad_change_h != 0:
        v_pad = torch.cat((x, x[..., :, :pad_change_h, :]), dim=-2)
    else:
        v_pad = x
    if pad_w != 0:
        h_pad = torch.cat((v_pad[..., :, :, -pad_w:],
                           v_pad, v_pad[..., :, :, :pad_w+pad_change_w]), dim=-1)
    elif pad_change_w != 0:
        h_pad = torch.cat((v_pad, v_pad[..., :, :, :+pad_change_w]), dim=-1)
    else:
        h_pad = v_pad
    return h_pad


def conv2d_cyclic_pad(x: torch.Tensor,
                      weight: torch.Tensor,
                      bias: Optional[torch.Tensor] = None,
                      stride: int = 1) -> torch.Tensor:
    """Implementation of cyclic padding followed by a normal convolution.

    Arguments
    ---------
        x: torch.Tensor
            Input image tensor
        weight: torch.Tensor
            Weight tensor of the 2D convolution
        bias: Optional[torch.Tensor]
            Bias tensor of the 2D convolution
        stride: int
            Stride to use

    Returns
    -------
        out: torch.Tensor   
            Output of a normal 2D convolution, applied to the cyclic-padded 
            image input tensor x.

    Remark: Striding was not implemented in the original implementation. 
    NOTE: check if power iteration is still corrected for stride = 2    
    """
    kh, kw = weight.size(-2), weight.size(-1)
    x = cyclic_pad_2d(x, [kh // 2, kw // 2], (kh % 2 == 0), (kw % 2 == 0))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return F.conv2d(x, weight, bias, stride=stride)


def power_it_forward(x, w):
    """Forward affine operator (without bias)"""
    return conv2d_cyclic_pad(
        x, w, bias=None
    )


def power_it_transpose_forward(x, w):
    """Transpose linear operator (used in power iteration)"""
    return conv2d_cyclic_pad(
        x, w.flip(
            [2, 3]).transpose(1, 0)
    )


def cyclic_power_iteration(W: torch.Tensor,
                           inp_dims: torch.Size,
                           out_dims: torch.Size,
                           us: Optional[torch.Tensor] = None,
                           stride: int = 1,
                           padding: int = 1,
                           n_iter: int = 25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Power iteration to compute largest singular value, as well as left and
    right singular vectors.

    Arguments
    ---------
        W: torch.Tensor 
            Weight tensor of 2D convolution
        inp_dims: torch.Size
            Size of input 
        out_dims: torch.Size
            Size of output
        us: Optional[torch.Tensor]
            Starting u vector
        stride: int 
            Striding to use
        padding: int 
            Amount of (cyclic) padding
        n_iter: int
            Number of power iterations 
    
    Returns
    -------
        u: torch.Tensor
            Left-singular vector corresponding to the largest singular value
        s: torch.Tensor
            Largest singular value
        v: Torch.Tensor
            Right-singular vector corresponding to the largest singular value
    """
    eps = 1e-12
    output_padding = inp_dims[-1] - ((out_dims[-1]-1)*stride-2*padding+1*(W.shape[-1]-1)+1)

    #no grad required to avoid memory leaks
    with torch.no_grad():
        if us == None:
            us = torch.randn(out_dims, device = W.device)
        else:
            us = us.to(W.device)
        u = F.normalize(us.view(-1), dim=0, eps=1e-12).view(us.shape)

    # power iteration using the stored vector u
    with torch.no_grad():
        for _ in range(n_iter):
            vs = power_it_transpose_forward(u, W)
            v = F.normalize(vs.contiguous().view(-1), dim=0,
                        eps=eps).view(vs.shape)
            us = power_it_forward(v, W)
            u = F.normalize(us.view(-1), dim=0, eps=eps).view(us.shape)

    # compute an estimate for the maximum singular value sigma
    weight_v = power_it_forward(v.detach(), W)
    weight_v = weight_v.view(-1)
    sigma = torch.dot(u.detach().view(-1), weight_v)

    return u, sigma, v