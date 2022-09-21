"""
taken from 

https://github.com/ColinQiyangLi/LConvNet/blob/abc14e1a17ae175bee02f57e35394a0c53fd725f/lconvnet/layers/utils.py#L60
"""

import torch
import torch.nn.functional as F


def cyclic_pad_2d(x, pads, even_h=False, even_w=False):
    """
    Implementation of cyclic padding for 2-D image input
    """
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


def conv2d_cyclic_pad(x, weight,  bias=None, stride = 1):
    # Seems strange that stride was not implemented in the repo.
    # Question: is the power iteration still right for stride = 2
    """
    Implementation of cyclic padding followed by a normal convolution; see
        https://github.com/ColinQiyangLi/LConvNet/blob/abc14e1a17ae175bee02f57e35394a0c53fd725f/lconvnet/layers/utils.py#L60
    """
    kh, kw = weight.size(-2), weight.size(-1)
    x = cyclic_pad_2d(x, [kh // 2, kw // 2], (kh % 2 == 0), (kw % 2 == 0))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return F.conv2d(x, weight, bias, stride=stride)

# define the forward affine operator (with bias)
#https://github.com/ColinQiyangLi/LConvNet/blob/abc14e1a17ae175bee02f57e35394a0c53fd725f/lconvnet/layers/utils.py#L60
def power_it_forward(x, w):
    return conv2d_cyclic_pad(
        x, w, bias = None
    )


# define the transpose linear operator (used in power iteration)
#https://github.com/ColinQiyangLi/LConvNet/blob/abc14e1a17ae175bee02f57e35394a0c53fd725f/lconvnet/layers/utils.py#L60
def power_it_transpose_forward(x, w):
    return conv2d_cyclic_pad(
        x, w.flip(
            [2, 3]).transpose(1, 0)
        )


def cyclic_power_iteration(W: torch.Tensor,
                inp_dims: list,
                out_dims: list,
                us: torch.Tensor = None,
                kernel_size: int = 3,
                stride: int = 1,
                padding: int = 1,
                n_iter: int = 25):
    # https://github.com/ColinQiyangLi/LConvNet/blob/abc14e1a17ae175bee02f57e35394a0c53fd725f/lconvnet/layers/utils.py#L60
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