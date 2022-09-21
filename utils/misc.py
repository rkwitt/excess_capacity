import os
import json
import torch
import torch.nn as nn
import numpy as np 
from typing import Tuple

from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader  
from torch.nn.functional import normalize, conv_transpose2d, conv2d

from .tiny import TinyImageNetDataset


__all__ = ['get_ds_stat', 'get_ds_and_dl', 'compute_margins', 'RampLoss']


def get_ds_stat(ds_name: str = 'cifar10') -> Tuple[Tuple,Tuple]:
    """Get dataset mean, std dev. statistics
    
    Parameters
    ----------
    ds_name: str 
        the name of the image dataset, i.e., either 
            'cifar10',
            'cifar100', or
            'tiny-imagenet-200'

    Returns
    -------
        channel_avg, channel_std: tuple, tuple
            two tuple containing the channel-wise average and std. deviation of the specified dataset 

    Raises
    ------
        NotImplementedError
            in case the dataset is not supported
    """
    if ds_name == 'cifar10':   
        return (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    elif ds_name == 'cifar100': 
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif ds_name == 'tiny-imagenet-200':
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError('Dataset {} not supported'.format(ds_name))
        

def get_ds_and_dl(ds_name: str, 
                  data_dir: str, 
                  batch_size: int = 128, 
                  num_workers: int = 4, 
                  limit_to: int = -1, 
                  randomize: bool = False) -> Tuple[Dataset, Dataset, DataLoader, DataLoader, int]:
    """Get PyTorch datasets and data loaders.

    Parameters
    ----------
        ds_name: str
            the name of the image dataset, i.e., either 
                'cifar10',
                'cifar100', or
                'tiny-imagenet-200'   
        data_dir: str
            the directory where the dataset is stored on harddisk
        batch_size: int (default: 128)
            batch size to use
        num_workers: int (default: 4)
            number of workers to use
        limit_to: int (default: -1)
            use a subset of size limit_to instead of full dataset if > 0
        randomize: bool (default: False)
            use randomized labels
    
    Returns
    -------
        ds_trn: torch.utils.data.Dataset 
            PyTorch dataset for training data
        ds_trn: torch.utils.data.Dataset 
            PyTorch dataset for testing data
        ds_trn: torch.utils.data.DataLoader 
            PyTorch data loader for training data
        ds_tst: torch.utils.data.DataLoader 
            PyTorch data loader for testing data
        num_classes: int 
            number of classes in dataset
    """

    trn_transforms=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*get_ds_stat(ds_name))])
    tst_transforms=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*get_ds_stat(ds_name))])
    
    if ds_name == 'cifar10':
        ds_trn = datasets.CIFAR10(
            os.path.join(data_dir, ds_name), 
            train=True, 
            download=True, 
            transform=trn_transforms)
        ds_tst = datasets.CIFAR10(
            os.path.join(data_dir, ds_name), 
            train=False, 
            download=True, 
            transform=tst_transforms)
    elif ds_name == 'cifar100':
        ds_trn = datasets.CIFAR100(
            os.path.join(data_dir, ds_name), 
            train=True, 
            download=True, 
            transform=trn_transforms)
        ds_tst = datasets.CIFAR100(
            os.path.join(data_dir, ds_name), 
            train=False, 
            download=True, 
            transform=tst_transforms)
    elif ds_name == 'tiny-imagenet-200':
        ds_trn = TinyImageNetDataset(
            os.path.join(data_dir, ds_name),
            mode='train', 
            preload=True, 
            transform=trn_transforms)
        ds_tst = TinyImageNetDataset(
            os.path.join(data_dir, ds_name),
            mode='val', 
            preload=True, 
            transform=tst_transforms)
    else:
        raise NotImplementedError('Dataset {} not supported'.format(ds_name))
    
    num_classes = len(ds_trn.classes)
    
    if randomize:
        if ds_name not in ['cifar10', 'cifar100']:
            raise NotImplementedError('Dataset {} not supported for randomization'.format(ds_name))
        with open(os.path.join(data_dir, '{}_rand.json'.format(ds_name)), 'r') as fp:
            random_labels = json.load(fp)
            ds_trn.targets = random_labels['ds_trn.targets']
            ds_tst.targets = random_labels['ds_tst.targets']    

    # TODO: check if range(...) is appropriate below
    if limit_to > 0:
        ds_trn = torch.utils.data.Subset(ds_trn, range(limit_to))

    dl_trn = DataLoader(
        ds_trn, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers)
    dl_tst = DataLoader(
        ds_tst,
        batch_size=256, 
        shuffle=False, 
        num_workers=num_workers)

    return ds_trn, ds_tst, dl_trn, dl_tst, num_classes


def compute_margins(inp: Tensor, tgt: Tensor) -> torch.Tensor:
        v_y = torch.gather(inp, 1, tgt.unsqueeze(1))

        n,d = inp.size()

        msk = torch.ones_like(inp).scatter_(1, tgt.unsqueeze(1), 0.)
        res = inp[msk.bool()].view(n, d-1)

        max_i_neq_j = res.topk(1,dim=1).values
        return -(v_y - max_i_neq_j)


class RampLoss(nn.Module):
    """Implements the ramp loss as defined in 

    Bartlett, Forster, Telgarsky
    Spectrally-normalized margin bounds for neural networks
    NeurIPS 2017
    arXiv: https://arxiv.org/abs/1706.08498

    Attributes
    ----------
        gamma: float (default: 1.0)
            the \gamma parameter of the ramp loss
    """
    def __init__(self, gamma: float=1.0):
        """
        Arguments
        ---------
            gamma: float (default: 1.0)
                the \gamma parameter of the ramp loss    
        """
        super(RampLoss, self).__init__()

        self.gamma = gamma

    def forward(self, inp: Tensor, tgt: Tensor) -> Tensor:
        r = compute_margins(inp, tgt)

        p0 = r<-self.gamma
        p1 = r>0.0

        out = 1+r.div(self.gamma)
        out[p0] = 0
        out[p1] = 1

        return out.mean(dim=0)