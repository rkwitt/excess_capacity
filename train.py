"""Main training/evaluation code"""

import os
import sys
import pickle
import socket
import argparse
import numpy as np
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils.logger import *
from utils.misc import get_ds_and_dl, compute_margins
from models.nostride_preact_resnet import nostride_preactresnet18
from models.simple_convnet import simple_convnet6, simple_convnet11
from flash.core.optimizers import LAMB, LARS


def setup_argparse():
    """Setup command-line parsing"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--lip",
                        nargs=4,
                        help="Lipschitz constraint in format: conv1, in block, shortcut, classifer",
                        type=float,
                        default=[-1, -1, -1, -1])
    parser.add_argument("--dist",
                        nargs=4,
                        help="Distance constraint in format: conv1, in block, shortcut, classifier",
                        type=float,
                        default=[-1, -1, -1, -1])
    parser.add_argument("--n_proj",
                        help="Number of alternating projections",
                        type=int,
                        default=1)
    parser.add_argument("--proj_freq",
                        help="Number of training batches after which the projection is applied",
                        type=int,
                        default=15)
    parser.add_argument("--bs",
                        help="Batch size",
                        type=int,
                        default=256)
    parser.add_argument("--epochs",
                        help="Number of training epochs",
                        type=int,
                        default=100)
    parser.add_argument("--lr",
                        help="Learning rate",
                        type=float,
                        default=0.003)
    parser.add_argument("--wd",
                        help="Weight decay",
                        type=float,
                        default=1e-4)
    parser.add_argument("--momentum",
                        help="Momentum",
                        type=float,
                        default=.9)
    parser.add_argument("--device",
                        help="Device to run on, e.g., cuda:0",
                        type=str,
                        default='cuda:0')
    parser.add_argument("--comment",
                        help="Comment for tensorboard",
                        type=str,
                        default='')
    parser.add_argument("--limit_to",
                        help="Limit training to N samples",
                        type=int,
                        default=-1)
    parser.add_argument("--n_power_it",
                        help="Number of iterations for computing Lipschitz constant",
                        type=int,
                        default=20)
    parser.add_argument('--logdir',
                        help="Directory for tensorboard logs",
                        type=str,
                        default=None)
    parser.add_argument('--datadir',
                        help="Data directory",
                        type=str,
                        default='/tmp/data')
    parser.add_argument('--bn',
                        help="Use batch normalization",
                        type=bool,
                        default=False)
    parser.add_argument('--rand',
                        help="Randomize training labels (works for --dataset cifar10 | cifar100)",
                        action="store_true",
                        default=False)
    parser.add_argument('--dataset',
                        help='Dataset to use (cifar10, cifar100)',
                        type=str,
                        choices=['cifar10', 'cifar100', 'tiny-imagenet-200'],
                        default='cifar10')
    parser.add_argument('--proj_mode',
                        help='Projection type',
                        choices=['orthogonal', 'radial'],
                        type=str,
                        default='orthogonal')
    parser.add_argument('--save_model',
                        help='Models (@epochs 0 and epochs) will be saved to logdir',
                        dest='save_model',
                        action='store_true',
                        default=False)
    parser.add_argument('--arch',
                        help='Architecture of the model',
                        choices=['no_stride_preactresnet18',
                                 'simple_convnet6',
                                 'simple_convnet11'],
                        type=str,
                        default='no_stride_preactresnet18')
    parser.add_argument('--optimizer',
                        help='Optimizer to use; for adam, --momentum and --wd are unused!',
                        choices=['sgd', 'adam', 'lars', 'lamb'],
                        type=str,
                        default='sgd')
    # The following arguments only apply to simple_convnet
    parser.add_argument('--n_channels',
                        help='Defines the number of channels of each layer of simple_convnetX',
                        type=int,
                        default=512)

    args = parser.parse_args()
    return args


def train_model(model,      # Model
                dl,         # Data loader
                optimizer,  # Optimizer
                loss_fn,    # Callable loss function
                device,     # Device to run on
                n_proj,     # Number of projection iterations
                proj_freq,  # Project every proj_freq update (within one epoch)
                i,          # Epoch counter
                writer,     # Summary writer
                ) -> Tuple[float, float, float]:
    """Training pass"""

    model.train()
    loss = 0
    acc = 0
    margins = []

    for j, (x, y) in enumerate(dl):
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)
        z = model(x)
        l = loss_fn(z, y)
        l.backward()

        optimizer.step()

        loss += l.item()*x.shape[0]
        acc += z.argmax(dim=1).eq(y).sum().item()
        margins += compute_margins(z, y).detach().tolist()

        with torch.no_grad():
            if n_proj > 0:
                if (j+1) % proj_freq == 0 or j == len(dl):
                    model.project(n_proj)
            if ((j+1) % (proj_freq//5) == 0 and n_proj > 0) or j+1 == len(dl):
                for attr in ["dist", "lip"]:
                    writer.add_scalars(
                        "Condition/"+attr, model.get_layer_attr(attr), j + 1 + i*len(dl))

    loss /= len(dl.dataset)
    acc /= len(dl.dataset)
    margins = np.array(margins)

    return acc, loss, margins


def eval_model(model,   # Model
               dl,      # Data loader
               device,  # Device 
               loss_fn  # Loss function
               ) -> Tuple[float, float, float]:
    """Evaluation pass"""

    model.eval()
    loss = 0
    acc = 0
    margins = []

    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            z = model(x)
            l = loss_fn(z, y)

            loss += l.item()*x.shape[0]
            acc += z.argmax(dim=1).eq(y).sum().item()
            margins += compute_margins(z, y).detach().tolist()

    loss /= len(dl.dataset)
    acc /= len(dl.dataset)
    margins = np.array(margins)

    return acc, loss, margins


def main():
    args = setup_argparse()
    for i in range(4):
        if args.dist[i]==-1:
            args.dist[i] = np.inf
        if args.lip[i]==-1:
            args.lip[i] = np.inf
    print(args)

    if args.logdir is None:
        log_dir = os.path.join('runs/{:}/lip_{:}__dist_{:}'.format(
            args.comment,
            args.lip,
            args.dist), datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    else: 
        log_dir = args.logdir

    writer = SummaryWriter(log_dir=log_dir)
    pickle.dump(args, open(os.path.join(log_dir, 'args.pkl'), "wb"))

    ds_trn, ds_tst, dl_trn, dl_tst, num_classes = get_ds_and_dl(args.dataset,
                                                                args.datadir,
                                                                batch_size=args.bs,
                                                                num_workers=4,
                                                                limit_to=args.limit_to,
                                                                randomize=args.rand)

    norm_layer = nn.BatchNorm2d if args.bn else nn.Identity
    if args.arch == 'no_stride_preactresnet18':
        model = nostride_preactresnet18(num_classes=num_classes,
                                        norm_layer=norm_layer,
                                        lip_cond=args.lip,
                                        dist_cond=args.dist,
                                        proj_mode=args.proj_mode).to(args.device)
    elif args.arch == 'simple_convnet6':
        model = simple_convnet6(num_classes,
                                args.n_channels).to(args.device)
    elif args.arch == 'simple_convnet11':
        model = simple_convnet11(num_classes,
                                 args.n_channels).to(args.device)
    else:
        raise NotImplementedError(args.arch + 'not a valid architecture type')

    # run one forward pass to initialize model
    model(torch.zeros_like(next(iter(dl_trn))[0].to(args.device)))  
    print(model)

    for attr in ["dist", "lip"]:
        writer.add_scalars("Condition/"+attr, model.get_layer_attr(attr), 0)
 
    if args.save_model:
        torch.save(model.state_dict(),
                   os.path.join(log_dir, 'model_0.pt'))

    try:
        optimizer = {
            'sgd':   SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum),
            'adam':  Adam(model.parameters(), lr=args.lr),
            'lars':  LARS(model.parameters(), lr=args.lr, momentum=args.momentum),
            'lamb':  LAMB(model.parameters(), lr=args.lr)
        }[args.optimizer]
    except KeyError:
        report_error('Optimizer {} not supported!'.format(args.optimizer))

    if args.arch in ["simple_convnet"+str(i) for i in [6, 11]] + ["nostride_simple_convnet"+str(i) for i in [6, 11]]:
        scheduler = CosineAnnealingLR(optimizer,
                                      args.epochs)
    else:
        scheduler = MultiStepLR(optimizer,
                                milestones=[60, 120, 160],
                                gamma=0.2)

    loss_fn = nn.CrossEntropyLoss()

    # currently, only works if full dataset is available in Dataset (e.g., cifar10/100)
    if hasattr(model, 'generalization_bound'):
        if hasattr(ds_trn, 'data') or hasattr(ds_trn, 'dataset'):
            gamma_range = list(range(1, 101))
            generalization_bounds = model.generalization_bound(ds_trn, gamma_range)
            for gen_bound, gamma in zip(generalization_bounds, gamma_range):
                writer.add_scalar('Generalization Bound', gen_bound, gamma)

    for epoch_i in range(args.epochs+1):
        trn_acc, trn_loss, trn_margins = train_model(model,
                                                     dl_trn,
                                                     optimizer,
                                                     loss_fn,
                                                     args.device,
                                                     args.n_proj,
                                                     args.proj_freq,
                                                     epoch_i,
                                                     writer)

        tst_acc, tst_loss, tst_margins = eval_model(model,
                                                    dl_tst,
                                                    args.device,
                                                    loss_fn)

        scheduler.step()
   
        writer.add_scalar('Accuracy/Train', trn_acc, epoch_i)
        writer.add_scalar('Accuracy/Test', tst_acc, epoch_i)
        writer.add_scalar('Accuracy/Gap', trn_acc - tst_acc, epoch_i)
        writer.add_scalar('Loss/Train', trn_loss, epoch_i)
        writer.add_scalar('Loss/Test', tst_loss, epoch_i)
        writer.add_scalar('Loss/Gap', trn_loss - tst_loss, epoch_i)
        
        if epoch_i % 10 == 0 or epoch_i == args.epochs - 1:
            model.reset_uv()
            spectral_comp = model.spectral_complexity()
            writer.add_histogram('Margins/Train', trn_margins, epoch_i)
            writer.add_histogram('Margins/Test', tst_margins, epoch_i)
            writer.add_histogram('Normalized Margins/Train', trn_margins/spectral_comp, epoch_i)
            writer.add_histogram('Normalized Margins/Test', tst_margins/spectral_comp, epoch_i)
            if args.save_model:
                torch.save(model.state_dict(),
                           os.path.join(log_dir, 'model_{}.pt'.format(epoch_i)))

        report_progress('Epoch {:04d} | Train-loss: {:0.4f} | Test-loss: {:0.4f} | Train acc.: {:0.4f} | Test acc.: {:0.4f}'.format(
            epoch_i + 1, trn_loss, tst_loss, trn_acc, tst_acc))

    if args.n_proj>0:
        model.project(15)
    trn_acc, trn_loss, _ = eval_model(model, dl_trn, args.device, loss_fn)
    tst_acc, tst_loss, _ = eval_model(model, dl_tst, args.device, loss_fn)

    writer.add_hparams(
        {
            'dist/conv1': args.dist[0], 'dist/block': args.dist[1],
            'dist/sc': args.dist[2], 'dist/fc': args.dist[3],
            'lip/conv1': args.lip[0], 'lip/block': args.lip[1],
            'lip/sc': args.lip[2], 'lip/fc': args.lip[3],
            'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.wd,
            'ds_size': args.limit_to, 'bs': args.bs, 'epochs': args.epochs
        },
        {
            'trn_acc': trn_acc,
            'tst_acc': tst_acc,
            'trn_loss': trn_loss,
            'tst_loss': tst_loss,
            'gap_acc': trn_acc - tst_acc,
            'gap_loss': trn_loss - tst_loss
        }
    )
    writer.close()

    if args.save_model:
        torch.save(model.state_dict(),
                   os.path.join(log_dir, 'model_{}.pt'.format(args.epochs)))

    return 0


if __name__ == '__main__':
    sys.exit(main())
