![Logo](assets/cat.png)
-----------------------

This is the **official repository** for the NeurIPS '22 paper

F. Graf, S. Zeng, B. Rieck, M. Niethammer and R. Kwitt    
**Measuring Excess Capacity in Neural Networks**    
NeurIPS '22    
[arXiv](https://arxiv.org/abs/2202.08070)

In case you use code provided in this repository, please cite as:
```bibtex
@inproceedings{Graf22a,
    author    = {Graf, F. and Zeng, S. and Rieck, B. and Niethammer, M. and Kwitt, R.},
    title     = {Measuring Excess Capacity in Neural Networks},
    booktitle = {NeurIPS},
    year      = 2022}
```
*The artwork (cat) used is licensed from Sniggle Sloth.*

*We are currently updating the repo with additional notebooks and tutorials - a final version will be available ASAP after NeurIPS '22*

## Overview

The following sections describe (1) how to use our code to reproduce the experiments of the paper and (2) how to use our constrained layers in your own experiments.

- [Requirements](#Requirements)
- [Layers](#Layers)
- [Models](#Models)
- [Training](#Training)
- [Notebooks](#Evaluation_notebooks)

## Requirements

Our code has been tested on a system with four NVIDIA GTX 3090, running Ubuntu Linux 22.04, CUDA 11.4 (driver version 470.141.03) and PyTorch v1.12.1. Additional required Python packages are:

- `tensorboard` (v2.10.0, for visualizing training progress)
- `lightning-flash` (v0.8.0, for LARS/LAMB optimizer support)
- `termcolor` (for colored console output)

These packages can be installed via, e.g., 

```bash
pip install termcolor lightning-flash tensorboard
```

## Layers

All layers that are used in our models can be found in [`core/layers.py`](core/layers.py). This includes an implementation of a linear layer and a 2D convolution layer that supports enforcing constraints (1) on the Lipschitz constant of the corresponding map and (2) a (2,1)-group norm distance of the layers current weight to its initialization. These are the two primary *capacity-driving* terms in our Rademacher complexity bounds. Both constraints can be disabled, enabled separately, or jointly. The implementations of these
layers are `ConstrainedLinear` and `ConstrainedConv2d`.

Additionally, we provide support for a fixed linear classifier where weights are set to the vertices of a regular (#classes-1) simplex in the representation space right before that layer (e.g., the output of a flattened pooling layer or 2D convolution). 

### ConstrainedLinear

```python
import torch
from core.layers import ConstrainedLinear

layer = ConstrainedLinear(20, 10, lip_cond=0.6)

x = torch.rand(16,20)
o = layer(x)
print('Desired Lipschitz constant: {:0.2f}'.format(layer.lipC))
print('Computed Lipschitz constant at init: {}'.format(layer.lip()))

eps = 0.1 * torch.randn_like(layer.weight)
layer.weight.data += eps
print('Computed Lipschitz constant after perturbation: {}'.format(layer.lip()))

layer.project()
print('Computed Lipschitz constant after projection: {}'.format(layer.lip()))
```

If we would want to quickly test (empirically) whether that layer enforces the Lipschitz constraint, we can do the following: taking the definition of the Lipschitz constant, we write down a small optimization problem where we seek to identify vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ that maximize

$$
-\frac{\lVert f(\mathbf{x})-f(\mathbf{y})\rVert_2}{\lVert \mathbf{x}-\mathbf{y}\rVert_2}
$$

Here, $f$ denotes the function implemented by `ConstrainedLinear`. This works, because $f$ is $K$-Lipschitz continuous if

$$
\forall \mathbf{x},\mathbf{y} \in \mathbb{R}^n: \frac{\lVert f(\mathbf{x})-f(\mathbf{y})\rVert_2}{\lVert \mathbf{x}-\mathbf{y}\rVert_2} \leq K
$$

A unit test for this case can be found in `tests/test_layers.py` (`TestConstrainedLinear`). 

### ConstrainedConv2d

tbd.


### Using layers in your own code

tbd.

## Models

All models we used in our experiments reside within the `models` subdirectory.

### Simple 6-layer network 

This network (`simple_convnet6`) is used in our bound comparison in the paper and is an instance of the `SimpleConvnet` class within `models/simple_convnet.py`. It is a very simple architecture consisting of `ConstrainedConv2d` layers with ReLU activations and optional maximum pooling (in case striding is set to 1). *All inputs are assumed to be of spatial size $32 \times 32$.

The following example instantiates such a 6-layer network with a fixed simplex classifier and 128 channels throughout the architecture. No constraints on the Lipschitz constants and (2,1)-group norm distance to initialization are given (i.e., the default setting).

```python
from models.simple_convnet import simple_convnet6

net = simple_convnet6(num_classes = 10,
                      n_channels = 64,
                      simplex=True,
                      dist_cond = [np.inf, 
                                   np.inf, 
                                   np.inf, 
                                   0 # use SimplexClassifier
                                   ],
                      lip_cond = [np.inf, 
                                  np.inf, 
                                  np.inf, 
                                  np.inf])
```

**Note**: to actually use the `SimplexClassifier` (from `core/layers.py`), we require that the last entry in `dist_cond = [np.inf, np.inf, np.inf, 0]` is 0 (for consistency reasons). The following table lists how to select different classifier types if `simplex=True` (in the table, $C$ denotes the number of classes):

| Classifier | `dist_cond`  | `lip_cond`  | $K$ |
|---|---|---|---|
| `SimplexClassifier`  |  `[np.inf, np.inf, np.inf, 0]` | `[np.inf, np.inf, np.inf, a]` | $a$, if $a<\sqrt\frac{C}{C-1}$| 
| `SimplexClassifier`  |  `[np.inf, np.inf, np.inf, 0]` | `[np.inf, np.inf, np.inf, 10.0]` | $\sqrt\frac{C}{C-1}$|  
| `ConstrainedLinear`  |  `[np.inf, np.inf, np.inf, np.inf]`| `[np.inf, np.inf, np.inf, a] `  | $a$ |   

In case `simplex=False`, then `ConstrainedConv2d` is used as the classification layer, with Lipschitz constant and distance to initialization set to the desired values.

## Training

The main training code can be found in [`train.py`](train.py). Using 
```python
python train.py --help
```
displays the description of all command-line parameters. At the moment, almost all logging is handeled via tensorboard. Hence, to monitor progress, we recommend a setup like 

```bash
mkdir log
screen -S ipys
tenorboard --logdir log
```
and then specifying `log` as the logging folder via `--logdir log`. As an example, consider training our variant of a  PreAct-Resnet18 model with a `SimplexClassifier`. In our implementation, we control the Lipschitz constaint and (2,1)-group norm distance constraint to a layers initialization for

1. the initial convolution layer (`conv1` in `PreActResNet`), 
2. the layers in each residual block, 
3. the shortcut, and 
4. the classifier (i.e., the final map of the network). 
    
In general, this yields a total of **eight** (2 $\times$ 4) constraints that need to be set. ]\ it is only six, as the shortcut implementation from the paper (via max-pooling) is a fixed map. The command-line arguments which control the constraints are `--lip` and `--dist`. For instance `--lip 2.0 1.0 1.0 1.0` sets the Lipschitz constraint for the initial convolutional layer in the residual network to $2.0$, the Lipschitz constraints for each layer within the residual blocks to $1.0$ and the Lipschitz constraint for the classifier to $1.0$ as well. Similarly, `--dist 90. 90. 90. 0` would set the corresponding (2,1)-group norm distance to initialization constraints. Note here that the last value is $0$ which selects the `SimplexClassifier` (i.e., a fixed map). On CIFAR10 with $C=10$ classes, the Lipschitz constant of this fixed classifier would be $\sqrt\frac{10}{9}=1.111$ and our choice of  `--lip 2.0 1.0 1.0 1.0` from above would scale the weights accordingly (using the largest singular value) to achieve the desired Lipschitz constraint of $1.0$ (last value for `--lip`). Below is a pretty self-explanatory call to `train.py` for a run on CIFAR10 (using SGD per default):

```python
python train.py --bs 256 \
    --lip 2.0 0.8 1.4 1.0 \ # see paragraph above
    --dist 90 90 90 0 \     # see paragraph above
    --device cuda:0  \      # run on cuda:0 device
    --comment testing \     # comment for tensorboard
    --epochs 200 \          # train for 200 epochs
    --n_proj 1 \            # perform one iteration of alternating proj.
    --proj_freq 15 \        # project every 15th optimizer step
    --datadir data \        # folder where cifar10 dataset resides
    --dataset cifar10 \     # cifar10 dataset
    --lr .003 \             # learning rate
    --logdir log
```

The training progress (train/test loss & error) as well as logging of the current contraints per layer are then visualized in the tensorboard. In similar manner, we can train on `cifar100` or `tiny-imagenet-200`.

## Notebooks



