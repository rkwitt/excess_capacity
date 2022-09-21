![Logo](assets/cat.png)
-----------------------

This is the **official repository** for the NeurIPS '22 paper

F. Graf, S. Zeng, B. Rieck, M. Niethammer and R. Kwitt    
**Measuring Excess Capacity in Neural Networks**    
NeurIPS '22    
[arXiv](https://arxiv.org/abs/2202.08070)

In case you use code provided here, please cite as:
```bibtex
@inproceedings{Graf22a,
    author    = {Graf, F. and Zeng, S. and Rieck, B. and Niethammer, M. and Kwitt, R.},
    title     = {Measuring Excess Capacity in Neural Networks},
    booktitle = {NeurIPS},
    year      = 2022}
```
*The artwork used is licensed from Sniggle Sloth.*

## Overview

The following sections describe how to use the code in order to reproduce the experiments of the paper.

- [Requirements](#Requirements)
- [Layers](#Layers)
- [Models](#Models)
- [Training](#Training)
- [Evaluation notebooks](#Evaluation_notebooks)

## Requirements

Our code has been tested on a system with four NVIDIA GTX 3090 cards, running Ubuntu Linux 22.04, CUDA 11.4 (driver version 470.141.03) and PyTorch v1.12.1. Additional required Python packages are:

- `tensorboard` (v2.10.0, for visualizing training progress)
- `lightning-flash` (v0.8.0, for LARS/LAMB optimizer support)
- `termcolor` (for colored console output)

which can all be installed using 

```bash
pip install termcolor lightning-flash tensorboard
```

### Layers

All layers that are used in our models can be found in [`core/layers.py`](core/layers.py). This includes an implementation of a linear layer and a 2D convolution layer that supports enforcing constraints (1) on the Lipschitz constant of the corresponding map and (2) a (2,1)-group norm distance of the layers current weight to its initialization. These are the two primary capacity driving terms in our Rademacher complexity bounds. Both constraints can be disabled, enabled separately, or jointly. The implementations of these
layers are `ConstrainedLinear` and `ConstrainedConv2d`.

Additionally, we provide support for a fixed classifier (i.e., a linear map) which weights set to the vertices of a regular (#classes-1) simplex in the 
representation space right before that layer (e.g., the output of a flattened pooling layer or 2D convolution). 

#### ConstrainedLinear

```python
from core.layers import ConstrainedLinear

layer = ConstrainedLinear(20,10, lipC=0.8)

x = torch.rand(16,20)
o = layer(x)
print('Desired Lipschitz constant: {:0.2f}'.format(layer.lipC))
print('Computed Lipschitz constant: {}'.format(layer.lipC()))
```

Taking the definition of the Lipschitz constant, we can easily write down a small optimization problem where we seek to identify vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ that maximize

$$
-\frac{\lVert f(\mathbf{x})-f(\mathbf{y})\rVert_2}{\lVert \mathbf{x}-\mathbf{y}\rVert_2}
$$

where $f$ denotes the function implemented by the `ConstrainedLinear` layer. This works, because $f$ is $K$-Lipschitz continuous if

$$
\forall \mathbf{x},\mathbf{y} \in \mathbb{R}^n: \frac{\lVert f(\mathbf{x})-f(\mathbf{y})\rVert_2}{\lVert \mathbf{x}-\mathbf{y}\rVert_2} \leq K
$$

This gives an *empirical assessment* of whether our layer enforces the desired Lipschitz constraint. A unit test for this case can be found in `tests/test_layers.py` (`TestConstrainedLinear`). An equivalent test for the `ConstrainedConv2d` layer is implemented in  `tests/test_layers.py` (`TestConstrainedConv2d`).

### Models

All models reside within the `models` subdirectory.

**Simple 6-layer network** (`simple_convnet6`)

This network is used in our bound comparison in the paper and is an instance of the `SimpleConvnet` class within `models/simple_convnet.py`. It is a very simple architecture consisting of `ConstrainedConv2d` layers with ReLU activations and optional maximum pooling (in case striding is set to 1). *All inputs are assumed to be of spatial size $32 \times 32$.

The following example instantiates such a 6-layer network with a fix simplex classifier and 128 channels throughout the architecture. No constraints on the Lipschitz constants and (2,1)-group norm distance to initialization are given (i.e., the default setting).

```python
from models.simple_convnet import simple_convnet6

net = simple_convnet6(n_classes = 10,
                      n_channels = 64,
                      simplex=True)
```
In similar manner, we can instantiate the same network, but with Lipschitz and distance to initialization constraints:

```python
from models.simple_convnet import simple_convnet6

net = simple_convnet6(num_classes = 10,
                      n_channels = 64,
                      simplex=True,
                      dist_cond = [np.inf, np.inf, np.inf, 0])
```

Note, that to actually use the `SimplexClassifier` (from `core/layers.py`), we require that the last entry in `dist_cond = [np.inf, np.inf, np.inf, 0]` is 0 for consistency reasons. The following table lists how to select different classifier types if `simplex=True` is set (here $C$ is the number of classes):

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
displays the description of all command-line parameters. As an example, consider 

```python
python train.py \
    --bs 256 \
    --lip 2.0 1.0 1.0 1.0 \
    --dist -1 -1 -1 0 \
    --device cuda:0 \
    --lr .03 \
    --comment simple_convnet6_cifar10 \
    --epochs 200 \
    --n_proj 1 \
    --proj_freq 10 \
    --datadir data/cifar10 \
    --dataset cifar10 \
    --arch simple_convnet6 \
    --n_channels  256
```

which trains a `simple_convnet6` (with 256 channels) with a fixed `SimplexClassifier` (Lipschitz constraint of $1.0$) and no distance to initialization constraint on CIFAR10, using SGD with learning rate set to 0.03 and a batch size of 256 for 200 epochs.

### Evaluation notebooks



