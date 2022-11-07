# On Measuring Excess Capacity in Neural Networks

```bibtex
@inproceedings{Graf22a,
    author    = {Graf, F. and Zeng, S. and Rieck, B. and Niethammer, M. and Kwitt, R.},
    title     = {Measuring Excess Capacity in Neural Networks},
    booktitle = {NeurIPS},
    year      = 2022}
```

# Documentation to rademacher_bounds.Bounds

This class allows computation of several Rademacher complexity bounds from the liteature. 
Implemented are
- ours_norms: First bound from Theorem 3.5
- ours_params: Second bound from Theoem 3.5
- bartlett: Lemma A.8 in Bartlett et al., NeurIPS 2017
- ledent_main: Theorem 11 in Ledent et al., AAAI 2021
- ledent_fixed_constraints: Considers Lipschitz constraints for all layers as in Section E ofLedent et al., AAAI 2021
- lin: Lemma 18 in Lin & Zhang, arxiv preprint 1910.01487
- neyshabur_1inf: Corollary 2 in Neyshabur et al., COLT 2015
- golowich_1inf: Theorem 2 in Golowich et al., COLT 2018
- gouk_1inf: Theorem 1 in Gouk et al., ICLR 2021
- neyshabur_fro:  Corollary 2 in Neyshabur et al., COLT 2015
- golowich_fro: Theorem 1 in Golowich et al., COLT 2018
- gouk_fro: Theorem 2 in Gouk et al., ICLR 2021

On init, the following quantities are computed
```python
    """
    Attributes
    ----------
        model: nn.Module
            model for which the Rademacher complexity bound is computed
        device: str
            model device
        data: torch.Tensor
            concatenation of all data points
        n: torch.Tensor 
            length of dataset
        classes: int
            number of classes in the data
        margin: torch.Tensor
            classification margins achieved by the model
        dl: DataLoader
            loader of training data
        layers: List[nn.Module]
            list of all parametric layers
        L: int
            network depth, number of parametric layers
        weights: list[torch.Tensor]
            list of weight tensors
        deltas: list[torch.Tensor]
            list of deviation from initialization of each layer
        strides: torch.Tensor
            strides of all layers
        kernel_sizes: torch.Tensor
            kernel sizes of all layers
        inp_dims: torch.Tensor
            spatial input dimension of each layer
        in_channels: torch.Tensor
            input channels to each layer
        params: torch.Tensor:
            number of parameters of each layer
        mat_params: torch.Tensor
            number of parameters the matrices corresponding to the convolutional map would have
        self.lip = torch.tensor([layer.lip() for layer in self.layers]).type(torch.float64)

        norm22: torch.Tensor
            l2 norms of weight tensors
        norm21: torch.Tensor
            l21 norms of weight tensors
        matrix_norm21: torch.Tensor
            l21 norms of corresponding matrices, divided by output_dim**2
        norm1inf: torch.Tensor
            l1inf norms of weight tensors

        dist22: torch.Tensor
            l2 distances to initialization
        dist21: torch.Tensor
            l21 distances to initialization
        matrix_dist21: torch.Tensor
            l21 distances of corresponding matrices, divided by output_dim**2
        dist1inf: torch.Tensor
            l1inf distances of to initialization

        bounds: List[str]
            list of implemented bounds
        quantities: Dict[str, float]
            {
            'lip_prod':  product of Lipschitz constants,
            'fro_prod':  product of l2 norms,
            '1inf_prod': product of l1inf norms
            }
    """
```

# Implementation of the bounds

## Architecture

We consider a hypothesis class $\mathcal F$ represented by a neural network of the form
$$f = \sigma_L \circ f_L \circ \dots \circ \sigma_1 \circ f_1$$ 
where $\sigma_i : x \mapsto \max(x, 0)$ denotes the ReLU activation function and $f_i$ identifies a convolutional layer parametrized by the tensor $K_i \in \mathbb R^{c_{i+1} c_i k_i^2}$. Depending on the bound, the layers need to satisfy different constraints. We will denote Lipschitz constraints by $s_i$. Norm/distance constraints will be denoted by the correspondong quanitiy, e.g. $\lVert K_i \rVert_2 should be understood as a contraint on the $l_2$ norm of the tensors $K_i$.

Note that fully-connected layers, e.g., a linear classifier at the last layer, can be handled by setting the spatial input dimension, the kernel size and the stride all equal to 1.

Bounds for fully-connected layers will be applied to the matrices corresponding to the linear map defined by the convolution.
Covering number based bounds will be for the class $\mathcal F_\gamma$ of the composition of functions $f\in \mathcal F$ with the ramp loss $\ell_\gamma$ with margin parameter $\gamma$. Layer-peeling based bounds, originally
presented for binary classification, are multiplied by the number of classes $\kappa$ (according to [A vector-contraction inequality for Rademacher complexities, A. Maurer, ALT, 2016]).


## Our bounds (Theorem 3.5)
The following terms $C_i$ quantify the part of a layer’s capacity attributed to weight and data norms.

$$
\tilde C_{i}(X)=
    \frac 4\gamma \, \frac{\lVert X\rVert}{\sqrt{n}} 
    \left(
        \prod_{\substack{l=1}}^{L}
            s_l
    \right)
    \frac{b_{i}}{s_{i}}
$$

Let $\gamma>0$ and let $H_{n-1}= \sum_{m=1}^{n-1} \frac{1}{m} = \mathcal O(\log(n))$ denote the $(n-1)$-th harmonic number. Then, the empirical Rademacher complexity of $\mathcal F_\gamma$ satisfies 

$$       
        {R}_S({\mathcal F_\gamma})
        \le
        \frac{4}{n}
        +
        \frac{12 H_{n-1}}{\sqrt{n}}
        \sqrt{\log(2W)}
        \left(
            \sum_{i=1}^L
            \lceil\tilde C_{i}^{{2}/{3}}\rceil
        \right)^{\!3/2}
    \tag{$\clubsuit$}
$$
and
$$
    {
        {R}_S({\mathcal F_\gamma})
        \le
        \frac{12}{\sqrt n}
        \sqrt{
            \sum_{i=1}^L
            2 W_{i}
            \log\left(1+\lceil{L ^2 \tilde C_{i}^2}\rceil\right) + 
            \psi\left(\lceil{L \tilde C_{i}}\rceil\right)
        }
    }
    \enspace, 
    \tag{$\spadesuit$}
$$
where $\psi$ is a monotonically increasing function, satisfying $\psi(0)=0$ and $\forall x: \psi(x)<2.7$.

```python
def ours_norms(self) -> float:
        """First bound from Theorem 3.5"""
        C_i = 4/self.margin * self.data.norm()/self.n.sqrt() * self.lip.prod() * self.dist21/self.lip
        W = self.params.max()

        norms_term = (C_i**(2/3)).sum()**(3/2)
        params_term = (2*W).log().sqrt()

        bound = 4/self.n + 12*self.n.log()/self.n.sqrt() * params_term * norms_term
        return bound.item()

def ours_params(self) -> float:
        """Second bound from Theorem 3.5"""
        C_i = 4/self.margin * self.data.norm()/self.n.sqrt() * self.lip.prod() * self.dist21/self.lip
        LC_i = self.L * C_i
        W_i = self.params        

        first_summand = 2*W_i * (1+LC_i**2).log()
        psi = lambda x: 2*x*(torch.pi/2-x.arctan())
        #take care of numerical instabilities in psi
        second_summand = psi(LC_i)*(LC_i<1000.).float() + 2*(LC_i>=1000.).float()

        bound = 12/self.n.sqrt() * (first_summand+second_summand).sum().sqrt()
        return bound.item()
```

## Bartlett et al.

The bound from Bartlett et al. is analogous to our first bound ($\clubsuit$). Essentially, it only differs in the definition of the terms $\tilde C_i$ as they do not account for the structure of convolutions. 
Similarly, the number $W_i$ of parameters will be replaced by $\bar W_i = (c_{i}d_i^2) \cdot(c_{i+1}d_{i+1}^2) =  W_i \frac{d_i^2 d_{i+1}^2}{k_i^2}$, the number of parameters of the corresponding matrix $M_i \in \mathbb R^{c_{i+1}d_{i+1}^2 \times c_{i}d_{i}^2}$. For more details see sections A.1 to A.4 in the paper.
$$
{R}_S({\mathcal F_\gamma})
        \le
        \frac{4}{n}
        +
        \frac{9 \log(n)}{\sqrt{n}}
        \sqrt{\log\left(2\bar W\right)}
        \left(
            \sum_{i=1}^L
            \tilde C_{i}^{{2}/{3}}
        \right)^{\!3/2}
$$
with
$$ 
\tilde C_{i}(X)=
    \frac 4\gamma \, \frac{\lVert X\rVert}{\sqrt{n}} 
    \left(
        \prod_{\substack{l=1}}^{L}
            s_l
    \right)
    \frac{d_{i+1}^2}{k_i}
    \frac{\lVert K_i-K^{(0)}_i\rVert_{2,1}}{s_{i}}
\qquad \text{and} \qquad
\bar W = \max_i W_i \frac{d_i^2 d_{i+1}^2}{k_i^2} 
$$

```python
def bartlett(self) -> float:
        """Lemma A.8 in Bartlett et al., NeurIPS 2017"""

        d_i = torch.tensor([l.inp_dims[-1] for l in self.layers]+[1])
        C_i = 4/self.margin * self.data.norm()/self.n.sqrt() * self.lip.prod()
        C_i = C_i * d_i[1:]**2/self.kernel_sizes * self.dist21/self.lip

        barW = self.params * (d_i[:-1]*d_i[1:]/self.kernel_sizes)**2
        barW = barW.max()

        norms_term = (C_i**(2/3)).sum()**(3/2)
        params_term = (2*barW).log().sqrt()

        bound = 4/self.n + 9*self.n.log()/self.n.sqrt() * params_term * norms_term
        return bound.item()
```

## ...