import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.layers import ConstrainedConv2d
from utils import compute_margins

class Bounds:  
    """Computes the empirical Rademacher complexity bounds of a convolutional network as in Table 1. Bounds for linear layers are applied to the matrices corresponding to the convolutional maps. 
    
    Assumes that model is of type nn.Sequential with ConstrainedConv2D layers and ReLU activation functions as in simple_conv6 | simple_conv11.


    Implemented bounds include    
    
    ours_norms: First bound from Theorem 3.5
    ours_params: Second bound from Theoem 3.5
    bartlett: Lemma A.8 in Bartlett et al., NeurIPS 2017
    ledent_main: Theorem 11 in Ledent et al., AAAI 2021
    ledent_fixed_constraints: Considers Lipschitz constraints for all layers as in Section E ofLedent et al., AAAI 2021
    lin: Lemma 18 in Lin & Zhang, arxiv preprint 1910.01487
    neyshabur_1inf: Corollary 2 in Neyshabur et al., COLT 2015
    golowich_1inf: Theorem 2 in Golowich et al., COLT 2018
    gouk_1inf: Theorem 1 in Gouk et al., ICLR 2021
    neyshabur_fro:  Corollary 2 in Neyshabur et al., COLT 2015
    golowich_fro: Theorem 1 in Golowich et al., COLT 2018
    gouk_fro: Theorem 2 in Gouk et al., ICLR 2021


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

    def __init__(self, model: nn.Module, dataloader: DataLoader):
        """On init, compute quantities required across bounds
        
        Arguments
        ---------
            model: nn.Module
                model for which the Rademacher complexity bounds should be computed
            dataloader: DataLoader
                loader of the data used for training the model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.data = torch.cat([x[0] for x in dataloader], dim=0)
        self.n = torch.tensor(len(self.data))
        self.classes = len(dataloader.dataset.classes)
        self.margin = -torch.cat([compute_margins(model(x[0].to(device)),x[1].to(device)).flatten().detach() for x in dataloader]).max().to('cpu')
        if self.margin <0:
            print('WARNING: negative margin')
        self.dl = dataloader

        self.layers = [layer for layer in model.modules() if hasattr(layer, 'weight')]
        #move cls  to end
        self.layers = self.layers[1:]+self.layers[:1]
        self.L = len(self.layers)
        self.weights = [layer.weight for layer in self.layers]
        self.deltas = [layer.weight - layer.init_weight for layer in self.layers]

        self.strides = torch.tensor([layer.stride[0] for layer in self.layers])
        self.kernel_sizes = torch.tensor([layer.kernel_size[0] for layer in self.layers])
        self.inp_dims = torch.tensor([layer.inp_dims[3] for layer in self.layers])
        self.in_channels = torch.tensor([layer.inp_dims[1] for layer in self.layers])

        self.params = torch.tensor([weight.numel() for weight in self.weights]).type(torch.float64)

        self.mat_params = self.params * (self.inp_dims/self.kernel_sizes/self.strides).type(torch.float64)
        model.reset_uv()
        self.lip = torch.tensor([layer.lip() for layer in self.layers]).type(torch.float64)

        self.norm22 = torch.tensor([weight.norm(p=2) for weight in self.weights]).type(torch.float64)
        self.norm21 = torch.tensor([weight.norm(dim=1).norm(p=1) for weight in self.weights]).type(torch.float64)
        # self.matrix_norm21 is NOT the norm of the matrix, it is only the part coming from the norm of the weight, i.e. the factor (d/t)^2 is discarded
        self.matrix_norm21 = torch.tensor([weight.norm(dim=[1,2,3], p=2).norm(p=1) for weight in self.weights]).type(torch.float64)
        self.norm1inf = torch.tensor([weight.norm(dim=[1,2,3],p=1).max() for weight in self.weights]).type(torch.float64)

        self.dist22 = torch.tensor([delta.norm(p=2) for delta in self.deltas]).type(torch.float64)
        self.dist21 = torch.tensor([delta.norm(dim=1).norm(p=1) for delta in self.deltas]).type(torch.float64)
        self.matrix_dist21 = torch.tensor([delta.norm(dim=[1,2,3], p=2).norm(p=1) for delta in self.deltas]).type(torch.float64)
        self.dist1inf = torch.tensor([delta.norm(dim=[1,2,3],p=1).max() for delta in self.deltas]).type(torch.float64)

        self.bounds = ['ours_norms', 'bartlett', 'ledent_main', 'ledent_fixed_constraints', 'ours_params','lin', 'neyshabur_1inf', 'golowich_1inf', 'gouk_1inf', 'neyshabur_fro', 'golowich_fro', 'gouk_fro']

        self.quantities = {
            'lip_prod': self.lip.prod(),
            'fro_prod': (self.norm22*self.inp_dims/self.strides).prod(),
            '1inf_prod': (self.norm1inf).prod()

        }

    def __call__(self, mode = 'default') -> dict:
        """Compute Rademacher complexity bounds
        Arguments
        ---------
            mode: default | full
                default mode excludes computation of ledent_main (requires ~ 5 min computation time)
        """

        if mode == 'default':
            list_of_bounds = [b for b in self.bounds if b != 'ledent_main']
        elif mode == 'full':
            list_of_bounds = self.bounds

        return {bound: getattr(self,bound)() for bound in list_of_bounds}
    

    def ours_norms(self) -> float:
        """First bound from Theorem 3.5"""
        C_i = 4/self.margin * self.data.norm()/self.n.sqrt() * self.lip.prod() * self.dist21/self.lip
        W = self.params.max()

        norms_term = (C_i**(2/3)).sum()**(3/2)
        params_term = (2*W).log().sqrt()
        
        bound = 4/self.n + 12*self.n.log()/self.n.sqrt() * params_term * norms_term
        return bound.item()

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

    def ledent_main(self) -> float:
        """Ledent et al., Theorem 11
        
        Variables
        ----------
            dists: np.array
                distances to init in 2,1 norms defined as the sum of the norm of all filter. Differs from self.dist21!
                Classification layer enters favorably with Frobenius norm
            lips: torch.tensor
                layerwise lipschitz constants, equals self.lip
                Classification layer enters favorably with maximal Frobenius norm of a filter.
            gamma: torch.Tensor
                margins, equals self.gamma
            d_i: np.array
                spatial input dimension to each layer. 
                Classification layer enters favorably with d=1
            c_i: np.array
                input channels to each layer. 
                Classification layer enters favorably with c=#classes
            b_l: np.array
                maximum l2 norm of an input patch to the l-th layer.Classification layer enters favorably with b=margin
            rho_lplus: torch.Tensor
                for each layer, the max_i(product of Lipschitz constants from l+1 to layer i+1, divided by b_l[i+1] and multplied with d_i[l+1] (max patchnorm, resp. spatial input dim).
            O_l: torch.tensor
                number of conv. patches, 
                equals spatial dim of output, i.e. O_l = d_i[1:]**2
            m_l: torch.tensor
                number of output channels, m_l = c_i[1:]
            R: torch.Tensor
                quantity as defined in Eq. 25, enters  bound linearly
            Gamma: torch.Tensor
                quantity as defined in Eq.25, enters bound logarithmically.
            barW: torch.Tensor
                max(O_l * m_l), enters bound logarithmically
        """
        new_model = []
        layer_idx = []
        i = 0
        model = self.model
        # self.model might be a sequential model of sequential models --> expand
        for m in model.model:
            for l in m:
                new_model.append(l)
                if isinstance(l, ConstrainedConv2d):
                    layer_idx.append(i)
                i += 1
        model = nn.Sequential(*new_model).to(self.device)
        #compute maximum norm of patch of activations:
        b_l = []
        gamma=self.margin
        lips = self.lip
        layers = [model[l] for l in layer_idx]
        
        dists = np.array([(l.weight-l.init_weight).norm(p=2, dim=[1,2,3]).norm(p=1).detach().cpu() for l in layers])
        l = layers[-1]
        dists[-1] = (l.weight-l.init_weight).norm(p=2).detach()
        lips[-1] = (l.weight).norm(p=2, dim=[1,2,3]).max().detach()

        d_i = np.array([l.inp_dims[-1] for l in layers]+[1])
        c_i = np.array([l.inp_dims[1] for l in layers]+[self.classes])

        for l in layer_idx:  
            #compute b_l  
            ks = model[l].kernel_size
            model_l = model[:l]
            max_norm = 0
            for b, (x,y) in enumerate(self.dl):
                x, y =  x.to(self.device), y.to(self.device)
                z = model_l(x)
                for i in range(z.shape[2]):
                    for j in range(z.shape[3]):
                        p = z[:,:,i:i+ks[0],j:j+ks[1]]
                        norm = p.norm(p=2, dim=[1,2,3]).cpu()
                        max_norm = max(max_norm, norm.max())
            b_l.append(max_norm.item())
        b_l.append(gamma)
        b_l = np.array(b_l)

        L =len(layers)
        rho_l = []
        for layer in layers:
            rho_l.append(layer.lip())
        rho_l = np.array(rho_l)
        rho_lplus = []
        for l in range(L-1):
            r = [np.array(rho_l[l+1:i+1]).prod() / b_l[i+1] * d_i[l+1] for i in range(l+1,L)]
            rho_lplus.append(max(r))
        rho_lplus.append(b_l[-2]/b_l[-1])
        rho_lplus = np.array(rho_lplus)

        #here rho_lplus does not include the factor from the spatial dim which enters because of the lipschitz constant being wrt different norms, i.e. d_i
        a_i = dists
        r_i = (b_l[:-1] * rho_lplus * a_i)
        R = (r_i**(2/3)).sum()**(3/2)

        O_l = d_i[1:]**2
        m_l = c_i[1:]
        Gamma = (r_i * O_l * m_l).max()
        barW = (O_l * m_l).max()

        bound = 4/self.n + 768 * R * np.sqrt(np.log2(32 * Gamma *self.n**2 + 7*barW*self.n)) * np.log(self.n)/np.sqrt(self.n)
        return bound


    def ledent_fixed_constraints(self):
        """Ledent et al., Section E.
        
        Variables
        ----------
            dists: np.array
                distances to init in 2,1 norms defined as the sum of the norm of all filter. Differs from self.dist21!
                Classification layer enters favorably with Frobenius norm
            lips: torch.tensor
                layerwise lipschitz constants
                Classification layer enters favorably with maximal Frobenius norm of a filter.
            gamma: torch.Tensor
                margins, equals self.gamma
            d_i: np.array
                spatial input dimension to each layer. 
                Classification layer enters favorably with d=1
            c_i: np.array
                input channels to each layer. 
                Classification layer enters favorably with c=#classes
            X_0: torch.Tensor
                maximal l2 norm of convolutional patch of input data
            rho_lplus: torch.Tensor
                for each layer, the max_i(product of Lipschitz constants from l+1 to layer i+1, divided by b_l[i+1]
            O_l: torch.tensor
                number of conv. patches, 
                equals spatial dim of output, i.e. O_l = d_i[1:]**2
            m_l: torch.tensor
                number of output channels, m_l = c_i[1:]
            R: torch.Tensor
                quantity as defined in Eq. 25, enters  bound linearly
            Gamma: torch.Tensor
                quantity as defined in Eq.25, enters bound logarithmically.
            barW: torch.Tensor
                max(O_l * m_l), enters bound logarithmically
        """
        layers = self.layers
        lips = np.array([l.lip() for l in layers])
        
        dists = np.array([(l.weight-l.init_weight).norm(p=2, dim=[1,2,3]).norm(p=1).detach().cpu() for l in self.layers])
        l = layers[-1]
        dists[-1] = l.weight.norm(p=2).detach()
        lips[-1] = (l.weight).norm(p=2, dim=[1,2,3]).max().detach()

        d_i = np.array([l.inp_dims[-1] for l in layers]+[1])
        c_i = np.array([l.inp_dims[1] for l in layers]+[self.classes])

        l = layers[0]
        ks = l.kernel_size

        #compute maximal norm of input patch
        max_norm = 0
        for b, (x,y) in enumerate(self.dl):
            x, y =  x.to(l.weight.device), y.to(l.weight.device)
            z = x
            for i in range(z.shape[2]):
                for j in range(z.shape[3]):
                    p = z[:,:,i:i+ks[0],j:j+ks[1]]
                    norm = p.norm(p=2, dim=[1,2,3]).cpu()
                    max_norm = max(max_norm, norm.max())
        X_0 = max_norm

        O_l = d_i[1:]**2
        m_l = c_i[1:]

        r_i = X_0/self.margin * np.prod(lips) * dists / lips * d_i[1:]
        R = (r_i**(2/3)).sum()**(3/2)

        Gamma = (r_i * O_l * m_l).max() 
        barW = (O_l * m_l).max()

        bound = 4/self.n + 768 * R * np.sqrt(np.log2(32 * Gamma *self.n**2 + 7*barW*self.n))
        return bound


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

    def lin(self) -> float:
        """Lemma 18 in Lin & Zhang, arxiv preprint 1910.01487"""
        root_term = 2/self.margin * self.data.norm()/self.n.sqrt() * self.lip.prod() * self.L**2

        summands = self.params**2 * self.inp_dims/self.strides * self.norm22 / self.lip
        root_term *= summands.sum()

        bound = 16 * root_term**(1/4) /self.n.sqrt()
        return bound.item()

    def neyshabur_1inf(self) -> float:
        """Corollary 2 in Neyshabur et al., COLT 2015"""
        bound = 2**self.L * self.classes * (self.norm1inf).prod() 
        bound *= (2*self.in_channels*self.inp_dims**2)[0].log().sqrt()
        bound *= self.data.abs().max()
        bound /= self.n.sqrt()
        return bound.item()

    def golowich_1inf(self) -> float:
        """heorem 2 in Golowich et al., COLT 2018"""
        bound =  2 * self.classes * self.norm1inf.prod()
        bound *= (self.L+1+(self.in_channels*self.inp_dims**2)[0].log() ).sqrt()
        bound *= (self.data.norm(dim=0, p=2).max()/self.n).sqrt()
        bound /= self.n.sqrt()
        return bound.item()

    def gouk_1inf(self) -> float:
        """Theorem 1 in Gouk et al., ICLR 2021"""
        bound = 2**(self.L+1) * self.classes * (self.norm1inf).prod()
        bound *= (self.in_channels*self.inp_dims**2)[0].log()

        summands = self.dist1inf/ self.norm1inf
        bound *= summands.sum()
        bound *= self.data.abs().max()
        bound /= self.n.sqrt()
        return bound.item()

    def neyshabur_fro(self) -> float:
        """Corollary 2 in Neyshabur et al., COLT 2015"""
        bound = 2**(self.L-1) * self.classes * self.data.norm() /(2*self.n).sqrt()
        bound *= (self.norm22*self.inp_dims/self.strides).prod()
        bound /= self.n.sqrt()
        return bound.item()

    def golowich_fro(self) -> float:
        """Theorem 1 in Golowich et al., COLT 2018"""
        bound = self.classes * self.data.norm() / self.n.sqrt()
        bound *= (self.norm22*self.inp_dims/self.strides).prod()
        bound = bound * (1 + (2 * (2*torch.ones(1)).log() * self.L ).sqrt())
        bound /= self.n.sqrt()
        return bound.item()
    
    def gouk_fro(self) -> float:
        """gouk_fro: Theorem 2 in Gouk et al., ICLR 2021"""
        #for numerical stability, compute logarithms
        bound = (2**(self.L+0.5) * self.classes * self.data.norm() / self.n.sqrt()).log10()
        bound += (self.inp_dims**2 / self.strides * self.in_channels.sqrt() * self.norm22).log10().sum()

        prods = [self.inp_dims[0] * self.in_channels.sqrt()[0] ]
        for i in range(1,self.L):
            prods.append(prods[-1] * self.inp_dims[i] * self.in_channels.sqrt()[i] )
        prods = torch.tensor(prods)
        summands = self.dist22 / self.norm22 / prods

        bound += summands.sum().log10()
        bound -= self.n.sqrt().log10()
        return 10**(bound.item())