import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F


# RBF Layer

# RBF Layer
# RBF Layer
class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centers: the learnable centers of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, num_func, num_filters, basis_func):
        super(RBF, self).__init__()
        self.num_func = num_func
        self.register_buffer('centers', torch.tensor(np.linspace(-310, 310, num_func)).float())
        self.num_filters = num_filters
        self.weight = nn.Parameter(torch.Tensor(num_func, 1, num_filters))
        self.gamma = 10 
        self.basis_func = basis_func  
        self.int_basis_func = erf_func

    def forward(self, input, shape_param=2.0, scale_param=1.0):
        # Generate and apply additive gamma noise
        noise = torch.tensor(np.random.gamma(shape_param, scale_param, input.shape)).float().to(input.device)
        input = input + noise
        
        size = [self.num_func] + list(input.shape)
        x = input.expand(size)
        c = self.centers.view(-1, 1, 1, 1, 1)
        weight = self.weight.view(-1, 1, self.num_filters, 1, 1)
        
        if self.basis_func == gaussian:
            distances = (x - c).div(self.gamma)
            return self.basis_func(distances).mul(weight).sum(0), self.int_basis_func(distances, self.gamma).mul(weight).sum(0)
        else:    
            distances = (x - c).abs()
            return self.basis_func(distances, self.gamma).mul(weight).sum(0), 0

# RBFs
def gaussian(alpha):
    phi = torch.exp(-0.5 * alpha.pow(2))
    return phi

def erf_func(alpha, gamma):
    phi = gamma * math.sqrt(math.pi / 2) * torch.erf(alpha.div(math.sqrt(2)))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5 * alpha) * torch.exp(-3**0.5 * alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(-5**0.5 * alpha)
    return phi

def triangular(alpha, gamma):
    out = 1 - alpha.div(gamma)
    out[alpha > gamma] = 0
    return out

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    bases = {
        'gaussian': gaussian,
        'linear': linear,
        'quadratic': quadratic,
        'inverse quadratic': inverse_quadratic,
        'multiquadric': multiquadric,
        'inverse multiquadric': inverse_multiquadric,
        'spline': spline,
        'poisson one': poisson_one,
        'poisson two': poisson_two,
        'matern32': matern32,
        'matern52': matern52
    }
    return bases