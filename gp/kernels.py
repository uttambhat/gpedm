import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist

class Kernel:
    """
    Base class for all kernels. Common functions for all kernels here.
    """

class SquaredExponential(kernel):
    def __init__(self, inv_lengthscale=1.0):
        self.phi = inv_lengthscale
    
    def covariance(self,X,parameters,eval_gradient=False):

class LinearExponential(kernel):
    def __init__(self, inv_lengthscale=1.0):
        self.phi = inv_lengthscale
    
    def covariance(self,X,parameters,eval_gradient=False):


