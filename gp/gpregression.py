import numpy as np
from .kernels import *

class GPregressor:
    def __init__(X,y,kernel=SquaredExponential,optimizer=rprop):
        self.X = X #training data
        self.y = y
        self.kernel=kernel
        self.optimizer=optimizer
    
    def log_marginal_likelihood(self,parameters,eval_gradient=False):
        #... (self.kernel.covariance(self.X,parameters,eval_gradient)) ...
