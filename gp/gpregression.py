import numpy as np
from .kernels import *

class GPregressor:
    def __init__(X_train,y_train,kernel=SquaredExponential,optimizer=rprop):
        self.X = X_train
        self.y = y_train
        self.kernel=kernel
        self.optimizer=optimizer

    def predict(X
    
    def log_marginal_likelihood(self,parameters,eval_gradient=False):
        #... (self.kernel.covariance(self.X,parameters,eval_gradient)) ...
