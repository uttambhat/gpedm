import numpy as np
from .covariance import *
from .predict import *

def log_marginal_likelihood(X_train,y_train,phi,tau=1.,Ve=1.e-10):
    """
    Description
    -----------
    Calculates the log marginal likelihood
    
    Parameters
    ----------
    X_train : (n x m) numpy array of training data (independent variables)
        The n rows are the different datapoints and the m columns represent
        the different features
    y_train : (n x 1) numpy array of training data (dependent variable)
    phi : scalar or (m,) shaped numpy array of inverse length-scales 
    tau : amplitude of the squared exponential kernel
        
    Returns
    -------
    log_marginal_likelihood : scalar
    
    """
