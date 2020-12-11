import numpy as np
from .covariance import *

def predict(X,X_train,y_train,phi,tau=1.,return_std=False,return_cov=False):
    """
    Description
    -----------
    Calculates the mean predictions and (optionally) covariances at new points X
    
    Parameters
    ----------
    X_train : (n x m) numpy array of training data (independent variables)
        The n rows are the different datapoints and the m columns represent
        the different features
    y_train : (n x 1) numpy array of training data (dependent variable)
    phi : scalar or (m,) shaped numpy array of inverse length-scales 
    tau : tau of the squared exponential kernel
    return_std : Switch to return the standard deviations at the new points
    return_cov : Switch to return the covariance matrix for the new points
        
    Returns
    -------
    y : (n x 1) numpy array of predictions
    std_y : (n x 1) numpy array of standard deviations on predictions
    cov_y : (n x n) numpy array of covariances on predictions
    
    """
    K_Xtrain_Xtrain = covariance(X,phi,tau)
    K_X_Xtrain = covariance((X,X_train),phi,tau)
    return K_X_Xtrain@np.linalg.inv(K_X_train_X_train)@y_train


