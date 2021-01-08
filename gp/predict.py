import numpy as np
from .covariance import *

#I think instead of _train we might want to call these _basis, or just X, because this function is used not just for model training
def predict(X_basis,Y_basis,phi,tau=None,Ve=None,locs=None,rate=None,site=None,rho=None,return_std=False,return_cov=False,X_new=None):
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

    K_X_X = covariance((X_basis,X_basis),phi,tau,locs,rate,site,rho)
    C = K_X_X+Ve*np.eye(X_basis.shape[0])
    L = np.linalg.cholesky(C)
    Linv = np.linalg.inv(L)
    Cinv = np.transpose(Linv)@Linv
    alpha = Cinv@Y_basis

    # when data points need to be predicted are the same as data points used as basis
    if X_new is None:
        Y_new = np.transpose(K_X_X)@alpha
        C_new = Ve*K_X_X@Cinv+Ve*np.eye(X_train.shape[0]) # do i need the last term?


    # when data points need to be predicted are different from data points used as basis
    if X_new is not None:
        K_X_Xnew = covariance((X_basis,X_new),phi,tau,locs,rate,site,rho)
        K_Xnew_Xnew = covariance((X_new,X_new),phi,tau,locs,rate,site,rho)
        
        Y_new = np.transpose(K_X_Xnew)@alpha
        C_new = K_Xnew_Xnew-np.transpose(K_X_Xnew)@Cinv@K_X_Xnew+Ve*np.eye(X_new.shape[0]) # do i need the last term?
      
    y=Y_new

    if return_cov:
        cov_y=np.diag(C_new).reshape((C_new.shape[0],1))
        #std_y=np.sqrt(cov_y)
        return (y, cov_y)
    else:
        return (y)

