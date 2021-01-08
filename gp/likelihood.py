import numpy as np
from .covariance import *
from .predict import *

# I prefer X and Y because sometimes I use this function for validation datasets
def log_marginal_likelihood(X,Y,phi,tau=None,Ve=None,locs=None,rate=None,site=None,rho=None):
    """
    Description
    -----------
    Calculates the log marginal likelihood
    
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    Y : (n x 1) numpy array or a tuple of (n1 x 1) and (n2 x 1) numpy arrays
    phi : scalar or (m,) shaped numpy array of inverse length-scales 
    tau : amplitude of the squared exponential kernel
    Ve : scalar of gaussian noise of Y
    locs : (n x k) numpy array of spatial locs or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rate : scalar or (k,) shaped numpy array of inverse length-scales for spatial locs
    site : (n x k) numpy array of site indices or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rho : scalar of correlation of mean function between sites
    
        
    Returns
    -------
    log_marginal_likelihood : scalar
    
    """

    # when data points need to be predicted are the same as data points used as basis
    if type(Y)!=tuple:
        if type(X)==tuple:
            print ("What???")

        K_X_X = covariance(X,phi,tau,locs,rate,site,rho)
        C = K_X_X+Ve*np.eye(X.shape[0])
        L = np.linalg.cholesky(C)
        Linv = np.linalg.inv(L)
        Cinv = np.transpose(Linv)@Linv
        alpha = Cinv@Y

        log_marginal_likelihood = -0.5*float(np.transpose(Y)@alpha)-np.sum(np.log(np.diag(L))) #omitted last term

    if type(Y)==tuple:
        if type(X)!=tuple:
            print ("What???")
        
        X1=(X[0],X[0])
        if locs is not None:
            locs1=(locs[0],locs[0])
        else:
            locs1=None
        if site is not None:
            site1=(site[0],site[0])
        else:
            site1=None
        K_X_X = covariance(X1,phi,tau,locs1,rate,site1,rho)

        X2=(X[0],X[1])
        if locs is not None:
            locs2=(locs[0],locs[1])
        else:
            locs2=None
        if site is not None:
            site2=(site[0],site[1])
        else:
            site2=None
        K_X_Xnew = covariance(X2,phi,tau,locs2,rate,site2,rho)

        X3=(X[1],X[1])
        if locs is not None:
            locs3=(locs[1],locs[1])
        else:
            locs3=None
        if site is not None:
            site3=(site[1],site[1])
        else:
            site3=None
        K_Xnew_Xnew = covariance(X3,phi,tau,locs3,rate,site3,rho)
       
        C = K_X_X+Ve*np.eye(X[0].shape[0])
        L = np.linalg.cholesky(C)
        Linv = np.linalg.inv(L)
        Cinv = np.transpose(Linv)@Linv
        alpha = Cinv@Y[0]
        
        C_new = K_Xnew_Xnew-np.transpose(K_X_Xnew)@Cinv@K_X_Xnew+Ve*np.eye(X[1].shape[0]) # do i need the last term?
        L_new = np.linalg.cholesky(C_new)
        Linv_new = np.linalg.inv(L_new)
        Cinv_new = np.transpose(Linv_new).dot(Linv_new)
        alpha_new = np.transpose(K_X_Xnew)@alpha

        log_marginal_likelihood = -0.5*float(np.transpose(Y[1]-alpha_new)@Cinv_new@(Y[1]-alpha_new))-np.sum(np.log(np.diag(L_new)))

    return (log_marginal_likelihood)
