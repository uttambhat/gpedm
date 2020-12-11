import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist

def covariance(X,phi,tau=1.,locs=None,rate=None,site=None,rho=None,eval_gradient=False):
    """
    Description
    -----------
    Calculates the covariance matrix.
    
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    phi : scalar or (m,) shaped numpy array of inverse length-scales 
    tau : tau of the squared exponential kernel
    locs : (n x k) numpy array of spatial locs or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rate : scalar or (k,) shaped numpy array of inverse length-scales for spatial locs
    site : (n x k) numpy array of site indices or a tuple of (n1 x k) and (n2 x k) numpy arrays
    
    Returns
    -------
    cov : (n x n) numpy array
    cov_grad : NOTE CODED YET!!!
    
    """
    #phi,tau=1.,locs=None,rate=None
    if type(X)!=tuple:
        lnC = (pdist(X*phi))**2
        if locs!=None:
            lnC += (pdist(locs*rate))**2
        if site!=None:
            lnC += rho*(pdist(site)>0)
        cov = squareform((tau**2)*np.exp(-0.5*lnC))
        np.fill_diagonal(cov,tau**2)
        
    else:
        lnC=(cdist(X[0]*phi,X[1]*phi))**2
        if locs!=None:
            lnC += (cdist(locs[0]*rate,locs[1]*rate))**2
        if site!=None:
            lnC += rho*(cdist(site[0],site[1])>0)
        cov = (tau**2)*np.exp(-0.5*lnC)
    
    if(eval_gradient):
        
        return cov,cov_grad
    else:
        return cov
