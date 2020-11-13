import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist

def covariance(X,inv_lengthscale,amplitude=1.,locations=None,spatial_decayrate=None,eval_gradient=False):
    """
    Description
    -----------
    Calculates the covariance matrix.
    
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    inv_lengthscale : scalar or (m,) shaped numpy array of inverse length-scales 
    amplitude : Amplitude of the squared exponential kernel
    locations : (n x k) numpy array of spatial locations or a tuple of (n1 x k) and (n2 x k) numpy arrays
    spatial_decayrate : scalar or (k,) shaped numpy array of inverse length-scales for spatial locations
    
    Returns
    -------
    covariance_matrix : (n x n) numpy array
    
    """
    if type(X)!=tuple:
        lnC = (pdist(X*inv_lengthscale))**2
        if locations!=None:
            lnC += (pdist(locations*spatial_decayrate))**2
        
        cov=squareform((amplitude**2)*np.exp(-0.5*lnC))
        
    else:
        lnC=(cdist(X[0]*inv_lengthscale,X[1]*inv_lengthscale))**2
        if locations!=None:
            lnC += (cdist(locations[0]*spatial_decayrate,locations[1]*spatial_decayrate))**2
        
        cov=(amplitude**2)*np.exp(-0.5*lnC)
    
    if(eval_gradient):
        
        return cov,cov_gradient
    else:
        return cov
