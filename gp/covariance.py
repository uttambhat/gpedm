import numpy as np
from scipy.spatial.distance import cdist,pdist

def covariance(X1,X2,phi,tau=1.):
    """
    Parameters
    ----------
    X1,X2 : (n x m) numpy array
        The n rows are the different datapoints and the m columns represent
        the different features
    phi : (m,) numpy array
        Inverse length-scales 
    tau : Amplitude of the squared exponential kernel
    additional_inputs : metric info
        DESCRIPTION.

    Returns
    -------
    covariance_matrix : (n x n) numpy array

    """
    return np.square(tau)*np.exp(-np.square(cdist(X1*phi,X2*phi)))
    

