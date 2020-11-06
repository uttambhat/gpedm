import numpy as np
from scipy.spatial.distance import cdist,pdist

def covariance(X1,X2,phi,tau):
    """
    Parameters
    ----------
    X : (n x m) numpy array
        The rows are the different datapoints and columns represent
        the different features
    additional_inputs : metric info
        DESCRIPTION.

    Returns
    -------
    covariance_matrix : (n x n) numpy array

    """
    return np.power(tau,2)*np.exp(-np.square(cdist(X*phi,X*phi)))
    

