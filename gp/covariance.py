import numpy as np
import scipy as sp

def covariance(X,additional_inputs):
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
    return sp.spatial.distance.pdist(X)
    
