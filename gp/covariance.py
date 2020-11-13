import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.linalg import cho_factor, cho_solve


def covariance(X,phi,tau=1.0,locs=None,s=None,sites=None,rho=None):
    """
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    tau : scalar for pointwise prior variance in function f
    phi : inverse length scale parameters; (m,)
    locs: (n x k) numpy array of spatial locations or a tuple of (n1 x k) and (n2 x k) numpy arrays
    s: inverse length scale parameters for spatial inputs (k,)
    sites: (n x k) numpy array of sites or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rho: scalar for correlation across sites

    Returns
    -------
    covariance_matrix : (n x n) numpy array

    """
    
    if type(X)!=tuple:
        lnC = (pdist(X*phi))**2
        C=(tau**2)*np.exp(-0.5*lnC)
        if locs!=None:
            lnC += (pdist(locs*s))**2
            C=(tau**2)*np.exp(-0.5*lnC)
        if sites!=None:
            C *= rho*(pdist(sites)>0)
    else:
        lnC=(cdist(X[0]*phi,X[1]*phi))**2
        C=(tau**2)*np.exp(-0.5*lnC)
        if locs!=None:
            lnC += (cdist(locs[0]*s,locs[1]*s))**2
            C=(tau**2)*np.exp(-0.5*lnC)
        if sites!=None:
            C *= rho*(cdist(sites[0],sites[1])>0)
        return squareform(C)
    
    
    
def evaluateGradient(X,Y,phi,Ve=0.01,tau=0.99,locs=None,s=None,sites=None,rho=None,lp=None,dlp=None):
    """
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    Y : (n x 1) numpy array or a tuple of (n1 x 1) and (n2 x 1) numpy arrays
    phi : inverse length scale parameters; (m,)
    Ve : scalar for process variance 
    tau : scalar for pointwise prior variance in function f
    locs: (n x k) numpy array of spatial locations or a tuple of (n1 x k) and (n2 x k) numpy arrays
    s: inverse length scale parameters for spatial inputs (k,)
    sites: (n x k) numpy array of sites or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rho: scalar for correlation across sites

    Returns
    -------
    nll : negative log likelihood (scalar)
    neglgrad : gradient of negative log likelihood with respect to the parameters, (m,) 

    """
    if type(X)!=tuple:
        # Compute covariance matrix
        C = covariance(X,phi,tau,locs,s,sites,rho)
    
        # Algorithm (2.1) from Rasmussen and Williams (2006)
        L=np.transpose(np.linalg.cholesky(C+Ve*np.eye(np.size(X,0))))
        c,low = cho_factor(C+Ve*np.eye(np.size(X,0)))
        a = cho_solve((c, low),Y)
        Linv=np.linalg.inv(L)
        iKVs=Linv.dot(np.transpose(Linv))
        #mpt=C.dot(a)  # don't need this for gradient
        #Ct=C-C.dot(iKVs.dot(C)) # don't need this for gradient
        like = -.5*np.transpose(Y).dot(a)-np.sum(np.log(np.diag(L)))
    
        # Calculate gradients (equation 5.9 in Rasmussen and Williams)
        dl=np.array([])
        vQ=a*np.transpose(a)-iKVs
    
        # Gradient wrt phi
        for i in range(np.size(X,1)):
            dC=C*(squareform(-(pdist(X[:,[i]])**2)))
            dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) 
    
        # Append gradient wrt Ve    
        dC=np.eye(np.size(X,0))
        dl=np.append(dl,np.trace(vQ.dot(dC)))
        # Append gradient wrt tau    
        dC=C/tau
        dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) 
        # Append gradient wrt s if applicable    
        if locs!=None:
            for i in range(np.size(locs,1)):
                dC=C*(squareform(-(pdist(locs[:,[i]])**2)))
                dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) 
                # Append gradient wrt rho if applicable    
        if sites!=None:
            dC=C*(pdist(sites)>0)/rho
            dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) 
    
        # Get negative log liklihood and gradient wrt parameters
        if (lp == None) & (dlp == None):
            nll=-like
            neglgrad=-dl
        else: 
        #To be continued 
    
        return nll, neglgrad
    else: 
        #To be continuted 

        
    


