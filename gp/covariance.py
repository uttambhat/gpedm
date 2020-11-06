import numpy as np
from scipy.spatial.distance import pdist, squareform

n=10
m=20
X=np.random.rand(n,m)
#P=np.random.randint(3, size=(n, 1))
C=np.random.rand(n,2)

X_new=X
#P_new=P
C_new=C

C_combined=np.concatenate((C, C_new), axis=0)
C_unique = np.unique(C_combined, axis=0)

x = np.where(C == C_unique[1,:])

P = []
for i in range (X.shape[0]):
    P.append(int(np.where((C_unique == C[i,:]).all(axis=1))[0]))

P_new = []
for i in range (X_new.shape[0]):
    P_new.append(int(np.where((C_unique == C_new[i,:]).all(axis=1))[0]))

distMat=squareform(pdist(C_unique))

phi=np.random.rand(m)

tau=1
gamma=0.1
rho=0.9

def covariance(X,X_new, P, P_new, phi, tau,simpleDist=False, rho=None,gamma=None, distMat=None):
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
    ndim=len(phi)
    lC0=np.zeros((X_new.shape[0], X.shape[0]))
    DD= []
    for i in range(ndim):
        DD.append(np.square(X_new[:,[i]].dot(np.ones((1,X.shape[0]) ))-np.ones((X_new.shape[0],1)).dot(X[:,[i]].transpose())))
        lC0=lC0-0.5*phi[i]*DD[i]
    
    Dist=np.zeros((X_new.shape[0], X.shape[0]))
    if simpleDist:
        for i in range(X_new.shape[0]):
            for j in range (X.shape[0]):
                if (P_new[i]==P[j]):
                    Dist[i,j]=0
                else:
                    Dist[i,j]=1
        kX_newX=tau*np.exp(lC0)*(Dist+rho*(1-Dist))

    if not simpleDist:
        for i in range(X_new.shape[0]):
            for j in range (X.shape[0]):
                Dist[i,j]=float(distMat[[P_new[i]],[P[j]]])

        kX_newX=tau*np.exp(lC0)*np.exp(-np.square(Dist)*gamma)

    return (kX_newX)

covariance(X,X_new, P, P_new, phi, tau,simpleDist=False, rho=None,gamma=gamma, distMat=distMat)
covariance(X,X_new, P, P_new, phi, tau,simpleDist=True, rho=rho)