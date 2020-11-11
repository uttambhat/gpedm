import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist

def transform_pars(hyper,pars=None, pars_tr=None):
    if (pars==None) & (pars_tr!=None):
        pars={}

        phi_tr=pars_tr["phi_tr"]
        phi=(hyper["phimax"]-hyper["phimin"])/(1+np.exp(-phi_tr))+hyper["phimin"]
        pars["phi"]=phi

        ve_tr=pars_tr["ve_tr"]
        ve=(hyper["vemax"]-hyper["vemin"])/(1+np.exp(-ve_tr))+hyper["vemin"]
        pars["ve"]=ve

        tau_tr=pars_tr["tau_tr"]
        tau=(hyper["taumax"]-hyper["taumin"])/(1+np.exp(-tau_tr))+hyper["taumin"]
        pars["tau"]=tau

        if "s_tr" in pars_tr:
            s_tr=pars_tr["s_tr"]
            s=(hyper["smax"]-hyper["smin"])/(1+np.exp(-s_tr))+hyper["smin"]
            pars["s"]=s

        if "rho_tr" in pars_tr:
            rho_tr=pars_tr["rho_tr"]
            rho=(hyper["rhomax"]-hyper["rhomin"])/(1+np.exp(-rho_tr))+hyper["rhomin"]
            pars["rho"]=rho
        
    return pars

def covariance(X,pars, coordinates=None,site_indices=None):
    """
    Description
    -----------
    Calculates the covariance matrix.
    
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    phi : scalar or (m,) shaped numpy array of inverse length-scales squared
    tau : scalar of amplitude of the squared exponential kernel squared
    coordinates : (n x k) numpy array of spatial coordinates or a tuple of (n1 x k) and (n2 x k) numpy arrays
    s : scalar or (k,) shaped numpy array of inverse length-scales squared for spatial locations
    site_indices : (n x k) numpy array of site indices or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rho : scalar of correlation of maps between sites
    
    Returns
    -------
    covariance_matrix : (n x n) or (n1 x n2) numpy array
    
    """
    phi=pars["phi"]
    tau=pars["tau"]
    if "s" in pars:
        s=pars["s"]
    if "rho" in pars:
        rho=pars["rho"]

    if type(X)!=tuple:
        X = (X,X)
        if "s" in pars:
            coordinates = (coordinates, coordinates)
        if "rho" in pars:
            site_indices = (site_indices,site_indices)
        
    lnC=(cdist(X[0]*np.sqrt(phi),X[1]*np.sqrt(phi)))**2 #using cdist for both cases cos when using pdist I dont get positive definite matrices
    C=tau*np.exp(-0.5*lnC)

    if "s" in pars:
        C *= np.exp(-0.5*(cdist(coordinates[0]*np.sqrt(s),coordinates[1]*np.sqrt(s)))**2) #make it more consistent with phi
    if "rho" in pars:
        C *= (cdist(site_indices[0],site_indices[1])==0)+rho*(cdist(site_indices[0],site_indices[1])>0) # is this correct?
    return C

def gradient (X, Y, pars=None, pars_tr=None,hyper=None, priors=None, priors_tr=None,coordinates=None, site_indices=None):
    """
    Description
    -----------
    Calculates the likelihood and gradient.
    
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    Y : (n x 1) numpy array or a tuple of (n1 x 1) and (n2 x 1) numpy arrays
    phi : scalar or (m,) shaped numpy array of inverse length-scales squared
    tau : scalar of amplitude of the squared exponential kernel squared
    coordinates : (n x k) numpy array of spatial coordinates or a tuple of (n1 x k) and (n2 x k) numpy arrays
    s : scalar or (k,) shaped numpy array of inverse length-scales squared for spatial locations
    site_indices : (n x k) numpy array of site indices or a tuple of (n1 x k) and (n2 x k) numpy arrays
    rho : scalar of correlation of maps between sites
    
    Returns
    -------
    neglpost : scalar, negative log likelihood
    neglgrad : (m x 1) numpy array, gradient of negative log likelihood wrt parameters
    
    """

    if (pars==None) & (pars_tr!=None):
        pars=transform_pars(hyper=hyper, pars_tr=pars_tr)

    phi=pars["phi"]
    ve=pars["ve"]
    tau=pars["tau"]
    if "s" in pars:
        s=pars["s"]
    if "rho" in pars:
        rho=pars["rho"]

    # likelihood and gradient from data
    if type(Y)!=tuple:
        if type(X)==tuple:
            print ("What???")

        X = (X,X)
        if "s" in pars:
            coordinates = (coordinates, coordinates)
        if "rho" in pars:
            site_indices = (site_indices,site_indices)

        covariance_matrix = covariance(X,pars,coordinates,site_indices)

        C = covariance_matrix+ve*np.eye(X[0].shape[0])
        L = np.linalg.cholesky(C)
        Linv = np.linalg.inv(L)
        Cinv = np.transpose(Linv).dot(Linv)
        m = Cinv.dot(Y) # conditional mean

        like = -0.5*float(np.transpose(Y).dot(m))-np.sum(np.log(np.diag(L)))
        
        vQ= m.dot(np.transpose(m))-Cinv

        dl=np.array([])
        for i in range(X[0].shape[1]):
            dC=covariance_matrix * (-0.5*(cdist(X[0][:,[i]],X[1][:,[i]]))**2)
            dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #phi
        dC=np.eye(X[0].shape[0])
        dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #ve
        dC=covariance_matrix/tau
        dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #tau
        if "s" in pars:
            if np.isscalar(s):
                dC=covariance_matrix * (-0.5*(cdist(coordinates[0],coordinates[1]))**2)
                dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #s
            else:
                for i in range(s.shape[0]):
                    dC=covariance_matrix * (-0.5*(cdist(coordinates[0][:,[i]],coordinates[1][:,[i]]))**2)
                    dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #s
        if "rho" in pars:
            dC=covariance_matrix * (cdist(site_indices[0],site_indices[1])>0)/rho
            dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #rho

    if type(Y)==tuple:
        if type(X)!=tuple:
            print ("What???")

        X1 = (X[0],X[0])
        if "s" in pars:
            coordinates1 = (coordinates[0], coordinates[0])
        else:
            coordinates1=None
        if "rho" in pars:
            site_indices1 = (site_indices[0],site_indices[0])
        else:
            site_indices1=None
        covariance_matrix1 = covariance(X1,pars,coordinates1,site_indices1)
        
        X2 = (X[1],X[0])
        if "s" in pars:
            coordinates2 = (coordinates[1], coordinates[0])
        else:
            coordinates2=None
        if "rho" in pars:
            site_indices2 = (site_indices[1],site_indices[0])
        else:
            site_indices2=None
        covariance_matrix2 = covariance(X2,pars,coordinates2,site_indices2)

        X3 = (X[1],X[1])
        if "s" in pars:
            coordinates3 = (coordinates[1], coordinates[1])
        else:
            coordinates2=None
        if "rho" in pars:
            site_indices3 = (site_indices[1],site_indices[1])
        else:
            site_indices3=None
        covariance_matrix3 = covariance(X3,pars,coordinates3,site_indices3)

        C = covariance_matrix1+ve*np.eye(X1[0].shape[0]) # variance
        L = np.linalg.cholesky(C)
        Linv = np.linalg.inv(L)
        Cinv = np.transpose(Linv).dot(Linv)
        m = Cinv.dot(Y[0]) # conditional mean
        
        Cn = covariance_matrix3-covariance_matrix2.dot(Cinv).dot(np.transpose(covariance_matrix2))+ve*np.eye(X3[0].shape[0]) # new variance # do i need the last term?
        Ln = np.linalg.cholesky(Cn)
        Lninv = np.linalg.inv(Ln)
        Cninv = np.transpose(Lninv).dot(Lninv)
        mn = covariance_matrix2.dot(m) # new conditional mean

        like = -0.5*float(np.transpose(Y[1]-mn).dot(Cninv).dot(Y[1]-mn))-np.sum(np.log(np.diag(Ln)))
        
        vQ= (Cninv.dot(Y[1]-mn)).dot(np.transpose(Cninv.dot(Y[1]-mn)))-Cninv

        dl=np.array([])
        for i in range(X1[0].shape[1]):
            dC  =covariance_matrix3 * (-0.5*(cdist(X3[0][:,[i]],X3[1][:,[i]]))**2)
            dC += -(covariance_matrix2 * (-0.5*(cdist(X2[0][:,[i]],X2[1][:,[i]]))**2)).dot(Cinv).dot(np.transpose(covariance_matrix2))
            dC += -covariance_matrix2.dot(Cinv).dot(np.transpose(covariance_matrix2 * (-0.5*(cdist(X2[0][:,[i]],X2[1][:,[i]]))**2)))
            dC += covariance_matrix2.dot(Cinv).dot(covariance_matrix1* (-0.5*(cdist(X1[0][:,[i]],X1[1][:,[i]]))**2)).dot(Cinv).dot(np.transpose(covariance_matrix2))
            dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #phi

        dC = np.eye(X3[0].shape[0])
        dC += covariance_matrix2.dot(Cinv).dot(Cinv).dot(np.transpose(covariance_matrix2))
        dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #ve

        dC = covariance_matrix3/tau
        dC += -(covariance_matrix2/tau).dot(Cinv).dot(np.transpose(covariance_matrix2))
        dC += -covariance_matrix2.dot(Cinv).dot(np.transpose(covariance_matrix2/tau))
        dC += covariance_matrix2.dot(Cinv).dot(covariance_matrix1/tau).dot(Cinv).dot(np.transpose(covariance_matrix2))
        dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #tau

        if "s" in pars:
            if np.isscalar(s):
                dC = covariance_matrix3 * (-0.5*(cdist(coordinates3[0],coordinates3[1]))**2)
                dC += -(covariance_matrix2 * (-0.5*(cdist(coordinates2[0],coordinates2[1]))**2)).dot(Cinv).dot(np.transpose(covariance_matrix2))
                dC += -covariance_matrix2.dot(Cinv).dot(np.transpose(covariance_matrix2 * (-0.5*(cdist(coordinates2[0],coordinates2[1]))**2)))
                dC += covariance_matrix2.dot(Cinv).dot(covariance_matrix1* (-0.5*(cdist(coordinates1[0],coordinates1[1]))**2)).dot(Cinv).dot(np.transpose(covariance_matrix2))
                dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #s
            else:     
                for i in range(s.shape[0]):
                    dC = covariance_matrix3 * (-0.5*(cdist(coordinates3[0][:,[i]],coordinates3[1][:,[i]]))**2)
                    dC += -(covariance_matrix2 * (-0.5*(cdist(coordinates2[0][:,[i]],coordinates2[1][:,[i]]))**2)).dot(Cinv).dot(np.transpose(covariance_matrix2))
                    dC += -covariance_matrix2.dot(Cinv).dot(np.transpose(covariance_matrix2 * (-0.5*(cdist(coordinates2[0][:,[i]],coordinates2[1][:,[i]]))**2)))
                    dC += covariance_matrix2.dot(Cinv).dot(covariance_matrix1* (-0.5*(cdist(coordinates1[0][:,[i]],coordinates1[1][:,[i]]))**2)).dot(Cinv).dot(np.transpose(covariance_matrix2))
                    dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #s
        
        if "rho" in pars:
            dC = covariance_matrix3 * (cdist(site_indices3[0],site_indices3[1])>0)/rho
            dC += -(covariance_matrix2 * (cdist(site_indices2[0],site_indices2[1])>0)/rho).dot(Cinv).dot(np.transpose(covariance_matrix2))
            dC += -covariance_matrix2.dot(Cinv).dot(np.transpose(covariance_matrix2 * (cdist(site_indices2[0],site_indices2[1])>0)/rho))
            dC += covariance_matrix2.dot(Cinv).dot(covariance_matrix1* (cdist(site_indices1[0],site_indices1[1])>0)/rho).dot(Cinv).dot(np.transpose(covariance_matrix2))
            dl=np.append(dl,0.5*np.trace(vQ.dot(dC))) #rho

    if (pars_tr!=None): # if using transformed parameters
        # derivative for parameters wrt transformed parameters
        dpars=np.array([])
        dpars=np.append(dpars,(phi-hyper["phimin"])*(1-(phi-hyper["phimin"])/(hyper["phimax"]-hyper["phimin"])))
        dpars=np.append(dpars,(ve-hyper["vemin"])*(1-(ve-hyper["vemin"])/(hyper["vemax"]-hyper["vemin"])))
        dpars=np.append(dpars,(tau-hyper["taumin"])*(1-(tau-hyper["taumin"])/(hyper["taumax"]-hyper["taumin"])))
        if s!=None:
            dpars=np.append(dpars,(s-hyper["smin"])*(1-(s-hyper["smin"])/(hyper["smax"]-hyper["smin"])))
        if rho!=None:
            dpars=np.append(dpars,(rho-hyper["rhomin"])*(1-(rho-hyper["rhomin"])/(hyper["rhomax"]-hyper["rhomin"])))
    else: # if using untransformed parameters
        dpars=1

    # likelihood and gradient from priors
    if (priors==None) & (priors_tr==None): # no priors
        lpost = like
        GradLpost = dl*dpars
    
    if (priors!=None) & (priors_tr==None): # priors on untransformed parameters
        lp=0
        dlp=np.array([])

        lp_phi = -0.5*np.sum(((phi-priors["E_phi"])**2)/priors["V_phi"])
        dlp_phi = -(phi-priors["E_phi"])/priors["V_phi"]
        lp+=lp_phi
        dlp=np.append(dlp, dlp_phi)

        if "E_ve" in priors:
            lp_ve = -0.5*np.sum(((ve-priors["E_ve"])**2)/priors["V_ve"])
            dlp_ve = -(ve-priors["E_ve"])/priors["V_ve"]
        if "a_ve" in priors:
            lp_ve = (priors["a_ve"]-1)*np.log(ve/hyper["vemax"])+(priors["b_ve"]-1)*np.log(1-ve/hyper["vemax"]) #??? need jacobian?
            dlp_ve = (priors["a_ve"]-1)/(ve/hyper["vemax"])-(priors["b_ve"]-1)/(1-(ve/hyper["vemax"]))
        lp+=lp_ve
        dlp=np.append(dlp, dlp_ve)

        if "E_tau" in priors:
            lp_tau = -0.5*np.sum(((tau-priors["E_tau"])**2)/priors["V_tau"])
            dlp_tau = -(tau-priors["E_tau"])/priors["V_tau"]
        if "a_tau" in priors:
            lp_tau = (priors["a_tau"]-1)*np.log(tau/hyper["taumax"])+(priors["b_tau"]-1)*np.log(1-tau/hyper["taumax"]) #??? need jacobian?
            dlp_tau = (priors["a_tau"]-1)/(tau/hyper["taumax"])-(priors["b_tau"]-1)/(1-(tau/hyper["taumax"]))
        lp+=lp_tau
        dlp=np.append(dlp, dlp_tau)

        if "s" in pars:
            if "E_s" in priors:
                lp_s = -0.5*np.sum(((s-priors["E_s"])**2)/priors["V_s"])
                dlp_s = -(s-priors["E_s"])/priors["V_s"]
            if "a_s" in priors:
                lp_s = (priors["a_s"]-1)*np.log(s/hyper["smax"])+(priors["b_s"]-1)*np.log(1-s/hyper["smax"]) #??? need jacobian?
                dlp_s = (priors["a_s"]-1)/(s/hyper["smax"])-(priors["b_s"]-1)/(1-(s/hyper["smax"]))
            lp+=lp_s
            dlp=np.append(dlp, dlp_s)
        
        if "rho" in pars:
            if "E_rho" in priors:
                lp_rho = -0.5*np.sum(((rho-priors["E_rho"])**2)/priors["V_rho"])
                dlp_rho = -(rho-priors["E_rho"])/priors["V_rho"]
            if "a_rho" in priors:
                lp_rho = (priors["a_rho"]-1)*np.log(rho/hyper["rhomax"])+(priors["b_rho"]-1)*np.log(1-rho/hyper["rhomax"]) #??? need jacobian?
                dlp_rho = (priors["a_rho"]-1)/(rho/hyper["rhomax"])-(priors["b_rho"]-1)/(1-(rho/hyper["rhomax"]))
            lp+=lp_rho
            dlp=np.append(dlp, dlp_rho)
        
        lpost = like+lp
        GradLpost = dl*dpars+dlp*dpars
    
    if (priors==None) & (priors_tr!=None): # priors on transformed parameters
        pars_tr_array=np.array([])
        pars_tr_array=np.append(pars_tr_array, pars_tr["phi_tr"])
        pars_tr_array=np.append(pars_tr_array, pars_tr["ve_tr"])
        pars_tr_array=np.append(pars_tr_array, pars_tr["tau_tr"])
        if "s_tr" in pars_tr:
            pars_tr_array=np.append(pars_tr_array, pars_tr["s_tr"])
        if "rho_tr" in pars_tr:
            pars_tr_array=np.append(pars_tr_array, pars_tr["rho_tr"])

        lp = -0.5*np.transpose(pars_tr_array-priors_tr["E"]).dot(np.linalg.inv(priors_tr["V"])).dot(pars_tr_array-priors_tr["E"])
        dlp = -np.linalg.inv(priors_tr["V"]).dot(pars_tr_array-E)

        # derivative for transformed parameters wrt parameters -- for jacobian in likelihood
        lj=0
        lj_phi = np.sum(np.log((hyper["phimax"]-hyper["phimin"])/(hyper["phimax"]-phi)/(phi-hyper["phimin"])))
        lj+=lj_phi
        lj_ve = np.log((hyper["vemax"]-hyper["vemin"])/(hyper["vemax"]-ve)/(ve-hyper["vemin"]))
        lj+=lj_ve
        lj_tau = np.log((hyper["taumax"]-hyper["taumin"])/(hyper["taumax"]-tau)/(tau-hyper["taumin"]))
        lj+=lj_tau
        if "s_tr" in pars_tr:
            lj_s = np.sum(np.log((hyper["smax"]-hyper["smin"])/(hyper["smax"]-s)/(s-hyper["smin"])))
            lj+=lj_s
        if "rho_tr" in pars_tr:
            lj_rho =np.log((hyper["rhomax"]-hyper["rhomin"])/(hyper["rhomax"]-rho)/(rho-hyper["rhomin"]))
            lj+=lj_rho

        # derivative for jacobian wrt transformed parameters -- for jacobian in gradients
        dlj=np.array([])
        dlj_phi = (np.exp(pars_tr["phi_tr"])-1)/(np.exp(pars_tr["phi_tr"])+1)
        dlj=np.append(dlj, dlj_phi)
        dlj_ve = (np.exp(pars_tr["ve_tr"])-1)/(np.exp(pars_tr["ve_tr"])+1)
        dlj=np.append(dlj, dlj_ve)
        dlj_tau = (np.exp(pars_tr["tau_tr"])-1)/(np.exp(pars_tr["tau_tr"])+1)
        dlj=np.append(dlj, dlj_tau)
        if "s_tr" in pars_tr:
            dlj_s = (np.exp(pars_tr["s_tr"])-1)/(np.exp(pars_tr["s_tr"])+1)
            dlj=np.append(dlj, dlj_s)
        if "rho_tr" in pars_tr:
            dlj_rho = (np.exp(pars_tr["rho_tr"])-1)/(np.exp(pars_tr["rho_tr"])+1)
            dlj=np.append(dlj, dlj_rho)
        
        lpost = like+lp +lj
        GradLpost = dl*dpars+dlp+dlj

    neglpost = -lpost
    neglgrad = -GradLpost

    return [neglpost, neglgrad]


############# testing
# n=10
# m=20
# X=np.random.rand(n,m)
# coordinates = np.random.rand(n,2)
# site_indices=np.random.randint(3, size=(n, 1))
# Y=np.random.rand(n,1)

n1=10
n2=30
m=20
X=(np.random.rand(n1,m),np.random.rand(n2,m))
coordinates = (np.random.rand(n1,2), np.random.rand(n2,2))
site_indices=(np.random.randint(3, size=(n1, 1)),np.random.randint(3, size=(n2, 1)))
Y=(np.random.rand(n1,1),np.random.rand(n2,1))

hyper={
  "phimin": 1e-50,
  "phimax": 0.95,
  "vemin": 0.0001,
  "vemax": 0.99,
  "taumin": 0.01,
  "taumax": 4.99,
  "smin": 0.0001,
  "smax": 1,
  "rhomin": 0.0001,
  "rhomax": 1
}

pars_tr = {
    "phi_tr": np.random.rand(m),
    "ve_tr": 0.1,
    "tau_tr": 1,
    "s_tr" : 0.1,
    "rho_tr" : 0.9
}

Dist=np.arange(m)
V_phi =np.exp(-Dist**2)
priors ={
  "E_phi" : np.zeros(m),
  # "V_phi" : np.full(m,0.05 ), #naive prior
  "V_phi" : V_phi, #informed prior
  "E_ve" : 0,
  "V_ve" : 1,
  "E_tau" : 0,
  "V_tau" : 5,
  "E_s" : 0,
  "V_s" : 1,
  "E_rho" : 0,
  "V_rho" : 1
}

# priors ={
#   "E_phi" : np.zeros(m),
#   # "V_phi" : np.full(m,0.05 ), #naive prior
#   "V_phi" : V_phi, #informed prior
#   "a_ve" : 2,
#   "b_ve" : 2,
#   "a_tau" : 2,
#   "b_tau" : 2,
#   "a_s" : 2,
#   "b_s" : 2,
#   "a_rho" : 2,
#   "b_rho" : 2
# }

# E=np.random.rand(m+4)
# V=np.random.rand(m+4,m+4)
# V=(V+np.transpose)/2
# priors_tr ={
#   "E" : E,
#   "V" : V
# }

[neglpost, neglgrad] = gradient (X, Y, pars_tr=pars_tr,hyper=hyper,priors=priors, coordinates=coordinates, site_indices=site_indices)