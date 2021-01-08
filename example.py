import matplotlib.pyplot as plt
import numpy as np
from gp.predict import *

X_train = np.arange(0,10,0.1).reshape(100,1) 
y_train = np.sin(X_train)
X_new = np.arange(0,10,0.5).reshape(20,1)
y_new = np.sin(X_new)
ypred, ycov = predict(X_basis=X_train,X_new=X,Y_basis=y_train,phi=0.5,tau=0.1,Ve=1.e-2,locs=None,rate=None,site=None,rho=None, return_cov=True)
#numerical issues when Ve too small: ycov goes below 0

plt.plot(X_train,y_train)
plt.fill_between(X.flatten(), (ypred-1.96*np.sqrt(ycov)/2).flatten(), (ypred+1.96*np.sqrt(ycov)/2).flatten())
plt.scatter(X,ypred)
plt.show()


log_marginal_likelihood(X=X_train,Y=y_train,phi=0.5,tau=0.1,Ve=1.e-2,locs=None,rate=None,site=None,rho=None)
log_marginal_likelihood(X=(X_train,X_new),Y=(y_train,y_new),phi=0.5,tau=0.1,Ve=1.e-2,locs=None,rate=None,site=None,rho=None)