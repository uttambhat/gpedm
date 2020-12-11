import matplotlib.pyplot as plt
import numpy as np
from gp.predict import *
X_train = np.arange(0,10,0.1).reshape(100,1)
y_train = np.sin(X_train)
X = np.arange(0,10,0.5).reshape(20,1)
y = np.sin(X)
ypred = predict(X,X_train,y_train,phi=0.1,tau=0.1)

plt.plot(X_train,y_train)
plt.scatter(X,ypred)
plt.show()

