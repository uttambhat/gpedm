import numpy as np
import gp
import edm
from helpers import dataprep, plotter

### Load data ###
data_original = np.loadtxt("example_timeseries_01.txt")

### Normalize and log-transform ###
data,mean,stdev = dataprep.normalize(np.log(data_original))

### Fit parameters ###
x_columns = [0,1,2] #independent variables
y_columns = [0]     #dependent variables
number_of_lags = [2,3,3] #number of time-lags for each independent variable
forecast_steps_ahead = 1 #number of time-steps ahead for forecasting
test_fraction = 0.2 #fraction of data to be used to calculate out-of-sample error (can be set to zero to use all data for training)
x_train,y_train,x_test,y_test = dataprep.construct(data,x_columns,y_columns,number_of_lags,forecast_steps_ahead,test_fraction)

### Fit GP ###
model = gp.gaussian_process_regressor(kernel='sq_exp',optimizer='rprop',prior='ard')
model.fit(x_train,y_train)

### Predict ###
y_predict = model.predict(x_test)

### Plot test vs. predictions ###
plotter.compare(x_test,y_test,y_predict)



