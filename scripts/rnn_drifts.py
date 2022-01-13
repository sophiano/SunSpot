# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:40:35 2021

@author: sopmathieu

This script applies the monitoring procedure based on recurrent neural 
networks on the composite (Nc=Ns+10Ng) smoothed on a year. 
This analysis highlights the long-term drifts of the stations. 

"""

import pickle
import numpy as np 
import pkg_resources as pkg
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14
from keras.models import model_from_json

from SunSpot import bb_methods as bb
from SunSpot import svm_training as svm
from SunSpot import NN_limits as nn_control
from SunSpot import NN_training as nn_train

### load data (loaded automatically with package)
data_path = pkg.resource_filename(pkg.Requirement.parse("SunSpot"), 'data')

with open(data_path + '/Nc_365', 'rb') as file:  
#with open('data/Nc_365', 'rb') as file: #local path
      my_depickler = pickle.Unpickler(file)
      data_stn = my_depickler.load() #standardized bias ('epsilon_{mu2}')
      time = my_depickler.load() #index of time
      station_names = my_depickler.load() #codenames of the series
      dataIC_stn = my_depickler.load() #IC standardized bias without deviations
      pool = my_depickler.load() #index of IC stations
      mu2 = my_depickler.load() #mu2 with levels ('eh')
      mu2_wht_levels = my_depickler.load() #mu2 without levels ('mu2')

(n_obs, n_series) = mu2.shape

data = np.copy(mu2_wht_levels)
dataIC = data[:,pool]

### compute the length of the input vector of the networks
m =  nn_train.input_in_autocorr(dataIC, 0.75) #close to 100

### set the block length (=length of the input vector of the networks)
block_length = 100

### train a regressor
#regressor, scores = nn_train.recurrent_reg(dataIC, block_length)

### serialize model to JSON (save model)
# model_json = regressor.to_json()
# with open("../nn_models/rnn_regression_365_test.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# regressor.save_weights("../nn_models/rnn_regression_365_test.h5")


### train a classifier
#classifier, scores, matrix = nn_train.recurrent_class(dataIC, block_length)

### serialize model to JSON (save model)
# model_json = classifier.to_json()
# with open("../nn_models/rnn_classification_365_test.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# classifier.save_weights("../nn_models/rnn_classification_365_test.h5")


### load regressor 
json_file = open('nn_models/rnn_regression_365.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
# load weights into new model
regressor.load_weights("nn_models/rnn_regression_365.h5")

### load classifier 
json_file = open('nn_models/rnn_classification_365.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("nn_models/rnn_classification_365.h5")

#========================================================================
### Compute the predictions (sizes and shapes) of the networks
#=======================================================================

### for a particular station
stat = [i for i in range(len(station_names)) if station_names[i] == 'MT'][0]

### separate the data from the selected station into blocks
blocks = np.zeros((n_obs, block_length)); blocks[:] = np.nan
blocks[block_length-1:,:] = bb.MBB(data[:,stat].reshape(-1,1), block_length, NaN=True, all_NaN=False) #mu2[:,stat]

### interpolate NaNs in input vectors
input_valid, ind = svm.fill_nan(blocks)
### reshape input vectors to match input dimensions
input_valid = np.reshape(input_valid, (input_valid.shape[0], 1, input_valid.shape[1]))

### apply classifier and regressor on (filled-up) input vectors
size_pred = np.zeros((n_obs,1)); size_pred[:] = np.nan
shape_pred = np.zeros((n_obs)); shape_pred[:] = np.nan
if len(ind) > 0: #at least one value
    shape_pred[ind] = classifier.predict_classes(input_valid)
    size_pred[ind] = regressor.predict(input_valid)

#========================================================================
### Compute the cut-off values of the networks
#=======================================================================

h = nn_control.rnn_cutoff(dataIC, regressor, L_plus=1, nmc=2000, 
                n=2000, block_length=block_length, BB_method='MBB')#0.0947265625

h = 0.1 #fixed value (reproduciability)


#========================================================================
### Plot the networks predictions
#=======================================================================

start = np.where(time == 1981)[0][0]
stop = np.where(time == 2019)[0][0]

#colors
#shape_pred = shape_pred.reshape(-1)
size_pred = size_pred.reshape(-1)
colorInd = np.where(~np.isnan(shape_pred))[0][:]
color_graph = np.ones((n_obs))*3
color_graph[colorInd] = shape_pred[colorInd]

#jumps
jump = np.zeros((n_obs)); jump[:]=np.nan
jump[np.where(color_graph == 0)[0]] = size_pred[np.where(color_graph == 0)[0]]
#trends
trend = np.zeros((n_obs)); trend[:] = np.nan
trend[np.where(color_graph == 1)[0]] = size_pred[np.where(color_graph == 1)[0]]

#oscillating shifts
oscill = np.zeros((n_obs)); oscill[:] = np.nan
oscill[np.where(color_graph == 2)[0]] = size_pred[np.where(color_graph == 2)[0]]


fig = plt.figure()
f1 = fig.add_subplot(2, 1, 1)
plt.title("Monitoring in %s" %station_names[stat])
plt.plot(time[start:stop], data[start:stop, stat])
plt.axis([time[start], time[stop], -0.5, 0.5])
plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
plt.ylabel('$\hat \mu_2(i,t)$')

f2 = fig.add_subplot(2, 1, 2)
plt.plot(time[start:stop], jump[start:stop], '--', c='tab:purple', label='jumps')
plt.plot(time[start:stop], trend[start:stop],  c='tab:green', label='trends')
plt.plot(time[start:stop], oscill[start:stop], ':', c='tab:orange', label='oscill')
plt.plot([time[start], time[stop]], [h, h], 'k-', lw=2)
plt.plot([time[start], time[stop]], [-h, -h], 'k-', lw=2)
plt.axis([time[start], time[stop], -0.4, 0.4])
plt.ylabel('NN predictions')
plt.legend(loc='upper right', ncol=3)
plt.xlabel('year')
plt.tick_params(axis='x', which='major')
plt.show()