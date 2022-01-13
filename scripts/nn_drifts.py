# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:29:24 2021

@author: sopmathieu

This script applies the monitoring procedure based on feed-forward neural 
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
from scipy.ndimage import gaussian_filter1d

from SunSpot import bb_methods as bb
from SunSpot import svm_training as svm
from SunSpot import NN_limits as nn_limit
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

### smooth the data with gaussian filter
# data = np.zeros((n_obs, n_series))
# data[:] = np.nan
# for i in range(n_series):
#     ind = np.where(~np.isnan(mu2_wht_levels[:,i]))
#     data[ind,i] = gaussian_filter1d(mu2_wht_levels[ind,i], 3)

dataIC = data[:,pool]

block_length = 100

### train a regressor
#regressor, scores = nn_train.feed_forward_reg(dataIC, block_length)

### serialize model to JSON (save model)
# model_json = regressor.to_json()
# with open("../nn_models/nn_regression_365_test.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# regressor.save_weights("../nn_models/nn_regression_365_test.h5")


### train a classifier
#classifier, scores, matrix = nn_train.feed_forward_class(dataIC, block_length)

### serialize model to JSON (save model)
# model_json = classifier.to_json()
# with open("../nn_models/nn_classification_365_test.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# classifier.save_weights("../nn_models/nn_classification_365_test.h5")


### load regressor 
json_file = open('nn_models/nn_regression_365.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
# load weights into new model
regressor.load_weights("nn_models/nn_regression_365.h5")

### load classifier 
json_file = open('nn_models/nn_classification_365.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("nn_models/nn_classification_365.h5")

#========================================================================
### Compute the cut-off values of the preditions
#=======================================================================

# h = nn_limit.cutoff(dataIC, regressor, ARL0_threshold=200, L_plus=1,
#                   nmc=1000, n=1000, block_length=block_length, 
#                   BB_method='MBB') #0.1171875

h = 0.11 #fixed value (reproduciability)

### with the heuristic decision procedure
n_L = 30 ; L_plus = 0.5
# h_high, h_low = nn_limit.cutoff_heuristic(dataIC, regressor, L_plus, l_plus=3/4*L_plus,
#                   nmc=1000, n=1000, block_length=block_length, 
#                   BB_method='MBB', n_L=n_L, alternance=True) 

h_high = 0.2119141 #0.20703125
h_low = 0.079834 #0.0791015625


#========================================================================
### Compute the predictions (sizes and shapes) of the networks
#=======================================================================

### for a particular station
stat = [i for i in range(len(station_names)) if station_names[i] == 'MT'][0]

### separate the data from the selected station into blocks
blocks = np.zeros((n_obs, block_length)); blocks[:] = np.nan
blocks[block_length-1:,:] = bb.MBB(data[:,stat].reshape(-1,1), block_length, NaN=True, all_NaN=False) #mu2[:,stat]

### smooth the blocks by gaussian filter
#blocks = gaussian_filter1d(blocks, 2)

### interpolate NaNs in input vectors
input_valid, ind = svm.fill_nan(blocks)

### apply classifier and regressor on (filled-up) input vectors
size_pred = np.zeros((n_obs,1)); size_pred[:] = np.nan
shape_pred = np.zeros((n_obs)); shape_pred[:] = np.nan
if len(ind) > 0: #at least one value
    size_pred[ind] = regressor.predict(input_valid)
    shape_pred[ind] = classifier.predict_classes(input_valid)


#========================================================================
### Plot the networks predictions
#=======================================================================

start = np.where(time == 1981)[0][0]
stop = np.where(time == 2019)[0][0]

#colors for the predictions 
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


fig1 = plt.figure()
f1 = fig1.add_subplot(2, 1, 1)
plt.title("Monitoring in %s" %station_names[stat])
plt.plot(time[start:stop], data[start:stop, stat])
plt.axis([time[start], time[stop]+1, -0.6, 0.6])
plt.plot([time[start], time[stop]+1], [0, 0], 'k-', lw=2)
plt.ylabel('$\hat \mu_2(i,t)$')
f1.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')
#plt.axis([time[start], time[stop]+1, 0,4])

f2 = fig1.add_subplot(2, 1, 2)
plt.plot(time[start:stop], jump[start:stop], '--', c='tab:purple', label='jumps')
plt.plot(time[start:stop], trend[start:stop],  c='tab:green', label='trends')
plt.plot(time[start:stop], oscill[start:stop], ':', c='tab:orange', label='oscill')
plt.plot([time[start], time[stop]+1], [h, h], 'k-', lw=2)
plt.plot([time[start], time[stop]+1], [-h, -h], 'k-', lw=2)
#plt.plot([time[start], time[stop]+1], [h_low, h_low], 'k-', lw=1)
#plt.plot([time[start], time[stop]+1], [-h_low, -h_low], 'k-', lw=1)
plt.axis([time[start], time[stop]+1, -0.6, 0.6])
plt.ylabel('NN predictions')
plt.legend(loc='lower right', ncol=3, fontsize=10)
f2.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')
plt.xlabel('year')
plt.tick_params(axis='x', which='major')
plt.show()

#fig1.savefig('../../figures_Nc/nn_drifts_%s.pdf' %station_names[stat]) #save figure

#===================================================================
### Plot the alerts in red (for the heuristic decision procedure)
#===================================================================

start = np.where(time == 1992)[0][0]
stop = np.where(time == 2005)[0][0]

#colors for the alerts
alerts = np.zeros((n_obs))

for i in range(n_obs):
    if size_pred[i] > h_high:
        alerts[i] = 1 
    elif i > n_L and np.all(size_pred[i-n_L+1:i+1] > h_low):
        alerts[i] = 1
    
for i in range(n_obs):
    if  size_pred[i] < -h_high:
        alerts[i] = 1 
    elif i > n_L and np.all(size_pred[i-n_L+1:i+1] < -h_low):
        alerts[i] = 1 

ic = np.zeros((n_obs)); ic[:] = np.nan
ic[np.where(alerts == 0)[0]] = size_pred[np.where(alerts == 0)[0]]
oc = np.zeros((n_obs)); oc[:] = np.nan
oc[np.where(alerts == 1)[0]] = size_pred[np.where(alerts == 1)[0]]

fig = plt.figure()
f1 = fig.add_subplot(2, 1, 1)
plt.title("Monitoring in %s" %station_names[stat])

plt.plot(time[start:stop], jump[start:stop], '--', c='tab:purple', label='jumps')
plt.plot(time[start:stop], trend[start:stop],  c='tab:green', label='trends')
plt.plot(time[start:stop], oscill[start:stop], ':', c='tab:orange', label='oscill')
plt.plot([time[start], time[stop]+1], [h_high, h_high], 'k-', lw=2)
plt.plot([time[start], time[stop]+1], [-h_high, -h_high], 'k-', lw=2)
plt.plot([time[start], time[stop]+1], [h_low, h_low], 'k-', lw=1)
plt.plot([time[start], time[stop]+1], [-h_low, -h_low], 'k-', lw=1)
plt.axis([time[start], time[stop]+1, -0.45, 0.45])
plt.ylabel('NN predictions')
plt.legend(loc='lower left', ncol=3, fontsize=10)
f1.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')

f2 = fig.add_subplot(2, 1, 2)
plt.plot(time[start:stop], ic[start:stop], '--', c='tab:green', label='IC')
plt.plot(time[start:stop], oc[start:stop],  c='tab:red', label='OC')
plt.plot([time[start], time[stop]+1], [h_high, h_high], 'k-', lw=2)
plt.plot([time[start], time[stop]+1], [-h_high, -h_high], 'k-', lw=2)
plt.plot([time[start], time[stop]+1], [h_low, h_low], 'k-', lw=1)
plt.plot([time[start], time[stop]+1], [-h_low, -h_low], 'k-', lw=1)
plt.axis([time[start], time[stop]+1, -0.45, 0.45])
plt.ylabel('status')
plt.legend(loc='lower left', ncol=3, fontsize=10)
f2.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')
plt.xlabel('year')
plt.tick_params(axis='x', which='major')
plt.show()

#fig.savefig('../../figures_Nc/nn_drifts_%s_2h.pdf' %station_names[stat]) #save figure
