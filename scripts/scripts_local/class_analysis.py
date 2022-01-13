# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:45:26 2021

@author: sopmathieu

This script studies the shapes of the deviations predicted by the neural networks
as a function of time. 
It also analyses the shapes of the shifts in the different stations. 

"""

import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14
import sys
#0 is the script path or working directory
sys.path.insert(1, '/Sunspot/') 
sys.path.insert(2, '../') 
sys.path.insert(3, '../../') 
from keras.models import model_from_json

from SunSpot import bb_methods as bb
from SunSpot import svm_training as svm

### load data
with open('../../data/Nc_27', 'rb') as fichier:
      my_depickler = pickle.Unpickler(fichier)
      data_stn = my_depickler.load() #contain all deviations, even from IC stations
      time = my_depickler.load() #time
      station_names = my_depickler.load() #code names of the series
      dataIC_stn = my_depickler.load() #IC data without deviations
      pool = my_depickler.load() #pool
      mu2 = my_depickler.load() #mu2 with levels
      mu2_wht_levels = my_depickler.load() #mu2 with levels

(n_obs, n_series) = mu2.shape

data = np.copy(mu2_wht_levels)
dataIC = data[:,pool]


# load json and create model
json_file = open('../../nn_models/nn_regression_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
# load weights into new model
regressor.load_weights("../../nn_models/nn_regression_27.h5")

# load json and create model
json_file = open('../../nn_models/nn_classification_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("../../nn_models/nn_classification_27.h5")


#========================================================================
### classification analysis
#=======================================================================

# #for a particular station
# stat = [i for i in range(len(station_names)) if station_names[i] == 'LO'][0]

# #separate the data from the selected station into blocks
# block_length = 50
# blocks = np.zeros((n_obs, block_length)); blocks[:] = np.nan
# blocks[block_length-1:,:] = bb.MBB(data[:,stat].reshape(-1,1), block_length, NaN=True, all_NaN=False) #mu2[:,stat]

# ## interpolate NaNs in input vectors
# input_valid, ind = svm.fill_nan(blocks)

# ##apply classifier and regressor on (filled-up) input vectors
# size_pred = np.zeros((n_obs,1)); size_pred[:] = np.nan
# shape_pred = np.zeros((n_obs)); shape_pred[:] = np.nan
# if len(ind)>0: #at least one value
#     size_pred[ind] = regressor.predict(input_valid)
#     shape_pred[ind] = classifier.predict_classes(input_valid)
    
# #regressor.predict(input_valid[154,:].reshape(-1,50))

# size_pred = size_pred.reshape(-1)
# perc_year = np.zeros((38, 3))

# for i in range(38):
    
#     start = np.where(time == 1981+i)[0][0]
#     stop = np.where(time == 1982+i)[0][0]
    
#     shape_year = shape_pred[start:stop]
#     l = len(shape_year[~np.isnan(shape_year)])
#     jump = size_pred[np.where(shape_year == 0)[0]]
#     trend = size_pred[np.where(shape_year == 1)[0]]
#     oscill = size_pred[np.where(shape_year == 2)[0]]
    
#     perc_year[i,0] = len(jump)*100/l
#     perc_year[i,1] = len(trend)*100/l
#     perc_year[i,2] = len(oscill)*100/l

# years = np.arange(1981, 2019)
# plt.plot(years, perc_year[:,0], 'r')
# plt.plot(years, perc_year[:,1], 'b')
# plt.plot(years, perc_year[:,2], 'g')
# plt.show()


#=====================================================================
#=====================================================================


block_length = 50
perc_station = np.zeros((n_series, 38, 3)) ; perc_station[:] = np.nan
    
for i in range(n_series): 
    
    blocks = np.zeros((n_obs, block_length)); blocks[:] = np.nan
    blocks[block_length-1:,:] = bb.MBB(data[:,i].reshape(-1,1), block_length, NaN=True, all_NaN=False) 
    
    ## interpolate NaNs in input vectors
    input_valid, ind = svm.fill_nan(blocks)
    
    ##apply classifier and regressor on (filled-up) input vectors
    size_pred = np.zeros((n_obs,1)); size_pred[:] = np.nan
    shape_pred = np.zeros((n_obs)); shape_pred[:] = np.nan
    if len(ind)>0: #at least one value
        size_pred[ind] = regressor.predict(input_valid)
        shape_pred[ind] = classifier.predict_classes(input_valid)
        
        size_pred = size_pred.reshape(-1)

        for j in range(38):
    
            start = np.where(time == 1981 + j)[0][0]
            stop = np.where(time == 1982 + j)[0][0]

            shape_year = shape_pred[start:stop]
            l = len(shape_year[~np.isnan(shape_year)])
            if l > 0: 
                jump = size_pred[np.where(shape_year == 0)[0]]
                trend = size_pred[np.where(shape_year == 1)[0]]
                oscill = size_pred[np.where(shape_year == 2)[0]]
            
                perc_station[i, j, 0] = len(jump)*100/l
                perc_station[i, j, 1] = len(trend)*100/l
                perc_station[i, j, 2] = len(oscill)*100/l


perc_years = np.nanmean(perc_station, axis=0) 
years = np.arange(1981, 2019)
fig = plt.figure()
plt.plot(years, perc_years[:,0], 'o-', label='jump')
plt.plot(years, perc_years[:,1], 'D--', label='drift')
plt.plot(years, perc_years[:,2], 'v:', label='oscill')
plt.xlabel('year')
plt.ylabel('Predicted shapes (in %)')
plt.legend()
plt.show()

#fig.savefig('../../figures/shapes_27.pdf') 


##############################################################################


sub= ['A3', 'BN', 'CA', 'CR', 'GE', 'FU', 'HD', 'HU', 'KH', 
             'KO', 'KS', 'KZ', 'LF', 'LO', 'MA', 'MO', 'PO', 'QU', 
             'SC', 'SK', 'SM', 'UC', 'MT']

#sub= ['CR', 'DB', 'GE', 'FU', 'HD',  'KO', 'KS', 'LF', 'LO', 'MO', 'QU','SC']
#sub = ['CA', 'HU', 'KH', 'KZ', 'MA', 'MT', 'PO', 'SK', 'SM', 'UC']

stats = [i for i in range(n_series) if station_names[i] in sub]
stats_names = [station_names[i]  for i in range(n_series) if station_names[i] in sub]

n_stats = len(stats)
perc = np.zeros((n_series, 3)) 
#perc = np.zeros((n_stats, 3)) 
for i in range(n_series): 
    
    blocks = np.zeros((n_obs, block_length)); blocks[:] = np.nan
    blocks[block_length-1:,:] = bb.MBB(data[:,i].reshape(-1,1), block_length, NaN=True, all_NaN=False) #mu2[:,stat]
    
    ## interpolate NaNs in input vectors
    input_valid, ind = svm.fill_nan(blocks)
    
    ##apply classifier and regressor on (filled-up) input vectors
    size_pred = np.zeros((n_obs,1)); size_pred[:] = np.nan
    shape_pred = np.zeros((n_obs)); shape_pred[:] = np.nan
    if len(ind)>0: #at least one value
        size_pred[ind] = regressor.predict(input_valid)
        shape_pred[ind] = classifier.predict_classes(input_valid)
        
        size_pred = size_pred.reshape(-1)
        l = len(shape_pred[~np.isnan(shape_pred)])
        jump = size_pred[np.where(shape_pred == 0)[0]]
        trend = size_pred[np.where(shape_pred == 1)[0]]
        oscill = size_pred[np.where(shape_pred == 2)[0]]
        
        perc[i, 0] = len(jump)*100/l
        perc[i, 1] = len(trend)*100/l
        perc[i, 2] = len(oscill)*100/l

# x = perc[:,2]
# y = perc[:,1]
# plt.scatter(x, y)
# for i, label in enumerate(station_names):
#     plt.annotate(label, (x[i], y[i]))
# plt.show()

# x = perc[:,0]
# y = perc[:,1]
# plt.scatter(x, y)
# for i, label in enumerate(station_names):
#     plt.annotate(label, (x[i], y[i]))
# plt.show()

# x = perc[:,0]
# y = perc[:,2]
# plt.scatter(x, y)
# for i, label in enumerate(station_names):
#     plt.annotate(label, (x[i], y[i]))
# plt.show()

