# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:29:45 2021

@author: sopmathieu

This script studies the shapes of the deviations predicted by the support
vector machine as a function of time.  

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

data = np.copy(data_stn)
dataIC = data[:,pool]

reg = pickle.load(open('../../svm_models/svr_Nc_27.sav', 'rb'))
clf = pickle.load(open('../../svm_models/svc_Nc_27.sav', 'rb'))


wdw_length = 70
perc_station = np.zeros((n_series, 38, 3)) ; perc_station[:] = np.nan
    
for i in range(n_series): 
    
    blocks = np.zeros((n_obs, wdw_length)); blocks[:] = np.nan
    blocks[wdw_length-1:,:] = bb.MBB(data[:,i].reshape(-1,1), wdw_length, NaN=True, all_NaN=False) 
    
    ## interpolate NaNs in input vectors
    input_valid, ind = svm.fill_nan(blocks)
    
    ##apply classifier and regressor on (filled-up) input vectors
    size_pred = np.zeros((n_obs)); size_pred[:] = np.nan
    shape_pred = np.zeros((n_obs)); shape_pred[:] = np.nan
    if len(ind)>0: #at least one value
        size_pred[ind] = reg.predict(input_valid)
        shape_pred[ind] = clf.predict(input_valid)
        
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
plt.plot(years, perc_years[:,0], label='jump')
plt.plot(years, perc_years[:,1], label='drift')
plt.plot(years, perc_years[:,2], label='oscill')
plt.xlabel('year')
plt.ylabel('Predicted shapes (in %)')
plt.legend()
plt.show()

#fig.savefig('../../figures/shapes_365.pdf') 
