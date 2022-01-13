# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:21:43 2020

@author: sopmathieu

This script applies the monitoring procedure based on the CUSUM chart 
and the support vector machine on the composite (Nc=Ns+10Ng) smoothed on two 
and half years. 

This analysis aims at identifying an upper value in time above which the 
monitoring method would be unvalid/inaproppriate. 
Smoothing the data on 2.5 years appears to work fine.

"""

### load packages/files 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
import sys
sys.path.insert(1, '../Sunspot/')
sys.path.insert(2, '../')

from SunSpot import errors as err
from SunSpot import preprocessing as pre
from SunSpot import cusum_design_bb as chart
from SunSpot import alerts as plot
from SunSpot import svm_training as svm
from SunSpot import autocorrelations as bbl

### load data
with open('data/data_1981', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    Ns = my_depickler.load() #number of spots
    Ng = my_depickler.load() #number of sunspot groups
    Nc = my_depickler.load() #composite: Ns+10Ng
    station_names = my_depickler.load() #codenames of the stations
    time = my_depickler.load() #time
    
### compute the long-term errors
mu2 = err.long_term_error(Ns, period_rescaling=8, wdw=900) #2 years and half

### discard stations with no values
ind_nan = []
for i in range(mu2.shape[1]):
    if not np.all(np.isnan(mu2[:,i])): 
        ind_nan.append(i)
mu2 = mu2[:,ind_nan]
station_names = [station_names[i] for i in range(len(station_names)) if i in ind_nan]
(n_obs, n_series) = mu2.shape

### apply preprocessing
dataNs = pre.PreProcessing(mu2)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels
dataNs.selection_pools(method='kmeans') #select the IC pool
pool = np.array(dataNs.pool) #pool (number)
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)
dataNs.outliers_removal(k=1) #remove outliers

### choose an appropriate value for K
# dataIC = dataNs.dataIC  #IC data without deviations
# data = dataNs.data #data (IC and OC) with deviations
# K = pre.choice_K(data, dataIC, plot=True)

### standardisation
dataNs.standardisation(K=4800) #standardisation of the data
dataIC = dataNs.dataIC  #IC data without deviations
data = dataNs.data #data (IC and OC) with deviations
     
(n_obs, n_series) = mu2.shape
     
#plot the (IC) data
plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("Data IC")
plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

#plot all data
plt.hist(data[~np.isnan(data)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("All Data (IC and OC)")
plt.text(2, 1, 'mean:' '%4f' %np.nanmean(data))
plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(data))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

n_obs_IC = len(dataIC[~np.isnan(dataIC)])*100 / len(data[~np.isnan(data)]) #29 

#=========================================================================
### Design of the chart
#=========================================================================
 
### choice of the block length 
#large range
#bb_length = bbl.block_length_choice(data, bbl_min=10, bbl_max=110, bbl_step=10) #50
#smaller range
#bb_length = bbl.block_length_choice(dataIC, bbl_min=44, bbl_max=64, bbl_step=2) #52
bb_length = 54

delta_min = 1 #intial value for the target shift size
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)

### adjust the control limits
delta_min = chart.shift_size(data, pool, dataIC=dataIC,
                delta=delta_min, ARL0_threshold=ARL0, block_length=bb_length, 
                qt=0.5, missing_values ='reset')[1] #1.7

delta_min = 1.7

control_limit = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                block_length=bb_length, missing_values='reset') #12.65-13.75

control_limit = 13


#=========================================================================
### Train support vector machine classifier and regressor 
#=========================================================================

### compute the control limit of the chart without MVs
control_limit_mv = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=ARL0,
              block_length=bb_length, missing_values='omit') #11.71
                             
control_limit_mv = 11.8

### select the length of the input vector 
wdw_length = svm.input_vector_length(dataIC, delta_min, control_limit_mv,
                                block_length=bb_length) #62_80

wdw_length = 80  

### train and validate the models
n = 21000*3 #number of testing and training instances
scale = 3.5 #scale parameter (~variance) of the halfnormal distribution
n_search = 12000*3 

### find an optimal value for C (regularization parameter)
C_choices = svm.choice_C(dataIC, control_limit_mv, delta_min, wdw_length, scale,
                    start = 5, stop = 15, step = 1,
              delay=True, n=n_search, block_length=bb_length, confusion=False)

C = C_choices[0] 

### train the classifier and regressor with selected C
reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
                wdw_length, scale, delay=True, n=n, C=C, block_length=bb_length)

### save models 
# filename = 'svr_Nc_limit.sav'
# pickle.dump(reg, open(filename, 'wb'))
# filename = 'svc_Nc_limit.sav'
# pickle.dump(clf, open(filename, 'wb'))

### or load the models previously trained
reg = pickle.load(open('svm_models/svr_Nc_limit.sav', 'rb'))
clf = pickle.load(open('svm_models/svc_Nc_limit.sav', 'rb'))


#=========================================================================
### Run the control chart and plot results (with svm predictions)
#=========================================================================


for i in range(n_series): 
    data_indv = data[:,i]
    level_indv = mu2[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = plot.alerts_info(data_indv, control_limit, 
            delta_min, wdw_length, clf, reg)
    
    fig = plot.plot_3panels(data_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, station_names[i], time_start=1981)
        


