# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:21:43 2020

@author: sopmathieu
"""

import pickle
import numpy as np 
import preprocessing as pp 
import CUSUM_design_BB as chart
import SVR_SVC_training as SVM
import plot_performances as plot 


### load data
#with open('Datasets/data_35_47', 'rb') as file:
#     my_depickler = pickle.Unpickler(file)
#     Ns = my_depickler.load() #number of spots
#     Ng = my_depickler.load() #number of sunspot groups
#     Nc = my_depickler.load() #Ns+10Ng
#     station_names = my_depickler.load()
#     time = my_depickler.load() 
     
### apply pre-processing
#dataNs = pp.PreProcessing(Ns, time, station_names) 
#
#dataNs.rescaling() #rescaled Ns with respect to the median of the network
#dataNs.commonSignalRemoval() #remove solar component
#dataNs.movingAverage() #smoothed ratio (remove short-term error)
#level = dataNs.ratio #data with intrinsic levels (mu2) 
#dataNs.levelRemoval() #remove intrisic levels
#dataNs.selectionIC(method='kmeans') #select P1 and P2
#pool = dataNs.pool #P1
#dataNs.outliersRemoval() #adaptive shewhart (remove deviations in P1 and P2)
#dataNs.standardisation() #standardization
#data = dataNs.data #data with deviations
#dataIC = dataNs.dataIC #IC data (P1), without deviations

#####################

### or direcctly load preprocessed data
with open('Datasets/Ns', 'rb') as file:     
     my_depickler = pickle.Unpickler(file)
     data = my_depickler.load() #data with deviations
     time = my_depickler.load() #time
     station_names = my_depickler.load() #code names of the stations
     dataIC = my_depickler.load() #IC stations without deviations
     IC_stations = my_depickler.load() #indexes of IC stations (P1)
     level = my_depickler.load() #stations with intrinsic levels (mu2)

(n_obs, n_series) = data.shape #dimensions 

####################################################
### design of the chart 

delta_min = 1.5 #min shift size to detect
BBL = 27 #block length 
k = delta_min/2 #allowance parameter
ARL0 = 200 #pre-specified ARL0
### Adjust the control limits
#L = chart.search_CUSUM_MV(dataIC, delta=delta_min, block_length=BBL, missing_values='reset') #8.6
L = 8.5
### Compute the performance of the chart
#ARL1 = chart.ARL1_CUSUM_MV(dataIC, L, delta=delta_min, missing_values='reset', 
#                                                       block_length=BBL) #45

### Compute the control limits and appropriate delta
L, delta_min = chart.reccurentDesign_CUSUM(data, IC_stations, dataIC=dataIC, delta=delta_min,
                                           block_length=BBL, missing_values='reset') #8.5-1.6

##################################################"

### train classifier and regressor 
n = 21000*3 #number of testing and training instances
scale = 3.5 #scale parameter of the halfnormal distribution

### compute the control limit of the chart without MV
L_wht_MV = chart.search_CUSUM_MV(dataIC, delta=delta_min, verbose=True, 
                                 block_length=BBL) #11.875
L_wht_MV = 11.9

### select the input vector 
#wdw_length = SVM.selection_input_vector(dataIC, delta_min, L_wht_MV, BBL) #24-26
wdw_length = 25     
     
### train and validate the models
#reg, clf = SVM.training_svr_svm(dataIC, L_wht_MV, delta_min, n, wdw_length,
#                                        scale, BBL)

### save models 
#filename = 'svr_Ns.sav'
#pickle.dump(reg, open(filename, 'wb'))
#filename = 'svc_Ns.sav'
#pickle.dump(clf, open(filename, 'wb'))

## or load the models previsouly trained
reg = pickle.load(open('Models/svr_Ns.sav', 'rb'))
clf = pickle.load(open('Models/svc_Ns.sav', 'rb'))

####################################################"
### run the control chart and plot results (with predictions)

start = int(np.where(time == 1985)[0]) #start of the period (plot)
length = 1800 #length of the period (plot)

for i in range(n_series): 
    data_indv = data[:,i]
    level_indv = level[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = plot.CUSUM_monitoring(data_indv, L, delta_min, wdw_length, clf, reg)
    
    fig = plot.plot_performances(data_indv, level_indv, L, time, form_plus, form_minus, size_plus, 
                          size_minus, C_plus, C_minus, station_names[i], start, length)
        
    #fig.savefig('Figures/Ns/%s.pdf' %station_names[i]) #save figures


